//! # SIMD Operations — Radical Rewrite
//!
//! All operations now use pre-resolved SIMD dispatch from `simd_dispatch`.
//! No more `is_x86_feature_detected!` in inner loops.

use crate::simd_dispatch::{self, dot_product, partial_sort_top_k, softmax_inplace, vec_argmax, vec_max, vec_scale, with_scratch_a};

/// Softmax in-place using pre-resolved SIMD.
pub fn simd_softmax_inplace(logits: &mut [f32]) {
    softmax_inplace(logits);
}

/// Find maximum value in slice using pre-resolved SIMD.
pub fn simd_max(arr: &[f32]) -> f32 {
    vec_max(arr)
}

/// Find argmax — returns (index, value).
pub fn simd_argmax(arr: &[f32]) -> Option<(usize, f32)> {
    if arr.is_empty() { return None; }
    let (idx, val) = vec_argmax(arr);
    Some((idx, val))
}

/// Scale logits in-place: logits[i] /= temperature.
pub fn scale_logits_inplace(logits: &mut [f32], temperature: f32) {
    if temperature == 0.0 || temperature == 1.0 { return; }
    let inv_t = 1.0 / temperature;
    vec_scale(logits, inv_t);
}

/// Fast top-k selection using O(n) partial sort.
/// Returns (token_id, logit) pairs, sorted descending.
pub fn fast_topk(arr: &[f32], k: usize) -> Vec<(usize, f32)> {
    let k = k.min(arr.len());
    if k == 0 { return Vec::new(); }

    with_scratch_a(arr.len() * 2, |scratch| {
        // Build indexed pairs in scratch (reinterpret as usize/f32 pairs via offset)
        let mut indexed: Vec<(usize, f32)> = arr.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        let top = partial_sort_top_k(&mut indexed, k);
        let mut result: Vec<(usize, f32)> = top.to_vec();
        result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    })
}

/// Prepare for nucleus (top-p) sampling.
/// Returns (sorted_indices, sorted_probs, cumulative_probs) for the nucleus.
pub fn prepare_nucleus_sampling(logits: &[f32], top_p: f32) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let n = logits.len();
    if n == 0 { return (Vec::new(), Vec::new(), Vec::new()); }

    // Convert logits to probabilities
    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);

    // Sort by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find nucleus (cumulative sum >= top_p)
    let mut cum = 0.0f32;
    let mut cutoff = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            cutoff = i + 1;
            break;
        }
    }

    let nucleus = &indexed[..cutoff];
    let indices: Vec<usize> = nucleus.iter().map(|&(i, _)| i).collect();
    let sorted_probs: Vec<f32> = nucleus.iter().map(|&(_, p)| p).collect();

    // Cumulative sum
    let mut cumsum = Vec::with_capacity(cutoff);
    let mut s = 0.0;
    for &p in &sorted_probs {
        s += p;
        cumsum.push(s);
    }

    (indices, sorted_probs, cumsum)
}

/// Apply repetition penalty to logits.
pub fn apply_repetition_penalty(logits: &mut [f32], token_ids: &[u32], penalty: f32) {
    if penalty == 1.0 { return; }
    for &tid in token_ids {
        let idx = tid as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Apply frequency and presence penalty.
pub fn apply_frequency_presence_penalty(
    logits: &mut [f32],
    token_counts: &std::collections::HashMap<u32, u32>,
    frequency_penalty: f32,
    presence_penalty: f32,
) {
    for (&tid, &count) in token_counts {
        let idx = tid as usize;
        if idx < logits.len() {
            logits[idx] -= frequency_penalty * count as f32;
            if count > 0 {
                logits[idx] -= presence_penalty;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        simd_softmax_inplace(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmax() {
        let data = vec![1.0, 5.0, 3.0, 10.0, 2.0];
        let (idx, val) = simd_argmax(&data).unwrap();
        assert_eq!(idx, 3);
        assert_eq!(val, 10.0);
    }

    #[test]
    fn test_fast_topk() {
        let data = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.6, 0.7, 0.0];
        let top3 = fast_topk(&data, 3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].0, 3);
        assert_eq!(top3[1].0, 5);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        apply_repetition_penalty(&mut logits, &[1, 3], 2.0);
        assert_eq!(logits[1], 1.0); // 2.0 / 2.0
        assert_eq!(logits[3], 2.0); // 4.0 / 2.0
        assert_eq!(logits[0], 1.0); // unchanged
    }
}
