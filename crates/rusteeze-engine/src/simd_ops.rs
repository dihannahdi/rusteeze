//! SIMD-optimized operations for the inference engine.
//!
//! This module provides vectorized implementations of performance-critical
//! operations used in token sampling and probability calculations.
//!
//! # Safety
//!
//! This module contains unsafe code for SIMD intrinsics. All unsafe
//! functions are marked appropriately and have safety documentation.

#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized softmax computation.
/// 
/// This is significantly faster than naive implementations for large vocabularies.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn simd_softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = simd_max(logits);

    // Subtract max and exp
    let mut sum = 0.0f32;
    for x in logits.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for x in logits.iter_mut() {
        *x *= inv_sum;
    }
}

/// SIMD-optimized maximum finding.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn simd_max(arr: &[f32]) -> f32 {
    if arr.is_empty() {
        return f32::NEG_INFINITY;
    }

    // Check for AVX2 support at runtime
    if is_x86_feature_detected!("avx2") {
        unsafe { simd_max_avx2(arr) }
    } else {
        // Fallback to scalar
        arr.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_max_avx2(arr: &[f32]) -> f32 {
    let n = arr.len();
    let chunks = n / 8;
    let ptr = arr.as_ptr();

    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);

    for i in 0..chunks {
        let vals = _mm256_loadu_ps(ptr.add(i * 8));
        max_vec = _mm256_max_ps(max_vec, vals);
    }

    // Horizontal max of the vector
    let mut max_arr = [0.0f32; 8];
    _mm256_storeu_ps(max_arr.as_mut_ptr(), max_vec);
    let mut max_val = max_arr.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Handle remainder
    for i in (chunks * 8)..n {
        max_val = max_val.max(*arr.get_unchecked(i));
    }

    max_val
}

/// SIMD-optimized argmax finding.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn simd_argmax(arr: &[f32]) -> Option<(usize, f32)> {
    if arr.is_empty() {
        return None;
    }

    // For now, use scalar for correctness
    // TODO: Implement full SIMD argmax with index tracking
    arr.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v))
}

/// SIMD-optimized log-sum-exp computation.
/// 
/// Computes log(sum(exp(x))) in a numerically stable way.
#[inline]
pub fn simd_log_sum_exp(arr: &[f32]) -> f32 {
    if arr.is_empty() {
        return f32::NEG_INFINITY;
    }

    let max_val = simd_max(arr);
    if max_val.is_infinite() && max_val < 0.0 {
        return f32::NEG_INFINITY;
    }

    let sum: f32 = arr.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum.ln()
}

/// Vectorized temperature scaling.
#[inline]
pub fn scale_logits_inplace(logits: &mut [f32], temperature: f32) {
    if temperature == 1.0 {
        return;
    }

    let inv_temp = 1.0 / temperature;
    
    // Use SIMD if available
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { scale_logits_avx2(logits, inv_temp) };
        return;
    }

    // Fallback to scalar
    for x in logits.iter_mut() {
        *x *= inv_temp;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_logits_avx2(logits: &mut [f32], inv_temp: f32) {
    let n = logits.len();
    let chunks = n / 8;
    let ptr = logits.as_mut_ptr();
    let scale = _mm256_set1_ps(inv_temp);

    for i in 0..chunks {
        let vals = _mm256_loadu_ps(ptr.add(i * 8));
        let scaled = _mm256_mul_ps(vals, scale);
        _mm256_storeu_ps(ptr.add(i * 8), scaled);
    }

    // Handle remainder
    for i in (chunks * 8)..n {
        *logits.get_unchecked_mut(i) *= inv_temp;
    }
}

/// Fast top-k selection using partial sort.
/// 
/// Returns indices of top-k elements in descending order by value.
#[inline]
pub fn fast_topk(arr: &[f32], k: usize) -> Vec<(usize, f32)> {
    if k == 0 || arr.is_empty() {
        return Vec::new();
    }

    let k = k.min(arr.len());

    // For small k, use simple selection
    if k <= 32 {
        let mut indexed: Vec<(usize, f32)> = arr.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        return indexed;
    }

    // For larger k, use heap-based selection
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    let mut heap: BinaryHeap<Reverse<(ordered_float::NotNan<f32>, usize)>> = BinaryHeap::with_capacity(k + 1);

    for (i, &val) in arr.iter().enumerate() {
        if let Ok(not_nan) = ordered_float::NotNan::new(val) {
            if heap.len() < k {
                heap.push(Reverse((not_nan, i)));
            } else if let Some(&Reverse((min_val, _))) = heap.peek() {
                if not_nan > min_val {
                    heap.pop();
                    heap.push(Reverse((not_nan, i)));
                }
            }
        }
    }

    let mut result: Vec<(usize, f32)> = heap
        .into_iter()
        .map(|Reverse((v, i))| (i, v.into_inner()))
        .collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Nucleus (top-p) sampling preparation.
/// 
/// Returns sorted indices and cumulative probabilities.
#[inline]
pub fn prepare_nucleus_sampling(probs: &[f32], top_p: f32) -> Vec<(usize, f32)> {
    // Sort by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find cutoff
    let mut cumsum = 0.0f32;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum > top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    indexed.truncate(cutoff_idx);
    indexed
}

/// Fast probability renormalization.
#[inline]
pub fn renormalize_probs(probs: &mut [(usize, f32)]) {
    let sum: f32 = probs.iter().map(|x| x.1).sum();
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for p in probs.iter_mut() {
            p.1 *= inv_sum;
        }
    }
}

/// Repetition penalty application.
#[inline]
pub fn apply_repetition_penalty(logits: &mut [f32], generated_tokens: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }

    for &token in generated_tokens {
        let idx = token as usize;
        if idx < logits.len() {
            let logit = logits[idx];
            // Apply penalty: divide positive logits, multiply negative logits
            logits[idx] = if logit > 0.0 {
                logit / penalty
            } else {
                logit * penalty
            };
        }
    }
}

/// Frequency and presence penalty application.
#[inline]
pub fn apply_frequency_presence_penalty(
    logits: &mut [f32],
    token_counts: &std::collections::HashMap<u32, usize>,
    frequency_penalty: f32,
    presence_penalty: f32,
) {
    if frequency_penalty == 0.0 && presence_penalty == 0.0 {
        return;
    }

    for (&token, &count) in token_counts {
        let idx = token as usize;
        if idx < logits.len() {
            logits[idx] -= frequency_penalty * count as f32;
            if count > 0 {
                logits[idx] -= presence_penalty;
            }
        }
    }
}

// Fallback implementations for non-x86_64 platforms
#[cfg(not(target_arch = "x86_64"))]
pub fn simd_softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }

    let max_val: f32 = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in logits.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    let inv_sum = 1.0 / sum;
    for x in logits.iter_mut() {
        *x *= inv_sum;
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_max(arr: &[f32]) -> f32 {
    arr.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_argmax(arr: &[f32]) -> Option<(usize, f32)> {
    arr.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v))
}

#[cfg(not(target_arch = "x86_64"))]
pub fn scale_logits_inplace(logits: &mut [f32], temperature: f32) {
    if temperature == 1.0 {
        return;
    }
    let inv_temp = 1.0 / temperature;
    for x in logits.iter_mut() {
        *x *= inv_temp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_max() {
        let arr = vec![1.0, 5.0, 3.0, 2.0, 4.0, 8.0, 6.0, 7.0, 9.0, 0.0];
        assert_eq!(simd_max(&arr), 9.0);
    }

    #[test]
    fn test_simd_argmax() {
        let arr = vec![1.0, 5.0, 3.0, 9.0, 4.0];
        let (idx, val) = simd_argmax(&arr).unwrap();
        assert_eq!(idx, 3);
        assert_eq!(val, 9.0);
    }

    #[test]
    fn test_simd_softmax() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        simd_softmax_inplace(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fast_topk() {
        let arr = vec![1.0, 5.0, 3.0, 9.0, 4.0, 7.0, 2.0, 8.0, 6.0, 0.0];
        let top3 = fast_topk(&arr, 3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].1, 9.0);
        assert_eq!(top3[1].1, 8.0);
        assert_eq!(top3[2].1, 7.0);
    }

    #[test]
    fn test_log_sum_exp() {
        let arr = vec![1.0, 2.0, 3.0];
        let lse = simd_log_sum_exp(&arr);
        let expected = (arr.iter().map(|x| x.exp()).sum::<f32>()).ln();
        assert!((lse - expected).abs() < 1e-5);
    }
}
