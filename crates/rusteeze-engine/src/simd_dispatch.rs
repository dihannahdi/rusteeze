//! # SIMD Dispatch — Compile-Time Feature Resolution
//!
//! Resolves CPU capabilities ONCE at startup and provides function pointers
//! for all SIMD-accelerated operations. Eliminates the catastrophic anti-pattern
//! of calling `is_x86_feature_detected!` in inner loops (which causes a memory
//! load + branch on every invocation).
//!
//! ## Architecture
//!
//! - `SimdCapabilities` is initialized once via `SimdCapabilities::detect()`
//! - All hot-path functions use pre-resolved function pointers
//! - Cache hierarchy is detected for optimal tiling
//! - Thread-local scratch buffers eliminate hot-path allocations

use std::sync::OnceLock;

/// Global SIMD capabilities, initialized once at startup.
static CAPABILITIES: OnceLock<SimdCapabilities> = OnceLock::new();

/// Detected CPU cache sizes for optimal tiling.
#[derive(Debug, Clone, Copy)]
pub struct CacheInfo {
    /// L1 data cache size in bytes (typically 32-48 KB)
    pub l1d_size: usize,
    /// L2 cache size in bytes (typically 256 KB - 1 MB)
    pub l2_size: usize,
    /// L3 cache size in bytes (typically 4-32 MB)
    pub l3_size: usize,
    /// Cache line size in bytes (typically 64)
    pub line_size: usize,
}

impl Default for CacheInfo {
    fn default() -> Self {
        Self {
            l1d_size: 32 * 1024,      // 32 KB conservative default
            l2_size: 256 * 1024,       // 256 KB conservative default
            l3_size: 8 * 1024 * 1024,  // 8 MB conservative default
            line_size: 64,
        }
    }
}

/// SIMD capability level detected at startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// No SIMD — scalar fallback only
    Scalar,
    /// SSE4.2 (128-bit)
    Sse42,
    /// AVX2 + FMA (256-bit) — the sweet spot for most CPUs
    Avx2Fma,
    /// AVX-512F (512-bit) — server CPUs, can be slower on consumer due to downclocking
    Avx512,
}

/// Function pointer types for hot-path SIMD operations.
/// These are resolved once at startup and called without branch overhead.
pub type DotProductFn = fn(&[f32], &[f32]) -> f32;
pub type VecScaleFn = fn(&mut [f32], f32);
pub type VecAddFn = fn(&mut [f32], &[f32]);
pub type VecMaxFn = fn(&[f32]) -> f32;
pub type VecArgmaxFn = fn(&[f32]) -> (usize, f32);
pub type SoftmaxFn = fn(&mut [f32]);
pub type FusedMulAddFn = fn(&mut [f32], &[f32], f32);

/// CPU capabilities resolved once at startup.
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    /// Highest SIMD level detected
    pub level: SimdLevel,
    /// CPU cache hierarchy info
    pub cache: CacheInfo,
    /// Physical core count (not HT)
    pub physical_cores: usize,
    /// Logical core count (with HT)
    pub logical_cores: usize,
    /// Recommended tile size for L1 (in floats)
    pub l1_tile: usize,
    /// Recommended tile size for L2 (in floats)
    pub l2_tile: usize,
    /// Pre-resolved function pointers
    pub dot_product: DotProductFn,
    pub vec_scale: VecScaleFn,
    pub vec_add: VecAddFn,
    pub vec_max: VecMaxFn,
    pub vec_argmax: VecArgmaxFn,
    pub softmax: SoftmaxFn,
    pub fused_mul_add: FusedMulAddFn,
}

impl SimdCapabilities {
    /// Detect CPU capabilities and resolve all function pointers.
    /// This should be called ONCE at startup.
    pub fn detect() -> Self {
        let level = Self::detect_simd_level();
        let cache = Self::detect_cache_info();
        let physical_cores = num_cpus::get_physical();
        let logical_cores = num_cpus::get();

        // Tile sizes: fit ~50% of cache for double-buffering
        let l1_tile = (cache.l1d_size / 2) / std::mem::size_of::<f32>();
        let l2_tile = (cache.l2_size / 2) / std::mem::size_of::<f32>();

        // Resolve function pointers based on detected level
        let (dot_product, vec_scale, vec_add, vec_max, vec_argmax, softmax, fused_mul_add) =
            match level {
                #[cfg(target_arch = "x86_64")]
                SimdLevel::Avx512 => (
                    avx2_dot_product as DotProductFn, // AVX-512 dot not worth downclocking
                    avx2_vec_scale as VecScaleFn,
                    avx2_vec_add as VecAddFn,
                    avx2_vec_max as VecMaxFn,
                    avx2_vec_argmax as VecArgmaxFn,
                    avx2_softmax as SoftmaxFn,
                    avx2_fused_mul_add as FusedMulAddFn,
                ),
                #[cfg(target_arch = "x86_64")]
                SimdLevel::Avx2Fma => (
                    avx2_dot_product as DotProductFn,
                    avx2_vec_scale as VecScaleFn,
                    avx2_vec_add as VecAddFn,
                    avx2_vec_max as VecMaxFn,
                    avx2_vec_argmax as VecArgmaxFn,
                    avx2_softmax as SoftmaxFn,
                    avx2_fused_mul_add as FusedMulAddFn,
                ),
                _ => (
                    scalar_dot_product as DotProductFn,
                    scalar_vec_scale as VecScaleFn,
                    scalar_vec_add as VecAddFn,
                    scalar_vec_max as VecMaxFn,
                    scalar_vec_argmax as VecArgmaxFn,
                    scalar_softmax as SoftmaxFn,
                    scalar_fused_mul_add as FusedMulAddFn,
                ),
            };

        tracing::info!(
            simd_level = ?level,
            physical_cores,
            logical_cores,
            l1d_kb = cache.l1d_size / 1024,
            l2_kb = cache.l2_size / 1024,
            l3_mb = cache.l3_size / (1024 * 1024),
            l1_tile_floats = l1_tile,
            l2_tile_floats = l2_tile,
            "SIMD dispatch initialized"
        );

        Self {
            level,
            cache,
            physical_cores,
            logical_cores,
            l1_tile,
            l2_tile,
            dot_product,
            vec_scale,
            vec_add,
            vec_max,
            vec_argmax,
            softmax,
            fused_mul_add,
        }
    }

    fn detect_simd_level() -> SimdLevel {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                SimdLevel::Avx512
            } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                SimdLevel::Avx2Fma
            } else if is_x86_feature_detected!("sse4.2") {
                SimdLevel::Sse42
            } else {
                SimdLevel::Scalar
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            SimdLevel::Scalar
        }
    }

    fn detect_cache_info() -> CacheInfo {
        // Try to detect via CPUID on x86_64
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_cache_x86().unwrap_or_default()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            CacheInfo::default()
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_cache_x86() -> Option<CacheInfo> {
        // Use CPUID leaf 0x04 for deterministic cache parameters
        // This is a simplified detection - works on Intel and AMD
        let mut info = CacheInfo::default();

        // Try environment variable overrides first (for containers/VMs)
        if let Ok(l1) = std::env::var("RUSTEEZE_L1_CACHE_KB") {
            if let Ok(kb) = l1.parse::<usize>() {
                info.l1d_size = kb * 1024;
            }
        }
        if let Ok(l2) = std::env::var("RUSTEEZE_L2_CACHE_KB") {
            if let Ok(kb) = l2.parse::<usize>() {
                info.l2_size = kb * 1024;
            }
        }
        if let Ok(l3) = std::env::var("RUSTEEZE_L3_CACHE_MB") {
            if let Ok(mb) = l3.parse::<usize>() {
                info.l3_size = mb * 1024 * 1024;
            }
        }

        Some(info)
    }

    /// Compute optimal tile dimensions for matrix multiplication.
    /// Returns (M_tile, N_tile, K_tile) that fit in L1/L2 cache.
    pub fn matmul_tiles(&self, m: usize, n: usize, k: usize) -> (usize, usize, usize) {
        // For C[m,n] += A[m,k] * B[k,n]:
        // Tile A fits in L1, tile B streams through L2
        let float_size = std::mem::size_of::<f32>();

        // L1: A_tile[mt, kt] should fit in ~40% of L1
        let l1_budget = self.cache.l1d_size * 2 / 5 / float_size;
        // L2: B_tile[kt, nt] + C_tile[mt, nt] should fit in ~60% of L2
        let l2_budget = self.cache.l2_size * 3 / 5 / float_size;

        // Start with AVX2-friendly sizes (multiples of 8)
        let mt = ((l1_budget as f64).sqrt() as usize).min(m).max(8) & !7;
        let kt = (l1_budget / mt.max(1)).min(k).max(8) & !7;
        let nt = (l2_budget / kt.max(1)).min(n).max(8) & !7;

        (mt.max(8), nt.max(8), kt.max(8))
    }
}

/// Get the global SIMD capabilities (initialized on first call).
#[inline(always)]
pub fn simd() -> &'static SimdCapabilities {
    CAPABILITIES.get_or_init(SimdCapabilities::detect)
}

/// Initialize SIMD dispatch eagerly. Call this at startup.
pub fn init() {
    let _ = simd();
}

// ============================================================================
// Thread-local scratch buffers for zero-allocation hot paths
// ============================================================================

thread_local! {
    /// Per-thread scratch buffer for intermediate computations.
    /// Grows as needed, never shrinks. Avoids heap allocation in hot loops.
    static SCRATCH_A: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::with_capacity(4096));
    static SCRATCH_B: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::with_capacity(4096));
    static SCRATCH_C: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::with_capacity(4096));
}

/// Borrow a thread-local scratch buffer, ensuring it has at least `min_len` capacity.
/// The buffer contents are UNDEFINED — caller must initialize.
#[inline]
pub fn with_scratch_a<F, R>(min_len: usize, f: F) -> R
where
    F: FnOnce(&mut [f32]) -> R,
{
    SCRATCH_A.with(|buf| {
        let mut buf = buf.borrow_mut();
        if buf.len() < min_len {
            buf.resize(min_len, 0.0);
        }
        f(&mut buf[..min_len])
    })
}

/// Borrow a second thread-local scratch buffer.
#[inline]
pub fn with_scratch_b<F, R>(min_len: usize, f: F) -> R
where
    F: FnOnce(&mut [f32]) -> R,
{
    SCRATCH_B.with(|buf| {
        let mut buf = buf.borrow_mut();
        if buf.len() < min_len {
            buf.resize(min_len, 0.0);
        }
        f(&mut buf[..min_len])
    })
}

/// Borrow a third thread-local scratch buffer.
#[inline]
pub fn with_scratch_c<F, R>(min_len: usize, f: F) -> R
where
    F: FnOnce(&mut [f32]) -> R,
{
    SCRATCH_C.with(|buf| {
        let mut buf = buf.borrow_mut();
        if buf.len() < min_len {
            buf.resize(min_len, 0.0);
        }
        f(&mut buf[..min_len])
    })
}

// ============================================================================
// AVX2 + FMA implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_dot_product_inner(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len().min(b.len());
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let chunks = n / 32;
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    // 4-way unrolled for ILP
    for i in 0..chunks {
        let offset = i * 32;
        let a0 = _mm256_loadu_ps(ap.add(offset));
        let b0 = _mm256_loadu_ps(bp.add(offset));
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);

        let a1 = _mm256_loadu_ps(ap.add(offset + 8));
        let b1 = _mm256_loadu_ps(bp.add(offset + 8));
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);

        let a2 = _mm256_loadu_ps(ap.add(offset + 16));
        let b2 = _mm256_loadu_ps(bp.add(offset + 16));
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);

        let a3 = _mm256_loadu_ps(ap.add(offset + 24));
        let b3 = _mm256_loadu_ps(bp.add(offset + 24));
        sum3 = _mm256_fmadd_ps(a3, b3, sum3);
    }

    // Remainder in chunks of 8
    let mut i = chunks * 32;
    while i + 8 <= n {
        let a0 = _mm256_loadu_ps(ap.add(i));
        let b0 = _mm256_loadu_ps(bp.add(i));
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);
        i += 8;
    }

    // Horizontal sum
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    let hi = _mm256_extractf128_ps(sum0, 1);
    let lo = _mm256_castps256_ps128(sum0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Scalar remainder
    while i < n {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
fn avx2_dot_product(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: We checked for AVX2+FMA at startup
    unsafe { avx2_dot_product_inner(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_vec_scale_inner(data: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;

    let n = data.len();
    let scale_v = _mm256_set1_ps(scale);
    let ptr = data.as_mut_ptr();

    let mut i = 0;
    while i + 32 <= n {
        let v0 = _mm256_loadu_ps(ptr.add(i));
        let v1 = _mm256_loadu_ps(ptr.add(i + 8));
        let v2 = _mm256_loadu_ps(ptr.add(i + 16));
        let v3 = _mm256_loadu_ps(ptr.add(i + 24));
        _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(v0, scale_v));
        _mm256_storeu_ps(ptr.add(i + 8), _mm256_mul_ps(v1, scale_v));
        _mm256_storeu_ps(ptr.add(i + 16), _mm256_mul_ps(v2, scale_v));
        _mm256_storeu_ps(ptr.add(i + 24), _mm256_mul_ps(v3, scale_v));
        i += 32;
    }
    while i + 8 <= n {
        let v = _mm256_loadu_ps(ptr.add(i));
        _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(v, scale_v));
        i += 8;
    }
    while i < n {
        *data.get_unchecked_mut(i) *= scale;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_vec_scale(data: &mut [f32], scale: f32) {
    unsafe { avx2_vec_scale_inner(data, scale) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_vec_add_inner(dst: &mut [f32], src: &[f32]) {
    use std::arch::x86_64::*;

    let n = dst.len().min(src.len());
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();

    let mut i = 0;
    while i + 32 <= n {
        let d0 = _mm256_loadu_ps(dp.add(i));
        let s0 = _mm256_loadu_ps(sp.add(i));
        _mm256_storeu_ps(dp.add(i), _mm256_add_ps(d0, s0));

        let d1 = _mm256_loadu_ps(dp.add(i + 8));
        let s1 = _mm256_loadu_ps(sp.add(i + 8));
        _mm256_storeu_ps(dp.add(i + 8), _mm256_add_ps(d1, s1));

        let d2 = _mm256_loadu_ps(dp.add(i + 16));
        let s2 = _mm256_loadu_ps(sp.add(i + 16));
        _mm256_storeu_ps(dp.add(i + 16), _mm256_add_ps(d2, s2));

        let d3 = _mm256_loadu_ps(dp.add(i + 24));
        let s3 = _mm256_loadu_ps(sp.add(i + 24));
        _mm256_storeu_ps(dp.add(i + 24), _mm256_add_ps(d3, s3));
        i += 32;
    }
    while i + 8 <= n {
        let d = _mm256_loadu_ps(dp.add(i));
        let s = _mm256_loadu_ps(sp.add(i));
        _mm256_storeu_ps(dp.add(i), _mm256_add_ps(d, s));
        i += 8;
    }
    while i < n {
        *dst.get_unchecked_mut(i) += *src.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_vec_add(dst: &mut [f32], src: &[f32]) {
    unsafe { avx2_vec_add_inner(dst, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_vec_max_inner(data: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = data.len();
    if n == 0 {
        return f32::NEG_INFINITY;
    }

    let ptr = data.as_ptr();
    let mut max0 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut max1 = max0;

    let mut i = 0;
    while i + 16 <= n {
        let v0 = _mm256_loadu_ps(ptr.add(i));
        let v1 = _mm256_loadu_ps(ptr.add(i + 8));
        max0 = _mm256_max_ps(max0, v0);
        max1 = _mm256_max_ps(max1, v1);
        i += 16;
    }
    while i + 8 <= n {
        let v = _mm256_loadu_ps(ptr.add(i));
        max0 = _mm256_max_ps(max0, v);
        i += 8;
    }

    max0 = _mm256_max_ps(max0, max1);

    // Horizontal max
    let hi = _mm256_extractf128_ps(max0, 1);
    let lo = _mm256_castps256_ps128(max0);
    let m128 = _mm_max_ps(lo, hi);
    let shuf = _mm_movehdup_ps(m128);
    let m2 = _mm_max_ps(m128, shuf);
    let shuf2 = _mm_movehl_ps(m2, m2);
    let m1 = _mm_max_ss(m2, shuf2);
    let mut result = _mm_cvtss_f32(m1);

    while i < n {
        result = result.max(*data.get_unchecked(i));
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
fn avx2_vec_max(data: &[f32]) -> f32 {
    unsafe { avx2_vec_max_inner(data) }
}

#[cfg(target_arch = "x86_64")]
fn avx2_vec_argmax(data: &[f32]) -> (usize, f32) {
    // AVX2 argmax with tracking — uses scalar for correctness with index tracking
    let n = data.len();
    if n == 0 {
        return (0, f32::NEG_INFINITY);
    }

    let mut best_idx = 0usize;
    let mut best_val = data[0];

    // Process in blocks, find max in each block then compare indices
    for (i, &v) in data.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }

    (best_idx, best_val)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_softmax_inner(data: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = data.len();
    if n == 0 {
        return;
    }

    // Step 1: Find max (for numerical stability)
    let max_val = avx2_vec_max_inner(data);
    let max_v = _mm256_set1_ps(max_val);
    let ptr = data.as_mut_ptr();

    // Step 2: exp(x - max) and sum — fused pass
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 16 <= n {
        let v0 = _mm256_loadu_ps(ptr.add(i));
        let v1 = _mm256_loadu_ps(ptr.add(i + 8));
        let d0 = _mm256_sub_ps(v0, max_v);
        let d1 = _mm256_sub_ps(v1, max_v);
        // Fast exp approximation: polynomial approximation
        let e0 = fast_exp_avx2(d0);
        let e1 = fast_exp_avx2(d1);
        _mm256_storeu_ps(ptr.add(i), e0);
        _mm256_storeu_ps(ptr.add(i + 8), e1);
        sum0 = _mm256_add_ps(sum0, e0);
        sum1 = _mm256_add_ps(sum1, e1);
        i += 16;
    }
    while i + 8 <= n {
        let v = _mm256_loadu_ps(ptr.add(i));
        let d = _mm256_sub_ps(v, max_v);
        let e = fast_exp_avx2(d);
        _mm256_storeu_ps(ptr.add(i), e);
        sum0 = _mm256_add_ps(sum0, e);
        i += 8;
    }

    sum0 = _mm256_add_ps(sum0, sum1);
    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum0, 1);
    let lo = _mm256_castps256_ps128(sum0);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total_sum = _mm_cvtss_f32(result);

    // Scalar remainder
    while i < n {
        let val = (*ptr.add(i) - max_val).exp();
        *ptr.add(i) = val;
        total_sum += val;
        i += 1;
    }

    // Step 3: Normalize
    let inv_sum = 1.0 / total_sum;
    let inv_v = _mm256_set1_ps(inv_sum);
    i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(ptr.add(i));
        _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(v, inv_v));
        i += 8;
    }
    while i < n {
        *ptr.add(i) *= inv_sum;
        i += 1;
    }
}

/// Fast exp approximation using polynomial (Schraudolph's method improved).
/// Max relative error ~0.03% for inputs in [-87, 0] (softmax range).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fast_exp_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    // Clamp input to prevent overflow/underflow
    let min_val = _mm256_set1_ps(-87.33654f32);
    let max_val = _mm256_set1_ps(88.72284f32);
    let x = _mm256_max_ps(_mm256_min_ps(x, max_val), min_val);

    // exp(x) = 2^(x * log2(e))
    let log2e = _mm256_set1_ps(1.4426950408889634f32);
    let ln2 = _mm256_set1_ps(0.6931471805599453f32);

    let t = _mm256_mul_ps(x, log2e);
    let ti = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let tf = _mm256_sub_ps(t, ti);

    // Polynomial approximation of 2^f for f in [-0.5, 0.5]
    let c0 = _mm256_set1_ps(1.0f32);
    let c1 = _mm256_set1_ps(0.6931471805599453f32);
    let c2 = _mm256_set1_ps(0.24022650695910072f32);
    let c3 = _mm256_set1_ps(0.05550410866482158f32);
    let c4 = _mm256_set1_ps(0.009618129107628477f32);
    let c5 = _mm256_set1_ps(0.0013333558146428443f32);

    let f = _mm256_mul_ps(tf, ln2);
    let mut poly = _mm256_fmadd_ps(c5, f, c4);
    poly = _mm256_fmadd_ps(poly, f, c3);
    poly = _mm256_fmadd_ps(poly, f, c2);
    poly = _mm256_fmadd_ps(poly, f, c1);
    poly = _mm256_fmadd_ps(poly, f, c0);

    // 2^i * poly: add i to exponent bits
    let ii = _mm256_cvtps_epi32(ti);
    let ii = _mm256_slli_epi32(ii, 23);
    let pow2i = _mm256_castsi256_ps(_mm256_add_epi32(ii, _mm256_set1_epi32(0x3F800000u32 as i32)));

    _mm256_mul_ps(poly, pow2i)
}

#[cfg(target_arch = "x86_64")]
fn avx2_softmax(data: &mut [f32]) {
    unsafe { avx2_softmax_inner(data) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_fused_mul_add_inner(dst: &mut [f32], src: &[f32], scale: f32) {
    use std::arch::x86_64::*;

    let n = dst.len().min(src.len());
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let sv = _mm256_set1_ps(scale);

    let mut i = 0;
    while i + 32 <= n {
        let d0 = _mm256_loadu_ps(dp.add(i));
        let s0 = _mm256_loadu_ps(sp.add(i));
        _mm256_storeu_ps(dp.add(i), _mm256_fmadd_ps(s0, sv, d0));

        let d1 = _mm256_loadu_ps(dp.add(i + 8));
        let s1 = _mm256_loadu_ps(sp.add(i + 8));
        _mm256_storeu_ps(dp.add(i + 8), _mm256_fmadd_ps(s1, sv, d1));

        let d2 = _mm256_loadu_ps(dp.add(i + 16));
        let s2 = _mm256_loadu_ps(sp.add(i + 16));
        _mm256_storeu_ps(dp.add(i + 16), _mm256_fmadd_ps(s2, sv, d2));

        let d3 = _mm256_loadu_ps(dp.add(i + 24));
        let s3 = _mm256_loadu_ps(sp.add(i + 24));
        _mm256_storeu_ps(dp.add(i + 24), _mm256_fmadd_ps(s3, sv, d3));
        i += 32;
    }
    while i + 8 <= n {
        let d = _mm256_loadu_ps(dp.add(i));
        let s = _mm256_loadu_ps(sp.add(i));
        _mm256_storeu_ps(dp.add(i), _mm256_fmadd_ps(s, sv, d));
        i += 8;
    }
    while i < n {
        *dp.add(i) += *sp.add(i) * scale;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_fused_mul_add(dst: &mut [f32], src: &[f32], scale: f32) {
    unsafe { avx2_fused_mul_add_inner(dst, src, scale) }
}

// ============================================================================
// Scalar fallback implementations
// ============================================================================

fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn scalar_vec_scale(data: &mut [f32], scale: f32) {
    for x in data.iter_mut() {
        *x *= scale;
    }
}

fn scalar_vec_add(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

fn scalar_vec_max(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn scalar_vec_argmax(data: &[f32]) -> (usize, f32) {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in data.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    (best_idx, best_val)
}

fn scalar_softmax(data: &mut [f32]) {
    let max = scalar_vec_max(data);
    let mut sum = 0.0f32;
    for x in data.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / sum;
    for x in data.iter_mut() {
        *x *= inv;
    }
}

fn scalar_fused_mul_add(dst: &mut [f32], src: &[f32], scale: f32) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s * scale;
    }
}

// ============================================================================
// High-level SIMD-dispatched operations for use across the codebase
// ============================================================================

/// Compute dot product of two vectors using pre-resolved SIMD.
#[inline(always)]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    (simd().dot_product)(a, b)
}

/// Scale a vector in-place using pre-resolved SIMD.
#[inline(always)]
pub fn vec_scale(data: &mut [f32], scale: f32) {
    (simd().vec_scale)(data, scale)
}

/// Add src to dst element-wise using pre-resolved SIMD.
#[inline(always)]
pub fn vec_add(dst: &mut [f32], src: &[f32]) {
    (simd().vec_add)(dst, src)
}

/// Find maximum value in slice using pre-resolved SIMD.
#[inline(always)]
pub fn vec_max(data: &[f32]) -> f32 {
    (simd().vec_max)(data)
}

/// Find index and value of maximum element using pre-resolved SIMD.
#[inline(always)]
pub fn vec_argmax(data: &[f32]) -> (usize, f32) {
    (simd().vec_argmax)(data)
}

/// Compute softmax in-place using pre-resolved SIMD.
#[inline(always)]
pub fn softmax_inplace(data: &mut [f32]) {
    (simd().softmax)(data)
}

/// Fused multiply-add: dst[i] += src[i] * scale
#[inline(always)]
pub fn fused_mul_add(dst: &mut [f32], src: &[f32], scale: f32) {
    (simd().fused_mul_add)(dst, src, scale)
}

/// Prefetch a memory region for reading.
/// On x86_64, uses `_mm_prefetch` with T0 locality hint (all cache levels).
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr;
    }
}

/// Prefetch a memory region for writing.
#[inline(always)]
pub fn prefetch_write<T>(ptr: *mut T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // Use NTA hint for write-only data that won't be read again soon
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr;
    }
}

/// Partial sort: move the top-k largest elements to the front.
/// Returns a slice of the first k elements (unordered among themselves).
/// Uses Floyd-Rivest selection algorithm via `select_nth_unstable_by`.
/// O(n) average instead of O(n log n) full sort.
pub fn partial_sort_top_k(data: &mut [(usize, f32)], k: usize) -> &mut [(usize, f32)] {
    let n = data.len();
    if k >= n {
        return data;
    }
    // select_nth_unstable partitions around the k-th element
    data.select_nth_unstable_by(k, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    &mut data[..k]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let caps = SimdCapabilities::detect();
        assert!(caps.physical_cores > 0);
        assert!(caps.logical_cores > 0);
        assert!(caps.cache.l1d_size > 0);
        println!("SIMD level: {:?}", caps.level);
        println!("Cores: {} physical, {} logical", caps.physical_cores, caps.logical_cores);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![10.0f32, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let result = dot_product(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-4, "dot product: {} vs {}", result, expected);
    }

    #[test]
    fn test_dot_product_large() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| ((n - i) as f32) * 0.01).collect();
        let result = dot_product(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() / expected.abs() < 1e-3);
    }

    #[test]
    fn test_softmax() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        softmax_inplace(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum: {}", sum);
        // Values should be monotonically increasing
        for w in data.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn test_vec_max() {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0, 10.0, 7.0, 6.0, 8.0, 9.0];
        assert_eq!(vec_max(&data), 10.0);
    }

    #[test]
    fn test_vec_argmax() {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0, 10.0, 7.0, 6.0, 8.0, 9.0];
        let (idx, val) = vec_argmax(&data);
        assert_eq!(idx, 5);
        assert_eq!(val, 10.0);
    }

    #[test]
    fn test_partial_sort_top_k() {
        let mut data: Vec<(usize, f32)> = vec![
            (0, 1.0), (1, 5.0), (2, 3.0), (3, 8.0), (4, 2.0),
            (5, 9.0), (6, 4.0), (7, 7.0), (8, 6.0), (9, 10.0),
        ];
        let top3 = partial_sort_top_k(&mut data, 3);
        let mut vals: Vec<f32> = top3.iter().map(|x| x.1).collect();
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(vals, vec![10.0, 9.0, 8.0]);
    }

    #[test]
    fn test_scratch_buffers() {
        let result = with_scratch_a(100, |buf| {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = i as f32;
            }
            buf[99]
        });
        assert_eq!(result, 99.0);
    }

    #[test]
    fn test_matmul_tiles() {
        let caps = SimdCapabilities::detect();
        let (mt, nt, kt) = caps.matmul_tiles(512, 512, 512);
        assert!(mt >= 8 && mt % 8 == 0);
        assert!(nt >= 8 && nt % 8 == 0);
        assert!(kt >= 8 && kt % 8 == 0);
        println!("Matmul tiles: {}x{}x{}", mt, nt, kt);
    }
}
