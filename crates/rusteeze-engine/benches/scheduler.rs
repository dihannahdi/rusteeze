//! Benchmark for scheduler operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rusteeze_engine::simd_ops::*;
use rand::prelude::*;

fn benchmark_simd_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_max");
    
    for size in [1000, 10000, 50000, 100000, 500000].iter() {
        let data: Vec<f32> = (0..*size).map(|i| (i as f32).sin()).collect();
        
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| simd_max(black_box(&data)))
        });
        
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            b.iter(|| {
                black_box(&data).iter().copied().fold(f32::NEG_INFINITY, f32::max)
            })
        });
    }
    
    group.finish();
}

fn benchmark_simd_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_softmax");
    
    for size in [1000, 10000, 50000, 128256].iter() {  // 128256 = typical LLaMA vocab
        let mut data: Vec<f32> = (0..*size).map(|i| ((i as f32) * 0.01).sin()).collect();
        let original = data.clone();
        
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                data.copy_from_slice(&original);
                simd_softmax_inplace(black_box(&mut data))
            })
        });
        
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            b.iter(|| {
                data.copy_from_slice(&original);
                // Scalar version
                let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for x in data.iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                for x in data.iter_mut() {
                    *x /= sum;
                }
            })
        });
    }
    
    group.finish();
}

fn benchmark_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("topk");
    
    for size in [10000, 50000, 128256].iter() {
        let data: Vec<f32> = {
            let mut rng = StdRng::seed_from_u64(42);
            (0..*size).map(|_| rng.gen::<f32>()).collect()
        };
        
        for k in [10, 50, 100].iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("fast_k{}", k), size), 
                size, 
                |b, _| {
                    b.iter(|| fast_topk(black_box(&data), *k))
                }
            );
            
            group.bench_with_input(
                BenchmarkId::new(format!("sort_k{}", k), size), 
                size, 
                |b, _| {
                    b.iter(|| {
                        let mut indexed: Vec<(usize, f32)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        indexed.truncate(*k);
                        indexed
                    })
                }
            );
        }
    }
    
    group.finish();
}

fn benchmark_log_sum_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_sum_exp");
    
    for size in [1000, 10000, 50000, 128256].iter() {
        let data: Vec<f32> = (0..*size).map(|i| ((i as f32) * 0.01).sin()).collect();
        
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| simd_log_sum_exp(black_box(&data)))
        });
        
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            b.iter(|| {
                let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = data.iter().map(|&x| (x - max).exp()).sum();
                max + sum.ln()
            })
        });
    }
    
    group.finish();
}

fn benchmark_scale_logits(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_logits");
    
    for size in [10000, 50000, 128256].iter() {
        let mut data: Vec<f32> = (0..*size).map(|i| (i as f32).sin()).collect();
        let original = data.clone();
        let temperature = 0.7f32;
        
        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                data.copy_from_slice(&original);
                scale_logits_inplace(black_box(&mut data), temperature)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, _| {
            b.iter(|| {
                data.copy_from_slice(&original);
                let inv_temp = 1.0 / temperature;
                for x in data.iter_mut() {
                    *x *= inv_temp;
                }
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_simd_max,
    benchmark_simd_softmax,
    benchmark_topk,
    benchmark_log_sum_exp,
    benchmark_scale_logits,
);
criterion_main!(benches);
