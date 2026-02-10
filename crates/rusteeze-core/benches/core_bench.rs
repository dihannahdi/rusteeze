//! Core benchmarks for Rusteeze
//!
//! Run with: cargo bench -p rusteeze-core

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;

/// Benchmark basic memory operations
fn bench_memory_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");
    group.measurement_time(Duration::from_secs(5));
    
    for size in [1024, 4096, 16384, 65536, 262144] {
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("vec_alloc", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let v: Vec<f32> = vec![0.0; size / 4];
                    black_box(v)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("vec_copy", size),
            &size,
            |b, &size| {
                let src: Vec<f32> = vec![1.0; size / 4];
                b.iter(|| {
                    let dst = src.clone();
                    black_box(dst)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark floating point operations
fn bench_float_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("float_ops");
    group.measurement_time(Duration::from_secs(5));
    
    let size = 1024 * 1024; // 1M elements
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
    
    group.throughput(Throughput::Elements(size as u64));
    
    group.bench_function("sum_iter", |b| {
        b.iter(|| {
            let sum: f32 = data.iter().sum();
            black_box(sum)
        });
    });
    
    group.bench_function("sum_fold", |b| {
        b.iter(|| {
            let sum = data.iter().fold(0.0f32, |acc, &x| acc + x);
            black_box(sum)
        });
    });
    
    group.bench_function("dot_product", |b| {
        let other: Vec<f32> = (0..size).map(|i| (i as f32) * 0.002).collect();
        b.iter(|| {
            let dot: f32 = data.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
            black_box(dot)
        });
    });
    
    group.finish();
}

/// Benchmark softmax operation
fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");
    group.measurement_time(Duration::from_secs(5));
    
    for vocab_size in [32000, 50257, 100000] {
        let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32 % 100.0) - 50.0).collect();
        
        group.throughput(Throughput::Elements(vocab_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("naive", vocab_size),
            &logits,
            |b, logits| {
                b.iter(|| {
                    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
                    let sum: f32 = exp.iter().sum();
                    let probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();
                    black_box(probs)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("online", vocab_size),
            &logits,
            |b, logits| {
                b.iter(|| {
                    // Online softmax (Flash Attention style)
                    let mut max = f32::NEG_INFINITY;
                    let mut sum = 0.0f32;
                    
                    for &x in logits.iter() {
                        if x > max {
                            sum = sum * (max - x).exp() + 1.0;
                            max = x;
                        } else {
                            sum += (x - max).exp();
                        }
                    }
                    
                    let probs: Vec<f32> = logits.iter()
                        .map(|&x| ((x - max).exp() / sum))
                        .collect();
                    
                    black_box(probs)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_memory_ops, bench_float_ops, bench_softmax);
criterion_main!(benches);
