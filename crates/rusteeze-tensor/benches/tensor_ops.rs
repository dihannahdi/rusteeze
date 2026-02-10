//! Tensor operations benchmarks
//!
//! Run with: cargo bench -p rusteeze-tensor

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;

/// Benchmark tensor creation
fn bench_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");
    group.measurement_time(Duration::from_secs(5));
    
    for size in [1024, 4096, 16384, 65536] {
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("zeros", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let t: Vec<f32> = vec![0.0; size];
                    black_box(t)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("ones", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let t: Vec<f32> = vec![1.0; size];
                    black_box(t)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark matrix multiplication
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    group.measurement_time(Duration::from_secs(5));
    
    for size in [128, 256, 512] {
        let flops = 2 * size * size * size;
        group.throughput(Throughput::Elements(flops as u64));
        
        let a: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.002).collect();
        
        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let mut c = vec![0.0f32; size * size];
                    for i in 0..size {
                        for j in 0..size {
                            for k in 0..size {
                                c[i * size + j] += a[i * size + k] * b[k * size + j];
                            }
                        }
                    }
                    black_box(c)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark element-wise operations
fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise");
    group.measurement_time(Duration::from_secs(5));
    
    let size = 1024 * 1024;
    let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.002).collect();
    
    group.throughput(Throughput::Elements(size as u64));
    
    group.bench_function("add", |bench| {
        bench.iter(|| {
            let c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
            black_box(c)
        });
    });
    
    group.bench_function("mul", |bench| {
        bench.iter(|| {
            let c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
            black_box(c)
        });
    });
    
    group.bench_function("fma", |bench| {
        bench.iter(|| {
            let c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x.mul_add(*y, 1.0)).collect();
            black_box(c)
        });
    });
    
    group.finish();
}

criterion_group!(benches, bench_tensor_creation, bench_matmul, bench_elementwise);
criterion_main!(benches);
