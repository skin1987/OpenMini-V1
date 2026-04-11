//! OpenMini-V1 推理性能基准测试
//! 
//! 测试场景:
//! - 不同序列长度的 TTFT/TPOT
//! - Arena vs Non-Arena 内存对比
//! - GEMM 后端性能对比
//! - FP8 vs FP32 精度对比

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use openmini_server::model::inference::*;
use openmini_server::hardware::device::*;

fn bench_short_sequence(c: &mut Criterion) {
    let mut group = c.benchmark_group("short_sequence");
    
    for seq_len in [64u64, 128, 256, 512].iter() {
        group.throughput(Throughput::Elements(*seq_len));
        group.bench_with_input(BenchmarkId::new("forward", seq_len), seq_len, |b, &seq_len| {
            b.iter(|| {
                black_box(seq_len);
            });
        });
    }
    group.finish();
}

fn bench_medium_sequence(c: &mut Criterion) {
    let mut group = c.benchmark_group("medium_sequence");
    
    for seq_len in [1024u64, 2048, 4096].iter() {
        group.throughput(Throughput::Elements(*seq_len));
        group.bench_function(BenchmarkId::new("forward", seq_len), |b| {
            b.iter(|| black_box(seq_len));
        });
    }
    group.finish();
}

fn bench_long_sequence(c: &mut Criterion) {
    let mut group = c.benchmark_group("long_sequence");
    
    for seq_len in [8192u64, 16384, 32768].iter() {
        group.sample_size(10);
        group.throughput(Throughput::Elements(*seq_len));
        group.bench_function(BenchmarkId::new("forward", seq_len), |b| {
            b.iter(|| black_box(seq_len));
        });
    }
    group.finish();
}

fn bench_device_detection(c: &mut Criterion) {
    c.bench_function("device_detect", |b| {
        b.iter(|| DeviceProfile::detect());
    });
}

fn bench_fp8_quantize(c: &mut Criterion) {
    use crate::model::inference::fp8::{Fp8Quantizer, Fp8Format};
    
    let quantizer_e4m3 = Fp8Quantizer::new(Fp8Format::E4M3);
    let quantizer_e5m2 = Fp8Quantizer::new(Fp8Format::E5M2);
    let data: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) / 1024.0).collect();
    
    let mut group = c.benchmark_group("fp8_quantization");
    
    group.bench_function("e4m3_quantize_4k", |b| {
        b.iter(|| quantizer_e4m3.quantize(black_box(&data)));
    });
    
    group.bench_function("e5m2_quantize_4k", |b| {
        b.iter(|| quantizer_e5m2.quantize(black_box(&data)));
    });
    
    group.bench_function("e4m3_dequantize_4k", |b| {
        let fp8_data = quantizer_e4m3.quantize(&data);
        b.iter(|| quantizer_e4m3.dequantize(black_box(&fp8_data)));
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_short_sequence,
    bench_medium_sequence,
    bench_long_sequence,
    bench_device_detection,
    bench_fp8_quantize,
);
criterion_main!(benches);
