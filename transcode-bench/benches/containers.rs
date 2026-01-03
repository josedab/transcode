//! Container benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use transcode_core::packet::{OwnedPacket, Packet};

fn generate_test_packet(size: usize) -> OwnedPacket {
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    Packet::new(data)
}

fn bench_packet_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("packet_creation");

    for size in &[1024usize, 65536, 1048576] {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(format!("{}KB", size / 1024)),
            size,
            |b, &size| {
                b.iter(|| {
                    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
                    let packet = Packet::new(data);
                    black_box(packet)
                });
            },
        );
    }

    group.finish();
}

fn bench_packet_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("packet_clone");

    for size in &[1024usize, 65536] {
        let packet = generate_test_packet(*size);
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(format!("{}KB", size / 1024)),
            &packet,
            |b, packet| {
                b.iter(|| {
                    let cloned = packet.clone();
                    black_box(cloned)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_packet_creation, bench_packet_clone);
criterion_main!(benches);
