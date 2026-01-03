//! VVC codec benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use transcode_vvc::{
    calculate_ctu_count, VvcConfig, VvcDecoder, VvcDecoderConfig, VvcEncoder, VvcEncoderConfig,
    VvcInfo, VvcLevel, VvcPreset, VvcProfile, VvcTier,
};

fn bench_ctu_count_calculation(c: &mut Criterion) {
    c.bench_function("calculate_ctu_count_1080p_128", |b| {
        b.iter(|| calculate_ctu_count(black_box(1920), black_box(1080), black_box(128)))
    });

    c.bench_function("calculate_ctu_count_4k_128", |b| {
        b.iter(|| calculate_ctu_count(black_box(3840), black_box(2160), black_box(128)))
    });

    c.bench_function("calculate_ctu_count_8k_128", |b| {
        b.iter(|| calculate_ctu_count(black_box(7680), black_box(4320), black_box(128)))
    });
}

fn bench_level_calculation(c: &mut Criterion) {
    use transcode_vvc::calculate_min_level;

    c.bench_function("calculate_min_level_1080p30", |b| {
        b.iter(|| {
            calculate_min_level(
                black_box(1920),
                black_box(1080),
                black_box(30.0),
                black_box(8000),
                black_box(VvcTier::Main),
            )
        })
    });

    c.bench_function("calculate_min_level_4k60", |b| {
        b.iter(|| {
            calculate_min_level(
                black_box(3840),
                black_box(2160),
                black_box(60.0),
                black_box(20000),
                black_box(VvcTier::Main),
            )
        })
    });
}

fn bench_config_creation(c: &mut Criterion) {
    c.bench_function("vvc_config_new", |b| {
        b.iter(|| VvcConfig::new())
    });

    c.bench_function("vvc_config_with_resolution", |b| {
        b.iter(|| {
            VvcConfig::new()
                .with_resolution(black_box(3840), black_box(2160))
                .with_profile(black_box(VvcProfile::Main10))
                .with_level(black_box(VvcLevel::L5_1))
        })
    });

    c.bench_function("vvc_encoder_config_preset_medium", |b| {
        b.iter(|| VvcEncoderConfig::with_preset(black_box(VvcPreset::Medium)))
    });

    c.bench_function("vvc_encoder_config_preset_ultrafast", |b| {
        b.iter(|| VvcEncoderConfig::with_preset(black_box(VvcPreset::Ultrafast)))
    });
}

fn bench_info_queries(c: &mut Criterion) {
    let info = VvcInfo::new();

    c.bench_function("vvc_info_supports_profile", |b| {
        b.iter(|| info.supports_profile(black_box(VvcProfile::Main10)))
    });

    c.bench_function("vvc_info_supports_level", |b| {
        b.iter(|| info.supports_level(black_box(VvcLevel::L5_1)))
    });

    c.bench_function("vvc_info_max_resolution_for_level", |b| {
        b.iter(|| VvcInfo::max_resolution_for_level(black_box(VvcLevel::L5_1)))
    });
}

fn bench_decoder_creation(c: &mut Criterion) {
    let config = VvcDecoderConfig::default();

    c.bench_function("vvc_decoder_new", |b| {
        b.iter(|| VvcDecoder::new(black_box(config.clone())))
    });
}

fn bench_encoder_creation(c: &mut Criterion) {
    c.bench_function("vvc_encoder_new_medium", |b| {
        let config = VvcEncoderConfig::with_preset(VvcPreset::Medium);
        b.iter(|| VvcEncoder::new(black_box(config.clone())))
    });

    c.bench_function("vvc_encoder_new_ultrafast", |b| {
        let config = VvcEncoderConfig::with_preset(VvcPreset::Ultrafast);
        b.iter(|| VvcEncoder::new(black_box(config.clone())))
    });
}

fn bench_profile_level_conversion(c: &mut Criterion) {
    c.bench_function("vvc_profile_from_idc", |b| {
        b.iter(|| VvcProfile::from_idc(black_box(1)))
    });

    c.bench_function("vvc_level_from_idc", |b| {
        b.iter(|| VvcLevel::from_idc(black_box(67)))
    });

    c.bench_function("vvc_tier_from_flag", |b| {
        b.iter(|| VvcTier::from_flag(black_box(true)))
    });
}

criterion_group!(
    benches,
    bench_ctu_count_calculation,
    bench_level_calculation,
    bench_config_creation,
    bench_info_queries,
    bench_decoder_creation,
    bench_encoder_creation,
    bench_profile_level_conversion,
);

criterion_main!(benches);
