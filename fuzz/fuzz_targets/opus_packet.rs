#![no_main]

//! Fuzz target for Opus packet parsing.
//!
//! Tests Opus TOC parsing and packet structure validation with arbitrary data.

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use transcode_opus::{OpusDecoder, OpusAudioDecoder};

#[derive(Arbitrary, Debug)]
struct OpusInput {
    data: Vec<u8>,
    sample_rate: OpusSampleRate,
    channels: OpusChannels,
}

#[derive(Arbitrary, Debug, Clone, Copy)]
enum OpusSampleRate {
    Rate8000,
    Rate12000,
    Rate16000,
    Rate24000,
    Rate48000,
}

impl OpusSampleRate {
    fn value(self) -> u32 {
        match self {
            OpusSampleRate::Rate8000 => 8000,
            OpusSampleRate::Rate12000 => 12000,
            OpusSampleRate::Rate16000 => 16000,
            OpusSampleRate::Rate24000 => 24000,
            OpusSampleRate::Rate48000 => 48000,
        }
    }
}

#[derive(Arbitrary, Debug, Clone, Copy)]
enum OpusChannels {
    Mono,
    Stereo,
}

impl OpusChannels {
    fn value(self) -> u8 {
        match self {
            OpusChannels::Mono => 1,
            OpusChannels::Stereo => 2,
        }
    }
}

fuzz_target!(|input: OpusInput| {
    // Limit input size to prevent OOM
    if input.data.len() > 1024 * 1024 {
        return;
    }

    // Skip empty input
    if input.data.is_empty() {
        return;
    }

    // Try to create decoder - may fail for invalid config
    let mut decoder = match OpusDecoder::new(input.sample_rate.value(), input.channels.value()) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Try to decode the fuzzed packet - should never panic
    let packet = transcode_core::Packet::new(input.data);
    let _ = decoder.decode_opus(&packet);
});
