/**
 * @file transcode.h
 * @brief Transcode C API - A memory-safe, high-performance codec library
 * @version 0.1.0
 * @author Transcode Contributors
 * @license MIT OR Apache-2.0
 *
 * @mainpage Transcode C API Documentation
 *
 * @section intro Introduction
 *
 * Transcode is a memory-safe, high-performance universal codec library written
 * in Rust with a C-compatible API. It provides FFmpeg-like functionality with
 * modern safety guarantees.
 *
 * @section features Key Features
 *
 * - **Memory Safety**: Written in Rust, preventing buffer overflows and memory leaks
 * - **High Performance**: SIMD-accelerated codec implementations
 * - **FFmpeg-like API**: Familiar packet/frame-based workflow
 * - **Multiple Codecs**: H.264, AAC, MP3, AV1, HEVC, VP9, Opus, and more
 * - **Container Support**: MP4, MKV, MPEG-TS, HLS, DASH
 *
 * @section quickstart Quick Start
 *
 * @code{.c}
 * #include <transcode.h>
 * #include <stdio.h>
 *
 * int main() {
 *     TranscodeContext* ctx = NULL;
 *     TranscodeError err;
 *
 *     // Open input file
 *     err = transcode_open_input("input.mp4", &ctx);
 *     if (err != TRANSCODE_SUCCESS) {
 *         fprintf(stderr, "Error: %s\n", transcode_error_string(err));
 *         return 1;
 *     }
 *
 *     // Print stream information
 *     printf("Found %u streams\n", ctx->num_streams);
 *     for (uint32_t i = 0; i < ctx->num_streams; i++) {
 *         TranscodeStreamInfo info;
 *         transcode_get_stream_info(ctx, i, &info);
 *         if (info.stream_type == TRANSCODE_STREAM_VIDEO) {
 *             printf("Video: %ux%u\n", info.width, info.height);
 *         } else if (info.stream_type == TRANSCODE_STREAM_AUDIO) {
 *             printf("Audio: %u Hz, %u channels\n", info.sample_rate, info.channels);
 *         }
 *     }
 *
 *     // Read and process packets
 *     TranscodePacket* packet = transcode_packet_alloc();
 *     TranscodeFrame* frame = transcode_frame_alloc();
 *
 *     while ((err = transcode_read_packet(ctx, packet)) == TRANSCODE_SUCCESS) {
 *         if (transcode_decode_packet(ctx, packet, frame) == TRANSCODE_SUCCESS) {
 *             // Process decoded frame...
 *             printf("Decoded frame: PTS=%lld\n", frame->pts);
 *         }
 *     }
 *
 *     // Cleanup
 *     transcode_frame_free(frame);
 *     transcode_packet_free(packet);
 *     transcode_close(ctx);
 *
 *     return 0;
 * }
 * @endcode
 *
 * @section transcoding Full Transcoding Example
 *
 * @code{.c}
 * #include <transcode.h>
 *
 * int transcode_file(const char* input_path, const char* output_path) {
 *     TranscodeContext* ctx = NULL;
 *     TranscodeConfig config = {0};
 *     TranscodePacket* in_packet = NULL;
 *     TranscodePacket* out_packet = NULL;
 *     TranscodeFrame* frame = NULL;
 *     TranscodeError err;
 *
 *     // Open input
 *     err = transcode_open_input(input_path, &ctx);
 *     if (err != TRANSCODE_SUCCESS) goto cleanup;
 *
 *     // Configure output encoding
 *     config.width = 1280;
 *     config.height = 720;
 *     config.bitrate = 2000000;  // 2 Mbps
 *     config.quality = 23;       // CRF 23
 *     config.preset = 5;         // Medium preset
 *
 *     // Open output
 *     err = transcode_open_output(ctx, output_path, &config);
 *     if (err != TRANSCODE_SUCCESS) goto cleanup;
 *
 *     // Allocate working structures
 *     in_packet = transcode_packet_alloc();
 *     out_packet = transcode_packet_alloc();
 *     frame = transcode_frame_alloc();
 *
 *     // Transcode loop
 *     while ((err = transcode_read_packet(ctx, in_packet)) == TRANSCODE_SUCCESS) {
 *         // Decode
 *         err = transcode_decode_packet(ctx, in_packet, frame);
 *         if (err != TRANSCODE_SUCCESS) continue;
 *
 *         // Encode
 *         err = transcode_encode_frame(ctx, frame, out_packet);
 *         if (err != TRANSCODE_SUCCESS) continue;
 *
 *         // Write output
 *         err = transcode_write_packet(ctx, out_packet);
 *         if (err != TRANSCODE_SUCCESS) goto cleanup;
 *     }
 *
 *     // Flush encoder
 *     while (transcode_encode_frame(ctx, NULL, out_packet) == TRANSCODE_SUCCESS) {
 *         transcode_write_packet(ctx, out_packet);
 *     }
 *
 *     err = TRANSCODE_SUCCESS;
 *
 * cleanup:
 *     transcode_frame_free(frame);
 *     transcode_packet_free(out_packet);
 *     transcode_packet_free(in_packet);
 *     transcode_close(ctx);
 *     return err;
 * }
 * @endcode
 *
 * @section building Building and Linking
 *
 * @subsection static Static Linking
 * @code{.sh}
 * # Build the library
 * cargo build --release -p transcode-capi
 *
 * # Link with your application
 * gcc -o myapp myapp.c -L target/release -ltranscode_capi -lpthread -ldl -lm
 * @endcode
 *
 * @subsection dynamic Dynamic Linking
 * @code{.sh}
 * # Build as shared library
 * cargo build --release -p transcode-capi
 *
 * # On Linux, set library path
 * export LD_LIBRARY_PATH=./target/release:$LD_LIBRARY_PATH
 *
 * # Link and run
 * gcc -o myapp myapp.c -L target/release -ltranscode_capi
 * ./myapp
 * @endcode
 */

#ifndef TRANSCODE_H
#define TRANSCODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* ============================================================================
 * Version Information
 * ============================================================================ */

/** Major version number */
#define TRANSCODE_VERSION_MAJOR 0

/** Minor version number */
#define TRANSCODE_VERSION_MINOR 1

/** Patch version number */
#define TRANSCODE_VERSION_PATCH 0

/** Full version string */
#define TRANSCODE_VERSION_STRING "0.1.0"

/* ============================================================================
 * Error Codes
 * ============================================================================ */

/**
 * @brief Error codes returned by transcode functions.
 *
 * All transcode functions return an error code. TRANSCODE_SUCCESS (0)
 * indicates success, while negative values indicate various error conditions.
 */
typedef enum TranscodeError {
    /** Operation completed successfully. */
    TRANSCODE_SUCCESS = 0,

    /** Invalid argument provided to function. */
    TRANSCODE_ERROR_INVALID_ARGUMENT = -1,

    /** Null pointer provided where non-null was expected. */
    TRANSCODE_ERROR_NULL_POINTER = -2,

    /** End of stream reached (not an error, just indicates completion). */
    TRANSCODE_ERROR_END_OF_STREAM = -3,

    /** I/O error occurred (file not found, permission denied, etc.). */
    TRANSCODE_ERROR_IO = -4,

    /** Codec error (corrupt data, unsupported profile, etc.). */
    TRANSCODE_ERROR_CODEC = -5,

    /** Container format error (invalid structure, missing atoms, etc.). */
    TRANSCODE_ERROR_CONTAINER = -6,

    /** Resource exhausted (out of memory, too many handles, etc.). */
    TRANSCODE_ERROR_RESOURCE_EXHAUSTED = -7,

    /** Unsupported feature or format requested. */
    TRANSCODE_ERROR_UNSUPPORTED = -8,

    /** Operation was cancelled by the user. */
    TRANSCODE_ERROR_CANCELLED = -9,

    /** Buffer too small to hold the requested data. */
    TRANSCODE_ERROR_BUFFER_TOO_SMALL = -10,

    /** Invalid state for this operation (e.g., calling decode before open). */
    TRANSCODE_ERROR_INVALID_STATE = -11,

    /** Unknown or internal error. */
    TRANSCODE_ERROR_UNKNOWN = -100
} TranscodeError;

/* ============================================================================
 * Pixel Formats
 * ============================================================================ */

/**
 * @brief Pixel format for video frames.
 *
 * Specifies the memory layout and color representation of video frame data.
 */
typedef enum TranscodePixelFormat {
    /** Unknown or unspecified format. */
    TRANSCODE_PIXEL_FORMAT_UNKNOWN = 0,

    /** Planar YUV 4:2:0, 12 bits per pixel (Y plane, then U, then V). */
    TRANSCODE_PIXEL_FORMAT_YUV420P = 1,

    /** Planar YUV 4:2:2, 16 bits per pixel. */
    TRANSCODE_PIXEL_FORMAT_YUV422P = 2,

    /** Planar YUV 4:4:4, 24 bits per pixel. */
    TRANSCODE_PIXEL_FORMAT_YUV444P = 3,

    /** Planar YUV 4:2:0, 10-bit little-endian. */
    TRANSCODE_PIXEL_FORMAT_YUV420P10LE = 4,

    /** Planar YUV 4:2:2, 10-bit little-endian. */
    TRANSCODE_PIXEL_FORMAT_YUV422P10LE = 5,

    /** Planar YUV 4:4:4, 10-bit little-endian. */
    TRANSCODE_PIXEL_FORMAT_YUV444P10LE = 6,

    /** Semi-planar YUV 4:2:0 (Y plane, then interleaved UV). */
    TRANSCODE_PIXEL_FORMAT_NV12 = 7,

    /** Semi-planar YUV 4:2:0 (Y plane, then interleaved VU). */
    TRANSCODE_PIXEL_FORMAT_NV21 = 8,

    /** Packed RGB, 24 bits per pixel (R, G, B). */
    TRANSCODE_PIXEL_FORMAT_RGB24 = 9,

    /** Packed BGR, 24 bits per pixel (B, G, R). */
    TRANSCODE_PIXEL_FORMAT_BGR24 = 10,

    /** Packed RGBA, 32 bits per pixel (R, G, B, A). */
    TRANSCODE_PIXEL_FORMAT_RGBA = 11,

    /** Packed BGRA, 32 bits per pixel (B, G, R, A). */
    TRANSCODE_PIXEL_FORMAT_BGRA = 12,

    /** Grayscale, 8 bits per pixel. */
    TRANSCODE_PIXEL_FORMAT_GRAY8 = 13,

    /** Grayscale, 16 bits per pixel. */
    TRANSCODE_PIXEL_FORMAT_GRAY16 = 14
} TranscodePixelFormat;

/* ============================================================================
 * Color Space and Range
 * ============================================================================ */

/**
 * @brief Color space for video frames.
 *
 * Specifies the color matrix coefficients used for YUV to RGB conversion.
 */
typedef enum TranscodeColorSpace {
    /** BT.601 coefficients (standard definition video). */
    TRANSCODE_COLOR_SPACE_BT601 = 0,

    /** BT.709 coefficients (high definition video). */
    TRANSCODE_COLOR_SPACE_BT709 = 1,

    /** BT.2020 coefficients (ultra-high definition/HDR video). */
    TRANSCODE_COLOR_SPACE_BT2020 = 2,

    /** sRGB color space. */
    TRANSCODE_COLOR_SPACE_SRGB = 3
} TranscodeColorSpace;

/**
 * @brief Color range for video frames.
 *
 * Specifies whether the frame uses limited (TV) or full (PC) range values.
 */
typedef enum TranscodeColorRange {
    /** Limited/TV range: Y=[16,235], UV=[16,240] for 8-bit. */
    TRANSCODE_COLOR_RANGE_LIMITED = 0,

    /** Full/PC range: Y=UV=[0,255] for 8-bit. */
    TRANSCODE_COLOR_RANGE_FULL = 1
} TranscodeColorRange;

/* ============================================================================
 * Stream Types
 * ============================================================================ */

/**
 * @brief Type of media stream.
 */
typedef enum TranscodeStreamType {
    /** Unknown stream type. */
    TRANSCODE_STREAM_UNKNOWN = 0,

    /** Video stream. */
    TRANSCODE_STREAM_VIDEO = 1,

    /** Audio stream. */
    TRANSCODE_STREAM_AUDIO = 2,

    /** Subtitle stream. */
    TRANSCODE_STREAM_SUBTITLE = 3,

    /** Data stream (e.g., timecodes). */
    TRANSCODE_STREAM_DATA = 4
} TranscodeStreamType;

/* ============================================================================
 * Flags
 * ============================================================================ */

/** @name Packet Flags
 *  @brief Flags for TranscodePacket.flags field.
 *  @{
 */

/** This packet contains a keyframe/IDR. */
#define TRANSCODE_PACKET_FLAG_KEYFRAME     0x0001

/** This packet is corrupted. */
#define TRANSCODE_PACKET_FLAG_CORRUPT      0x0002

/** This packet should be discarded. */
#define TRANSCODE_PACKET_FLAG_DISCARD      0x0004

/** This packet is disposable (can be dropped without affecting decode). */
#define TRANSCODE_PACKET_FLAG_DISPOSABLE   0x0008

/** @} */

/** @name Frame Flags
 *  @brief Flags for TranscodeFrame.flags field.
 *  @{
 */

/** This frame is a keyframe. */
#define TRANSCODE_FRAME_FLAG_KEYFRAME      0x0001

/** This frame is corrupted. */
#define TRANSCODE_FRAME_FLAG_CORRUPT       0x0002

/** This frame should be discarded. */
#define TRANSCODE_FRAME_FLAG_DISCARD       0x0004

/** This frame is interlaced. */
#define TRANSCODE_FRAME_FLAG_INTERLACED    0x0008

/** For interlaced content, top field comes first. */
#define TRANSCODE_FRAME_FLAG_TOP_FIELD_FIRST 0x0010

/** @} */

/** Maximum number of planes in a video frame. */
#define TRANSCODE_MAX_PLANES 4

/* ============================================================================
 * Core Structures
 * ============================================================================ */

/**
 * @brief Information about a media stream.
 *
 * Contains metadata about a single stream within a container file.
 */
typedef struct TranscodeStreamInfo {
    /** Stream index (0-based). */
    uint32_t index;

    /** Type of stream (video, audio, etc.). */
    TranscodeStreamType stream_type;

    /** Codec identifier (fourcc-style, e.g., 'avc1' for H.264). */
    uint32_t codec_id;

    /** Video width in pixels (0 for non-video streams). */
    uint32_t width;

    /** Video height in pixels (0 for non-video streams). */
    uint32_t height;

    /** Pixel format (for video streams). */
    TranscodePixelFormat pixel_format;

    /** Audio sample rate in Hz (0 for non-audio streams). */
    uint32_t sample_rate;

    /** Number of audio channels (0 for non-audio streams). */
    uint32_t channels;

    /** Bits per audio sample (0 for non-audio streams). */
    uint32_t bits_per_sample;

    /** Time base numerator (pts = pts_raw * time_base_num / time_base_den). */
    int32_t time_base_num;

    /** Time base denominator. */
    int32_t time_base_den;

    /** Stream duration in time base units (-1 if unknown). */
    int64_t duration;

    /** Bitrate in bits per second (0 if unknown). */
    uint64_t bitrate;
} TranscodeStreamInfo;

/**
 * @brief Opaque context handle for transcoding operations.
 *
 * The context manages all state for reading, decoding, encoding, and writing
 * media files. Create with transcode_open_input() and destroy with transcode_close().
 */
typedef struct TranscodeContext {
    /** Internal pointer (opaque to C code). */
    void* _private;

    /** Input file path. */
    char* input_path;

    /** Output file path (NULL if no output opened). */
    char* output_path;

    /** Number of streams in the input. */
    uint32_t num_streams;

    /** Array of stream information. */
    TranscodeStreamInfo* streams;

    /** Context flags. */
    uint32_t flags;

    /** Last error code. */
    TranscodeError last_error;
} TranscodeContext;

/**
 * @brief An encoded media packet.
 *
 * Contains compressed data before decoding or after encoding.
 * Allocate with transcode_packet_alloc() and free with transcode_packet_free().
 */
typedef struct TranscodePacket {
    /** Pointer to packet data. */
    uint8_t* data;

    /** Size of packet data in bytes. */
    size_t size;

    /** Allocated capacity of data buffer (internal use). */
    size_t capacity;

    /** Presentation timestamp in time base units. */
    int64_t pts;

    /** Decode timestamp in time base units. */
    int64_t dts;

    /** Duration in time base units. */
    int64_t duration;

    /** Stream index this packet belongs to. */
    uint32_t stream_index;

    /** Packet flags (TRANSCODE_PACKET_FLAG_*). */
    uint32_t flags;

    /** Byte position in the input stream (-1 if unknown). */
    int64_t pos;

    /** Time base numerator. */
    int32_t time_base_num;

    /** Time base denominator. */
    int32_t time_base_den;
} TranscodePacket;

/**
 * @brief A decoded video frame.
 *
 * Contains raw pixel data in a specific format. The data is stored in planes
 * (e.g., Y, U, V for planar YUV formats).
 * Allocate with transcode_frame_alloc() and free with transcode_frame_free().
 */
typedef struct TranscodeFrame {
    /** Pointers to plane data. */
    uint8_t* data[TRANSCODE_MAX_PLANES];

    /** Stride (bytes per row) for each plane. */
    size_t linesize[TRANSCODE_MAX_PLANES];

    /** Number of valid planes. */
    uint32_t num_planes;

    /** Frame width in pixels. */
    uint32_t width;

    /** Frame height in pixels. */
    uint32_t height;

    /** Pixel format. */
    TranscodePixelFormat format;

    /** Color space. */
    TranscodeColorSpace color_space;

    /** Color range. */
    TranscodeColorRange color_range;

    /** Presentation timestamp in time base units. */
    int64_t pts;

    /** Decode timestamp in time base units. */
    int64_t dts;

    /** Frame duration in time base units. */
    int64_t duration;

    /** Frame flags (TRANSCODE_FRAME_FLAG_*). */
    uint32_t flags;

    /** Picture order count (for B-frame reordering). */
    int32_t poc;

    /** Time base numerator. */
    int32_t time_base_num;

    /** Time base denominator. */
    int32_t time_base_den;

    /** Internal buffer (opaque to C code). */
    void* _buffer;
} TranscodeFrame;

/**
 * @brief Configuration for encoding/transcoding.
 *
 * Specifies output parameters for transcoding operations.
 */
typedef struct TranscodeConfig {
    /** Output width in pixels (0 = same as input). */
    uint32_t width;

    /** Output height in pixels (0 = same as input). */
    uint32_t height;

    /** Output pixel format (UNKNOWN = same as input). */
    TranscodePixelFormat pixel_format;

    /** Target bitrate in bits per second (0 = auto/CRF mode). */
    uint64_t bitrate;

    /** Maximum bitrate for VBR (0 = no limit). */
    uint64_t max_bitrate;

    /** Quality level (0-51 for H.264/H.265, lower = better, default 23). */
    int32_t quality;

    /** GOP size / keyframe interval (0 = auto). */
    uint32_t gop_size;

    /** Number of B-frames (0 = none). */
    uint32_t b_frames;

    /** Frame rate numerator (0 = same as input). */
    uint32_t framerate_num;

    /** Frame rate denominator (default 1). */
    uint32_t framerate_den;

    /** Audio sample rate in Hz (0 = same as input). */
    uint32_t sample_rate;

    /** Number of audio channels (0 = same as input). */
    uint32_t channels;

    /** Encoder preset (0-9, lower = slower but better quality). */
    uint32_t preset;

    /** Number of encoding threads (0 = auto). */
    uint32_t threads;
} TranscodeConfig;

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Get the version string of the transcode library.
 *
 * @return A null-terminated version string. Do not free this pointer.
 */
const char* transcode_version(void);

/**
 * @brief Get a human-readable error message for an error code.
 *
 * @param error The error code to describe.
 * @return A null-terminated error description. Do not free this pointer.
 */
const char* transcode_error_string(TranscodeError error);

/* ============================================================================
 * Context Management
 * ============================================================================ */

/**
 * @brief Open an input file for reading.
 *
 * This function opens a media file and probes its contents to determine
 * the available streams and their parameters.
 *
 * @param path Path to the input file (null-terminated).
 * @param ctx Pointer to receive the allocated context.
 * @return TRANSCODE_SUCCESS on success, or an error code on failure.
 *
 * @note The caller must call transcode_close() to free the context.
 *
 * Example:
 * @code{.c}
 * TranscodeContext* ctx = NULL;
 * TranscodeError err = transcode_open_input("video.mp4", &ctx);
 * if (err != TRANSCODE_SUCCESS) {
 *     fprintf(stderr, "Failed to open: %s\n", transcode_error_string(err));
 *     return 1;
 * }
 * // Use ctx...
 * transcode_close(ctx);
 * @endcode
 */
TranscodeError transcode_open_input(const char* path, TranscodeContext** ctx);

/**
 * @brief Open an output file for writing.
 *
 * @param ctx The transcode context (must have input open).
 * @param path Path to the output file (null-terminated).
 * @param config Encoding configuration (may be NULL for defaults).
 * @return TRANSCODE_SUCCESS on success, or an error code on failure.
 */
TranscodeError transcode_open_output(
    TranscodeContext* ctx,
    const char* path,
    const TranscodeConfig* config
);

/**
 * @brief Close the transcode context and free all resources.
 *
 * @param ctx The context to close. Safe to call with NULL.
 */
void transcode_close(TranscodeContext* ctx);

/**
 * @brief Get information about a specific stream.
 *
 * @param ctx The transcode context.
 * @param stream_index Index of the stream to query (0-based).
 * @param info Pointer to receive the stream information.
 * @return TRANSCODE_SUCCESS on success, or an error code on failure.
 */
TranscodeError transcode_get_stream_info(
    const TranscodeContext* ctx,
    uint32_t stream_index,
    TranscodeStreamInfo* info
);

/* ============================================================================
 * Packet Management
 * ============================================================================ */

/**
 * @brief Allocate a new packet.
 *
 * @return A pointer to the allocated packet, or NULL on failure.
 *
 * @note The caller must call transcode_packet_free() to release the packet.
 */
TranscodePacket* transcode_packet_alloc(void);

/**
 * @brief Free a packet and its data.
 *
 * @param packet The packet to free. Safe to call with NULL.
 */
void transcode_packet_free(TranscodePacket* packet);

/**
 * @brief Ensure packet has sufficient buffer capacity.
 *
 * @param packet The packet to resize.
 * @param size Required size in bytes.
 * @return TRANSCODE_SUCCESS on success, or an error code on failure.
 */
TranscodeError transcode_packet_grow(TranscodePacket* packet, size_t size);

/**
 * @brief Read the next packet from the input.
 *
 * @param ctx The transcode context.
 * @param packet Packet to receive the data.
 * @return TRANSCODE_SUCCESS on success, TRANSCODE_ERROR_END_OF_STREAM at end,
 *         or another error code on failure.
 */
TranscodeError transcode_read_packet(TranscodeContext* ctx, TranscodePacket* packet);

/**
 * @brief Write a packet to the output.
 *
 * @param ctx The transcode context (must have output open).
 * @param packet The packet to write.
 * @return TRANSCODE_SUCCESS on success, or an error code on failure.
 */
TranscodeError transcode_write_packet(TranscodeContext* ctx, const TranscodePacket* packet);

/* ============================================================================
 * Frame Management
 * ============================================================================ */

/**
 * @brief Allocate a new frame.
 *
 * @return A pointer to the allocated frame, or NULL on failure.
 *
 * @note The caller must call transcode_frame_free() to release the frame.
 */
TranscodeFrame* transcode_frame_alloc(void);

/**
 * @brief Allocate frame buffer with specified dimensions.
 *
 * @param frame The frame to allocate buffer for.
 * @param width Frame width in pixels.
 * @param height Frame height in pixels.
 * @param format Pixel format.
 * @return TRANSCODE_SUCCESS on success, or an error code on failure.
 */
TranscodeError transcode_frame_alloc_buffer(
    TranscodeFrame* frame,
    uint32_t width,
    uint32_t height,
    TranscodePixelFormat format
);

/**
 * @brief Free frame buffer data (but not the frame structure itself).
 *
 * @param frame The frame whose buffer to free.
 */
void transcode_frame_free_buffer(TranscodeFrame* frame);

/**
 * @brief Free a frame and its data.
 *
 * @param frame The frame to free. Safe to call with NULL.
 */
void transcode_frame_free(TranscodeFrame* frame);

/**
 * @brief Copy frame data from source to destination.
 *
 * @param dst Destination frame (must have matching buffer allocated).
 * @param src Source frame.
 * @return TRANSCODE_SUCCESS on success, or an error code on failure.
 */
TranscodeError transcode_frame_copy(TranscodeFrame* dst, const TranscodeFrame* src);

/* ============================================================================
 * Decoding and Encoding
 * ============================================================================ */

/**
 * @brief Decode a packet into a frame.
 *
 * @param ctx The transcode context.
 * @param packet The packet to decode.
 * @param frame Frame to receive the decoded data.
 * @return TRANSCODE_SUCCESS on success, TRANSCODE_ERROR_RESOURCE_EXHAUSTED
 *         if more input is needed, or another error code on failure.
 */
TranscodeError transcode_decode_packet(
    TranscodeContext* ctx,
    const TranscodePacket* packet,
    TranscodeFrame* frame
);

/**
 * @brief Encode a frame into a packet.
 *
 * @param ctx The transcode context.
 * @param frame The frame to encode (NULL to flush encoder).
 * @param packet Packet to receive the encoded data.
 * @return TRANSCODE_SUCCESS on success, TRANSCODE_ERROR_RESOURCE_EXHAUSTED
 *         if no output is available, or another error code on failure.
 */
TranscodeError transcode_encode_frame(
    TranscodeContext* ctx,
    const TranscodeFrame* frame,
    TranscodePacket* packet
);

/* ============================================================================
 * Seek and Flush Operations
 * ============================================================================ */

/**
 * @brief Seek to a specific timestamp in the input.
 *
 * @param ctx The transcode context.
 * @param stream_index Stream to seek in (-1 for default stream).
 * @param timestamp Target timestamp in stream time base.
 * @param flags Seek flags (reserved, pass 0).
 * @return TRANSCODE_SUCCESS on success, or an error code on failure.
 *
 * @note After seeking, call transcode_flush_decoder() to clear buffered frames.
 */
TranscodeError transcode_seek(
    TranscodeContext* ctx,
    int stream_index,
    int64_t timestamp,
    int flags
);

/**
 * @brief Flush decoder buffers.
 *
 * Call this after seeking or at end of stream to get remaining frames.
 *
 * @param ctx The transcode context.
 * @return TRANSCODE_SUCCESS on success, or an error code on failure.
 */
TranscodeError transcode_flush_decoder(TranscodeContext* ctx);

/**
 * @brief Flush encoder buffers.
 *
 * Call this at end of stream to get remaining encoded packets.
 *
 * @param ctx The transcode context.
 * @return TRANSCODE_SUCCESS on success, or an error code on failure.
 */
TranscodeError transcode_flush_encoder(TranscodeContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCODE_H */
