/*
 * Transcode C API
 *
 * A memory-safe, high-performance universal codec library.
 *
 * Copyright (c) Transcode Contributors
 * Licensed under MIT OR Apache-2.0
 */


#ifndef TRANSCODE_H
#define TRANSCODE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Packet flags.
 */
#define TRANSCODE_PACKET_FLAG_KEYFRAME 1

#define TRANSCODE_PACKET_FLAG_CORRUPT 2

#define TRANSCODE_PACKET_FLAG_DISCARD 4

#define TRANSCODE_PACKET_FLAG_DISPOSABLE 8

/**
 * Frame flags.
 */
#define TRANSCODE_FRAME_FLAG_KEYFRAME 1

#define TRANSCODE_FRAME_FLAG_CORRUPT 2

#define TRANSCODE_FRAME_FLAG_DISCARD 4

#define TRANSCODE_FRAME_FLAG_INTERLACED 8

#define TRANSCODE_FRAME_FLAG_TOP_FIELD_FIRST 16

/**
 * Maximum number of planes in a frame.
 */
#define TRANSCODE_MAX_PLANES 4

/**
 * Color range for video frames.
 */
typedef enum TranscodeColorRange {
    /**
     * Limited/TV range (16-235 for Y, 16-240 for UV).
     */
    TRANSCODE_COLOR_RANGE_LIMITED = 0,
    /**
     * Full/PC range (0-255).
     */
    TRANSCODE_COLOR_RANGE_FULL = 1,
} TranscodeColorRange;

/**
 * Color space for video frames.
 */
typedef enum TranscodeColorSpace {
    /**
     * BT.601 (SD video).
     */
    TRANSCODE_COLOR_SPACE_BT601 = 0,
    /**
     * BT.709 (HD video).
     */
    TRANSCODE_COLOR_SPACE_BT709 = 1,
    /**
     * BT.2020 (UHD/HDR video).
     */
    TRANSCODE_COLOR_SPACE_BT2020 = 2,
    /**
     * sRGB.
     */
    TRANSCODE_COLOR_SPACE_SRGB = 3,
} TranscodeColorSpace;

/**
 * Error codes returned by transcode functions.
 */
typedef enum TranscodeError {
    /**
     * Operation completed successfully.
     */
    TRANSCODE_ERROR_SUCCESS = 0,
    /**
     * Invalid argument provided.
     */
    TRANSCODE_ERROR_INVALID_ARGUMENT = -1,
    /**
     * Null pointer provided where non-null was expected.
     */
    TRANSCODE_ERROR_NULL_POINTER = -2,
    /**
     * End of stream reached.
     */
    TRANSCODE_ERROR_END_OF_STREAM = -3,
    /**
     * I/O error occurred.
     */
    TRANSCODE_ERROR_IO_ERROR = -4,
    /**
     * Codec error occurred.
     */
    TRANSCODE_ERROR_CODEC_ERROR = -5,
    /**
     * Container/format error occurred.
     */
    TRANSCODE_ERROR_CONTAINER_ERROR = -6,
    /**
     * Resource exhausted (out of memory, etc.).
     */
    TRANSCODE_ERROR_RESOURCE_EXHAUSTED = -7,
    /**
     * Unsupported feature or format.
     */
    TRANSCODE_ERROR_UNSUPPORTED = -8,
    /**
     * Operation was cancelled.
     */
    TRANSCODE_ERROR_CANCELLED = -9,
    /**
     * Buffer too small for operation.
     */
    TRANSCODE_ERROR_BUFFER_TOO_SMALL = -10,
    /**
     * Invalid state for this operation.
     */
    TRANSCODE_ERROR_INVALID_STATE = -11,
    /**
     * Unknown or internal error.
     */
    TRANSCODE_ERROR_UNKNOWN = -100,
} TranscodeError;

/**
 * Pixel format for video frames.
 */
typedef enum TranscodePixelFormat {
    /**
     * Unknown pixel format.
     */
    TRANSCODE_PIXEL_FORMAT_UNKNOWN = 0,
    /**
     * Planar YUV 4:2:0, 12bpp.
     */
    TRANSCODE_PIXEL_FORMAT_YUV420P = 1,
    /**
     * Planar YUV 4:2:2, 16bpp.
     */
    TRANSCODE_PIXEL_FORMAT_YUV422P = 2,
    /**
     * Planar YUV 4:4:4, 24bpp.
     */
    TRANSCODE_PIXEL_FORMAT_YUV444P = 3,
    /**
     * Planar YUV 4:2:0, 10-bit.
     */
    TRANSCODE_PIXEL_FORMAT_YUV420P10LE = 4,
    /**
     * Planar YUV 4:2:2, 10-bit.
     */
    TRANSCODE_PIXEL_FORMAT_YUV422P10LE = 5,
    /**
     * Planar YUV 4:4:4, 10-bit.
     */
    TRANSCODE_PIXEL_FORMAT_YUV444P10LE = 6,
    /**
     * NV12 (Y plane + interleaved UV).
     */
    TRANSCODE_PIXEL_FORMAT_NV12 = 7,
    /**
     * NV21 (Y plane + interleaved VU).
     */
    TRANSCODE_PIXEL_FORMAT_NV21 = 8,
    /**
     * Packed RGB, 24bpp.
     */
    TRANSCODE_PIXEL_FORMAT_RGB24 = 9,
    /**
     * Packed BGR, 24bpp.
     */
    TRANSCODE_PIXEL_FORMAT_BGR24 = 10,
    /**
     * Packed RGBA, 32bpp.
     */
    TRANSCODE_PIXEL_FORMAT_RGBA = 11,
    /**
     * Packed BGRA, 32bpp.
     */
    TRANSCODE_PIXEL_FORMAT_BGRA = 12,
    /**
     * Grayscale, 8bpp.
     */
    TRANSCODE_PIXEL_FORMAT_GRAY8 = 13,
    /**
     * Grayscale, 16bpp.
     */
    TRANSCODE_PIXEL_FORMAT_GRAY16 = 14,
} TranscodePixelFormat;

/**
 * Type of media stream.
 */
typedef enum TranscodeStreamType {
    /**
     * Unknown stream type.
     */
    TRANSCODE_STREAM_TYPE_UNKNOWN = 0,
    /**
     * Video stream.
     */
    TRANSCODE_STREAM_TYPE_VIDEO = 1,
    /**
     * Audio stream.
     */
    TRANSCODE_STREAM_TYPE_AUDIO = 2,
    /**
     * Subtitle stream.
     */
    TRANSCODE_STREAM_TYPE_SUBTITLE = 3,
    /**
     * Data stream.
     */
    TRANSCODE_STREAM_TYPE_DATA = 4,
} TranscodeStreamType;

/**
 * Information about a media stream.
 */
typedef struct TranscodeStreamInfo {
    /**
     * Stream index.
     */
    uint32_t index;
    /**
     * Stream type.
     */
    enum TranscodeStreamType stream_type;
    /**
     * Codec identifier (fourcc-style).
     */
    uint32_t codec_id;
    /**
     * For video: width in pixels.
     */
    uint32_t width;
    /**
     * For video: height in pixels.
     */
    uint32_t height;
    /**
     * For video: pixel format.
     */
    enum TranscodePixelFormat pixel_format;
    /**
     * For audio: sample rate in Hz.
     */
    uint32_t sample_rate;
    /**
     * For audio: number of channels.
     */
    uint32_t channels;
    /**
     * For audio: bits per sample.
     */
    uint32_t bits_per_sample;
    /**
     * Time base numerator.
     */
    int32_t time_base_num;
    /**
     * Time base denominator.
     */
    int32_t time_base_den;
    /**
     * Duration in time base units (-1 if unknown).
     */
    int64_t duration;
    /**
     * Bitrate in bits per second (0 if unknown).
     */
    uint64_t bitrate;
} TranscodeStreamInfo;

/**
 * Opaque context handle for transcoding operations.
 *
 * This structure manages the state for reading, decoding, encoding,
 * and writing media files.
 */
typedef struct TranscodeContext {
    /**
     * Pointer to internal Rust context (opaque to C code).
     */
    void *private_;
    /**
     * Input file path (null-terminated).
     */
    char *input_path;
    /**
     * Output file path (null-terminated, may be null).
     */
    char *output_path;
    /**
     * Number of streams in the input.
     */
    uint32_t num_streams;
    /**
     * Stream information array.
     */
    struct TranscodeStreamInfo *streams;
    /**
     * Context flags.
     */
    uint32_t flags;
    /**
     * Last error code.
     */
    enum TranscodeError last_error;
} TranscodeContext;

/**
 * Configuration for encoding/transcoding.
 */
typedef struct TranscodeConfig {
    /**
     * Output width (0 = same as input).
     */
    uint32_t width;
    /**
     * Output height (0 = same as input).
     */
    uint32_t height;
    /**
     * Output pixel format.
     */
    enum TranscodePixelFormat pixel_format;
    /**
     * Target bitrate in bits per second (0 = auto).
     */
    uint64_t bitrate;
    /**
     * Maximum bitrate for VBR (0 = no limit).
     */
    uint64_t max_bitrate;
    /**
     * Quality level (0-51 for H.264/H.265, codec-specific).
     */
    int32_t quality;
    /**
     * GOP size (frames between keyframes, 0 = auto).
     */
    uint32_t gop_size;
    /**
     * Number of B-frames (0 = none).
     */
    uint32_t b_frames;
    /**
     * Frame rate numerator.
     */
    uint32_t framerate_num;
    /**
     * Frame rate denominator.
     */
    uint32_t framerate_den;
    /**
     * Sample rate for audio (0 = same as input).
     */
    uint32_t sample_rate;
    /**
     * Number of audio channels (0 = same as input).
     */
    uint32_t channels;
    /**
     * Encoder preset (0-9, lower = slower/better quality).
     */
    uint32_t preset;
    /**
     * Threading mode: 0 = auto, N = use N threads.
     */
    uint32_t threads;
} TranscodeConfig;

/**
 * An encoded media packet.
 *
 * Contains compressed data before decoding or after encoding.
 */
typedef struct TranscodePacket {
    /**
     * Pointer to packet data.
     */
    uint8_t *data;
    /**
     * Size of packet data in bytes.
     */
    uintptr_t size;
    /**
     * Allocated capacity of data buffer.
     */
    uintptr_t capacity;
    /**
     * Presentation timestamp.
     */
    int64_t pts;
    /**
     * Decode timestamp.
     */
    int64_t dts;
    /**
     * Duration in time base units.
     */
    int64_t duration;
    /**
     * Stream index this packet belongs to.
     */
    uint32_t stream_index;
    /**
     * Packet flags (see TRANSCODE_PACKET_FLAG_*).
     */
    uint32_t flags;
    /**
     * Position in the input stream (bytes), -1 if unknown.
     */
    int64_t pos;
    /**
     * Time base numerator.
     */
    int32_t time_base_num;
    /**
     * Time base denominator.
     */
    int32_t time_base_den;
} TranscodePacket;

/**
 * A decoded video frame.
 *
 * Contains raw pixel data in a specific format.
 */
typedef struct TranscodeFrame {
    /**
     * Pointers to plane data.
     */
    uint8_t *data[TRANSCODE_MAX_PLANES];
    /**
     * Size of each plane in bytes.
     */
    uintptr_t linesize[TRANSCODE_MAX_PLANES];
    /**
     * Number of valid planes.
     */
    uint32_t num_planes;
    /**
     * Frame width in pixels.
     */
    uint32_t width;
    /**
     * Frame height in pixels.
     */
    uint32_t height;
    /**
     * Pixel format.
     */
    enum TranscodePixelFormat format;
    /**
     * Color space.
     */
    enum TranscodeColorSpace color_space;
    /**
     * Color range.
     */
    enum TranscodeColorRange color_range;
    /**
     * Presentation timestamp.
     */
    int64_t pts;
    /**
     * Decode timestamp.
     */
    int64_t dts;
    /**
     * Frame duration in time base units.
     */
    int64_t duration;
    /**
     * Frame flags (see TRANSCODE_FRAME_FLAG_*).
     */
    uint32_t flags;
    /**
     * Picture order count (for B-frame reordering).
     */
    int32_t poc;
    /**
     * Time base numerator.
     */
    int32_t time_base_num;
    /**
     * Time base denominator.
     */
    int32_t time_base_den;
    /**
     * Internal buffer (opaque to C code).
     */
    void *buffer;
} TranscodeFrame;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Get the version string of the transcode library.
 *
 * Returns a null-terminated string. The caller must not free this pointer.
 */
const char *transcode_version(void);

/**
 * Get a human-readable error message for an error code.
 *
 * Returns a null-terminated string. The caller must not free this pointer.
 */
const char *transcode_error_string(enum TranscodeError error);

/**
 * Open an input file for reading.
 *
 * # Arguments
 *
 * * `path` - Path to the input file (null-terminated string).
 * * `ctx` - Pointer to receive the allocated context.
 *
 * # Returns
 *
 * `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
 */
enum TranscodeError transcode_open_input(const char *path, struct TranscodeContext **ctx);

/**
 * Open an output file for writing.
 *
 * # Arguments
 *
 * * `ctx` - The transcode context.
 * * `path` - Path to the output file (null-terminated string).
 * * `config` - Optional encoding configuration (may be null for defaults).
 *
 * # Returns
 *
 * `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
 */
enum TranscodeError transcode_open_output(struct TranscodeContext *ctx,
                                          const char *path,
                                          const struct TranscodeConfig *config);

/**
 * Close the transcode context and free all resources.
 *
 * # Arguments
 *
 * * `ctx` - The context to close (may be null).
 */
void transcode_close(struct TranscodeContext *ctx);

/**
 * Get stream information.
 *
 * # Arguments
 *
 * * `ctx` - The transcode context.
 * * `stream_index` - Index of the stream to query.
 * * `info` - Pointer to receive stream information.
 *
 * # Returns
 *
 * `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
 */
enum TranscodeError transcode_get_stream_info(const struct TranscodeContext *ctx,
                                              uint32_t stream_index,
                                              struct TranscodeStreamInfo *info);

/**
 * Allocate a new packet.
 *
 * # Returns
 *
 * A pointer to the allocated packet, or null on failure.
 */
struct TranscodePacket *transcode_packet_alloc(void);

/**
 * Free a packet and its data.
 *
 * # Arguments
 *
 * * `packet` - The packet to free (may be null).
 */
void transcode_packet_free(struct TranscodePacket *packet);

/**
 * Reallocate packet data to ensure capacity.
 *
 * # Arguments
 *
 * * `packet` - The packet to resize.
 * * `size` - Required size in bytes.
 *
 * # Returns
 *
 * `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
 */
enum TranscodeError transcode_packet_grow(struct TranscodePacket *packet, uintptr_t size);

/**
 * Read the next packet from the input.
 *
 * # Arguments
 *
 * * `ctx` - The transcode context.
 * * `packet` - Packet to receive the data.
 *
 * # Returns
 *
 * `TRANSCODE_ERROR_SUCCESS` on success, `TRANSCODE_ERROR_END_OF_STREAM` at end,
 * or another error code on failure.
 */
enum TranscodeError transcode_read_packet(struct TranscodeContext *ctx,
                                          struct TranscodePacket *packet);

/**
 * Write a packet to the output.
 *
 * # Arguments
 *
 * * `ctx` - The transcode context.
 * * `packet` - The packet to write.
 *
 * # Returns
 *
 * `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
 */
enum TranscodeError transcode_write_packet(struct TranscodeContext *ctx,
                                           const struct TranscodePacket *packet);

/**
 * Allocate a new frame.
 *
 * # Returns
 *
 * A pointer to the allocated frame, or null on failure.
 */
struct TranscodeFrame *transcode_frame_alloc(void);

/**
 * Allocate frame buffer with specified dimensions.
 *
 * # Arguments
 *
 * * `frame` - The frame to allocate buffer for.
 * * `width` - Frame width in pixels.
 * * `height` - Frame height in pixels.
 * * `format` - Pixel format.
 *
 * # Returns
 *
 * `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
 */
enum TranscodeError transcode_frame_alloc_buffer(struct TranscodeFrame *frame,
                                                 uint32_t width,
                                                 uint32_t height,
                                                 enum TranscodePixelFormat format);

/**
 * Free frame buffer data (but not the frame structure itself).
 *
 * # Arguments
 *
 * * `frame` - The frame whose buffer to free.
 */
void transcode_frame_free_buffer(struct TranscodeFrame *frame);

/**
 * Free a frame and its data.
 *
 * # Arguments
 *
 * * `frame` - The frame to free (may be null).
 */
void transcode_frame_free(struct TranscodeFrame *frame);

/**
 * Copy frame data from source to destination.
 *
 * # Arguments
 *
 * * `dst` - Destination frame (must have buffer allocated).
 * * `src` - Source frame.
 *
 * # Returns
 *
 * `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
 */
enum TranscodeError transcode_frame_copy(struct TranscodeFrame *dst,
                                         const struct TranscodeFrame *src);

/**
 * Decode a packet into a frame.
 *
 * # Arguments
 *
 * * `ctx` - The transcode context.
 * * `packet` - The packet to decode.
 * * `frame` - Frame to receive the decoded data.
 *
 * # Returns
 *
 * `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
 * May return `TRANSCODE_ERROR_RESOURCE_EXHAUSTED` if more input is needed.
 */
enum TranscodeError transcode_decode_packet(struct TranscodeContext *ctx,
                                            const struct TranscodePacket *packet,
                                            struct TranscodeFrame *frame);

/**
 * Encode a frame into a packet.
 *
 * # Arguments
 *
 * * `ctx` - The transcode context.
 * * `frame` - The frame to encode (may be null to flush encoder).
 * * `packet` - Packet to receive the encoded data.
 *
 * # Returns
 *
 * `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
 * May return `TRANSCODE_ERROR_RESOURCE_EXHAUSTED` if no output is available yet.
 */
enum TranscodeError transcode_encode_frame(struct TranscodeContext *ctx,
                                           const struct TranscodeFrame *frame,
                                           struct TranscodePacket *packet);

/**
 * Seek to a specific timestamp in the input.
 *
 * # Arguments
 *
 * * `ctx` - The transcode context.
 * * `stream_index` - Stream to seek in (-1 for default).
 * * `timestamp` - Target timestamp in stream time base.
 * * `flags` - Seek flags (reserved for future use, pass 0).
 *
 * # Returns
 *
 * `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
 */
enum TranscodeError transcode_seek(struct TranscodeContext *ctx,
                                   int stream_index,
                                   int64_t timestamp,
                                   int flags);

/**
 * Flush decoder buffers.
 *
 * Call this when seeking or at end of stream to get remaining frames.
 *
 * # Arguments
 *
 * * `ctx` - The transcode context.
 *
 * # Returns
 *
 * `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
 */
enum TranscodeError transcode_flush_decoder(struct TranscodeContext *ctx);

/**
 * Flush encoder buffers.
 *
 * Call this at end of stream to get remaining packets.
 *
 * # Arguments
 *
 * * `ctx` - The transcode context.
 *
 * # Returns
 *
 * `TRANSCODE_ERROR_SUCCESS` on success, or an error code on failure.
 */
enum TranscodeError transcode_flush_encoder(struct TranscodeContext *ctx);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* TRANSCODE_H */
