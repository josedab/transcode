/**
 * @file transcode.h
 * @brief Transcode C API - A memory-safe, high-performance codec library
 * @version 0.1.0
 *
 * This is a copy of the transcode C API header for Swift bindings.
 * See the main transcode-capi crate for the canonical version.
 */

#ifndef TRANSCODE_H
#define TRANSCODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* Version Information */
#define TRANSCODE_VERSION_MAJOR 0
#define TRANSCODE_VERSION_MINOR 1
#define TRANSCODE_VERSION_PATCH 0
#define TRANSCODE_VERSION_STRING "0.1.0"

/* Error Codes */
typedef enum TranscodeError {
    TRANSCODE_SUCCESS = 0,
    TRANSCODE_ERROR_INVALID_ARGUMENT = -1,
    TRANSCODE_ERROR_NULL_POINTER = -2,
    TRANSCODE_ERROR_END_OF_STREAM = -3,
    TRANSCODE_ERROR_IO = -4,
    TRANSCODE_ERROR_CODEC = -5,
    TRANSCODE_ERROR_CONTAINER = -6,
    TRANSCODE_ERROR_RESOURCE_EXHAUSTED = -7,
    TRANSCODE_ERROR_UNSUPPORTED = -8,
    TRANSCODE_ERROR_CANCELLED = -9,
    TRANSCODE_ERROR_BUFFER_TOO_SMALL = -10,
    TRANSCODE_ERROR_INVALID_STATE = -11,
    TRANSCODE_ERROR_UNKNOWN = -100
} TranscodeError;

/* Pixel Formats */
typedef enum TranscodePixelFormat {
    TRANSCODE_PIXEL_FORMAT_UNKNOWN = 0,
    TRANSCODE_PIXEL_FORMAT_YUV420P = 1,
    TRANSCODE_PIXEL_FORMAT_YUV422P = 2,
    TRANSCODE_PIXEL_FORMAT_YUV444P = 3,
    TRANSCODE_PIXEL_FORMAT_YUV420P10LE = 4,
    TRANSCODE_PIXEL_FORMAT_YUV422P10LE = 5,
    TRANSCODE_PIXEL_FORMAT_YUV444P10LE = 6,
    TRANSCODE_PIXEL_FORMAT_NV12 = 7,
    TRANSCODE_PIXEL_FORMAT_NV21 = 8,
    TRANSCODE_PIXEL_FORMAT_RGB24 = 9,
    TRANSCODE_PIXEL_FORMAT_BGR24 = 10,
    TRANSCODE_PIXEL_FORMAT_RGBA = 11,
    TRANSCODE_PIXEL_FORMAT_BGRA = 12,
    TRANSCODE_PIXEL_FORMAT_GRAY8 = 13,
    TRANSCODE_PIXEL_FORMAT_GRAY16 = 14
} TranscodePixelFormat;

/* Color Space and Range */
typedef enum TranscodeColorSpace {
    TRANSCODE_COLOR_SPACE_BT601 = 0,
    TRANSCODE_COLOR_SPACE_BT709 = 1,
    TRANSCODE_COLOR_SPACE_BT2020 = 2,
    TRANSCODE_COLOR_SPACE_SRGB = 3
} TranscodeColorSpace;

typedef enum TranscodeColorRange {
    TRANSCODE_COLOR_RANGE_LIMITED = 0,
    TRANSCODE_COLOR_RANGE_FULL = 1
} TranscodeColorRange;

/* Stream Types */
typedef enum TranscodeStreamType {
    TRANSCODE_STREAM_UNKNOWN = 0,
    TRANSCODE_STREAM_VIDEO = 1,
    TRANSCODE_STREAM_AUDIO = 2,
    TRANSCODE_STREAM_SUBTITLE = 3,
    TRANSCODE_STREAM_DATA = 4
} TranscodeStreamType;

/* Flags */
#define TRANSCODE_PACKET_FLAG_KEYFRAME     0x0001
#define TRANSCODE_PACKET_FLAG_CORRUPT      0x0002
#define TRANSCODE_PACKET_FLAG_DISCARD      0x0004
#define TRANSCODE_PACKET_FLAG_DISPOSABLE   0x0008

#define TRANSCODE_FRAME_FLAG_KEYFRAME      0x0001
#define TRANSCODE_FRAME_FLAG_CORRUPT       0x0002
#define TRANSCODE_FRAME_FLAG_DISCARD       0x0004
#define TRANSCODE_FRAME_FLAG_INTERLACED    0x0008
#define TRANSCODE_FRAME_FLAG_TOP_FIELD_FIRST 0x0010

#define TRANSCODE_MAX_PLANES 4

/* Core Structures */
typedef struct TranscodeStreamInfo {
    uint32_t index;
    TranscodeStreamType stream_type;
    uint32_t codec_id;
    uint32_t width;
    uint32_t height;
    TranscodePixelFormat pixel_format;
    uint32_t sample_rate;
    uint32_t channels;
    uint32_t bits_per_sample;
    int32_t time_base_num;
    int32_t time_base_den;
    int64_t duration;
    uint64_t bitrate;
} TranscodeStreamInfo;

typedef struct TranscodeContext {
    void* _private;
    char* input_path;
    char* output_path;
    uint32_t num_streams;
    TranscodeStreamInfo* streams;
    uint32_t flags;
    TranscodeError last_error;
} TranscodeContext;

typedef struct TranscodePacket {
    uint8_t* data;
    size_t size;
    size_t capacity;
    int64_t pts;
    int64_t dts;
    int64_t duration;
    uint32_t stream_index;
    uint32_t flags;
    int64_t pos;
    int32_t time_base_num;
    int32_t time_base_den;
} TranscodePacket;

typedef struct TranscodeFrame {
    uint8_t* data[TRANSCODE_MAX_PLANES];
    size_t linesize[TRANSCODE_MAX_PLANES];
    uint32_t num_planes;
    uint32_t width;
    uint32_t height;
    TranscodePixelFormat format;
    TranscodeColorSpace color_space;
    TranscodeColorRange color_range;
    int64_t pts;
    int64_t dts;
    int64_t duration;
    uint32_t flags;
    int32_t poc;
    int32_t time_base_num;
    int32_t time_base_den;
    void* _buffer;
} TranscodeFrame;

typedef struct TranscodeConfig {
    uint32_t width;
    uint32_t height;
    TranscodePixelFormat pixel_format;
    uint64_t bitrate;
    uint64_t max_bitrate;
    int32_t quality;
    uint32_t gop_size;
    uint32_t b_frames;
    uint32_t framerate_num;
    uint32_t framerate_den;
    uint32_t sample_rate;
    uint32_t channels;
    uint32_t preset;
    uint32_t threads;
} TranscodeConfig;

/* Utility Functions */
const char* transcode_version(void);
const char* transcode_error_string(TranscodeError error);

/* Context Management */
TranscodeError transcode_open_input(const char* path, TranscodeContext** ctx);
TranscodeError transcode_open_output(
    TranscodeContext* ctx,
    const char* path,
    const TranscodeConfig* config
);
void transcode_close(TranscodeContext* ctx);
TranscodeError transcode_get_stream_info(
    const TranscodeContext* ctx,
    uint32_t stream_index,
    TranscodeStreamInfo* info
);

/* Packet Management */
TranscodePacket* transcode_packet_alloc(void);
void transcode_packet_free(TranscodePacket* packet);
TranscodeError transcode_packet_grow(TranscodePacket* packet, size_t size);
TranscodeError transcode_read_packet(TranscodeContext* ctx, TranscodePacket* packet);
TranscodeError transcode_write_packet(TranscodeContext* ctx, const TranscodePacket* packet);

/* Frame Management */
TranscodeFrame* transcode_frame_alloc(void);
TranscodeError transcode_frame_alloc_buffer(
    TranscodeFrame* frame,
    uint32_t width,
    uint32_t height,
    TranscodePixelFormat format
);
void transcode_frame_free_buffer(TranscodeFrame* frame);
void transcode_frame_free(TranscodeFrame* frame);
TranscodeError transcode_frame_copy(TranscodeFrame* dst, const TranscodeFrame* src);

/* Decoding and Encoding */
TranscodeError transcode_decode_packet(
    TranscodeContext* ctx,
    const TranscodePacket* packet,
    TranscodeFrame* frame
);
TranscodeError transcode_encode_frame(
    TranscodeContext* ctx,
    const TranscodeFrame* frame,
    TranscodePacket* packet
);

/* Seek and Flush Operations */
TranscodeError transcode_seek(
    TranscodeContext* ctx,
    int stream_index,
    int64_t timestamp,
    int flags
);
TranscodeError transcode_flush_decoder(TranscodeContext* ctx);
TranscodeError transcode_flush_encoder(TranscodeContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCODE_H */
