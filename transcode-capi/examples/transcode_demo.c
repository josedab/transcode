/**
 * @file transcode_demo.c
 * @brief Demonstration of the Transcode C API
 *
 * This example shows how to:
 * 1. Open an input media file
 * 2. Read stream information
 * 3. Read and decode packets
 * 4. Encode and write to output
 *
 * Build:
 *   gcc -o transcode_demo transcode_demo.c -I../include -L../../target/release \
 *       -ltranscode_capi -lpthread -ldl -lm
 *
 * Usage:
 *   ./transcode_demo input.mp4 output.mp4
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/transcode.h"

/**
 * Print stream information in a human-readable format.
 */
static void print_stream_info(const TranscodeStreamInfo* info) {
    printf("  Stream #%u: ", info->index);

    switch (info->stream_type) {
        case TRANSCODE_STREAM_VIDEO:
            printf("Video");
            printf(" %ux%u", info->width, info->height);
            if (info->bitrate > 0) {
                printf(" @ %llu kbps", (unsigned long long)(info->bitrate / 1000));
            }
            break;

        case TRANSCODE_STREAM_AUDIO:
            printf("Audio");
            printf(" %u Hz, %u channel(s)", info->sample_rate, info->channels);
            if (info->bits_per_sample > 0) {
                printf(", %u-bit", info->bits_per_sample);
            }
            if (info->bitrate > 0) {
                printf(" @ %llu kbps", (unsigned long long)(info->bitrate / 1000));
            }
            break;

        case TRANSCODE_STREAM_SUBTITLE:
            printf("Subtitle");
            break;

        case TRANSCODE_STREAM_DATA:
            printf("Data");
            break;

        default:
            printf("Unknown");
            break;
    }

    /* Print codec as fourcc */
    if (info->codec_id != 0) {
        char fourcc[5] = {0};
        fourcc[0] = (info->codec_id >> 24) & 0xFF;
        fourcc[1] = (info->codec_id >> 16) & 0xFF;
        fourcc[2] = (info->codec_id >> 8) & 0xFF;
        fourcc[3] = info->codec_id & 0xFF;
        printf(" (codec: %s)", fourcc);
    }

    /* Print duration if known */
    if (info->duration > 0 && info->time_base_den > 0) {
        double duration_sec = (double)info->duration * info->time_base_num / info->time_base_den;
        printf(" - %.2f sec", duration_sec);
    }

    printf("\n");
}

/**
 * Probe a media file and print its information.
 */
static int probe_file(const char* path) {
    TranscodeContext* ctx = NULL;
    TranscodeError err;

    printf("Probing: %s\n", path);
    printf("Transcode version: %s\n\n", transcode_version());

    /* Open the input file */
    err = transcode_open_input(path, &ctx);
    if (err != TRANSCODE_SUCCESS) {
        fprintf(stderr, "Error opening file: %s\n", transcode_error_string(err));
        return 1;
    }

    /* Print stream information */
    printf("Found %u stream(s):\n", ctx->num_streams);

    for (uint32_t i = 0; i < ctx->num_streams; i++) {
        TranscodeStreamInfo info;
        err = transcode_get_stream_info(ctx, i, &info);
        if (err == TRANSCODE_SUCCESS) {
            print_stream_info(&info);
        }
    }

    printf("\n");

    transcode_close(ctx);
    return 0;
}

/**
 * Transcode a file from input to output.
 */
static int transcode_file(const char* input_path, const char* output_path) {
    TranscodeContext* ctx = NULL;
    TranscodeConfig config = {0};
    TranscodePacket* in_packet = NULL;
    TranscodePacket* out_packet = NULL;
    TranscodeFrame* frame = NULL;
    TranscodeError err;
    uint64_t packets_read = 0;
    uint64_t frames_decoded = 0;
    uint64_t packets_written = 0;
    int result = 0;

    printf("Transcoding: %s -> %s\n", input_path, output_path);

    /* Open input */
    err = transcode_open_input(input_path, &ctx);
    if (err != TRANSCODE_SUCCESS) {
        fprintf(stderr, "Error opening input: %s\n", transcode_error_string(err));
        return 1;
    }

    /* Configure output */
    config.width = 1280;              /* 720p */
    config.height = 720;
    config.bitrate = 2000000;         /* 2 Mbps */
    config.quality = 23;              /* CRF 23 */
    config.preset = 5;                /* Medium preset */
    config.gop_size = 60;             /* 2 seconds at 30fps */
    config.b_frames = 2;
    config.threads = 0;               /* Auto */

    /* Open output */
    err = transcode_open_output(ctx, output_path, &config);
    if (err != TRANSCODE_SUCCESS) {
        fprintf(stderr, "Error opening output: %s\n", transcode_error_string(err));
        transcode_close(ctx);
        return 1;
    }

    /* Allocate working structures */
    in_packet = transcode_packet_alloc();
    out_packet = transcode_packet_alloc();
    frame = transcode_frame_alloc();

    if (!in_packet || !out_packet || !frame) {
        fprintf(stderr, "Error allocating packet/frame\n");
        result = 1;
        goto cleanup;
    }

    printf("Transcoding (press Ctrl+C to cancel)...\n");

    /* Main transcode loop */
    while ((err = transcode_read_packet(ctx, in_packet)) == TRANSCODE_SUCCESS) {
        packets_read++;

        /* Decode */
        err = transcode_decode_packet(ctx, in_packet, frame);
        if (err == TRANSCODE_SUCCESS) {
            frames_decoded++;

            /* Encode */
            err = transcode_encode_frame(ctx, frame, out_packet);
            if (err == TRANSCODE_SUCCESS) {
                /* Write */
                err = transcode_write_packet(ctx, out_packet);
                if (err == TRANSCODE_SUCCESS) {
                    packets_written++;
                } else if (err != TRANSCODE_ERROR_RESOURCE_EXHAUSTED) {
                    fprintf(stderr, "Error writing packet: %s\n", transcode_error_string(err));
                    result = 1;
                    goto cleanup;
                }
            }
        }

        /* Progress indicator */
        if (packets_read % 100 == 0) {
            printf("\r  Packets: %llu read, %llu frames decoded, %llu written",
                   (unsigned long long)packets_read,
                   (unsigned long long)frames_decoded,
                   (unsigned long long)packets_written);
            fflush(stdout);
        }
    }

    if (err != TRANSCODE_ERROR_END_OF_STREAM) {
        fprintf(stderr, "\nError reading: %s\n", transcode_error_string(err));
    }

    /* Flush encoder */
    printf("\nFlushing encoder...\n");
    while ((err = transcode_encode_frame(ctx, NULL, out_packet)) == TRANSCODE_SUCCESS) {
        err = transcode_write_packet(ctx, out_packet);
        if (err == TRANSCODE_SUCCESS) {
            packets_written++;
        }
    }

    printf("\nTranscode complete!\n");
    printf("  Packets read:    %llu\n", (unsigned long long)packets_read);
    printf("  Frames decoded:  %llu\n", (unsigned long long)frames_decoded);
    printf("  Packets written: %llu\n", (unsigned long long)packets_written);

cleanup:
    transcode_frame_free(frame);
    transcode_packet_free(out_packet);
    transcode_packet_free(in_packet);
    transcode_close(ctx);

    return result;
}

/**
 * Print usage information.
 */
static void print_usage(const char* progname) {
    printf("Transcode Demo - C API Example\n\n");
    printf("Usage:\n");
    printf("  %s probe <input>           - Probe file and show stream info\n", progname);
    printf("  %s <input> <output>        - Transcode input to output\n", progname);
    printf("\nExamples:\n");
    printf("  %s probe video.mp4\n", progname);
    printf("  %s input.mp4 output.mp4\n", progname);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* Probe mode */
    if (argc == 3 && strcmp(argv[1], "probe") == 0) {
        return probe_file(argv[2]);
    }

    /* Transcode mode */
    if (argc == 3) {
        return transcode_file(argv[1], argv[2]);
    }

    print_usage(argv[0]);
    return 1;
}
