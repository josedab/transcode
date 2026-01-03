/**
 * @file decode_frames.c
 * @brief Simple frame decoding example
 *
 * This example shows the minimal code needed to:
 * 1. Open a video file
 * 2. Decode all frames
 * 3. Access frame data
 *
 * Build:
 *   gcc -o decode_frames decode_frames.c -I../include -L../../target/release \
 *       -ltranscode_capi -lpthread -ldl -lm
 *
 * Usage:
 *   ./decode_frames video.mp4
 */

#include <stdio.h>
#include <stdlib.h>
#include "../include/transcode.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    TranscodeContext* ctx = NULL;
    TranscodePacket* packet = NULL;
    TranscodeFrame* frame = NULL;
    TranscodeError err;
    int frame_count = 0;

    /* Open input file */
    err = transcode_open_input(argv[1], &ctx);
    if (err != TRANSCODE_SUCCESS) {
        fprintf(stderr, "Failed to open '%s': %s\n", argv[1], transcode_error_string(err));
        return 1;
    }

    printf("Opened: %s\n", argv[1]);
    printf("Streams: %u\n\n", ctx->num_streams);

    /* Allocate packet and frame */
    packet = transcode_packet_alloc();
    frame = transcode_frame_alloc();

    if (!packet || !frame) {
        fprintf(stderr, "Failed to allocate packet/frame\n");
        transcode_close(ctx);
        return 1;
    }

    /* Read and decode all packets */
    while ((err = transcode_read_packet(ctx, packet)) == TRANSCODE_SUCCESS) {
        /* Decode packet to frame */
        err = transcode_decode_packet(ctx, packet, frame);
        if (err == TRANSCODE_SUCCESS) {
            frame_count++;

            /* Print frame info every 30 frames */
            if (frame_count % 30 == 1) {
                printf("Frame #%d: %ux%u, PTS=%lld",
                       frame_count,
                       frame->width,
                       frame->height,
                       (long long)frame->pts);

                if (frame->flags & TRANSCODE_FRAME_FLAG_KEYFRAME) {
                    printf(" [KEYFRAME]");
                }
                printf("\n");
            }
        }
    }

    if (err == TRANSCODE_ERROR_END_OF_STREAM) {
        printf("\nDecoding complete!\n");
    } else {
        fprintf(stderr, "\nDecoding stopped with error: %s\n", transcode_error_string(err));
    }

    printf("Total frames decoded: %d\n", frame_count);

    /* Cleanup */
    transcode_frame_free(frame);
    transcode_packet_free(packet);
    transcode_close(ctx);

    return 0;
}
