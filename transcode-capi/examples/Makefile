# Transcode C API Examples Makefile

CC = gcc
CFLAGS = -Wall -Wextra -I../include
LDFLAGS = -L../../target/release -ltranscode_capi -lpthread -ldl -lm

# On macOS, use different linker flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS = -L../../target/release -ltranscode_capi -lpthread -lm
endif

EXAMPLES = transcode_demo decode_frames

.PHONY: all clean build-lib

all: build-lib $(EXAMPLES)

# Build the Rust library first
build-lib:
	@echo "Building transcode-capi library..."
	cd ../.. && cargo build --release -p transcode-capi

transcode_demo: transcode_demo.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

decode_frames: decode_frames.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(EXAMPLES)

# Run examples (requires building first)
run-demo: transcode_demo
	@echo "Running transcode_demo probe..."
	LD_LIBRARY_PATH=../../target/release DYLD_LIBRARY_PATH=../../target/release ./transcode_demo probe ../../test_data/sample.mp4 || echo "(no test file available)"

run-decode: decode_frames
	@echo "Running decode_frames..."
	LD_LIBRARY_PATH=../../target/release DYLD_LIBRARY_PATH=../../target/release ./decode_frames ../../test_data/sample.mp4 || echo "(no test file available)"

help:
	@echo "Transcode C API Examples"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build all examples (default)"
	@echo "  clean       - Remove built examples"
	@echo "  run-demo    - Run transcode_demo"
	@echo "  run-decode  - Run decode_frames"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build all examples"
	@echo "  make transcode_demo     # Build just transcode_demo"
	@echo "  make run-demo           # Build and run demo"
