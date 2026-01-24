// Package main demonstrates basic usage of the transcode Go bindings.
package main

import (
	"fmt"
	"os"

	"github.com/anthropics/transcode-go/transcode"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <input-file> [output-file]\n", os.Args[0])
		os.Exit(1)
	}

	inputPath := os.Args[1]
	outputPath := "output.mp4"
	if len(os.Args) >= 3 {
		outputPath = os.Args[2]
	}

	// Print library version
	fmt.Printf("Transcode version: %s\n", transcode.Version())

	// Open input file
	ctx, err := transcode.OpenInput(inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open input: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Close()

	// Print stream information
	numStreams := ctx.NumStreams()
	fmt.Printf("Input file: %s\n", inputPath)
	fmt.Printf("Number of streams: %d\n", numStreams)

	for i := 0; i < numStreams; i++ {
		info, err := ctx.StreamInfo(i)
		if err != nil {
			fmt.Printf("  Stream %d: error getting info: %v\n", i, err)
			continue
		}

		fmt.Printf("  Stream %d:\n", i)
		fmt.Printf("    Type: %s\n", info.Type)
		fmt.Printf("    Codec: %s\n", info.CodecName)

		if info.Type == transcode.StreamTypeVideo {
			fmt.Printf("    Resolution: %dx%d\n", info.Width, info.Height)
			if info.FramerateDen > 0 {
				fps := float64(info.FramerateNum) / float64(info.FramerateDen)
				fmt.Printf("    Frame rate: %.2f fps\n", fps)
			}
			fmt.Printf("    Pixel format: %d\n", info.PixelFormat)
		} else if info.Type == transcode.StreamTypeAudio {
			fmt.Printf("    Sample rate: %d Hz\n", info.SampleRate)
			fmt.Printf("    Channels: %d\n", info.Channels)
		}

		if info.Bitrate > 0 {
			fmt.Printf("    Bitrate: %d kbps\n", info.Bitrate/1000)
		}
		if info.Duration > 0 {
			fmt.Printf("    Duration: %.2f seconds\n", info.Duration)
		}
	}

	// Configure output
	config := transcode.Config{
		Width:        1280,
		Height:       720,
		PixelFormat:  transcode.PixelFormatYUV420P,
		Bitrate:      2500000, // 2.5 Mbps
		GOPSize:      60,
		FramerateNum: 30,
		FramerateDen: 1,
		Preset:       5, // Medium preset
		Threads:      4,
	}

	// Open output
	err = ctx.OpenOutput(outputPath, &config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open output: %v\n", err)
		os.Exit(1)
	}

	// Process packets
	pkt := transcode.NewPacket()
	defer pkt.Free()

	frm := transcode.NewFrame()
	defer frm.Free()

	packetCount := 0
	frameCount := 0

	fmt.Println("\nProcessing...")

	for {
		// Read packet from input
		err := ctx.ReadPacket(pkt)
		if err == transcode.ErrEndOfStream {
			break
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading packet: %v\n", err)
			break
		}

		packetCount++

		// Decode packet to frame
		err = ctx.DecodePacket(pkt, frm)
		if err == transcode.ErrEndOfStream {
			// Need more packets to produce a frame
			continue
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error decoding packet: %v\n", err)
			continue
		}

		frameCount++

		// Encode and write frame
		err = ctx.EncodeFrame(frm, pkt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error encoding frame: %v\n", err)
			continue
		}

		err = ctx.WritePacket(pkt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error writing packet: %v\n", err)
			continue
		}

		// Progress indicator
		if frameCount%100 == 0 {
			fmt.Printf("\rProcessed %d frames...", frameCount)
		}
	}

	// Flush encoders
	if err := ctx.FlushEncoder(); err != nil && err != transcode.ErrEndOfStream {
		fmt.Fprintf(os.Stderr, "Error flushing encoder: %v\n", err)
	}

	fmt.Printf("\n\nComplete!\n")
	fmt.Printf("  Packets read: %d\n", packetCount)
	fmt.Printf("  Frames processed: %d\n", frameCount)
	fmt.Printf("  Output file: %s\n", outputPath)
}
