// Transcode Example - Basic Usage
// Demonstrates reading and transcoding a video file

import Foundation
import Transcode

// MARK: - Main

@main
struct TranscodeExample {
    static func main() async {
        // Parse command line arguments
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("Usage: \(args[0]) <input-file> [output-file]")
            exit(1)
        }

        let inputPath = args[1]
        let outputPath = args.count >= 3 ? args[2] : "output.mp4"

        // Print version
        print("Transcode version: \(version())")

        do {
            try await transcode(input: inputPath, output: outputPath)
        } catch {
            print("Error: \(error)")
            exit(1)
        }
    }

    static func transcode(input inputPath: String, output outputPath: String) async throws {
        // Open input file
        let context = try TranscodeContext(path: inputPath)
        defer { context.close() }

        // Print stream information
        print("Input file: \(inputPath)")
        print("Number of streams: \(context.numStreams)")

        for i in 0..<context.numStreams {
            let info = try context.streamInfo(at: i)
            print("  Stream \(i):")
            print("    Type: \(info.type)")

            if info.type == .video {
                print("    Resolution: \(info.width)x\(info.height)")
                if info.timeBaseDen > 0 {
                    let fps = Double(info.timeBaseNum) / Double(info.timeBaseDen)
                    print("    Frame rate: \(String(format: "%.2f", fps)) fps")
                }
            } else if info.type == .audio {
                print("    Sample rate: \(info.sampleRate) Hz")
                print("    Channels: \(info.channels)")
            }

            if info.bitrate > 0 {
                print("    Bitrate: \(info.bitrate / 1000) kbps")
            }
        }

        // Configure output
        var config = TranscodeConfig()
        config.width = 1280
        config.height = 720
        config.pixelFormat = .yuv420p
        config.bitrate = 2_500_000  // 2.5 Mbps
        config.gopSize = 60
        config.framerateNum = 30
        config.framerateDen = 1
        config.preset = 5  // Medium

        // Open output
        try context.openOutput(path: outputPath, config: config)

        // Process using async/await
        let packet = Packet()
        let frame = Frame()
        var packetCount = 0
        var frameCount = 0

        print("\nProcessing...")

        // Read packets using async sequence
        for try await pkt in context.packets() {
            packetCount += 1

            // Decode
            do {
                try context.decodePacket(pkt, into: frame)
                frameCount += 1

                // Encode and write
                try context.encodeFrame(frame, into: packet)
                try context.writePacket(packet)

                // Progress indicator
                if frameCount % 100 == 0 {
                    print("\rProcessed \(frameCount) frames...", terminator: "")
                    fflush(stdout)
                }
            } catch TranscodeError.resourceExhausted {
                // Need more packets for a frame
                continue
            }
        }

        // Flush encoder
        do {
            while true {
                try context.encodeFlush(into: packet)
                try context.writePacket(packet)
            }
        } catch TranscodeError.endOfStream {
            // Expected
        }

        try context.flushEncoder()

        print("\n\nComplete!")
        print("  Packets read: \(packetCount)")
        print("  Frames processed: \(frameCount)")
        print("  Output file: \(outputPath)")
    }
}
