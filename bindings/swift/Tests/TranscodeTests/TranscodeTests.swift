import XCTest
@testable import Transcode

final class TranscodeTests: XCTestCase {

    func testVersion() {
        let v = version()
        XCTAssertFalse(v.isEmpty, "Version should not be empty")
        print("Transcode version: \(v)")
    }

    func testStreamType() {
        XCTAssertEqual(StreamType.unknown.description, "unknown")
        XCTAssertEqual(StreamType.video.description, "video")
        XCTAssertEqual(StreamType.audio.description, "audio")
        XCTAssertEqual(StreamType.subtitle.description, "subtitle")
        XCTAssertEqual(StreamType.data.description, "data")
    }

    func testPixelFormats() {
        // Test that all pixel formats have unique values
        let formats: [PixelFormat] = [
            .unknown, .yuv420p, .yuv422p, .yuv444p,
            .nv12, .rgb24, .rgba
        ]

        var seen = Set<Int32>()
        for format in formats {
            XCTAssertFalse(seen.contains(format.rawValue),
                          "Duplicate pixel format value: \(format.rawValue)")
            seen.insert(format.rawValue)
        }
    }

    func testPacketCreation() {
        let packet = Packet()
        XCTAssertEqual(packet.size, 0, "New packet should have size 0")
        XCTAssertNil(packet.data, "New packet should have no data")
        XCTAssertEqual(packet.pts, 0)
        XCTAssertEqual(packet.dts, 0)
        XCTAssertFalse(packet.isKeyframe)
    }

    func testFrameCreation() {
        let frame = Frame()
        XCTAssertEqual(frame.width, 0, "New frame should have width 0")
        XCTAssertEqual(frame.height, 0, "New frame should have height 0")
        XCTAssertEqual(frame.format, .unknown)
        XCTAssertFalse(frame.isKeyframe)
    }

    func testConfigDefaults() {
        let config = TranscodeConfig()
        XCTAssertEqual(config.width, 0)
        XCTAssertEqual(config.height, 0)
        XCTAssertEqual(config.quality, 23)
        XCTAssertEqual(config.preset, 5)
        XCTAssertEqual(config.framerateDen, 1)
    }

    func testConfigToCConfig() {
        var config = TranscodeConfig()
        config.width = 1920
        config.height = 1080
        config.bitrate = 5_000_000
        config.quality = 20
        config.gopSize = 60
        config.preset = 3

        let cConfig = config.toCConfig()
        XCTAssertEqual(cConfig.width, 1920)
        XCTAssertEqual(cConfig.height, 1080)
        XCTAssertEqual(cConfig.bitrate, 5_000_000)
        XCTAssertEqual(cConfig.quality, 20)
        XCTAssertEqual(cConfig.gop_size, 60)
        XCTAssertEqual(cConfig.preset, 3)
    }

    func testOpenInputNotFound() {
        XCTAssertThrowsError(try TranscodeContext(path: "/nonexistent/path/to/file.mp4")) { error in
            guard case TranscodeError.io = error else {
                XCTFail("Expected io error, got \(error)")
                return
            }
        }
    }

    func testErrorDescriptions() {
        let errors: [TranscodeError] = [
            .invalidArgument,
            .nullPointer,
            .endOfStream,
            .io(nil),
            .codec(nil),
            .container(nil),
            .resourceExhausted,
            .unsupported(nil),
            .cancelled,
            .bufferTooSmall,
            .invalidState,
            .unknown(-999)
        ]

        for error in errors {
            XCTAssertFalse(error.description.isEmpty,
                          "Error description should not be empty for \(error)")
        }
    }

    func testColorSpaces() {
        XCTAssertEqual(ColorSpace.bt601.rawValue, 0)
        XCTAssertEqual(ColorSpace.bt709.rawValue, 1)
        XCTAssertEqual(ColorSpace.bt2020.rawValue, 2)
        XCTAssertEqual(ColorSpace.srgb.rawValue, 3)
    }

    func testColorRanges() {
        XCTAssertEqual(ColorRange.limited.rawValue, 0)
        XCTAssertEqual(ColorRange.full.rawValue, 1)
    }
}
