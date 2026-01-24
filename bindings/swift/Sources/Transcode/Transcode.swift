// Transcode Swift SDK
// High-level Swift wrapper for the Transcode C API

import Foundation
import CTranscode

// MARK: - Errors

/// Errors thrown by Transcode operations.
public enum TranscodeError: Error, CustomStringConvertible {
    case invalidArgument
    case nullPointer
    case endOfStream
    case io(String?)
    case codec(String?)
    case container(String?)
    case resourceExhausted
    case unsupported(String?)
    case cancelled
    case bufferTooSmall
    case invalidState
    case unknown(Int32)

    init(from error: CTranscode.TranscodeError) {
        switch error {
        case TRANSCODE_SUCCESS:
            // This shouldn't happen, but handle gracefully
            self = .unknown(0)
        case TRANSCODE_ERROR_INVALID_ARGUMENT:
            self = .invalidArgument
        case TRANSCODE_ERROR_NULL_POINTER:
            self = .nullPointer
        case TRANSCODE_ERROR_END_OF_STREAM:
            self = .endOfStream
        case TRANSCODE_ERROR_IO:
            self = .io(nil)
        case TRANSCODE_ERROR_CODEC:
            self = .codec(nil)
        case TRANSCODE_ERROR_CONTAINER:
            self = .container(nil)
        case TRANSCODE_ERROR_RESOURCE_EXHAUSTED:
            self = .resourceExhausted
        case TRANSCODE_ERROR_UNSUPPORTED:
            self = .unsupported(nil)
        case TRANSCODE_ERROR_CANCELLED:
            self = .cancelled
        case TRANSCODE_ERROR_BUFFER_TOO_SMALL:
            self = .bufferTooSmall
        case TRANSCODE_ERROR_INVALID_STATE:
            self = .invalidState
        default:
            self = .unknown(error.rawValue)
        }
    }

    public var description: String {
        switch self {
        case .invalidArgument:
            return "Invalid argument"
        case .nullPointer:
            return "Null pointer"
        case .endOfStream:
            return "End of stream"
        case .io(let msg):
            return msg ?? "I/O error"
        case .codec(let msg):
            return msg ?? "Codec error"
        case .container(let msg):
            return msg ?? "Container error"
        case .resourceExhausted:
            return "Resource exhausted"
        case .unsupported(let msg):
            return msg ?? "Unsupported operation"
        case .cancelled:
            return "Operation cancelled"
        case .bufferTooSmall:
            return "Buffer too small"
        case .invalidState:
            return "Invalid state"
        case .unknown(let code):
            return "Unknown error (\(code))"
        }
    }
}

// MARK: - Stream Types

/// Type of media stream.
public enum StreamType: UInt32, CustomStringConvertible {
    case unknown = 0
    case video = 1
    case audio = 2
    case subtitle = 3
    case data = 4

    public var description: String {
        switch self {
        case .unknown: return "unknown"
        case .video: return "video"
        case .audio: return "audio"
        case .subtitle: return "subtitle"
        case .data: return "data"
        }
    }
}

/// Pixel format for video frames.
public enum PixelFormat: Int32 {
    case unknown = 0
    case yuv420p = 1
    case yuv422p = 2
    case yuv444p = 3
    case yuv420p10le = 4
    case yuv422p10le = 5
    case yuv444p10le = 6
    case nv12 = 7
    case nv21 = 8
    case rgb24 = 9
    case bgr24 = 10
    case rgba = 11
    case bgra = 12
    case gray8 = 13
    case gray16 = 14
}

/// Color space for video frames.
public enum ColorSpace: Int32 {
    case bt601 = 0
    case bt709 = 1
    case bt2020 = 2
    case srgb = 3
}

/// Color range for video frames.
public enum ColorRange: Int32 {
    case limited = 0
    case full = 1
}

// MARK: - Stream Info

/// Information about a media stream.
public struct StreamInfo {
    /// Stream index (0-based).
    public let index: Int

    /// Type of stream.
    public let type: StreamType

    /// Codec identifier.
    public let codecId: UInt32

    /// Video width in pixels (0 for non-video).
    public let width: Int

    /// Video height in pixels (0 for non-video).
    public let height: Int

    /// Pixel format (for video streams).
    public let pixelFormat: PixelFormat

    /// Sample rate in Hz (for audio streams).
    public let sampleRate: Int

    /// Number of audio channels (for audio streams).
    public let channels: Int

    /// Bits per audio sample.
    public let bitsPerSample: Int

    /// Time base numerator.
    public let timeBaseNum: Int32

    /// Time base denominator.
    public let timeBaseDen: Int32

    /// Duration in time base units.
    public let duration: Int64

    /// Bitrate in bits per second.
    public let bitrate: UInt64

    init(from info: TranscodeStreamInfo) {
        self.index = Int(info.index)
        self.type = StreamType(rawValue: info.stream_type.rawValue) ?? .unknown
        self.codecId = info.codec_id
        self.width = Int(info.width)
        self.height = Int(info.height)
        self.pixelFormat = PixelFormat(rawValue: Int32(info.pixel_format.rawValue)) ?? .unknown
        self.sampleRate = Int(info.sample_rate)
        self.channels = Int(info.channels)
        self.bitsPerSample = Int(info.bits_per_sample)
        self.timeBaseNum = info.time_base_num
        self.timeBaseDen = info.time_base_den
        self.duration = info.duration
        self.bitrate = info.bitrate
    }
}

// MARK: - Config

/// Configuration for encoding/transcoding.
public struct TranscodeConfig {
    /// Output width in pixels (0 = same as input).
    public var width: Int = 0

    /// Output height in pixels (0 = same as input).
    public var height: Int = 0

    /// Output pixel format.
    public var pixelFormat: PixelFormat = .unknown

    /// Target bitrate in bits per second.
    public var bitrate: UInt64 = 0

    /// Maximum bitrate for VBR.
    public var maxBitrate: UInt64 = 0

    /// Quality level (0-51, lower = better).
    public var quality: Int32 = 23

    /// GOP size / keyframe interval.
    public var gopSize: Int = 0

    /// Number of B-frames.
    public var bFrames: Int = 0

    /// Frame rate numerator.
    public var framerateNum: Int = 0

    /// Frame rate denominator.
    public var framerateDen: Int = 1

    /// Audio sample rate in Hz.
    public var sampleRate: Int = 0

    /// Number of audio channels.
    public var channels: Int = 0

    /// Encoder preset (0-9, lower = slower but better).
    public var preset: Int = 5

    /// Number of encoding threads.
    public var threads: Int = 0

    public init() {}

    func toCConfig() -> CTranscode.TranscodeConfig {
        var config = CTranscode.TranscodeConfig()
        config.width = UInt32(width)
        config.height = UInt32(height)
        config.pixel_format = CTranscode.TranscodePixelFormat(rawValue: UInt32(pixelFormat.rawValue))
        config.bitrate = bitrate
        config.max_bitrate = maxBitrate
        config.quality = quality
        config.gop_size = UInt32(gopSize)
        config.b_frames = UInt32(bFrames)
        config.framerate_num = UInt32(framerateNum)
        config.framerate_den = UInt32(framerateDen)
        config.sample_rate = UInt32(sampleRate)
        config.channels = UInt32(channels)
        config.preset = UInt32(preset)
        config.threads = UInt32(threads)
        return config
    }
}

// MARK: - Packet

/// An encoded media packet.
public final class Packet {
    internal var ptr: UnsafeMutablePointer<TranscodePacket>?

    /// Create a new empty packet.
    public init() {
        ptr = transcode_packet_alloc()
    }

    deinit {
        if let ptr = ptr {
            transcode_packet_free(ptr)
        }
    }

    /// Size of packet data in bytes.
    public var size: Int {
        guard let ptr = ptr else { return 0 }
        return ptr.pointee.size
    }

    /// Raw packet data.
    public var data: Data? {
        guard let ptr = ptr, ptr.pointee.data != nil, ptr.pointee.size > 0 else {
            return nil
        }
        return Data(bytes: ptr.pointee.data, count: ptr.pointee.size)
    }

    /// Presentation timestamp.
    public var pts: Int64 {
        get { ptr?.pointee.pts ?? 0 }
        set { ptr?.pointee.pts = newValue }
    }

    /// Decode timestamp.
    public var dts: Int64 {
        get { ptr?.pointee.dts ?? 0 }
        set { ptr?.pointee.dts = newValue }
    }

    /// Duration in time base units.
    public var duration: Int64 {
        get { ptr?.pointee.duration ?? 0 }
        set { ptr?.pointee.duration = newValue }
    }

    /// Stream index.
    public var streamIndex: Int {
        get { Int(ptr?.pointee.stream_index ?? 0) }
        set { ptr?.pointee.stream_index = UInt32(newValue) }
    }

    /// Packet flags.
    public var flags: UInt32 {
        get { ptr?.pointee.flags ?? 0 }
        set { ptr?.pointee.flags = newValue }
    }

    /// Whether this packet is a keyframe.
    public var isKeyframe: Bool {
        (flags & UInt32(TRANSCODE_PACKET_FLAG_KEYFRAME)) != 0
    }
}

// MARK: - Frame

/// A decoded video frame.
public final class Frame {
    internal var ptr: UnsafeMutablePointer<TranscodeFrame>?

    /// Create a new empty frame.
    public init() {
        ptr = transcode_frame_alloc()
    }

    deinit {
        if let ptr = ptr {
            transcode_frame_free(ptr)
        }
    }

    /// Frame width in pixels.
    public var width: Int {
        Int(ptr?.pointee.width ?? 0)
    }

    /// Frame height in pixels.
    public var height: Int {
        Int(ptr?.pointee.height ?? 0)
    }

    /// Pixel format.
    public var format: PixelFormat {
        guard let ptr = ptr else { return .unknown }
        return PixelFormat(rawValue: Int32(ptr.pointee.format.rawValue)) ?? .unknown
    }

    /// Presentation timestamp.
    public var pts: Int64 {
        get { ptr?.pointee.pts ?? 0 }
        set { ptr?.pointee.pts = newValue }
    }

    /// Decode timestamp.
    public var dts: Int64 {
        get { ptr?.pointee.dts ?? 0 }
        set { ptr?.pointee.dts = newValue }
    }

    /// Duration in time base units.
    public var duration: Int64 {
        get { ptr?.pointee.duration ?? 0 }
        set { ptr?.pointee.duration = newValue }
    }

    /// Frame flags.
    public var flags: UInt32 {
        get { ptr?.pointee.flags ?? 0 }
        set { ptr?.pointee.flags = newValue }
    }

    /// Whether this frame is a keyframe.
    public var isKeyframe: Bool {
        (flags & UInt32(TRANSCODE_FRAME_FLAG_KEYFRAME)) != 0
    }
}

// MARK: - Context

/// Main transcoding context.
public final class TranscodeContext {
    private var ptr: UnsafeMutablePointer<CTranscode.TranscodeContext>?

    /// Open an input file for reading.
    public init(path: String) throws {
        var ctx: UnsafeMutablePointer<CTranscode.TranscodeContext>?
        let result = transcode_open_input(path, &ctx)
        guard result == TRANSCODE_SUCCESS else {
            throw TranscodeError(from: result)
        }
        self.ptr = ctx
    }

    deinit {
        close()
    }

    /// Close the context and release resources.
    public func close() {
        if let ptr = ptr {
            transcode_close(ptr)
            self.ptr = nil
        }
    }

    /// Number of streams in the input.
    public var numStreams: Int {
        guard let ptr = ptr else { return 0 }
        return Int(ptr.pointee.num_streams)
    }

    /// Get information about a specific stream.
    public func streamInfo(at index: Int) throws -> StreamInfo {
        guard let ptr = ptr else {
            throw TranscodeError.invalidState
        }
        var info = TranscodeStreamInfo()
        let result = transcode_get_stream_info(ptr, UInt32(index), &info)
        guard result == TRANSCODE_SUCCESS else {
            throw TranscodeError(from: result)
        }
        return StreamInfo(from: info)
    }

    /// Open an output file for writing.
    public func openOutput(path: String, config: TranscodeConfig? = nil) throws {
        guard let ptr = ptr else {
            throw TranscodeError.invalidState
        }
        var cConfig = config?.toCConfig() ?? CTranscode.TranscodeConfig()
        let result = transcode_open_output(ptr, path, &cConfig)
        guard result == TRANSCODE_SUCCESS else {
            throw TranscodeError(from: result)
        }
    }

    /// Read the next packet from the input.
    public func readPacket(_ packet: Packet) throws {
        guard let ptr = ptr, let pktPtr = packet.ptr else {
            throw TranscodeError.invalidState
        }
        let result = transcode_read_packet(ptr, pktPtr)
        guard result == TRANSCODE_SUCCESS else {
            throw TranscodeError(from: result)
        }
    }

    /// Write a packet to the output.
    public func writePacket(_ packet: Packet) throws {
        guard let ptr = ptr, let pktPtr = packet.ptr else {
            throw TranscodeError.invalidState
        }
        let result = transcode_write_packet(ptr, pktPtr)
        guard result == TRANSCODE_SUCCESS else {
            throw TranscodeError(from: result)
        }
    }

    /// Decode a packet into a frame.
    public func decodePacket(_ packet: Packet, into frame: Frame) throws {
        guard let ptr = ptr, let pktPtr = packet.ptr, let frmPtr = frame.ptr else {
            throw TranscodeError.invalidState
        }
        let result = transcode_decode_packet(ptr, pktPtr, frmPtr)
        guard result == TRANSCODE_SUCCESS else {
            throw TranscodeError(from: result)
        }
    }

    /// Encode a frame into a packet.
    public func encodeFrame(_ frame: Frame, into packet: Packet) throws {
        guard let ptr = ptr, let frmPtr = frame.ptr, let pktPtr = packet.ptr else {
            throw TranscodeError.invalidState
        }
        let result = transcode_encode_frame(ptr, frmPtr, pktPtr)
        guard result == TRANSCODE_SUCCESS else {
            throw TranscodeError(from: result)
        }
    }

    /// Flush encoder to get remaining packets.
    public func encodeFlush(into packet: Packet) throws {
        guard let ptr = ptr, let pktPtr = packet.ptr else {
            throw TranscodeError.invalidState
        }
        let result = transcode_encode_frame(ptr, nil, pktPtr)
        guard result == TRANSCODE_SUCCESS else {
            throw TranscodeError(from: result)
        }
    }

    /// Flush decoder buffers.
    public func flushDecoder() throws {
        guard let ptr = ptr else {
            throw TranscodeError.invalidState
        }
        let result = transcode_flush_decoder(ptr)
        guard result == TRANSCODE_SUCCESS else {
            throw TranscodeError(from: result)
        }
    }

    /// Flush encoder buffers.
    public func flushEncoder() throws {
        guard let ptr = ptr else {
            throw TranscodeError.invalidState
        }
        let result = transcode_flush_encoder(ptr)
        guard result == TRANSCODE_SUCCESS else {
            throw TranscodeError(from: result)
        }
    }

    /// Seek to a specific timestamp.
    public func seek(to timestamp: Int64, streamIndex: Int = -1, flags: Int = 0) throws {
        guard let ptr = ptr else {
            throw TranscodeError.invalidState
        }
        let result = transcode_seek(ptr, Int32(streamIndex), timestamp, Int32(flags))
        guard result == TRANSCODE_SUCCESS else {
            throw TranscodeError(from: result)
        }
    }
}

// MARK: - AsyncSequence Support

/// An async sequence of packets from a transcode context.
public struct AsyncPacketSequence: AsyncSequence {
    public typealias Element = Packet

    private let context: TranscodeContext

    init(context: TranscodeContext) {
        self.context = context
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(context: context)
    }

    public struct AsyncIterator: AsyncIteratorProtocol {
        private let context: TranscodeContext
        private let packet = Packet()

        init(context: TranscodeContext) {
            self.context = context
        }

        public mutating func next() async throws -> Packet? {
            do {
                try context.readPacket(packet)
                return packet
            } catch TranscodeError.endOfStream {
                return nil
            }
        }
    }
}

extension TranscodeContext {
    /// Returns an async sequence of packets from the input.
    public func packets() -> AsyncPacketSequence {
        AsyncPacketSequence(context: self)
    }
}

// MARK: - Utility Functions

/// Get the library version string.
public func version() -> String {
    guard let cStr = transcode_version() else {
        return "unknown"
    }
    return String(cString: cStr)
}

/// Get a human-readable error description.
public func errorString(_ error: CTranscode.TranscodeError) -> String {
    guard let cStr = transcode_error_string(error) else {
        return "Unknown error"
    }
    return String(cString: cStr)
}
