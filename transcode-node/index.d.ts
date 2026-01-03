/**
 * Node.js bindings for the Transcode codec library.
 * A memory-safe, high-performance universal codec library.
 *
 * @example
 * ```javascript
 * const { transcode, probe } = require('@transcode/node');
 *
 * // Probe a media file
 * const info = await probe('input.mp4');
 * console.log(info.duration, info.video_streams[0].width);
 *
 * // Transcode with progress callback
 * await transcode('input.mp4', 'output.mp4', {
 *   videoBitrate: 5_000_000,
 *   onProgress: (p) => console.log(`${p.percent}%`)
 * });
 * ```
 */

/**
 * Video stream information.
 */
export interface VideoStreamInfo {
  /** Stream index. */
  index: number;
  /** Codec name (e.g., "h264", "h265", "vp9"). */
  codec: string;
  /** Width in pixels. */
  width: number;
  /** Height in pixels. */
  height: number;
  /** Frame rate (frames per second). */
  frameRate?: number;
  /** Bit depth. */
  bitDepth: number;
  /** Duration in seconds. */
  duration?: number;
}

/**
 * Audio stream information.
 */
export interface AudioStreamInfo {
  /** Stream index. */
  index: number;
  /** Codec name (e.g., "aac", "mp3", "opus"). */
  codec: string;
  /** Sample rate in Hz. */
  sampleRate: number;
  /** Number of channels. */
  channels: number;
  /** Bits per sample. */
  bitsPerSample: number;
  /** Duration in seconds. */
  duration?: number;
}

/**
 * Media file information returned by probe().
 */
export interface MediaInfo {
  /** Container format (e.g., "mp4", "mkv", "webm"). */
  format: string;
  /** Total duration in seconds. */
  duration?: number;
  /** File size in bytes. */
  size: number;
  /** Video streams in the file. */
  videoStreams: VideoStreamInfo[];
  /** Audio streams in the file. */
  audioStreams: AudioStreamInfo[];
  /** Total bitrate in bits per second. */
  bitrate?: number;
}

/**
 * Progress information emitted during transcoding.
 */
export interface Progress {
  /** Progress percentage (0-100). */
  percent: number;
  /** Frames processed. */
  frames: number;
  /** Current speed (realtime multiplier, e.g., 2.5x). */
  speed: number;
  /** Estimated time remaining in seconds. */
  eta?: number;
  /** Current output size in bytes. */
  size: number;
  /** Current bitrate in bits per second. */
  bitrate: number;
}

/**
 * Transcoding statistics returned after completion.
 */
export interface TranscodeStats {
  /** Total frames decoded. */
  framesDecoded: number;
  /** Total frames encoded. */
  framesEncoded: number;
  /** Input file size in bytes. */
  inputSize: number;
  /** Output file size in bytes. */
  outputSize: number;
  /** Compression ratio achieved. */
  compressionRatio: number;
  /** Average encoding speed (realtime multiplier). */
  averageSpeed: number;
  /** Total duration processed in seconds. */
  duration: number;
}

/**
 * Transcoding options.
 */
export interface TranscodeOptions {
  /** Video codec (e.g., "h264", "h265", "av1"). */
  videoCodec?: string;
  /** Audio codec (e.g., "aac", "mp3", "opus"). */
  audioCodec?: string;
  /** Video bitrate in bits per second. */
  videoBitrate?: number;
  /** Audio bitrate in bits per second. */
  audioBitrate?: number;
  /** Output width in pixels. */
  width?: number;
  /** Output height in pixels. */
  height?: number;
  /** Frame rate. */
  frameRate?: number;
  /** Audio sample rate in Hz. */
  sampleRate?: number;
  /** Number of audio channels. */
  channels?: number;
  /** Number of threads for encoding (0 = auto). */
  threads?: number;
  /** Enable hardware acceleration. */
  hardwareAcceleration?: boolean;
  /** Overwrite output file if exists. */
  overwrite?: boolean;
  /** Start time in seconds (for trimming). */
  startTime?: number;
  /** Duration in seconds (for trimming). */
  duration?: number;
  /** Encoder preset (e.g., "ultrafast", "medium", "slow"). */
  preset?: string;
  /** CRF value for quality-based encoding (0-51, lower is better). */
  crf?: number;
  /** Progress callback function. */
  onProgress?: (progress: Progress) => void;
}

/**
 * SIMD capabilities detected on the current system.
 */
export interface SimdCapabilities {
  /** SSE4.2 support (x86_64). */
  sse42: boolean;
  /** AVX2 support (x86_64). */
  avx2: boolean;
  /** AVX-512 support (x86_64). */
  avx512: boolean;
  /** FMA support (x86_64). */
  fma: boolean;
  /** NEON support (ARM). */
  neon: boolean;
  /** SVE support (ARM). */
  sve: boolean;
}

/**
 * Build information about the library.
 */
export interface BuildInfo {
  /** Library version. */
  version: string;
  /** Target architecture. */
  arch: string;
  /** Operating system. */
  os: string;
  /** Debug build. */
  debug: boolean;
}

/**
 * Transcoder class for performing media transcoding operations.
 *
 * @example
 * ```javascript
 * const transcoder = new Transcoder('input.mp4', 'output.mp4', {
 *   videoBitrate: 5_000_000,
 *   audioCodec: 'aac'
 * });
 *
 * const stats = await transcoder.run((progress) => {
 *   console.log(`Progress: ${progress.percent}%`);
 * });
 *
 * console.log(`Compression ratio: ${stats.compressionRatio}`);
 * ```
 */
export class Transcoder {
  /**
   * Create a new Transcoder instance.
   * @param input - Path to the input media file.
   * @param output - Path for the output file.
   * @param options - Optional transcoding options.
   */
  constructor(input: string, output: string, options?: TranscodeOptions);

  /**
   * Run the transcoding operation asynchronously.
   * @param onProgress - Optional callback for progress updates.
   * @returns Promise that resolves to TranscodeStats on completion.
   */
  run(onProgress?: (progress: Progress) => void): Promise<TranscodeStats>;

  /**
   * Cancel an ongoing transcoding operation.
   */
  cancel(): void;

  /**
   * Get the current progress (frames processed).
   */
  getProgress(): number;
}

/**
 * Probe a media file to get information about its streams.
 *
 * @param input - Path to the media file.
 * @returns Promise that resolves to MediaInfo.
 *
 * @example
 * ```javascript
 * const info = await probe('video.mp4');
 * console.log(`Duration: ${info.duration}s`);
 * console.log(`Resolution: ${info.videoStreams[0].width}x${info.videoStreams[0].height}`);
 * ```
 */
export function probe(input: string): Promise<MediaInfo>;

/**
 * High-level transcode function.
 *
 * @param input - Path to the input media file.
 * @param output - Path for the output file.
 * @param options - Optional transcoding options.
 * @returns Promise that resolves to TranscodeStats on completion.
 *
 * @example
 * ```javascript
 * const stats = await transcode('input.mp4', 'output.mp4', {
 *   videoBitrate: 5_000_000,
 *   videoCodec: 'h264',
 *   audioCodec: 'aac',
 *   overwrite: true
 * });
 * ```
 */
export function transcode(
  input: string,
  output: string,
  options?: TranscodeOptions
): Promise<TranscodeStats>;

/**
 * Extract a thumbnail from a video file.
 *
 * @param input - Path to the input video file.
 * @param timestamp - Timestamp in seconds to extract the thumbnail from.
 * @param output - Optional output path (defaults to input path with .jpg extension).
 * @returns Promise that resolves to the output path.
 *
 * @example
 * ```javascript
 * const thumbnailPath = await extractThumbnail('video.mp4', 10.5);
 * console.log(`Thumbnail saved to: ${thumbnailPath}`);
 * ```
 */
export function extractThumbnail(
  input: string,
  timestamp: number,
  output?: string
): Promise<string>;

/**
 * Detect SIMD capabilities of the current CPU.
 *
 * @returns SimdCapabilities object.
 *
 * @example
 * ```javascript
 * const simd = detectSimd();
 * if (simd.avx2) {
 *   console.log('AVX2 acceleration available');
 * }
 * ```
 */
export function detectSimd(): SimdCapabilities;

/**
 * Get version information about the transcode library.
 *
 * @returns Version string.
 */
export function version(): string;

/**
 * Get build information about the library.
 *
 * @returns BuildInfo object.
 */
export function buildInfo(): BuildInfo;
