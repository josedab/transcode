"""
Transcode - A memory-safe, high-performance universal codec library.

This module provides Python bindings for the Transcode Rust library,
enabling safe and efficient media transcoding operations.

Example usage:
    >>> import transcode_py
    >>>
    >>> # Simple transcoding
    >>> stats = transcode_py.transcode('input.mp4', 'output.mp4')
    >>> print(f"Processed {stats.frames_encoded} frames")
    >>>
    >>> # Using the builder pattern
    >>> options = transcode_py.TranscodeOptions()
    >>> options = options.input('input.mp4')
    >>> options = options.output('output.mp4')
    >>> options = options.video_bitrate(5_000_000)
    >>> options = options.overwrite(True)
    >>>
    >>> transcoder = transcode_py.Transcoder(options)
    >>> stats = transcoder.run()
    >>>
    >>> # Check SIMD capabilities
    >>> caps = transcode_py.detect_simd()
    >>> print(f"Best SIMD level: {caps.best_level()}")
"""

from .transcode_py import (
    # Classes
    TranscodeOptions,
    TranscodeStats,
    Transcoder,
    SimdCapabilities,
    # Functions
    transcode,
    detect_simd,
    version,
)

__all__ = [
    "TranscodeOptions",
    "TranscodeStats",
    "Transcoder",
    "SimdCapabilities",
    "transcode",
    "detect_simd",
    "version",
]

__version__ = version()
