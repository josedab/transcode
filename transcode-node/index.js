/**
 * Node.js bindings for the Transcode codec library.
 * A memory-safe, high-performance universal codec library.
 *
 * @module @transcode/node
 */

'use strict';

const { existsSync, readFileSync } = require('fs');
const { join } = require('path');

const { platform, arch } = process;

let nativeBinding = null;
let localFileExisted = false;
let loadError = null;

/**
 * Get the platform-specific binary name.
 * @returns {string} Binary filename
 */
function getBindingName() {
  switch (platform) {
    case 'android':
      switch (arch) {
        case 'arm64':
          return 'transcode.android-arm64.node';
        case 'arm':
          return 'transcode.android-arm-eabi.node';
        default:
          throw new Error(`Unsupported architecture on Android: ${arch}`);
      }
    case 'win32':
      switch (arch) {
        case 'x64':
          return 'transcode.win32-x64-msvc.node';
        case 'ia32':
          return 'transcode.win32-ia32-msvc.node';
        case 'arm64':
          return 'transcode.win32-arm64-msvc.node';
        default:
          throw new Error(`Unsupported architecture on Windows: ${arch}`);
      }
    case 'darwin':
      switch (arch) {
        case 'x64':
          return 'transcode.darwin-x64.node';
        case 'arm64':
          return 'transcode.darwin-arm64.node';
        default:
          throw new Error(`Unsupported architecture on macOS: ${arch}`);
      }
    case 'freebsd':
      if (arch !== 'x64') {
        throw new Error(`Unsupported architecture on FreeBSD: ${arch}`);
      }
      return 'transcode.freebsd-x64.node';
    case 'linux':
      switch (arch) {
        case 'x64':
          if (isMusl()) {
            return 'transcode.linux-x64-musl.node';
          }
          return 'transcode.linux-x64-gnu.node';
        case 'arm64':
          if (isMusl()) {
            return 'transcode.linux-arm64-musl.node';
          }
          return 'transcode.linux-arm64-gnu.node';
        case 'arm':
          return 'transcode.linux-arm-gnueabihf.node';
        default:
          throw new Error(`Unsupported architecture on Linux: ${arch}`);
      }
    default:
      throw new Error(`Unsupported OS: ${platform}, architecture: ${arch}`);
  }
}

/**
 * Check if we're running on musl libc.
 * @returns {boolean}
 */
function isMusl() {
  // For Node.js, check the binary
  if (process.report && process.report.getReport) {
    const report = process.report.getReport();
    if (report.header && report.header.glibcVersionRuntime) {
      return false;
    }
  }
  // Fallback: check for musl in ldd output
  try {
    const { execSync } = require('child_process');
    const lddOutput = execSync('ldd --version 2>&1 || true').toString();
    return lddOutput.includes('musl');
  } catch {
    return false;
  }
}

/**
 * Try to load the native binding from various locations.
 */
function loadBinding() {
  const bindingName = getBindingName();

  // Try loading from the package directory
  const localPath = join(__dirname, bindingName);
  if (existsSync(localPath)) {
    localFileExisted = true;
    try {
      nativeBinding = require(localPath);
      return;
    } catch (e) {
      loadError = e;
    }
  }

  // Try loading from platform-specific package
  const platformPackage = `@transcode/node-${platform}-${arch}`;
  try {
    nativeBinding = require(platformPackage);
    return;
  } catch (e) {
    loadError = e;
  }

  // Try loading from node_modules
  const modulePath = join(__dirname, 'node_modules', platformPackage);
  if (existsSync(modulePath)) {
    try {
      nativeBinding = require(modulePath);
      return;
    } catch (e) {
      loadError = e;
    }
  }

  throw new Error(
    `Failed to load native binding for ${platform}-${arch}.\n` +
      `Tried:\n` +
      `  - ${localPath}\n` +
      `  - ${platformPackage}\n` +
      (loadError ? `Last error: ${loadError.message}` : '')
  );
}

// Load the native binding
try {
  loadBinding();
} catch (e) {
  // Allow requiring the module even if native binding is not available
  // This is useful for TypeScript type checking and documentation
  console.warn(`Warning: Native binding not loaded: ${e.message}`);
}

// Re-export all native functions and classes
if (nativeBinding) {
  module.exports = nativeBinding;
} else {
  // Export stub functions for development/testing without native module
  module.exports = {
    Transcoder: class Transcoder {
      constructor(input, output, options) {
        this.input = input;
        this.output = output;
        this.options = options || {};
      }
      async run(onProgress) {
        throw new Error('Native module not loaded');
      }
      cancel() {
        throw new Error('Native module not loaded');
      }
      getProgress() {
        return 0;
      }
    },
    probe: async function probe(input) {
      throw new Error('Native module not loaded');
    },
    transcode: async function transcode(input, output, options) {
      throw new Error('Native module not loaded');
    },
    extractThumbnail: async function extractThumbnail(input, timestamp, output) {
      throw new Error('Native module not loaded');
    },
    detectSimd: function detectSimd() {
      return {
        sse42: false,
        avx2: false,
        avx512: false,
        fma: false,
        neon: false,
        sve: false,
      };
    },
    version: function version() {
      return '0.1.0';
    },
    buildInfo: function buildInfo() {
      return {
        version: '0.1.0',
        arch: arch,
        os: platform,
        debug: false,
      };
    },
  };
}

/**
 * High-level transcode function with progress callback support.
 *
 * This is a wrapper around the native transcode function that provides
 * a more convenient API with the onProgress callback in options.
 *
 * @param {string} input - Path to the input media file.
 * @param {string} output - Path for the output file.
 * @param {Object} options - Transcoding options.
 * @returns {Promise<Object>} Transcoding statistics.
 *
 * @example
 * const { transcodeWithProgress } = require('@transcode/node');
 *
 * await transcodeWithProgress('input.mp4', 'output.mp4', {
 *   videoBitrate: 5_000_000,
 *   onProgress: (p) => console.log(`${p.percent}%`)
 * });
 */
module.exports.transcodeWithProgress = async function transcodeWithProgress(
  input,
  output,
  options = {}
) {
  const { onProgress, ...transcodeOptions } = options;
  const transcoder = new module.exports.Transcoder(input, output, transcodeOptions);
  return transcoder.run(onProgress);
};
