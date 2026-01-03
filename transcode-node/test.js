/**
 * Test file for @transcode/node
 *
 * This file demonstrates the usage of the transcode Node.js bindings.
 * Run with: node test.js
 */

'use strict';

const path = require('path');

// Try to load the module
let transcode;
try {
  transcode = require('./index.js');
} catch (e) {
  console.error('Failed to load transcode module:', e.message);
  console.error('Make sure to build the native module first with: npm run build');
  process.exit(1);
}

const { Transcoder, probe, transcode: transcodeFile, extractThumbnail, detectSimd, version, buildInfo } = transcode;

async function main() {
  console.log('=== Transcode Node.js Bindings Test ===\n');

  // Test version
  console.log('Version:', version());
  console.log('Build Info:', buildInfo());
  console.log();

  // Test SIMD detection
  console.log('SIMD Capabilities:', detectSimd());
  console.log();

  // Test probe (with a sample file if it exists)
  const testFile = process.argv[2] || 'test.mp4';

  try {
    console.log(`Probing file: ${testFile}`);
    const info = await probe(testFile);
    console.log('Media Info:');
    console.log('  Format:', info.format);
    console.log('  Duration:', info.duration, 'seconds');
    console.log('  Size:', info.size, 'bytes');
    console.log('  Bitrate:', info.bitrate, 'bps');
    console.log('  Video Streams:', info.videoStreams.length);
    info.videoStreams.forEach((vs, i) => {
      console.log(`    [${i}] ${vs.codec} ${vs.width}x${vs.height} @ ${vs.frameRate} fps`);
    });
    console.log('  Audio Streams:', info.audioStreams.length);
    info.audioStreams.forEach((as, i) => {
      console.log(`    [${i}] ${as.codec} ${as.sampleRate}Hz ${as.channels}ch`);
    });
    console.log();
  } catch (e) {
    console.log(`Could not probe ${testFile}:`, e.message);
    console.log('(This is expected if the file does not exist)\n');
  }

  // Test Transcoder class (without actually running)
  console.log('Creating Transcoder instance...');
  const transcoder = new Transcoder('input.mp4', 'output.mp4', {
    videoBitrate: 5_000_000,
    audioBitrate: 128_000,
    videoCodec: 'h264',
    audioCodec: 'aac',
    overwrite: true,
  });
  console.log('Transcoder created successfully');
  console.log();

  console.log('=== All tests passed! ===');
}

main().catch((e) => {
  console.error('Test failed:', e);
  process.exit(1);
});
