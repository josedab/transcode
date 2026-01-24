/**
 * Transcode Playground - Browser Video Transcoding App
 *
 * This application demonstrates the transcode-wasm library capabilities
 * for client-side video transcoding.
 */

// WASM module and transcoder instance
let wasm = null;
let transcoder = null;
let currentTranscode = null;

// DOM Elements
const elements = {
    // Browser support
    supportStatus: document.getElementById('support-status'),

    // Input
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    inputInfo: document.getElementById('input-info'),

    // Settings
    settingsSection: document.getElementById('settings-section'),
    presetSelect: document.getElementById('preset-select'),
    customSettings: document.getElementById('custom-settings'),
    resolutionSelect: document.getElementById('resolution-select'),
    videoBitrate: document.getElementById('video-bitrate'),
    audioBitrate: document.getElementById('audio-bitrate'),
    transcodeBtn: document.getElementById('transcode-btn'),

    // Progress
    progressSection: document.getElementById('progress-section'),
    progressFill: document.getElementById('progress-fill'),
    progressPercent: document.getElementById('progress-percent'),
    progressEta: document.getElementById('progress-eta'),
    statFps: document.getElementById('stat-fps'),
    statFrames: document.getElementById('stat-frames'),
    statSize: document.getElementById('stat-size'),
    cancelBtn: document.getElementById('cancel-btn'),

    // Output
    outputSection: document.getElementById('output-section'),
    outputVideo: document.getElementById('output-video'),
    outputDuration: document.getElementById('output-duration'),
    outputSize: document.getElementById('output-size'),
    outputCompression: document.getElementById('output-compression'),
    downloadBtn: document.getElementById('download-btn'),
    resetBtn: document.getElementById('reset-btn'),

    // Footer
    versionInfo: document.getElementById('version-info'),
};

// Current input file
let inputFile = null;
let inputSize = 0;

// Presets configuration
const presets = {
    web: {
        resolution: '1280x720',
        videoBitrate: 2500,
        audioBitrate: 128,
    },
    high: {
        resolution: '1920x1080',
        videoBitrate: 8000,
        audioBitrate: 192,
    },
    small: {
        resolution: '854x480',
        videoBitrate: 1000,
        audioBitrate: 96,
    },
};

/**
 * Initialize the application
 */
async function init() {
    try {
        // Try to load WASM module
        await loadWasm();

        // Check browser support
        checkBrowserSupport();

        // Setup event listeners
        setupEventListeners();

        console.log('Transcode Playground initialized');
    } catch (error) {
        console.error('Initialization error:', error);
        showError('Failed to initialize. Please check console for details.');
    }
}

/**
 * Load the WASM module
 */
async function loadWasm() {
    try {
        // Dynamic import of the WASM module
        // In production, this would be: import init, { Transcoder, ... } from '../pkg/transcode_wasm.js';
        wasm = await import('../pkg/transcode_wasm.js');
        await wasm.default();

        // Update version info
        elements.versionInfo.textContent = `v${wasm.version()}`;

        console.log('WASM module loaded');
    } catch (error) {
        console.warn('WASM module not found. Running in demo mode.');
        elements.versionInfo.textContent = 'Demo Mode';
    }
}

/**
 * Check browser support for required features
 */
function checkBrowserSupport() {
    const features = [
        { name: 'WebAssembly', supported: typeof WebAssembly !== 'undefined' },
        { name: 'Web Workers', supported: typeof Worker !== 'undefined' },
        { name: 'File API', supported: typeof FileReader !== 'undefined' },
        { name: 'Streams', supported: typeof ReadableStream !== 'undefined' },
    ];

    elements.supportStatus.innerHTML = features.map(f => `
        <span class="support-badge ${f.supported ? 'supported' : 'unsupported'}">
            ${f.supported ? checkIcon() : crossIcon()}
            ${f.name}
        </span>
    `).join('');
}

function checkIcon() {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';
}

function crossIcon() {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Drop zone
    elements.dropZone.addEventListener('click', () => elements.fileInput.click());
    elements.dropZone.addEventListener('dragover', handleDragOver);
    elements.dropZone.addEventListener('dragleave', handleDragLeave);
    elements.dropZone.addEventListener('drop', handleDrop);
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Preset selection
    elements.presetSelect.addEventListener('change', handlePresetChange);

    // Transcode button
    elements.transcodeBtn.addEventListener('click', startTranscode);

    // Cancel button
    elements.cancelBtn.addEventListener('click', cancelTranscode);

    // Output actions
    elements.downloadBtn.addEventListener('click', downloadOutput);
    elements.resetBtn.addEventListener('click', reset);
}

/**
 * Handle drag over event
 */
function handleDragOver(e) {
    e.preventDefault();
    elements.dropZone.classList.add('dragover');
}

/**
 * Handle drag leave event
 */
function handleDragLeave(e) {
    e.preventDefault();
    elements.dropZone.classList.remove('dragover');
}

/**
 * Handle file drop
 */
function handleDrop(e) {
    e.preventDefault();
    elements.dropZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

/**
 * Handle file selection from input
 */
function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
}

/**
 * Process selected file
 */
function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('video/')) {
        showError('Please select a video file.');
        return;
    }

    inputFile = file;
    inputSize = file.size;

    // Show file info
    elements.inputInfo.innerHTML = `
        <p class="filename">${file.name}</p>
        <p>Size: ${formatSize(file.size)}</p>
        <p>Type: ${file.type}</p>
    `;
    elements.inputInfo.classList.remove('hidden');

    // Show settings
    elements.settingsSection.classList.remove('hidden');

    console.log('File selected:', file.name, formatSize(file.size));
}

/**
 * Handle preset change
 */
function handlePresetChange() {
    const preset = elements.presetSelect.value;

    if (preset === 'custom') {
        elements.customSettings.classList.remove('hidden');
    } else {
        elements.customSettings.classList.add('hidden');

        // Apply preset values
        const config = presets[preset];
        if (config) {
            elements.resolutionSelect.value = config.resolution;
            elements.videoBitrate.value = config.videoBitrate;
            elements.audioBitrate.value = config.audioBitrate;
        }
    }
}

/**
 * Start transcoding
 */
async function startTranscode() {
    if (!inputFile) {
        showError('Please select a file first.');
        return;
    }

    // Show progress section
    elements.settingsSection.classList.add('hidden');
    elements.progressSection.classList.remove('hidden');

    // Update button state
    setButtonLoading(elements.transcodeBtn, true);

    // Get settings
    const resolution = elements.resolutionSelect.value;
    const videoBitrate = parseInt(elements.videoBitrate.value) * 1000;
    const audioBitrate = parseInt(elements.audioBitrate.value) * 1000;

    console.log('Starting transcode:', { resolution, videoBitrate, audioBitrate });

    try {
        if (wasm) {
            // Real transcoding with WASM
            await transcodeWithWasm(resolution, videoBitrate, audioBitrate);
        } else {
            // Demo mode - simulate transcoding
            await simulateTranscode();
        }
    } catch (error) {
        console.error('Transcode error:', error);
        showError(`Transcoding failed: ${error.message}`);
        reset();
    }
}

/**
 * Transcode using WASM module
 */
async function transcodeWithWasm(resolution, videoBitrate, audioBitrate) {
    const options = new wasm.TranscodeOptions()
        .withVideoCodec('h264')
        .withVideoBitrate(videoBitrate)
        .withAudioCodec('aac')
        .withAudioBitrate(audioBitrate);

    if (resolution !== 'original') {
        const [width, height] = resolution.split('x').map(Number);
        options.withResolution(width, height);
    }

    transcoder = new wasm.Transcoder();

    // Setup progress callback
    transcoder.onProgress((progress, frame, fps) => {
        updateProgress(progress, frame, fps);
    });

    // Read file as ArrayBuffer
    const buffer = await inputFile.arrayBuffer();

    // Start transcoding
    currentTranscode = transcoder.transcode(new Uint8Array(buffer), options);
    const outputBuffer = await currentTranscode;

    // Show output
    showOutput(outputBuffer);
}

/**
 * Simulate transcoding for demo mode
 */
async function simulateTranscode() {
    const totalFrames = 1000;
    const startTime = Date.now();

    for (let frame = 0; frame <= totalFrames; frame += 10) {
        if (!currentTranscode) {
            // Cancelled
            return;
        }

        const progress = (frame / totalFrames) * 100;
        const elapsed = (Date.now() - startTime) / 1000;
        const fps = elapsed > 0 ? frame / elapsed : 0;

        updateProgress(progress, frame, fps);

        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 50));
    }

    // Create a demo output (copy of input for demo)
    const buffer = await inputFile.arrayBuffer();
    showOutput(new Uint8Array(buffer));
}

/**
 * Update progress display
 */
function updateProgress(progress, frame, fps) {
    elements.progressFill.style.width = `${progress}%`;
    elements.progressPercent.textContent = `${Math.round(progress)}%`;

    // Calculate ETA
    if (progress > 0 && progress < 100) {
        const remaining = (100 - progress) / progress;
        // Rough estimate based on current speed
        const etaSeconds = remaining * (frame / fps);
        elements.progressEta.textContent = `ETA: ${formatDuration(etaSeconds)}`;
    } else {
        elements.progressEta.textContent = '';
    }

    // Update stats
    elements.statFps.textContent = `${fps.toFixed(1)} fps`;
    elements.statFrames.textContent = frame.toString();
    elements.statSize.textContent = formatSize(inputSize * (progress / 100) * 0.5); // Estimate
}

/**
 * Cancel transcoding
 */
function cancelTranscode() {
    if (transcoder && transcoder.cancel) {
        transcoder.cancel();
    }
    currentTranscode = null;
    reset();
}

/**
 * Show output section with result
 */
function showOutput(outputBuffer) {
    elements.progressSection.classList.add('hidden');
    elements.outputSection.classList.remove('hidden');

    // Create blob URL for video
    const blob = new Blob([outputBuffer], { type: 'video/mp4' });
    const url = URL.createObjectURL(blob);
    elements.outputVideo.src = url;

    // Update stats
    elements.outputSize.textContent = formatSize(outputBuffer.length);
    elements.outputCompression.textContent = `${(inputSize / outputBuffer.length).toFixed(2)}x`;

    // Get video duration when metadata loads
    elements.outputVideo.onloadedmetadata = () => {
        elements.outputDuration.textContent = formatDuration(elements.outputVideo.duration);
    };

    console.log('Transcode complete:', formatSize(outputBuffer.length));
}

/**
 * Download the output file
 */
function downloadOutput() {
    const url = elements.outputVideo.src;
    const filename = inputFile.name.replace(/\.[^/.]+$/, '') + '_transcoded.mp4';

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

/**
 * Reset to initial state
 */
function reset() {
    inputFile = null;
    inputSize = 0;
    currentTranscode = null;
    transcoder = null;

    // Reset UI
    elements.inputInfo.classList.add('hidden');
    elements.settingsSection.classList.add('hidden');
    elements.progressSection.classList.add('hidden');
    elements.outputSection.classList.add('hidden');

    elements.progressFill.style.width = '0%';
    elements.progressPercent.textContent = '0%';

    // Clean up video URL
    if (elements.outputVideo.src) {
        URL.revokeObjectURL(elements.outputVideo.src);
        elements.outputVideo.src = '';
    }

    setButtonLoading(elements.transcodeBtn, false);
}

/**
 * Show error message
 */
function showError(message) {
    // Could be enhanced with a toast notification
    alert(message);
}

/**
 * Set button loading state
 */
function setButtonLoading(button, loading) {
    const text = button.querySelector('.btn-text');
    const spinner = button.querySelector('.btn-spinner');

    if (loading) {
        button.disabled = true;
        if (text) text.textContent = 'Processing...';
        if (spinner) spinner.classList.remove('hidden');
    } else {
        button.disabled = false;
        if (text) text.textContent = 'Start Transcoding';
        if (spinner) spinner.classList.add('hidden');
    }
}

/**
 * Format file size
 */
function formatSize(bytes) {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }

    return `${size.toFixed(2)} ${units[unitIndex]}`;
}

/**
 * Format duration in seconds to readable string
 */
function formatDuration(seconds) {
    if (!isFinite(seconds) || seconds < 0) return '--';

    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);

    if (mins > 0) {
        return `${mins}m ${secs}s`;
    }
    return `${secs}s`;
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);
