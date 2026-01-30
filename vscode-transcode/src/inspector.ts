import * as path from 'path';
import { TranscodeRunner } from './runner';

export interface MediaInfo {
    filename: string;
    format: string;
    duration: string;
    fileSize: string;
    bitrate: string;
    videoCodec?: string;
    audioCodec?: string;
    resolution?: string;
    framerate?: string;
    sampleRate?: string;
}

export async function inspectMediaFile(filePath: string, runner: TranscodeRunner): Promise<MediaInfo | null> {
    try {
        const raw = await runner.getMediaInfo(filePath);
        return parseMediaInfo(filePath, raw);
    } catch {
        // Fall back to basic file info
        return {
            filename: path.basename(filePath),
            format: path.extname(filePath).slice(1).toUpperCase(),
            duration: 'Unknown',
            fileSize: 'Unknown',
            bitrate: 'Unknown',
        };
    }
}

function parseMediaInfo(filePath: string, raw: string): MediaInfo {
    const info: MediaInfo = {
        filename: path.basename(filePath),
        format: path.extname(filePath).slice(1).toUpperCase(),
        duration: 'Unknown',
        fileSize: 'Unknown',
        bitrate: 'Unknown',
    };

    // Parse CLI output lines
    for (const line of raw.split('\n')) {
        const trimmed = line.trim();
        if (trimmed.startsWith('Duration:')) {
            info.duration = trimmed.replace('Duration:', '').trim();
        } else if (trimmed.startsWith('Size:')) {
            info.fileSize = trimmed.replace('Size:', '').trim();
        } else if (trimmed.startsWith('Bitrate:')) {
            info.bitrate = trimmed.replace('Bitrate:', '').trim();
        } else if (trimmed.startsWith('Video:')) {
            info.videoCodec = trimmed.replace('Video:', '').trim();
        } else if (trimmed.startsWith('Audio:')) {
            info.audioCodec = trimmed.replace('Audio:', '').trim();
        } else if (trimmed.startsWith('Resolution:')) {
            info.resolution = trimmed.replace('Resolution:', '').trim();
        } else if (trimmed.startsWith('Frame rate:')) {
            info.framerate = trimmed.replace('Frame rate:', '').trim();
        } else if (trimmed.startsWith('Sample rate:')) {
            info.sampleRate = trimmed.replace('Sample rate:', '').trim();
        }
    }

    return info;
}
