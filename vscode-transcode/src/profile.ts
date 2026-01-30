import * as vscode from 'vscode';

export interface EncodingProfile {
    videoCodec: string;
    audioCodec: string;
    preset: string;
    videoBitrate?: number;
    audioBitrate?: number;
    crf?: number;
    resolution?: string;
}

export class ProfileBuilder {
    async build(): Promise<EncodingProfileResult | undefined> {
        const videoCodec = await vscode.window.showQuickPick(
            ['h264', 'hevc', 'av1', 'vp9', 'vp8', 'prores', 'copy'],
            { placeHolder: 'Select video codec' }
        );
        if (!videoCodec) { return undefined; }

        const audioCodec = await vscode.window.showQuickPick(
            ['aac', 'opus', 'flac', 'mp3', 'ac3', 'copy'],
            { placeHolder: 'Select audio codec' }
        );
        if (!audioCodec) { return undefined; }

        const preset = await vscode.window.showQuickPick(
            ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
            { placeHolder: 'Select encoding preset' }
        );
        if (!preset) { return undefined; }

        const qualityMode = await vscode.window.showQuickPick(
            ['CRF (Constant Quality)', 'Bitrate (Constant Bitrate)'],
            { placeHolder: 'Select quality mode' }
        );

        let crf: number | undefined;
        let videoBitrate: number | undefined;

        if (qualityMode?.startsWith('CRF')) {
            const crfStr = await vscode.window.showInputBox({
                prompt: 'CRF value (0-51, lower = better quality)',
                value: '23',
                validateInput: (v) => {
                    const n = parseInt(v);
                    return (isNaN(n) || n < 0 || n > 51) ? 'Must be 0-51' : null;
                }
            });
            crf = crfStr ? parseInt(crfStr) : undefined;
        } else {
            const brStr = await vscode.window.showInputBox({
                prompt: 'Video bitrate (kbps)',
                value: '5000',
            });
            videoBitrate = brStr ? parseInt(brStr) : undefined;
        }

        return new EncodingProfileResult({
            videoCodec, audioCodec, preset, crf, videoBitrate
        });
    }
}

export class EncodingProfileResult {
    constructor(private profile: EncodingProfile) {}

    toCliCommand(): string {
        const args: string[] = ['transcode', '-i', 'INPUT_FILE'];
        args.push('--video-codec', this.profile.videoCodec);
        args.push('--audio-codec', this.profile.audioCodec);
        args.push('--preset', this.profile.preset);
        if (this.profile.crf !== undefined) {
            args.push('--crf', this.profile.crf.toString());
        }
        if (this.profile.videoBitrate !== undefined) {
            args.push('--video-bitrate', (this.profile.videoBitrate * 1000).toString());
        }
        args.push('-o', 'OUTPUT_FILE');
        return args.join(' ');
    }

    toJson(): string {
        return JSON.stringify(this.profile, null, 2);
    }
}
