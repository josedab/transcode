import { exec } from 'child_process';
import { promisify } from 'util';
import * as vscode from 'vscode';

const execAsync = promisify(exec);

export interface TranscodeOptions {
    videoCodec?: string;
    audioCodec?: string;
    preset?: string;
    videoBitrate?: number;
    audioBitrate?: number;
    crf?: number;
}

export interface QualityResult {
    psnr: number;
    ssim: number;
}

export class TranscodeRunner {
    private getCliPath(): string {
        return vscode.workspace.getConfiguration('transcode').get<string>('cliPath', 'transcode');
    }

    async transcode(
        input: string,
        output: string,
        options: TranscodeOptions,
        progress: vscode.Progress<{ message?: string; increment?: number }>,
        token: vscode.CancellationToken
    ): Promise<void> {
        const cli = this.getCliPath();
        const args: string[] = ['-i', `"${input}"`, '-o', `"${output}"`];

        if (options.videoCodec) { args.push('--video-codec', options.videoCodec); }
        if (options.audioCodec) { args.push('--audio-codec', options.audioCodec); }
        if (options.preset) { args.push('--preset', options.preset); }
        if (options.videoBitrate) { args.push('--video-bitrate', options.videoBitrate.toString()); }
        if (options.crf) { args.push('--crf', options.crf.toString()); }
        args.push('--overwrite');

        const cmd = `${cli} ${args.join(' ')}`;
        progress.report({ message: 'Starting transcode...' });

        return new Promise((resolve, reject) => {
            const child = exec(cmd, (error, stdout, stderr) => {
                if (error) {
                    reject(new Error(stderr || error.message));
                } else {
                    resolve();
                }
            });

            token.onCancellationRequested(() => {
                child.kill();
                reject(new Error('Cancelled'));
            });
        });
    }

    async listCodecs(): Promise<string[]> {
        try {
            const { stdout } = await execAsync(`${this.getCliPath()} codecs`);
            return stdout.trim().split('\n').filter(l => l.length > 0);
        } catch {
            return ['h264', 'hevc', 'av1', 'vp9', 'vp8', 'aac', 'opus', 'flac', 'mp3'];
        }
    }

    async getMediaInfo(filePath: string): Promise<string> {
        try {
            const { stdout } = await execAsync(`${this.getCliPath()} info "${filePath}"`);
            return stdout;
        } catch (err: any) {
            throw new Error(`Failed to inspect file: ${err.message}`);
        }
    }

    async compareQuality(reference: string, distorted: string): Promise<QualityResult> {
        try {
            const { stdout } = await execAsync(
                `${this.getCliPath()} quality --reference "${reference}" --distorted "${distorted}" --json`
            );
            const result = JSON.parse(stdout);
            return { psnr: result.psnr ?? 0, ssim: result.ssim ?? 0 };
        } catch {
            return { psnr: 0, ssim: 0 };
        }
    }
}
