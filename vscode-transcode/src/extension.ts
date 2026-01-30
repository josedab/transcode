import * as vscode from 'vscode';
import { inspectMediaFile, MediaInfo } from './inspector';
import { TranscodeRunner } from './runner';
import { ProfileBuilder } from './profile';

export function activate(context: vscode.ExtensionContext) {
    const runner = new TranscodeRunner();
    const profileBuilder = new ProfileBuilder();

    // Inspect media file
    context.subscriptions.push(
        vscode.commands.registerCommand('transcode.inspectFile', async (uri?: vscode.Uri) => {
            const fileUri = uri ?? await pickMediaFile();
            if (!fileUri) { return; }

            const info = await inspectMediaFile(fileUri.fsPath, runner);
            if (info) {
                showMediaInfoPanel(info, context);
            }
        })
    );

    // Transcode file
    context.subscriptions.push(
        vscode.commands.registerCommand('transcode.transcode', async (uri?: vscode.Uri) => {
            const fileUri = uri ?? await pickMediaFile();
            if (!fileUri) { return; }

            const config = vscode.workspace.getConfiguration('transcode');
            const codec = config.get<string>('defaultVideoCodec', 'h264');
            const preset = config.get<string>('defaultPreset', 'medium');

            const outputPath = await vscode.window.showSaveDialog({
                defaultUri: vscode.Uri.file(fileUri.fsPath.replace(/\.[^.]+$/, `.${codec}.mp4`)),
                filters: { 'Video': ['mp4', 'mkv', 'webm'], 'Audio': ['aac', 'flac', 'opus'] }
            });

            if (!outputPath) { return; }

            await vscode.window.withProgress(
                { location: vscode.ProgressLocation.Notification, title: 'Transcoding...', cancellable: true },
                async (progress, token) => {
                    try {
                        await runner.transcode(fileUri.fsPath, outputPath.fsPath, {
                            videoCodec: codec,
                            preset: preset,
                        }, progress, token);

                        if (config.get<boolean>('showNotifications', true)) {
                            vscode.window.showInformationMessage(`Transcode complete: ${outputPath.fsPath}`);
                        }
                    } catch (err: any) {
                        vscode.window.showErrorMessage(`Transcode failed: ${err.message}`);
                    }
                }
            );
        })
    );

    // Show supported codecs
    context.subscriptions.push(
        vscode.commands.registerCommand('transcode.showCodecs', async () => {
            const codecs = await runner.listCodecs();
            const panel = vscode.window.createWebviewPanel(
                'transcodeCodecs', 'Supported Codecs', vscode.ViewColumn.One, {}
            );
            panel.webview.html = formatCodecList(codecs);
        })
    );

    // Build encoding profile
    context.subscriptions.push(
        vscode.commands.registerCommand('transcode.buildProfile', async () => {
            const profile = await profileBuilder.build();
            if (profile) {
                const doc = await vscode.workspace.openTextDocument({
                    content: profile.toCliCommand(),
                    language: 'shellscript'
                });
                vscode.window.showTextDocument(doc);
            }
        })
    );

    // Quality comparison
    context.subscriptions.push(
        vscode.commands.registerCommand('transcode.compareQuality', async () => {
            const files = await vscode.window.showOpenDialog({
                canSelectMany: true,
                filters: { 'Video': ['mp4', 'mkv', 'webm', 'avi'] },
                openLabel: 'Select reference and distorted files (2 files)'
            });

            if (!files || files.length !== 2) {
                vscode.window.showWarningMessage('Please select exactly 2 files for quality comparison');
                return;
            }

            const result = await runner.compareQuality(files[0].fsPath, files[1].fsPath);
            vscode.window.showInformationMessage(
                `Quality: PSNR=${result.psnr.toFixed(2)}dB, SSIM=${result.ssim.toFixed(4)}`
            );
        })
    );
}

export function deactivate() {}

async function pickMediaFile(): Promise<vscode.Uri | undefined> {
    const files = await vscode.window.showOpenDialog({
        canSelectMany: false,
        filters: {
            'Media': ['mp4', 'mkv', 'avi', 'mov', 'webm', 'ts', 'flv', 'mxf',
                       'wav', 'flac', 'aac', 'mp3', 'opus']
        }
    });
    return files?.[0];
}

function showMediaInfoPanel(info: MediaInfo, context: vscode.ExtensionContext) {
    const panel = vscode.window.createWebviewPanel(
        'transcodeInspector', `Media: ${info.filename}`, vscode.ViewColumn.Two, {}
    );
    panel.webview.html = `
        <html><body style="font-family: var(--vscode-font-family); padding: 20px;">
        <h2>${info.filename}</h2>
        <table>
            <tr><td><b>Format</b></td><td>${info.format}</td></tr>
            <tr><td><b>Duration</b></td><td>${info.duration}</td></tr>
            <tr><td><b>Size</b></td><td>${info.fileSize}</td></tr>
            ${info.videoCodec ? `<tr><td><b>Video</b></td><td>${info.videoCodec} ${info.resolution ?? ''}</td></tr>` : ''}
            ${info.audioCodec ? `<tr><td><b>Audio</b></td><td>${info.audioCodec} ${info.sampleRate ?? ''}Hz</td></tr>` : ''}
            <tr><td><b>Bitrate</b></td><td>${info.bitrate}</td></tr>
        </table>
        </body></html>`;
}

function formatCodecList(codecs: string[]): string {
    const rows = codecs.map(c => `<tr><td>${c}</td></tr>`).join('');
    return `<html><body style="font-family: var(--vscode-font-family); padding: 20px;">
        <h2>Supported Codecs</h2><table>${rows}</table></body></html>`;
}
