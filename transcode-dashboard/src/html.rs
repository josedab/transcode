//! HTML rendering for the quality dashboard.

use crate::collector::Dashboard;

pub fn render_dashboard(dashboard: &Dashboard) -> String {
    let summary = dashboard.summary();
    let frames_json = dashboard.export_json();

    format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Transcode Quality Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #1e1e2e; color: #cdd6f4; }}
        h1 {{ color: #89b4fa; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0; }}
        .card {{ background: #313244; border-radius: 8px; padding: 16px; }}
        .card .label {{ color: #a6adc8; font-size: 12px; text-transform: uppercase; }}
        .card .value {{ font-size: 28px; font-weight: bold; margin-top: 4px; }}
        .psnr {{ color: #a6e3a1; }}
        .ssim {{ color: #89b4fa; }}
        .vmaf {{ color: #f9e2af; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #45475a; }}
        th {{ color: #89b4fa; font-size: 12px; text-transform: uppercase; }}
        .chart {{ background: #313244; border-radius: 8px; padding: 20px; margin: 20px 0; min-height: 200px; }}
    </style>
</head>
<body>
    <h1>Transcode Quality Dashboard</h1>

    <div class="summary">
        <div class="card">
            <div class="label">Total Frames</div>
            <div class="value">{frames}</div>
        </div>
        <div class="card">
            <div class="label">Avg PSNR</div>
            <div class="value psnr">{avg_psnr:.2} dB</div>
        </div>
        <div class="card">
            <div class="label">Avg SSIM</div>
            <div class="value ssim">{avg_ssim:.4}</div>
        </div>
        <div class="card">
            <div class="label">Avg VMAF</div>
            <div class="value vmaf">{avg_vmaf:.1}</div>
        </div>
        <div class="card">
            <div class="label">PSNR Range</div>
            <div class="value">{min_psnr:.1} – {max_psnr:.1} dB</div>
        </div>
        <div class="card">
            <div class="label">SSIM Range</div>
            <div class="value">{min_ssim:.4} – {max_ssim:.4}</div>
        </div>
    </div>

    <div class="chart" id="chart">
        <p style="color: #a6adc8;">Per-frame quality chart (requires Chart.js for interactive view)</p>
    </div>

    <h2>Per-Frame Data</h2>
    <div style="max-height: 400px; overflow-y: auto;">
        <table>
            <thead><tr><th>Frame</th><th>PSNR (dB)</th><th>SSIM</th><th>VMAF</th></tr></thead>
            <tbody id="frame-data"></tbody>
        </table>
    </div>

    <script>
        const data = {data_json};
        const tbody = document.getElementById('frame-data');
        data.forEach(f => {{
            const row = document.createElement('tr');
            row.innerHTML = `<td>${{f.frame}}</td><td>${{f.psnr?.toFixed(2) ?? '—'}}</td><td>${{f.ssim?.toFixed(4) ?? '—'}}</td><td>${{f.vmaf?.toFixed(1) ?? '—'}}</td>`;
            tbody.appendChild(row);
        }});
    </script>
</body>
</html>"#,
        frames = summary.total_frames,
        avg_psnr = summary.avg_psnr,
        avg_ssim = summary.avg_ssim,
        avg_vmaf = summary.avg_vmaf,
        min_psnr = summary.min_psnr,
        max_psnr = summary.max_psnr,
        min_ssim = summary.min_ssim,
        max_ssim = summary.max_ssim,
        data_json = frames_json,
    )
}
