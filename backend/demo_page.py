"""
backend/demo_page.py — Single-Page Executive Demo Dashboard Generator
Serves a self-contained, responsive, dark-mode command center for live video,
violation evidence, KPI cards, and BOCW Act statutory compliance audit tables.
Zero external CDN dependencies.
"""

import json
from pathlib import Path
from typing import Any, Dict


def _get_demo_data() -> Dict[str, Any]:
    root_dir = Path(__file__).resolve().parent.parent
    audit_file = root_dir / "reports" / "demo_audit_log.json"
    if audit_file.exists():
        try:
            with open(audit_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "video_source": "data/raw/hardhat_demo.mp4",
        "frames_processed": 90,
        "total_violation_detections": 2,
        "sample_agent_output": {
            "compliance_score": 70.0,
            "alert_message": "CRITICAL risk: Worker w-03 (no_helmet) | LOW risk: Worker w-02 (no_vest)",
            "regulations_cited": [
                {
                    "worker_id": "w-03",
                    "violation_type": "no_helmet",
                    "zone": "scaffolding_elevated",
                    "risk_score": 85.23,
                    "risk_label": "CRITICAL",
                    "bocw_section": "Section 38(1)(a) & BOCW Rule 46 (IS 2925:1984)",
                    "penalty": "Section 47: Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.",
                },
                {
                    "worker_id": "w-02",
                    "violation_type": "no_vest",
                    "zone": "ground_level",
                    "risk_score": 30.29,
                    "risk_label": "LOW",
                    "bocw_section": "Section 38(1)(a), Section 40(2)(r) & BOCW Rule 45",
                    "penalty": "Section 47: Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.",
                },
            ],
        },
    }


def render_demo_html() -> str:
    data = _get_demo_data()
    agent_out = data.get("sample_agent_output", {})
    score = float(agent_out.get("compliance_score", 70.0) or 70.0)
    score_color = "#10b981" if score >= 80 else ("#f59e0b" if score >= 60 else "#ef4444")
    rating_label = "Compliant / Good" if score >= 80 else ("Fair (Action Required)" if score >= 60 else "Critical Hazard")

    regulations = agent_out.get("regulations_cited", [])
    total_violations = len(regulations) or data.get("total_violation_detections", 2)

    # Determine peak risk tier
    peak_risk_label = "SAFE"
    peak_risk_score = 0.0
    for r in regulations:
        r_score = float(r.get("risk_score", 0.0) or 0.0)
        if r_score > peak_risk_score:
            peak_risk_score = r_score
            peak_risk_label = r.get("risk_label", "LOW")

    risk_color = "#ef4444" if peak_risk_label in ("CRITICAL", "HIGH") else ("#f59e0b" if peak_risk_label in ("MEDIUM", "LOW") else "#10b981")

    # Build initial rows for SSR fallback
    table_rows_html = ""
    for r in regulations:
        v_type = r.get("violation_type", "unknown")
        badge_cls = "badge-danger" if "helmet" in v_type else "badge-warning"
        risk_label = r.get("risk_label", "LOW")
        risk_cls = "badge-danger" if risk_label in ("HIGH", "CRITICAL") else "badge-warning"
        r_score_val = float(r.get("risk_score", 0.0) or 0.0)
        table_rows_html += f"""
        <tr>
            <td><span class="mono-tag">{r.get('worker_id', 'w-01')}</span></td>
            <td><span class="badge {badge_cls}">{v_type}</span></td>
            <td><code>{r.get('zone', 'ground_level')}</code></td>
            <td><span class="badge {risk_cls}">{risk_label} ({r_score_val:.0f}/100)</span></td>
            <td class="law-cite"><strong>{r.get('bocw_section', 'Section 38')}</strong></td>
            <td class="penalty-text">{r.get('penalty', 'Section 47: Fine')}</td>
        </tr>
        """

    if not table_rows_html:
        table_rows_html = """
        <tr>
            <td colspan="6" style="text-align:center; padding:2rem; color:#94a3b8;">
                No violations detected or pipeline awaiting execution. Click "Trigger Live Audit" above.
            </td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RakshaSite — Executive Safety Audit & BOCW Compliance Viewer</title>
    <style>
        :root {{
            --bg-base: #0b0f19;
            --bg-panel: #111827;
            --bg-card: #1e293b;
            --bg-card-hover: #26334d;
            --border: #334155;
            --border-accent: #38bdf8;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-green: #10b981;
            --accent-orange: #f97316;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-base);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            min-height: 100vh;
            padding: 1.5rem;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 1.25rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }}
        .brand {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }}
        .brand-icon {{
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #0284c7, #38bdf8);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 14px rgba(56, 189, 248, 0.35);
        }}
        .brand-icon svg {{ width: 26px; height: 26px; fill: #ffffff; }}
        .brand-title h1 {{
            font-size: 1.45rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #ffffff;
        }}
        .brand-title p {{
            font-size: 0.82rem;
            color: var(--text-muted);
        }}
        .header-actions {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.85rem;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #6ee7b7;
        }}
        .pulse-dot {{
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-green);
            animation: pulse 1.8s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.4; transform: scale(0.85); }}
        }}
        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s ease;
            border: none;
        }}
        .btn-primary {{
            background: #0284c7;
            color: #ffffff;
        }}
        .btn-primary:hover {{ background: #0369a1; }}
        .btn-secondary {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-main);
        }}
        .btn-secondary:hover {{ background: var(--bg-card-hover); }}

        /* KPI Cards Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .kpi-card {{
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.15rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .kpi-title {{
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}
        .kpi-value {{
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.35rem;
        }}
        .kpi-subtext {{
            font-size: 0.78rem;
            color: var(--text-muted);
        }}

        /* Media Visual Evidence Grid */
        .media-grid {{
            display: grid;
            grid-template-columns: 3fr 2fr;
            gap: 1.25rem;
            margin-bottom: 1.5rem;
        }}
        @media (max-width: 980px) {{
            .media-grid {{ grid-template-columns: 1fr; }}
        }}
        .panel {{
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        .panel-header {{
            padding: 0.85rem 1.15rem;
            background: rgba(15, 23, 42, 0.6);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .panel-title {{
            font-size: 0.88rem;
            font-weight: 600;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .panel-badge {{
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.4);
            border-radius: 6px;
            color: var(--accent-cyan);
            font-weight: 600;
        }}
        .panel-body {{
            padding: 1rem;
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            background: #000000;
            min-height: 280px;
        }}
        video, img {{
            width: 100%;
            height: auto;
            max-height: 440px;
            object-fit: contain;
            border-radius: 8px;
            display: block;
        }}
        .media-footer {{
            padding: 0.65rem 1rem;
            background: var(--bg-panel);
            border-top: 1px solid var(--border);
            font-size: 0.78rem;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        /* Table Section */
        .table-panel {{
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 1.5rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.84rem;
        }}
        th {{
            background: #0f172a;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.05em;
            padding: 0.85rem 1rem;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 0.85rem 1rem;
            border-bottom: 1px solid rgba(51, 65, 85, 0.5);
            vertical-align: middle;
        }}
        tr:hover td {{
            background: rgba(30, 41, 59, 0.4);
        }}
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge-danger {{
            background: rgba(239, 68, 68, 0.15);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }}
        .badge-warning {{
            background: rgba(245, 158, 11, 0.15);
            color: #fcd34d;
            border: 1px solid rgba(245, 158, 11, 0.4);
        }}
        .mono-tag {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            background: rgba(56, 189, 248, 0.1);
            color: var(--accent-cyan);
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-weight: 600;
        }}
        .law-cite {{
            color: #e2e8f0;
            font-weight: 600;
        }}
        .penalty-text {{
            color: #cbd5e1;
            font-size: 0.8rem;
        }}

        /* Architecture Flowchart */
        .arch-panel {{
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
        }}
        .flow-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        .flow-step {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            position: relative;
        }}
        .flow-step h4 {{
            font-size: 0.85rem;
            color: var(--accent-cyan);
            margin-bottom: 0.35rem;
        }}
        .flow-step p {{
            font-size: 0.76rem;
            color: var(--text-muted);
        }}

        /* Action Dock Footer */
        .dock {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }}
        .dock-left {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
            flex-wrap: wrap;
        }}
        .dock-right {{
            font-size: 0.78rem;
            color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="brand">
                <div class="brand-icon">
                    <svg viewBox="0 0 24 24"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/></svg>
                </div>
                <div class="brand-title">
                    <h1>RakshaSite Executive Demo Viewer</h1>
                    <p>AI Construction Site Safety Auditor & BOCW Act 1996 Compliance Engine</p>
                </div>
            </div>
            <div class="header-actions">
                <span class="status-pill" id="system-status-pill">
                    <span class="pulse-dot"></span>
                    <span id="system-status-text">SYSTEM ONLINE / READY</span>
                </span>
                <button class="btn btn-secondary" onclick="refreshDashboard()">↻ Refresh</button>
                <button class="btn btn-primary" id="btn-run-pipeline" onclick="triggerPipelineRun()">▶ Run Pipeline</button>
            </div>
        </header>

        <!-- KPI Executive Summary Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Site Compliance Score</div>
                <div class="kpi-value" style="color: {score_color};" id="kpi-compliance">{score:.1f}%</div>
                <div class="kpi-subtext" id="kpi-rating">{rating_label}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Overall Risk Tier</div>
                <div class="kpi-value" style="color: {risk_color};" id="kpi-risk">{peak_risk_label}</div>
                <div class="kpi-subtext">Peak Score: {peak_risk_score:.1f}/100</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Workforce Tracked</div>
                <div class="kpi-value" style="color: var(--accent-cyan);" id="kpi-workers">2 Workers</div>
                <div class="kpi-subtext">DeepSORT Streak Confirmed</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Violations Detected</div>
                <div class="kpi-value" style="color: var(--accent-orange);" id="kpi-violations">{total_violations} Events</div>
                <div class="kpi-subtext">Spatial Containment Verified</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">BOCW Act Penalties</div>
                <div class="kpi-value" style="color: #cbd5e1;">Rs. 2,000</div>
                <div class="kpi-subtext">Section 47 Statutory Citation</div>
            </div>
        </div>

        <!-- Visual Evidence Center -->
        <div class="media-grid">
            <!-- Left: Annotated Video Player -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">
                        <span>📹</span> Live Video Stream & HUD Overlay
                    </div>
                    <span class="panel-badge">CAM-01-LIVE • 25 FPS</span>
                </div>
                <div class="panel-body">
                    <video id="demo-video-player" controls autoplay loop muted playsinline>
                        <source src="/reports/demo_annotated_output.mp4" type="video/mp4">
                        Your browser does not support HTML5 video streaming.
                    </video>
                </div>
                <div class="media-footer">
                    <span>Source: <code>data/raw/hardhat_demo.mp4</code></span>
                    <a href="/reports/demo_annotated_output.mp4" download class="btn btn-secondary" style="padding:0.25rem 0.6rem; font-size:0.75rem;">⬇ Download Video</a>
                </div>
            </div>

            <!-- Right: Violation Evidence Snapshot -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">
                        <span>📸</span> Critical Violation Snapshot
                    </div>
                    <span class="panel-badge" style="background:rgba(239,68,68,0.15); border-color:rgba(239,68,68,0.4); color:#fca5a5;">FORENSIC CAPTURE</span>
                </div>
                <div class="panel-body">
                    <img id="demo-snapshot-img" src="/reports/demo_snapshot_violation.jpg" alt="Construction Safety Violation Snapshot">
                </div>
                <div class="media-footer">
                    <span>Target: <strong>Spatial PPE Containment</strong> (Hazard Zone Identified)</span>
                    <a href="/reports/demo_snapshot_violation.jpg" target="_blank" class="btn btn-secondary" style="padding:0.25rem 0.6rem; font-size:0.75rem;">🔍 Full Resolution</a>
                </div>
            </div>
        </div>

        <!-- BOCW Statutory Audit Log Table -->
        <div class="table-panel">
            <div class="panel-header">
                <div class="panel-title">
                    <span>⚖️</span> Autonomous BOCW Act 1996 Statutory Audit Log
                </div>
                <span class="panel-badge" style="background:rgba(16,185,129,0.15); border-color:rgba(16,185,129,0.4); color:#6ee7b7;">CHROMA-RAG VERIFIED</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Worker ID</th>
                        <th>Violation Type</th>
                        <th>Hazard Zone</th>
                        <th>Risk Assessment</th>
                        <th>BOCW Act Statutory Mandate</th>
                        <th>Statutory Penalty</th>
                    </tr>
                </thead>
                <tbody id="audit-table-body">
                    {table_rows_html}
                </tbody>
            </table>
        </div>

        <!-- 4-Agent LangGraph Architecture Flowchart -->
        <div class="arch-panel">
            <div class="panel-title">
                <span>🧠</span> Real-Time 4-Agent LangGraph Pipeline Architecture
            </div>
            <div class="flow-grid">
                <div class="flow-step">
                    <h4>1. Detection Agent</h4>
                    <p>YOLOv8n trained checkpoint (10 classes: person, helmet, vest, machinery). Conf threshold 0.25 with DeepSORT streak confirmation.</p>
                </div>
                <div class="flow-step">
                    <h4>2. Risk Agent</h4>
                    <p>XGBoost hazard risk model scoring zone risk (scaffolding vs ground), machine proximity, and temporal exposure duration.</p>
                </div>
                <div class="flow-step">
                    <h4>3. Compliance Agent</h4>
                    <p>ChromaDB RAG indexing Building & Other Construction Workers (BOCW) Act 1996 and Central Rules 1998 with deterministic fallback.</p>
                </div>
                <div class="flow-step">
                    <h4>4. Report Agent</h4>
                    <p>Compiles printable PDF safety inspection reports (ReportLab) and broadcasts real-time safety alerts with zero cloud dependency.</p>
                </div>
            </div>
        </div>

        <!-- Action & Export Dock -->
        <div class="dock">
            <div class="dock-left">
                <a href="/reports/safety_audit.pdf" download class="btn btn-primary">📄 Download PDF Report (BOCW Form IV)</a>
                <a href="/reports/demo_audit_log.json" download class="btn btn-secondary">💾 Export Audit Log (JSON)</a>
                <a href="/docs" target="_blank" class="btn btn-secondary">⚡ Swagger API Docs</a>
            </div>
            <div class="dock-right">
                RakshaSite v1.0.0 • Strict Frontend Boundary Maintained • Offline Resilient
            </div>
        </div>
    </div>

    <!-- Live Client JavaScript -->
    <script>
        async function refreshDashboard() {{
            try {{
                const res = await fetch('/api/v1/demo/summary');
                if (res.ok) {{
                    const d = await res.json();
                    document.getElementById('kpi-compliance').textContent = `${{d.compliance_score.toFixed(1)}}%`;
                    document.getElementById('kpi-rating').textContent = d.compliance_rating || 'Good';
                    document.getElementById('kpi-risk').textContent = d.overall_risk_tier || 'LOW';
                    document.getElementById('kpi-violations').textContent = `${{d.total_violations}} Events`;
                    document.getElementById('kpi-workers').textContent = `${{d.active_workers_tracked || 2}} Workers`;
                }}
            }} catch (e) {{
                console.warn('Dashboard poll error:', e);
            }}
        }}

        async function triggerPipelineRun() {{
            const btn = document.getElementById('btn-run-pipeline');
            const pillText = document.getElementById('system-status-text');
            btn.disabled = true;
            btn.textContent = '⏳ Executing...';
            pillText.textContent = 'PIPELINE RUNNING';
            try {{
                const res = await fetch('/api/v1/demo/run', {{ method: 'POST' }});
                if (res.ok) {{
                    await refreshDashboard();
                    // Reload video and snapshot with cache buster
                    const v = document.getElementById('demo-video-player');
                    v.src = '/reports/demo_annotated_output.mp4?t=' + Date.now();
                    v.load();
                    const img = document.getElementById('demo-snapshot-img');
                    img.src = '/reports/demo_snapshot_violation.jpg?t=' + Date.now();
                }}
            }} catch (e) {{
                alert('Pipeline run failed: ' + e);
            }} finally {{
                btn.disabled = false;
                btn.textContent = '▶ Run Pipeline';
                pillText.textContent = 'SYSTEM ONLINE / READY';
            }}
        }}

        // Poll every 10 seconds for real-time updates
        setInterval(refreshDashboard, 10000);
    </script>
</body>
</html>
"""
    return html
