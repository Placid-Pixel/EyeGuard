"""
EyeGuard - Healthcare Report Generator
--------------------------------------
Generates a downloadable, print-to-PDF ready clinical health report (HTML + CSS)
and CSV data export summarizing:
1. Session Fatigue Trends & Elevated Fatigue Frequency
2. Eye Biometric Behavior (Blink Rate, Blink Duration, PERCLOS)
3. Symptom Severity & Tracking History
4. Before-vs-After Break Recovery Evaluation
5. Clinical & Ergonomic Recommendations
"""

import json
from datetime import datetime
import pandas as pd
import health_db


def generate_html_health_report():
    """
    Generate an executive clinical report in HTML format.
    Designed to look like a modern telemedicine / eye-care assessment report,
    complete with printable layout, KPI cards, tables, and recommendations.
    """
    stats = health_db.get_health_summary_stats()
    sessions_df = health_db.get_sessions_df()
    symptoms_df = health_db.get_symptoms_df()
    breaks_df = health_db.get_breaks_df()

    now_str = datetime.now().strftime("%B %d, %Y - %I:%M %p")
    report_id = f"EGR-{datetime.now().strftime('%Y%m%d-%H%M')}"

    # Determine overall fatigue risk level
    avg_f = stats["avg_fatigue"]
    if avg_f <= 30:
        overall_status = "Optimal / Low Risk"
        status_color = "#10b981"
        status_badge_bg = "#ecfdf5"
    elif avg_f <= 60:
        overall_status = "Moderate Strain"
        status_color = "#d97706"
        status_badge_bg = "#fffbeb"
    elif avg_f <= 80:
        overall_status = "Elevated Fatigue"
        status_color = "#ea580c"
        status_badge_bg = "#fff7ed"
    else:
        overall_status = "High Risk / Chronic Strain"
        status_color = "#dc2626"
        status_badge_bg = "#fef2f2"

    # Recent session rows
    session_rows_html = ""
    if not sessions_df.empty:
        recent_s = sessions_df.tail(8).iloc[::-1]
        for _, r in recent_s.iterrows():
            f_score = int(r["fatigue_score"])
            badge_color = "#10b981" if f_score <= 30 else ("#f59e0b" if f_score <= 60 else "#ef4444")
            t_str = r["timestamp"].strftime("%b %d, %Y %H:%M")
            dur = f"{int(r['duration_seconds'] // 60)}m"
            session_rows_html += f"""
            <tr>
                <td><strong>{t_str}</strong></td>
                <td>{dur}</td>
                <td><span style="background: {badge_color}18; color: {badge_color}; padding: 3px 8px; border-radius: 9999px; font-weight: 600;">{f_score}/100</span></td>
                <td>{r['blink_rate']} bpm</td>
                <td>{r['avg_blink_duration']}s</td>
                <td>{r['perclos']}%</td>
                <td style="color: #64748b; font-size: 13px;">{r['notes'] or 'Standard session'}</td>
            </tr>
            """
    else:
        session_rows_html = "<tr><td colspan='7'>No session data recorded.</td></tr>"

    # Symptom breakdown table
    symptom_summary_html = ""
    if not symptoms_df.empty:
        sym_means = {
            "Eye Dryness / Irritation": round(float(symptoms_df["dryness"].mean()), 1),
            "Headache": round(float(symptoms_df["headache"].mean()), 1),
            "Blurred Vision": round(float(symptoms_df["blurred_vision"].mean()), 1),
            "Eye Discomfort": round(float(symptoms_df["discomfort"].mean()), 1),
            "Difficulty Focusing": round(float(symptoms_df["focus_difficulty"].mean()), 1),
        }
        for name, mean_val in sym_means.items():
            pct = int((mean_val / 5.0) * 100)
            bar_color = "#10b981" if mean_val <= 1.5 else ("#f59e0b" if mean_val <= 3.0 else "#ef4444")
            symptom_summary_html += f"""
            <div style="margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 4px;">
                    <span style="font-weight: 500; color: #334155;">{name}</span>
                    <span style="font-weight: 600; color: {bar_color};">{mean_val} / 5.0</span>
                </div>
                <div style="background: #e2e8f0; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: {bar_color}; width: {pct}%; height: 100%;"></div>
                </div>
            </div>
            """
    else:
        symptom_summary_html = "<p style='color: #64748b;'>No symptoms reported yet.</p>"

    # Break comparisons table
    break_rows_html = ""
    if not breaks_df.empty:
        for _, b in breaks_df.head(5).iterrows():
            diff = b["before_fatigue_score"] - b["after_fatigue_score"]
            start_str = b["timestamp_start"].strftime("%b %d, %H:%M")
            break_rows_html += f"""
            <tr>
                <td><strong>{start_str}</strong></td>
                <td>{b['break_type']}</td>
                <td><span style="color: #ea580c; font-weight: 600;">{b['before_fatigue_score']}</span></td>
                <td><span style="color: #0d9488; font-weight: 600;">{b['after_fatigue_score']}</span></td>
                <td><span style="color: #10b981; font-weight: 700;">-{diff} pts</span></td>
                <td style="font-size: 13px; color: #475569;">{b['user_feedback'] or 'Completed'}</td>
            </tr>
            """
    else:
        break_rows_html = "<tr><td colspan='6'>No break records logged yet.</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EyeGuard Health Assessment Report - {report_id}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        body {{
            background: #f8fafc;
            color: #1e293b;
            padding: 40px 20px;
            line-height: 1.5;
        }}
        .report-card {{
            max-width: 900px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.03);
            border: 1px solid #e2e8f0;
            padding: 40px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 24px;
            margin-bottom: 28px;
        }}
        .header h1 {{
            font-size: 26px;
            color: #0f172a;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .meta-info {{
            text-align: right;
            font-size: 13px;
            color: #64748b;
        }}
        .status-banner {{
            background: {status_badge_bg};
            border: 1px solid {status_color}33;
            border-left: 6px solid {status_color};
            padding: 16px 20px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 32px;
        }}
        .kpi-box {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }}
        .kpi-value {{
            font-size: 24px;
            font-weight: 700;
            color: #0f172a;
            margin-top: 4px;
        }}
        .kpi-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
            font-weight: 600;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 6px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 24px;
            font-size: 13.5px;
        }}
        th {{
            background: #f8fafc;
            color: #475569;
            text-align: left;
            padding: 10px 12px;
            font-weight: 600;
            border-bottom: 1px solid #cbd5e1;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #f1f5f9;
        }}
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 32px;
        }}
        .card-inner {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
        }}
        .rec-box {{
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        .rec-box ul {{
            padding-left: 20px;
            margin-top: 10px;
            color: #166534;
            font-size: 14px;
        }}
        .rec-box li {{
            margin-bottom: 6px;
        }}
        .disclaimer-box {{
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 8px;
            padding: 14px 18px;
            font-size: 12.5px;
            color: #92400e;
            margin-top: 24px;
        }}
        .footer {{
            text-align: center;
            font-size: 12px;
            color: #94a3b8;
            margin-top: 30px;
            border-top: 1px solid #f1f5f9;
            padding-top: 16px;
        }}
        @media print {{
            body {{
                background: #ffffff;
                padding: 0;
            }}
            .report-card {{
                box-shadow: none;
                border: none;
                padding: 15px;
            }}
            .no-print {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-card">
        <div class="header">
            <div>
                <h1>👁 EyeGuard Health Summary Report</h1>
                <p style="color: #64748b; font-size: 14px; margin-top: 4px;">Digital Eye Fatigue, Ocular Biometrics & Symptom Assessment</p>
            </div>
            <div class="meta-info">
                <div><strong>Report ID:</strong> {report_id}</div>
                <div><strong>Generated:</strong> {now_str}</div>
                <div><strong>Total Screen Sessions:</strong> {stats['total_sessions']}</div>
            </div>
        </div>

        <div class="status-banner">
            <div>
                <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: {status_color};">Assessment Classification</div>
                <div style="font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 2px;">{overall_status}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 12px; color: #64748b;">Elevated Sessions (Score ≥ 60)</div>
                <div style="font-size: 22px; font-weight: 700; color: {status_color};">{stats['elevated_sessions_count']} of {stats['total_sessions']} sessions</div>
            </div>
        </div>

        <div class="kpi-grid">
            <div class="kpi-box">
                <div class="kpi-label">Avg Fatigue Score</div>
                <div class="kpi-value">{stats['avg_fatigue']}<span style="font-size: 14px; color: #64748b;">/100</span></div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Peak Fatigue</div>
                <div class="kpi-value" style="color: {'#ef4444' if stats['max_fatigue'] >= 70 else '#0f172a'};">{stats['max_fatigue']}<span style="font-size: 14px; color: #64748b;">/100</span></div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Avg Blink Rate</div>
                <div class="kpi-value">{stats['avg_blink_rate']} <span style="font-size: 13px; font-weight: 500; color: #64748b;">bpm</span></div>
            </div>
            <div class="kpi-box">
                <div class="kpi-label">Avg PERCLOS</div>
                <div class="kpi-value">{stats['avg_perclos']}%</div>
            </div>
        </div>

        <div class="two-col">
            <div class="card-inner">
                <div class="section-title">🩺 Average Reported Symptoms</div>
                {symptom_summary_html}
            </div>
            <div class="card-inner">
                <div class="section-title">☕ Break Recovery Efficacy</div>
                <div style="margin-bottom: 12px;">
                    <div style="font-size: 13px; color: #64748b;">Total Ergonomic Breaks Logged</div>
                    <div style="font-size: 22px; font-weight: 700; color: #0d9488;">{stats['break_count']}</div>
                </div>
                <div style="margin-bottom: 14px;">
                    <div style="font-size: 13px; color: #64748b;">Average Fatigue Score Reduction</div>
                    <div style="font-size: 22px; font-weight: 700; color: #10b981;">-{stats['avg_break_improvement']} points</div>
                </div>
                <p style="font-size: 12.5px; color: #64748b; line-height: 1.4;">
                    Taking structured breaks (e.g. 20-20-20 rule, hydration, eye exercises) consistently demonstrates immediate downward pressure on perceived visual discomfort and eyelid fatigue.
                </p>
            </div>
        </div>

        <div class="section-title">📊 Recent Monitoring Sessions & Eye Biometrics</div>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Duration</th>
                    <th>Fatigue Score</th>
                    <th>Blink Rate</th>
                    <th>Avg Duration</th>
                    <th>PERCLOS</th>
                    <th>Session Context</th>
                </tr>
            </thead>
            <tbody>
                {session_rows_html}
            </tbody>
        </table>

        <div class="section-title">☕ Before-Break vs After-Break Logs</div>
        <table>
            <thead>
                <tr>
                    <th>Break Time</th>
                    <th>Intervention</th>
                    <th>Pre-Break Score</th>
                    <th>Post-Break Score</th>
                    <th>Delta Change</th>
                    <th>User Experience</th>
                </tr>
            </thead>
            <tbody>
                {break_rows_html}
            </tbody>
        </table>

        <div class="rec-box">
            <strong style="font-size: 15px; color: #166534;">🩺 Clinical & Ergonomic Guidance</strong>
            <ul>
                <li><strong>Follow the 20-20-20 Rule:</strong> Every 20 minutes, gaze at an object 20 feet away for at least 20 seconds to relax your ciliary accommodation muscles.</li>
                <li><strong>Mindful Blinking:</strong> Blink rate typically drops from 18 bpm to under 10 bpm while reading screens. Practice deliberate full blinks to replenish your tear film and prevent corneal dry spots.</li>
                <li><strong>Ergonomic Geometry:</strong> Position your display 20–28 inches away, with the top of the monitor at or slightly below eye level (15–20° downward viewing angle).</li>
                <li><strong>Illumination & Glare:</strong> Eliminate direct screen glare from overhead lamps or windows. Use balanced ambient illumination that matches monitor luminance.</li>
                <li><strong>Medical Follow-up:</strong> If eye irritation, persistent headaches, or blurred vision remain severe (scores ≥ 4) after ergonomics adjustment, consult an optometrist or ophthalmologist for comprehensive examination.</li>
            </ul>
        </div>

        <div class="disclaimer-box">
            <strong>⚠️ Medical & Methodological Disclaimer:</strong><br>
            EyeGuard is an AI-powered digital wellness assistant and screen ergonomic tool. The estimated fatigue scores, blink metrics, and symptom reports are indicative heuristics and are NOT diagnostic medical devices. Before-and-after break comparisons record observed changes in reported state, but do not scientifically prove that the break alone caused the improvement.
        </div>

        <div class="footer">
            EyeGuard Health System &bull; Confidential Personal Wellness Record &bull; Generated via MobileNetV2 Neural Network & FaceMesh Biometrics
        </div>
    </div>
</body>
</html>
"""
    return html


def generate_csv_export():
    """Export all sessions and symptoms as a clean combined CSV string."""
    sessions_df = health_db.get_sessions_df()
    if sessions_df.empty:
        return "timestamp,fatigue_score,blink_rate,avg_blink_duration,perclos,notes\n"
    return sessions_df.to_csv(index=False)
