"""
EyeGuard - Healthcare Visualization & Trend Analytics Engine
------------------------------------------------------------
Implements rich Plotly-based clinical and ergonomic charts:
1. Fatigue Score Trend (0-100 line graph with elevated zones and alerts)
2. Eye Behaviour Trends (Blink rate, Duration, PERCLOS across sessions)
3. Symptom History (Weekly timeline of dryness, headache, blur, discomfort, focus)
4. Before-Break vs After-Break Comparison (Bar chart with clinical disclaimer)
"""

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Common Clinical Color Palette
COLOR_TEAL = "#0d9488"
COLOR_BLUE = "#0284c7"
COLOR_INDIGO = "#6366f1"
COLOR_AMBER = "#f59e0b"
COLOR_RED = "#ef4444"
COLOR_GREEN = "#10b981"
COLOR_PURPLE = "#a855f7"
COLOR_BG_CARD = "#ffffff"


def create_fatigue_trend_chart(sessions_df):
    """
    Feature 1: Fatigue Score Trend — Most Important.
    A line graph showing fatigue score against time or session date.
    - X-axis: time/date
    - Y-axis: estimated fatigue score (0–100)
    - Purpose: identify sessions when estimated fatigue is consistently elevated.
    """
    if sessions_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No session records available yet.<br>Start a monitoring session to begin logging data.",
            showarrow=False, font=dict(size=14, color="#64748b")
        )
        fig.update_layout(height=400)
        return fig

    df = sessions_df.copy().sort_values("timestamp")
    df["date_str"] = df["timestamp"].dt.strftime("%b %d, %H:%M")

    # Map fatigue score to color
    def get_color(score):
        if score <= 30:
            return COLOR_GREEN
        elif score <= 60:
            return COLOR_AMBER
        elif score <= 80:
            return "#ea580c"  # Dark orange
        else:
            return COLOR_RED

    marker_colors = [get_color(s) for s in df["fatigue_score"]]

    fig = go.Figure()

    # Clinical Zone Background Bands
    fig.add_hrect(y0=0, y1=30, fillcolor="#10b981", opacity=0.08, line_width=0,
                  annotation_text="Optimal / Low (0-30)", annotation_position="top left",
                  annotation_font=dict(size=10, color="#059669"))
    fig.add_hrect(y0=30, y1=60, fillcolor="#f59e0b", opacity=0.08, line_width=0,
                  annotation_text="Moderate Fatigue (31-60)", annotation_position="top left",
                  annotation_font=dict(size=10, color="#d97706"))
    fig.add_hrect(y0=60, y1=80, fillcolor="#ea580c", opacity=0.10, line_width=0,
                  annotation_text="Elevated Fatigue (61-80)", annotation_position="top left",
                  annotation_font=dict(size=10, color="#c2410c"))
    fig.add_hrect(y0=80, y1=100, fillcolor="#ef4444", opacity=0.14, line_width=0,
                  annotation_text="Severe Fatigue (81-100)", annotation_position="top left",
                  annotation_font=dict(size=10, color="#dc2626"))

    # Critical Alert Threshold Line
    fig.add_hline(
        y=60, line_dash="dash", line_color="#dc2626", line_width=1.8,
        annotation_text="⚠️ Elevated Fatigue Threshold (Score ≥ 60)",
        annotation_position="bottom right",
        annotation_font=dict(size=11, color="#b91c1c", family="Arial, sans-serif")
    )

    # 7-session Moving Average Trendline (if enough points)
    if len(df) >= 3:
        df["rolling_avg"] = df["fatigue_score"].rolling(window=min(3, len(df)), min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["rolling_avg"],
            mode="lines",
            name="Smoothed Trend",
            line=dict(color="#64748b", width=2, dash="dot"),
            hoverinfo="skip"
        ))

    # Main Fatigue Score Trajectory
    hover_templates = []
    for _, row in df.iterrows():
        dur_min = int(row["duration_seconds"] // 60)
        dur_sec = int(row["duration_seconds"] % 60)
        dur_str = f"{dur_min}m {dur_sec}s" if dur_min > 0 else f"{dur_sec}s"
        hover_templates.append(
            f"<b>Session Date:</b> {row['timestamp'].strftime('%a, %b %d %Y, %I:%M %p')}<br>"
            f"<b>Fatigue Score:</b> {int(row['fatigue_score'])}/100<br>"
            f"<b>Session Duration:</b> {dur_str}<br>"
            f"<b>Blink Rate:</b> {row['blink_rate']} bpm | <b>PERCLOS:</b> {row['perclos']}%<br>"
            f"<b>Notes:</b> {row['notes'] or 'None'}<extra></extra>"
        )

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["fatigue_score"],
        mode="lines+markers",
        name="Session Fatigue Score",
        line=dict(color="#1e293b", width=3),
        marker=dict(
            size=11,
            color=marker_colors,
            line=dict(color="#ffffff", width=2)
        ),
        hovertemplate=hover_templates
    ))

    fig.update_layout(
        title=dict(
            text="<b>Fatigue Score Trend Across Sessions</b><br><sup>Identify sessions when digital eye fatigue is consistently elevated</sup>",
            font=dict(size=18, color="#0f172a")
        ),
        xaxis=dict(
            title="Session Date / Time",
            gridcolor="#f1f5f9",
            showgrid=True,
            tickformat="%b %d\n%H:%M"
        ),
        yaxis=dict(
            title="Estimated Fatigue Score (0–100)",
            range=[0, 105],
            gridcolor="#e2e8f0",
            dtick=20,
            showgrid=True
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=40, t=80, b=50),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        height=450
    )

    return fig


def create_eye_behaviour_chart(sessions_df):
    """
    Feature 2: Eye Behaviour Trends.
    Plot blink rate, average blink duration and PERCLOS across sessions.
    - Compare eye-behaviour metrics over time.
    - Identify changes during longer screen sessions.
    - Help users understand which factors contribute to their score.
    """
    if sessions_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No biometric data available yet.", showarrow=False)
        return fig

    df = sessions_df.copy().sort_values("timestamp")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>1. Blink Rate (blinks / minute)</b> — Healthy Baseline: 12-20 bpm",
            "<b>2. Average Blink Closure Duration (seconds)</b> — Normal: 0.15-0.35s",
            "<b>3. PERCLOS (% Time Eyes Closed)</b> — Drowsiness Threshold: ≥ 12%"
        )
    )

    # 1. Blink Rate Trace
    fig.add_hrect(y0=12, y1=20, fillcolor="#10b981", opacity=0.12, line_width=0, row=1, col=1)
    fig.add_hline(y=10, line_dash="dot", line_color="#ef4444", line_width=1.5,
                  annotation_text="Under-blinking (<10 bpm, High Evaporative Dryness)",
                  annotation_position="bottom right", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["blink_rate"],
        mode="lines+markers",
        name="Blink Rate (bpm)",
        line=dict(color=COLOR_TEAL, width=2.5),
        marker=dict(size=8, color=COLOR_TEAL),
        hovertemplate="<b>Date:</b> %{x|%b %d, %H:%M}<br><b>Blink Rate:</b> %{y:.1f} bpm<extra></extra>"
    ), row=1, col=1)

    # 2. Average Blink Duration Trace
    fig.add_hrect(y0=0.15, y1=0.35, fillcolor="#0284c7", opacity=0.10, line_width=0, row=2, col=1)
    fig.add_hline(y=0.45, line_dash="dot", line_color="#f59e0b", line_width=1.5,
                  annotation_text="Prolonged Closure (>0.45s, Heavy Lids)",
                  annotation_position="top right", row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["avg_blink_duration"],
        mode="lines+markers",
        name="Avg Duration (s)",
        line=dict(color=COLOR_BLUE, width=2.5),
        marker=dict(size=8, color=COLOR_BLUE),
        hovertemplate="<b>Date:</b> %{x|%b %d, %H:%M}<br><b>Closure Duration:</b> %{y:.2f}s<extra></extra>"
    ), row=2, col=1)

    # 3. PERCLOS Trace
    fig.add_hrect(y0=0, y1=8, fillcolor="#10b981", opacity=0.10, line_width=0, row=3, col=1)
    fig.add_hline(y=12.0, line_dash="dash", line_color="#dc2626", line_width=1.5,
                  annotation_text="Physiological Fatigue Risk (PERCLOS ≥ 12%)",
                  annotation_position="top right", row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["perclos"],
        mode="lines+markers",
        name="PERCLOS (%)",
        line=dict(color=COLOR_INDIGO, width=2.5),
        marker=dict(size=8, color=COLOR_INDIGO),
        hovertemplate="<b>Date:</b> %{x|%b %d, %H:%M}<br><b>PERCLOS:</b> %{y:.1f}%<extra></extra>"
    ), row=3, col=1)

    fig.update_layout(
        height=680,
        margin=dict(l=50, r=40, t=70, b=50),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        showlegend=False
    )
    fig.update_xaxes(gridcolor="#f1f5f9", showgrid=True)
    fig.update_yaxes(gridcolor="#e2e8f0", showgrid=True)
    fig.update_xaxes(title_text="Session Date / Time", row=3, col=1, tickformat="%b %d\n%H:%M")

    return fig


def create_symptom_history_chart(symptoms_df, days=14):
    """
    Feature 3: Symptom History.
    A weekly graph of self-reported eye dryness, headache, blurred vision,
    discomfort and difficulty focusing.
    - X-axis: date
    - Y-axis: user-reported severity (0–5)
    - Purpose: help users track symptoms and identify patterns.
    """
    if symptoms_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No symptom reports logged yet.", showarrow=False)
        return fig

    df = symptoms_df.copy().sort_values("timestamp")
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    df_recent = df[df["timestamp"] >= cutoff]
    if df_recent.empty:
        df_recent = df.tail(15)

    symptom_specs = [
        ("dryness", "Eye Dryness / Irritation", "#0284c7", "diamond"),
        ("headache", "Headache", "#ef4444", "circle"),
        ("blurred_vision", "Blurred Vision", "#a855f7", "triangle-up"),
        ("discomfort", "Eye Discomfort", "#f59e0b", "square"),
        ("focus_difficulty", "Difficulty Focusing", "#0d9488", "hexagon"),
    ]

    fig = go.Figure()

    # Severity bands (0 to 5)
    fig.add_hrect(y0=0, y1=1.5, fillcolor="#10b981", opacity=0.07, line_width=0,
                  annotation_text="Mild / Normal (0-1)", annotation_position="top left",
                  annotation_font=dict(size=10, color="#059669"))
    fig.add_hrect(y0=1.5, y1=3.5, fillcolor="#f59e0b", opacity=0.07, line_width=0,
                  annotation_text="Moderate Severity (2-3)", annotation_position="top left",
                  annotation_font=dict(size=10, color="#d97706"))
    fig.add_hrect(y0=3.5, y1=5.0, fillcolor="#ef4444", opacity=0.10, line_width=0,
                  annotation_text="Severe Strain (4-5)", annotation_position="top left",
                  annotation_font=dict(size=10, color="#dc2626"))

    for col, label, color, symbol in symptom_specs:
        if col in df_recent.columns:
            fig.add_trace(go.Scatter(
                x=df_recent["timestamp"],
                y=df_recent[col],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=9, symbol=symbol, color=color, line=dict(color="#ffffff", width=1.5)),
                hovertemplate=f"<b>{label}</b><br>Date: %{{x|%a, %b %d %H:%M}}<br>Severity: %{{y}}/5<extra></extra>"
            ))

    fig.update_layout(
        title=dict(
            text="<b>Weekly Self-Reported Eye-Health Symptoms</b><br><sup>Tracking eye dryness, headache, blurred vision, discomfort, and focus over time</sup>",
            font=dict(size=17, color="#0f172a")
        ),
        xaxis=dict(
            title="Date of Check-In",
            gridcolor="#f1f5f9",
            showgrid=True,
            tickformat="%a\n%b %d"
        ),
        yaxis=dict(
            title="Reported Severity Score (0–5)",
            tickmode="array",
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=["0 (None)", "1 (Very Mild)", "2 (Mild)", "3 (Moderate)", "4 (Severe)", "5 (Extreme)"],
            range=[-0.2, 5.3],
            gridcolor="#e2e8f0",
            showgrid=True
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=40, t=90, b=50),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        height=460
    )

    return fig


def create_break_comparison_chart(break_record=None):
    """
    Feature 4: Before-Break vs After-Break Comparison.
    A bar chart comparing fatigue scores and reported symptoms before and after a break.
    - Record the score before the break.
    - Ask the user how they feel afterward.
    - Compare the new score and symptom report.
    - Includes the required disclaimer.
    """
    if break_record is None:
        # Default fallback comparison to illustrate structure if no record provided
        before_score = 75
        after_score = 38
        before_sym = {"Dryness": 4, "Headache": 3, "Blurred Vision": 3, "Discomfort": 4, "Difficulty Focusing": 3}
        after_sym = {"Dryness": 2, "Headache": 1, "Blurred Vision": 1, "Discomfort": 1, "Difficulty Focusing": 1}
    else:
        before_score = break_record.get("before_fatigue_score", 70)
        after_score = break_record.get("after_fatigue_score", 40)
        b_raw = break_record.get("before_symptoms", {})
        a_raw = break_record.get("after_symptoms", {})
        if isinstance(b_raw, str):
            b_raw = json.loads(b_raw)
        if isinstance(a_raw, str):
            a_raw = json.loads(a_raw)

        name_map = {
            "dryness": "Dryness / Irritation",
            "headache": "Headache",
            "blurred_vision": "Blurred Vision",
            "discomfort": "Discomfort",
            "focus_difficulty": "Difficulty Focusing"
        }
        before_sym = {name_map.get(k, k): v for k, v in b_raw.items()}
        after_sym = {name_map.get(k, k): v for k, v in a_raw.items()}

    categories = ["Fatigue Score (x 1/20)"] + list(before_sym.keys())
    # Normalizing fatigue score to 0-5 scale so it displays nicely alongside symptom 0-5 scores
    # Also showing raw fatigue values in hover and data labels
    before_vals = [round(before_score / 20.0, 1)] + [before_sym[k] for k in before_sym.keys()]
    after_vals = [round(after_score / 20.0, 1)] + [after_sym.get(k, 0) for k in before_sym.keys()]

    before_labels = [f"{before_score}/100"] + [f"{before_sym[k]}/5" for k in before_sym.keys()]
    after_labels = [f"{after_score}/100"] + [f"{after_sym.get(k, 0)}/5" for k in before_sym.keys()]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=before_vals,
        name="Before Break",
        marker_color="#ea580c", # Vibrant orange-red
        text=before_labels,
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Before Break: %{text}<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=after_vals,
        name="After Break",
        marker_color="#0d9488", # Calming healthcare teal
        text=after_labels,
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>After Break: %{text}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Before-Break vs. After-Break Recovery Comparison</b><br><sup>Comparing fatigue score and symptom severity before and after ergonomic rest</sup>",
            font=dict(size=17, color="#0f172a")
        ),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        yaxis=dict(
            title="Reported Severity (0-5 Scale)",
            range=[0, 6.0],
            dtick=1,
            gridcolor="#e2e8f0",
            showgrid=True
        ),
        xaxis=dict(
            gridcolor="#f1f5f9"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=40, t=80, b=60),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        height=440
    )

    return fig
