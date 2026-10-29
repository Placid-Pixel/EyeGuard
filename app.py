"""
EyeGuard - AI-Powered Digital Eye Fatigue & Healthcare Monitor
--------------------------------------------------------------
A comprehensive clinical & ergonomic eye-health application:
  1. Live Webcam Feed & Eye Tracking (MediaPipe Face Mesh + MobileNetV2)
  2. Real-time Eye State Detection (OPEN / CLOSED) & Blink Telemetry
  3. Continuous Explainable Fatigue Scoring (0-100)
  4. Eye-Health Symptom Check-In (Dryness, Headache, Blurred Vision, Discomfort, Focus)
  5. Fatigue Score Trend Analysis (Historical line graph with elevated zones)
  6. Eye Behaviour Trends (Blink rate, Duration, PERCLOS across sessions)
  7. Symptom History (Weekly timeline, 0-5 severity)
  8. Before-Break vs After-Break Comparison (Bar chart with clinical disclaimer)
  9. Downloadable Healthcare Report (Printable HTML / PDF ready + CSV export)
"""

import time
import collections
from datetime import datetime
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import mediapipe as mp

from MobileNetV2 import mobilenet_v2
import health_db
import health_charts
import health_report


# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

MODEL_PATH = "best_model.pt"
CLASS_NAMES = ["CLOSED", "OPEN"]

IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

MIN_CLOSURE_SECONDS = 0.08
PERCLOS_WINDOW_SECONDS = 60

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


# ============================================================
# MODEL & MESH INITIALIZATION (Cached)
# ============================================================

@st.cache_resource(show_spinner="Loading EyeGuard MobileNetV2 Model...")
def load_model():
    """Load trained MobileNetV2 model weights onto CUDA or CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mobilenet_v2(pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location=device)

    if "classifier.weight" in state_dict:
        ckpt_classes = state_dict["classifier.weight"].shape[0]
        if model.classifier.out_features != ckpt_classes:
            model.classifier = nn.Linear(model.classifier.in_features, ckpt_classes)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


@st.cache_resource(show_spinner=False)
def load_face_mesh():
    """MediaPipe Face Mesh for eye landmark localization."""
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]
)


# ============================================================
# COMPUTER VISION & INFERENCE PIPELINE
# ============================================================

def get_eye_crop(frame_bgr, face_mesh):
    """Find eye region using MediaPipe FaceMesh and return cropped BGR patch."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark

    def crop_from_indices(idx_list):
        xs = [int(landmarks[i].x * w) for i in idx_list]
        ys = [int(landmarks[i].y * h) for i in idx_list]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        pad_x = int((x_max - x_min) * 0.6) + 5
        pad_y = int((y_max - y_min) * 0.9) + 5

        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)

        if x_max <= x_min or y_max <= y_min:
            return None
        return frame_bgr[y_min:y_max, x_min:x_max]

    crop = crop_from_indices(LEFT_EYE_IDX)
    box_pts = LEFT_EYE_IDX
    if crop is None or crop.size == 0:
        crop = crop_from_indices(RIGHT_EYE_IDX)
        box_pts = RIGHT_EYE_IDX

    if crop is None or crop.size == 0:
        return None, None

    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in box_pts]
    return crop, pts


def predict_eye_state(eye_crop_bgr, model, device):
    """Run MobileNetV2 on cropped eye image patch."""
    tensor = preprocess(eye_crop_bgr).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
    label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
    return label, confidence


def compute_fatigue_score(history, blink_durations, session_seconds):
    """
    Compute explainable 0-100 fatigue score based on:
    1. PERCLOS (% of time eyes closed in recent window)
    2. Average blink closure duration
    3. Blink rate deviation from healthy resting norms
    4. Continuous session duration
    """
    now = time.time()

    # 1. PERCLOS component
    recent = [c for (t, c) in history if now - t <= PERCLOS_WINDOW_SECONDS]
    perclos_pct = (sum(recent) / len(recent) * 100) if recent else 0.0
    perclos_score = min(perclos_pct * 2.5, 100)

    # 2. Average closure duration
    recent_durations = blink_durations[-10:] if blink_durations else []
    avg_duration = sum(recent_durations) / len(recent_durations) if recent_durations else 0.0
    duration_score = min((avg_duration / 1.5) * 100, 100)

    # 3. Blink rate
    minutes = max(session_seconds / 60, 0.05)
    blink_rate = len(blink_durations) / minutes
    if blink_rate < 10:
        rate_score = min((10 - blink_rate) * 10, 100)
    elif blink_rate > 30:
        rate_score = min((blink_rate - 30) * 5, 100)
    else:
        rate_score = 0

    # 4. Session duration
    session_score = min((session_seconds / 1800) * 100, 100)

    weighted = (
        0.40 * perclos_score
        + 0.30 * duration_score
        + 0.20 * rate_score
        + 0.10 * session_score
    )
    score = int(round(min(max(weighted, 0), 100)))

    breakdown = {
        "PERCLOS (% time eyes closed)": round(perclos_pct, 1),
        "Avg blink duration (s)": round(avg_duration, 2),
        "Blink rate (blinks/min)": round(blink_rate, 1),
        "Session length factor": round(session_score, 1),
    }
    return score, breakdown, perclos_pct, avg_duration, blink_rate


def fatigue_level(score):
    if score <= 30:
        return "LOW", "🟢", "Optimal Comfort"
    elif score <= 60:
        return "MODERATE", "🟡", "Mild Eye Strain"
    elif score <= 80:
        return "HIGH", "🟠", "Elevated Visual Fatigue"
    else:
        return "VERY HIGH", "🔴", "Severe Fatigue - Rest Required"


def recommendation_for(level):
    return {
        "LOW": "Ocular state is optimal. Maintain good posture and ergonomic lighting.",
        "MODERATE": "Mild fatigue detected. Consider a brief 20-second gaze shift away from the display.",
        "HIGH": "Elevated fatigue detected! Take a 5-minute break now. Practice deliberate blinking and the 20-20-20 rule.",
        "VERY HIGH": "Critical visual strain. Cease continuous screen work immediately to allow corneal rehydration.",
    }[level]


# ============================================================
# STATE MANAGEMENT
# ============================================================

def init_state():
    health_db.init_db()
    defaults = {
        "running": False,
        "blink_count": 0,
        "is_closed": False,
        "closure_start": None,
        "blink_durations": [],
        "history": collections.deque(),
        "session_start": None,
        "last_label": "—",
        "last_conf": 0.0,
        "no_face_frames": 0,
        "last_finished_session": None,
        # Break tracking state
        "break_in_progress": False,
        "break_start_time": None,
        "break_pre_score": 65,
        "break_pre_symptoms": {"dryness": 3, "headache": 2, "blurred_vision": 1, "discomfort": 3, "focus_difficulty": 2},
        "break_type": "20-20-20 Rule + Ocular Palming",
        "active_tab": "👁️ Live AI Monitor",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    st.session_state.blink_count = 0
    st.session_state.is_closed = False
    st.session_state.closure_start = None
    st.session_state.blink_durations = []
    st.session_state.history = collections.deque()
    st.session_state.session_start = time.time()
    st.session_state.last_label = "—"
    st.session_state.last_conf = 0.0
    st.session_state.no_face_frames = 0


# ============================================================
# MAIN STREAMLIT APPLICATION
# ============================================================

def main():
    st.set_page_config(
        page_title="EyeGuard - Digital Eye Health & Fatigue Monitor",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    init_state()

    # Custom Healthcare Design Stylesheet
    st.markdown("""
        <style>
        .healthcare-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #ffffff;
            padding: 20px 26px;
            border-radius: 14px;
            margin-bottom: 22px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .healthcare-badge {
            display: inline-block;
            background: rgba(13, 148, 136, 0.2);
            color: #2dd4bf;
            border: 1px solid rgba(45, 212, 191, 0.3);
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 12px;
        }
        .disclaimer-badge {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 13px;
            color: #92400e;
            margin-bottom: 16px;
        }
        .elevated-tag {
            background: #fef2f2;
            color: #dc2626;
            border: 1px solid #fecaca;
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Application Header
    st.markdown("""
        <div class="healthcare-banner">
            <div>
                <span class="healthcare-badge">Clinical Ergonomics & AI Telemetry</span>
                <h1 style="color: #ffffff; margin: 6px 0 2px 0; font-size: 28px;">👁️ EyeGuard Healthcare</h1>
                <p style="color: #94a3b8; margin: 0; font-size: 14px;">
                    Digital Eye Fatigue Monitor, Biometric Trend Analytics & Symptom History
                </p>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 12px; color: #94a3b8;">Neural Architecture</span><br>
                <strong style="color: #38bdf8; font-size: 14px;">MobileNetV2 + MediaPipe</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation & Clinical Tools
    with st.sidebar:
        st.subheader("🏥 Clinical Navigation")
        app_mode = st.radio(
            "Select Module:",
            [
                "👁️ Live AI Monitor",
                "🩺 Symptom Check-In",
                "📈 Health Trends & Analytics",
                "☕ Break Recovery Tracker",
                "📑 Downloadable Health Report"
            ],
            key="selected_module"
        )

        st.markdown("---")
        st.subheader("📊 Health Overview")
        stats = health_db.get_health_summary_stats()
        c1, c2 = st.columns(2)
        c1.metric("Sessions", stats["total_sessions"])
        c2.metric("Avg Fatigue", f"{stats['avg_fatigue']}")

        c3, c4 = st.columns(2)
        c3.metric("Elevated", f"{stats['elevated_sessions_count']}", delta_color="inverse")
        c4.metric("Avg Blink", f"{stats['avg_blink_rate']} bpm")

        st.markdown("---")
        st.subheader("⚡ Quick Export")
        quick_html = health_report.generate_html_health_report()
        st.download_button(
            label="📥 Download Clinical Report",
            data=quick_html,
            file_name=f"EyeGuard_Report_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html",
            use_container_width=True
        )

        with st.expander("⚙️ Data Management"):
            if st.button("🔄 Reset / Re-seed Sample History", use_container_width=True):
                health_db.seed_sample_data_if_empty(force=True)
                st.success("Historical database refreshed with realistic sample records!")
                st.rerun()

    # ============================================================
    # MODULE 1: LIVE AI MONITOR
    # ============================================================
    if app_mode == "👁️ Live AI Monitor":
        render_live_monitor()

    # ============================================================
    # MODULE 2: SYMPTOM CHECK-IN
    # ============================================================
    elif app_mode == "🩺 Symptom Check-In":
        render_symptom_checkin()

    # ============================================================
    # MODULE 3: HEALTH TRENDS & ANALYTICS (Features 1, 2, 3)
    # ============================================================
    elif app_mode == "📈 Health Trends & Analytics":
        render_health_trends()

    # ============================================================
    # MODULE 4: BREAK RECOVERY TRACKER (Feature 4)
    # ============================================================
    elif app_mode == "☕ Break Recovery Tracker":
        render_break_tracker()

    # ============================================================
    # MODULE 5: DOWNLOADABLE HEALTH REPORT
    # ============================================================
    elif app_mode == "📑 Downloadable Health Report":
        render_report_view()


# ============================================================
# COMPONENT RENDERERS
# ============================================================

def render_live_monitor():
    st.markdown("""
        <div class="disclaimer-badge">
            <strong>⚠️ Clinical Notice:</strong> The fatigue score is an explainable heuristic powered by 
            blink rate, eyelid closure duration (PERCLOS), and session length. It is intended for ergonomic self-monitoring.
        </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("Live Biometric Stream")
        frame_placeholder = st.empty()
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        start_clicked = btn_col1.button("▶ Start Monitoring", use_container_width=True, type="primary")
        stop_clicked = btn_col2.button("■ Stop Monitoring", use_container_width=True)
        break_btn = btn_col3.button("☕ Take a Break", use_container_width=True)

    with right_col:
        st.subheader("Ocular Telemetry")
        status_ph = st.empty()
        metrics_ph = st.empty()
        st.subheader("Visual Fatigue Index")
        score_ph = st.empty()
        bar_ph = st.empty()
        level_ph = st.empty()
        rec_ph = st.empty()
        with st.expander("🔍 Physiological Score Breakdown"):
            breakdown_ph = st.empty()

    if break_btn:
        st.session_state.selected_module = "☕ Break Recovery Tracker"
        st.rerun()

    if start_clicked:
        reset_session()
        st.session_state.running = True
        st.session_state.last_finished_session = None

    if stop_clicked:
        st.session_state.running = False

    # Handle post-session saving when stopped
    if not st.session_state.running:
        frame_placeholder.info("Click '▶ Start Monitoring' to initialize webcam detection and real-time fatigue analysis.")
        status_ph.metric("Current Eye Status", st.session_state.last_label)

        if st.session_state.last_finished_session is not None:
            sess = st.session_state.last_finished_session
            st.success(f"Session finished! Final Fatigue Score: **{sess['fatigue_score']}/100** ({sess['level']}).")
            save_col1, save_col2 = st.columns(2)
            if save_col1.button("💾 Save Session to Health Records", use_container_width=True, type="primary"):
                sess_id = health_db.save_session(
                    duration_seconds=sess["duration_seconds"],
                    fatigue_score=sess["fatigue_score"],
                    blink_rate=sess["blink_rate"],
                    avg_blink_duration=sess["avg_blink_duration"],
                    perclos=sess["perclos"],
                    notes="Live monitoring session logged by user."
                )
                st.session_state.last_finished_session = None
                st.toast(f"Session saved to database ({sess_id})!")
                st.rerun()

            if save_col2.button("🩺 Report Symptoms for this Session", use_container_width=True):
                st.session_state.selected_module = "🩺 Symptom Check-In"
                st.rerun()
        return

    # Real-time monitoring loop
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam. Please verify camera permissions and ensure no other application is using it.")
        st.session_state.running = False
        return

    model, device = load_model()
    face_mesh = load_face_mesh()

    try:
        final_score = 0
        final_level = "LOW"
        final_breakdown = {}
        final_perclos = 0
        final_duration = 0
        final_blink_rate = 0
        session_seconds = 0

        while st.session_state.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                st.warning("Webcam feed disconnected.")
                break

            frame = cv2.flip(frame, 1)
            eye_crop, pts = get_eye_crop(frame, face_mesh)
            display_frame = frame.copy()

            if eye_crop is not None and eye_crop.size > 0:
                st.session_state.no_face_frames = 0
                label, conf = predict_eye_state(eye_crop, model, device)
                st.session_state.last_label = label
                st.session_state.last_conf = conf

                is_closed_now = (label == "CLOSED")
                now = time.time()
                st.session_state.history.append((now, is_closed_now))

                while (
                    st.session_state.history
                    and now - st.session_state.history[0][0] > PERCLOS_WINDOW_SECONDS
                ):
                    st.session_state.history.popleft()

                # Blink tracking
                if is_closed_now and not st.session_state.is_closed:
                    st.session_state.closure_start = now
                    st.session_state.is_closed = True
                elif not is_closed_now and st.session_state.is_closed:
                    st.session_state.is_closed = False
                    if st.session_state.closure_start is not None:
                        duration = now - st.session_state.closure_start
                        if duration >= MIN_CLOSURE_SECONDS:
                            st.session_state.blink_count += 1
                            st.session_state.blink_durations.append(duration)
                    st.session_state.closure_start = None

                # Bounding box & telemetry overlay
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    box_color = (0, 0, 255) if is_closed_now else (0, 200, 0)
                    cv2.rectangle(
                        display_frame,
                        (min(xs) - 10, min(ys) - 10),
                        (max(xs) + 10, max(ys) + 10),
                        box_color, 2
                    )
                    cv2.putText(
                        display_frame, f"{label} ({conf*100:.0f}%)",
                        (min(xs) - 10, min(ys) - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2
                    )
            else:
                st.session_state.no_face_frames += 1

            frame_placeholder.image(
                cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            # Update metrics
            session_seconds = time.time() - st.session_state.session_start
            minutes = int(session_seconds // 60)
            seconds = int(session_seconds % 60)

            score, breakdown, perclos_val, avg_dur_val, blink_rate_val = compute_fatigue_score(
                st.session_state.history,
                st.session_state.blink_durations,
                session_seconds
            )
            level, emoji, title_desc = fatigue_level(score)

            final_score = score
            final_level = level
            final_breakdown = breakdown
            final_perclos = perclos_val
            final_duration = avg_dur_val
            final_blink_rate = blink_rate_val

            if st.session_state.no_face_frames > 15:
                status_ph.warning("⚠️ No face or eyes detected in camera frame. Adjust angle/lighting.")
            else:
                status_ph.metric(
                    "Eye State",
                    f"{st.session_state.last_label}",
                    f"{st.session_state.last_conf * 100:.0f}% confidence"
                )

            with metrics_ph.container():
                m1, m2 = st.columns(2)
                m1.metric("Blink Count", st.session_state.blink_count)
                m2.metric("Blink Rate", f"{blink_rate_val:.1f} bpm")
                m3, m4 = st.columns(2)
                m3.metric("Session Time", f"{minutes:02d}:{seconds:02d}")
                m4.metric("PERCLOS", f"{perclos_val:.1f}%")

            score_ph.markdown(f"## {emoji} Fatigue Score: **{score}/100**")
            bar_ph.progress(score / 100.0)
            level_ph.markdown(f"**Classification:** `{level}` — *{title_desc}*")
            rec_ph.info(f"**Ergonomic Recommendation:** {recommendation_for(level)}")
            breakdown_ph.table(breakdown)

            time.sleep(0.03)

        # Store session summary when loop ends
        if session_seconds > 5:
            st.session_state.last_finished_session = {
                "duration_seconds": session_seconds,
                "fatigue_score": final_score,
                "level": final_level,
                "blink_rate": round(final_blink_rate, 1),
                "avg_blink_duration": round(final_duration, 2),
                "perclos": round(final_perclos, 1),
            }
            # Auto-save to break pre-score for smooth transition
            st.session_state.break_pre_score = final_score

    finally:
        cap.release()


def render_symptom_checkin():
    st.subheader("🩺 Eye-Health Symptom Check-In")
    st.markdown("""
        Self-reporting ocular discomfort enables EyeGuard to correlate digital eye strain with your subjective experience.
        Rate your current symptoms on a **0 to 5 severity scale**:
        - **0:** None (No symptoms)
        - **1:** Very Mild (Barely noticeable)
        - **2:** Mild (Noticeable but does not disrupt work)
        - **3:** Moderate (Distracting discomfort)
        - **4:** Severe (Significant strain, difficult to focus)
        - **5:** Extreme (Intense pain / unable to continue screen work)
    """)

    with st.form("symptom_form"):
        col1, col2 = st.columns(2)

        with col1:
            dryness = st.slider(
                "🌵 Eye Dryness or Irritation",
                min_value=0, max_value=5, value=1,
                help="Gritty, scratchy feeling, burning, or redness in the eyes."
            )
            headache = st.slider(
                "⚡ Headache",
                min_value=0, max_value=5, value=0,
                help="Frontal or temple tension headache associated with screen viewing."
            )
            blurred_vision = st.slider(
                "🌫️ Blurred Vision",
                min_value=0, max_value=5, value=0,
                help="Difficulty seeing text clearly on the screen or momentary fuzziness."
            )

        with col2:
            discomfort = st.slider(
                "👁️ Eye Discomfort",
                min_value=0, max_value=5, value=1,
                help="General aching, fatigue behind the eyes, or heavy eyelids."
            )
            focus_difficulty = st.slider(
                "🎯 Difficulty Focusing",
                min_value=0, max_value=5, value=1,
                help="Trouble shifting focus between near and distant objects or staying focused."
            )
            context = st.selectbox(
                "Trigger Context:",
                ["Manual Check-In", "Post-Screen Session", "Pre-Break Check", "Post-Break Check", "End of Workday"]
            )

        notes = st.text_input("Additional Observations / Ergonomic Notes (Optional):", placeholder="e.g. Worked under fluorescent lights for 3 hours")

        submitted = st.form_submit_button("💾 Submit Symptom Check-In", use_container_width=True, type="primary")
        if submitted:
            health_db.save_symptom_checkin(
                dryness=dryness,
                headache=headache,
                blurred_vision=blurred_vision,
                discomfort=discomfort,
                focus_difficulty=focus_difficulty,
                trigger_context=context.lower().replace(" ", "_"),
                notes=notes
            )
            st.success("✅ Symptom check-in successfully logged to your health records!")
            st.toast("Symptom record saved!")

    st.markdown("---")
    st.subheader("📋 Recent Symptom Log History")
    symptoms_df = health_db.get_symptoms_df()
    if not symptoms_df.empty:
        display_df = symptoms_df.copy().sort_values("timestamp", ascending=False).head(8)
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%b %d, %Y - %I:%M %p")
        st.dataframe(
            display_df[["timestamp", "dryness", "headache", "blurred_vision", "discomfort", "focus_difficulty", "trigger_context", "notes"]],
            use_container_width=True,
            column_config={
                "dryness": st.column_config.ProgressColumn("Dryness (0-5)", min_value=0, max_value=5, format="%d"),
                "headache": st.column_config.ProgressColumn("Headache (0-5)", min_value=0, max_value=5, format="%d"),
                "blurred_vision": st.column_config.ProgressColumn("Blur (0-5)", min_value=0, max_value=5, format="%d"),
                "discomfort": st.column_config.ProgressColumn("Discomfort (0-5)", min_value=0, max_value=5, format="%d"),
                "focus_difficulty": st.column_config.ProgressColumn("Focus (0-5)", min_value=0, max_value=5, format="%d"),
            }
        )
    else:
        st.info("No symptom reports logged yet.")


def render_health_trends():
    st.subheader("📈 Longitudinal Health & Biometric Trends")
    st.markdown("""
        Review changes in visual fatigue, underlying eye behavior (blinking and PERCLOS), and self-reported symptoms.
    """)

    trend_tab1, trend_tab2, trend_tab3 = st.tabs([
        "1. Fatigue Score Trend (Most Important)",
        "2. Eye Behaviour Trends",
        "3. Symptom History"
    ])

    sessions_df = health_db.get_sessions_df()
    symptoms_df = health_db.get_symptoms_df()

    # ----------------------------------------------------
    # FEATURE 1: FATIGUE SCORE TREND
    # ----------------------------------------------------
    with trend_tab1:
        st.markdown("#### 1. Fatigue Score Trend — Most Important")
        st.caption("A line graph showing estimated fatigue score (0–100) across sessions to identify elevated fatigue clusters.")

        if not sessions_df.empty:
            elevated_sessions = sessions_df[sessions_df["fatigue_score"] >= 60]
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Sessions Recorded", len(sessions_df))
            col_b.metric("Elevated Fatigue Sessions (≥60)", f"{len(elevated_sessions)} ({len(elevated_sessions)/len(sessions_df)*100:.0f}%)")
            col_c.metric("Average Fatigue Score", f"{sessions_df['fatigue_score'].mean():.1f} / 100")

            fig1 = health_charts.create_fatigue_trend_chart(sessions_df)
            st.plotly_chart(fig1, use_container_width=True)

            with st.expander("💡 Clinical Interpretation of Fatigue Trends"):
                st.markdown("""
                    - **Green Zone (0–30):** Healthy ocular resting state with steady blink frequency and rapid closures.
                    - **Yellow Zone (31–60):** Mild digital eye strain. Common during sustained screen sessions of 30–60 minutes.
                    - **Orange Zone (61–80):** **Elevated Fatigue.** Marked by reduced blink rates (<10 bpm) or prolonged lid closure (>0.45s).
                    - **Red Zone (81–100):** **Severe Fatigue.** High risk of corneal desiccation and severe visual fatigue. Immediate screen breaks are strongly recommended.
                """)
        else:
            st.info("No sessions available yet. Run a live monitoring session to generate data.")

    # ----------------------------------------------------
    # FEATURE 2: EYE BEHAVIOUR TRENDS
    # ----------------------------------------------------
    with trend_tab2:
        st.markdown("#### 2. Eye Behaviour Trends Across Sessions")
        st.caption("Plot blink rate, average blink duration and PERCLOS across sessions to understand which factors drive fatigue.")

        if not sessions_df.empty:
            fig2 = health_charts.create_eye_behaviour_chart(sessions_df)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("""
                <div class="metric-card">
                    <strong>🔬 Biometric Insights & Factors:</strong>
                    <ul style="margin-top: 6px; font-size: 14px; color: #475569;">
                        <li><strong>Blink Rate:</strong> Resting baseline is 12–20 bpm. Prolonged screen engagement often causes "computer staring", cutting blink rate by up to 60% and triggering dry eye symptoms.</li>
                        <li><strong>Blink Duration:</strong> Normal blinks take 0.15–0.35s. When drowsy, eyelid velocity slows and closure durations exceed 0.45s.</li>
                        <li><strong>PERCLOS (Percentage of Eye Closure):</strong> Gold-standard physiological metric measuring the percentage of time eyes are closed over a rolling window. PERCLOS > 12% strongly correlates with degraded vigilance.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No biometric session data found.")

    # ----------------------------------------------------
    # FEATURE 3: SYMPTOM HISTORY
    # ----------------------------------------------------
    with trend_tab3:
        st.markdown("#### 3. Weekly Symptom History")
        st.caption("Weekly timeline of self-reported eye dryness, headache, blurred vision, discomfort, and focus difficulty (0–5 severity).")

        if not symptoms_df.empty:
            fig3 = health_charts.create_symptom_history_chart(symptoms_df, days=14)
            st.plotly_chart(fig3, use_container_width=True)

            # Summary metrics
            mean_dry = symptoms_df["dryness"].mean()
            mean_head = symptoms_df["headache"].mean()
            mean_blur = symptoms_df["blurred_vision"].mean()
            mean_disc = symptoms_df["discomfort"].mean()
            mean_foc = symptoms_df["focus_difficulty"].mean()

            st.markdown("##### Average Symptom Severity Ranking")
            s_c1, s_c2, s_c3, s_c4, s_c5 = st.columns(5)
            s_c1.metric("Dryness", f"{mean_dry:.1f}/5")
            s_c2.metric("Headache", f"{mean_head:.1f}/5")
            s_c3.metric("Blur", f"{mean_blur:.1f}/5")
            s_c4.metric("Discomfort", f"{mean_disc:.1f}/5")
            s_c5.metric("Focus", f"{mean_foc:.1f}/5")
        else:
            st.info("No symptom history available yet. Complete a check-in in the 'Symptom Check-In' tab.")


def render_break_tracker():
    st.subheader("☕ Before-Break vs After-Break Comparison")
    st.markdown("""
        A structured workflow to evaluate how an ergonomic rest break impacts visual fatigue and symptom severity.
    """)

    st.markdown("""
        <div class="disclaimer-badge">
            <strong>⚠️ Methodological Note:</strong> This comparison helps track changes before and after a rest period, 
            <strong>but does not prove that the break alone caused the improvement</strong>.
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 1. Pre-Break State")
        pre_score = st.slider("Recorded Fatigue Score Before Break (0-100):", min_value=10, max_value=100, value=int(st.session_state.break_pre_score))
        st.markdown("**Reported Symptoms Before Break (0–5):**")
        b_dry = st.slider("Pre-Break Eye Dryness", 0, 5, 3, key="b_dry")
        b_head = st.slider("Pre-Break Headache", 0, 5, 2, key="b_head")
        b_blur = st.slider("Pre-Break Blurred Vision", 0, 5, 2, key="b_blur")
        b_disc = st.slider("Pre-Break Eye Discomfort", 0, 5, 3, key="b_disc")
        b_focus = st.slider("Pre-Break Difficulty Focusing", 0, 5, 3, key="b_focus")

    with col2:
        st.markdown("### 2. Post-Break Evaluation")
        st.markdown("*Ask the user how they feel afterward:*")
        post_score = st.slider("Re-evaluated Fatigue Score After Break (0-100):", min_value=0, max_value=100, value=max(15, pre_score - 30))
        st.markdown("**Reported Symptoms After Break (0–5):**")
        a_dry = st.slider("Post-Break Eye Dryness", 0, 5, max(0, b_dry - 2), key="a_dry")
        a_head = st.slider("Post-Break Headache", 0, 5, max(0, b_head - 1), key="a_head")
        a_blur = st.slider("Post-Break Blurred Vision", 0, 5, max(0, b_blur - 1), key="a_blur")
        a_disc = st.slider("Post-Break Eye Discomfort", 0, 5, max(0, b_disc - 2), key="a_disc")
        a_focus = st.slider("Post-Break Difficulty Focusing", 0, 5, max(0, b_focus - 2), key="a_focus")

    b_type = st.selectbox(
        "Ergonomic Intervention Type:",
        ["20-20-20 Rule (Gaze 20 ft away for 20s)", "Ocular Palming & Eye Rolling", "Hydration + Walk Away From Screen", "Full 15-Minute Ergonomic Rest"]
    )
    user_notes = st.text_input("User Feedback / Sensations afterward:", placeholder="e.g. Eyes feel noticeably more lubricated, headache faded.")

    if st.button("📊 Save Break Comparison & Update Chart", type="primary", use_container_width=True):
        now = datetime.now()
        start_t = (now - pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        end_t = now.strftime("%Y-%m-%d %H:%M:%S")

        before_sym = {"dryness": b_dry, "headache": b_head, "blurred_vision": b_blur, "discomfort": b_disc, "focus_difficulty": b_focus}
        after_sym = {"dryness": a_dry, "headache": a_head, "blurred_vision": a_blur, "discomfort": a_disc, "focus_difficulty": a_focus}

        health_db.save_break_comparison(
            timestamp_start=start_t,
            timestamp_end=end_t,
            duration_seconds=300,
            before_score=pre_score,
            after_score=post_score,
            before_symptoms=before_sym,
            after_symptoms=after_sym,
            break_type=b_type,
            user_feedback=user_notes
        )
        st.success("✅ Break comparison logged successfully!")

    st.markdown("---")
    st.markdown("### 📊 Before vs After Comparison Chart")
    breaks_df = health_db.get_breaks_df()

    active_record = None
    if not breaks_df.empty:
        active_record = breaks_df.iloc[0]

    fig4 = health_charts.create_break_comparison_chart(active_record)
    st.plotly_chart(fig4, use_container_width=True)

    st.caption("ℹ️ Note: This helps track changes, but does not prove that the break caused the improvement.")

    if not breaks_df.empty:
        st.markdown("##### 📜 Recent Break History Table")
        st.dataframe(
            breaks_df[["timestamp_start", "break_type", "before_fatigue_score", "after_fatigue_score", "user_feedback"]],
            use_container_width=True
        )


def render_report_view():
    st.subheader("📑 Downloadable Healthcare Report")
    st.markdown("""
        Generate an executive digital eye health report suitable for personal tracking, ergonomic reviews, or sharing with an eye care specialist.
    """)

    html_content = health_report.generate_html_health_report()
    csv_content = health_report.generate_csv_export()

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.download_button(
            label="📥 Download Clinical Health Report (.HTML / Print to PDF)",
            data=html_content,
            file_name=f"EyeGuard_Clinical_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            use_container_width=True,
            type="primary"
        )
        st.caption("Tip: Open the downloaded HTML file in any browser and press **Ctrl+P** to save as a clean PDF.")

    with btn_col2:
        st.download_button(
            label="📊 Download Raw Session Logs (.CSV)",
            data=csv_content,
            file_name=f"EyeGuard_Session_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("Raw telemetry data containing timestamp, fatigue score, blink rate, duration, and PERCLOS.")

    st.markdown("---")
    st.markdown("### 📄 In-App Report Preview")
    with st.expander("Click to Preview Complete Formatted Health Report", expanded=True):
        st.components.v1.html(html_content, height=800, scrolling=True)


if __name__ == "__main__":
    main()
