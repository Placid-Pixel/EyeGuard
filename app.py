"""
EyeGuard - AI-Powered Digital Eye Fatigue Monitor
---------------------------------------------------
A single-page Streamlit app that:
  1. Reads frames from your webcam (OpenCV)
  2. Finds the eyes in each frame (MediaPipe Face Mesh)
  3. Crops + preprocesses the eye region exactly the way the model was trained
  4. Runs your existing MobileNetV2 classifier (best_model.pt) to get OPEN/CLOSED
  5. Tracks blinks and eye-closure time
  6. Turns that into a simple, explainable 0-100 "fatigue score"

This file assumes the following already exist in the SAME folder:
    best_model.pt       <- your trained weights
    MobileNetV2.py       <- defines mobilenet_v2(), used EXACTLY as you trained it

>>> READ ME FIRST: ONE THING YOU MUST VERIFY <<<
Your model outputs class *indices* (0, 1, ...), not the words "OPEN"/"CLOSED".
We don't have your dataset's folder listing, so CLASS_NAMES below is a best
guess based on the common convention (PyTorch's ImageFolder sorts class
folders alphabetically, so "Closed" -> index 0, "Open" -> index 1).
If the app runs but Status is backwards (shows CLOSED when your eyes are open,
or vice versa), just flip the two strings in CLASS_NAMES below. That's it.
"""

import time
import collections

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import mediapipe as mp

from MobileNetV2 import mobilenet_v2  # your existing architecture file


# ============================================================
# CONFIG - the handful of things you may need to tweak
# ============================================================

MODEL_PATH = "best_model.pt"

# Index 0 -> ? , Index 1 -> ?  (see big warning above the imports)
# Default assumes alphabetical ImageFolder ordering: Closed=0, Open=1
CLASS_NAMES = ["CLOSED", "OPEN"]

IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# A blink/closure shorter than this many seconds is treated as noise
# (a single misread frame), not a real eye-closed event.
MIN_CLOSURE_SECONDS = 0.08

# How many recent seconds of eye-state history we look at for the
# "percentage of time closed" part of the fatigue score.
PERCLOS_WINDOW_SECONDS = 60

# Eye-landmark indices from MediaPipe Face Mesh (468-point mesh) used to
# build a bounding box around each eye.
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


# ============================================================
# MODEL LOADING (cached so it only happens once)
# ============================================================

@st.cache_resource(show_spinner="Loading EyeGuard model...")
def load_model():
    """Load the trained MobileNetV2 and pick CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pretrained=False: we're about to load OUR OWN trained weights below,
    # not the original 1000-class ImageNet ones - this also skips an
    # unnecessary download, which matters on event-day WiFi.
    model = mobilenet_v2(pretrained=False)

    # map_location=device makes this work whether the weights were saved
    # from a GPU machine or a CPU machine - it always loads correctly here.
    state_dict = torch.load(MODEL_PATH, map_location=device)

    # The architecture file always builds a 1000-class head by default.
    # Your checkpoint was trained for far fewer classes (OPEN/CLOSED), so
    # we read the real class count straight out of the checkpoint itself
    # and rebuild the final layer to match - no guessing required.
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
    """MediaPipe Face Mesh, used only to locate the eyes in each frame."""
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# Preprocessing pipeline - MUST match training exactly.
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]
)


# ============================================================
# EYE DETECTION + CROPPING
# ============================================================

def get_eye_crop(frame_bgr, face_mesh):
    """
    Find the eye region in a BGR frame using MediaPipe Face Mesh.
    Returns a cropped BGR image of one eye, or None if no face was found.

    We use whichever single eye gives a valid crop (left eye first, then
    right) - for a simple fatigue demo, one eye is enough and it keeps
    things fast and simple.
    """
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

        # pad the tight box so we capture the full eye, not just the inner
        # landmarks (otherwise the crop is too tight around the eyelid)
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

    # also return landmark points so we can draw a box on the preview
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in box_pts]
    return crop, pts


def predict_eye_state(eye_crop_bgr, model, device):
    """Run the model on a cropped eye image. Returns (label_str, confidence)."""
    tensor = preprocess(eye_crop_bgr).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
    label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
    return label, confidence


# ============================================================
# FATIGUE SCORE - simple, explainable heuristic (NOT medical advice)
# ============================================================

def compute_fatigue_score(history, blink_durations, session_seconds):
    """
    history: deque of (timestamp, is_closed_bool) for the last PERCLOS window
    blink_durations: list of completed blink durations (seconds), most recent last
    session_seconds: how long monitoring has been running

    Returns (score 0-100, dict of the sub-scores for transparency)
    """
    now = time.time()

    # 1) PERCLOS-style component: % of recent time spent with eyes closed.
    #    Normal blinking keeps this low; drowsy eyes stay closed longer/more often.
    recent = [c for (t, c) in history if now - t <= PERCLOS_WINDOW_SECONDS]
    perclos_pct = (sum(recent) / len(recent) * 100) if recent else 0.0
    perclos_score = min(perclos_pct * 2.5, 100)  # scaled so it's sensitive

    # 2) Average closure duration of recent blinks. A normal blink is
    #    quick (~0.1-0.4s). Long closures suggest drowsy/heavy eyelids.
    recent_durations = blink_durations[-10:] if blink_durations else []
    avg_duration = sum(recent_durations) / len(recent_durations) if recent_durations else 0.0
    duration_score = min((avg_duration / 1.5) * 100, 100)

    # 3) Blink rate. A healthy resting blink rate is roughly 12-20/min.
    #    Both unusually low (staring, early drowsiness) and very high rates
    #    push the score up a little.
    minutes = max(session_seconds / 60, 0.05)
    blink_rate = len(blink_durations) / minutes
    if blink_rate < 10:
        rate_score = min((10 - blink_rate) * 10, 100)
    elif blink_rate > 30:
        rate_score = min((blink_rate - 30) * 5, 100)
    else:
        rate_score = 0

    # 4) Mild escalation just from a long continuous session (eye strain
    #    builds up over time even without obvious drowsiness signs).
    session_score = min((session_seconds / 1800) * 100, 100)  # 30 min -> 100

    weighted = (
        0.40 * perclos_score
        + 0.30 * duration_score
        + 0.20 * rate_score
        + 0.10 * session_score
    )
    score = int(round(min(max(weighted, 0), 100)))

    breakdown = {
        "PERCLOS (time eyes closed)": round(perclos_score, 1),
        "Avg blink duration": round(duration_score, 1),
        "Blink rate": round(rate_score, 1),
        "Session length": round(session_score, 1),
    }
    return score, breakdown


def fatigue_level(score):
    if score <= 30:
        return "LOW", "🟢"
    elif score <= 60:
        return "MODERATE", "🟡"
    elif score <= 80:
        return "HIGH", "🟠"
    else:
        return "VERY HIGH", "🔴"


def recommendation_for(level):
    return {
        "LOW": "You're doing fine. Keep an eye on your posture and lighting.",
        "MODERATE": "Consider a short 1-2 minute break and blink a few times deliberately.",
        "HIGH": "Take a 5-10 minute break and look away from the screen (try the 20-20-20 rule).",
        "VERY HIGH": "Strongly consider stopping screen work for a while and resting your eyes.",
    }[level]


# ============================================================
# STREAMLIT APP
# ============================================================

def init_state():
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


def main():
    st.set_page_config(page_title="EyeGuard", page_icon="👁", layout="wide")
    init_state()

    st.title("👁 EyeGuard")
    st.caption("AI-Powered Digital Eye Fatigue Monitor")
    st.caption(
        "⚠️ Prototype for demonstration only. The fatigue score is a simple "
        "heuristic, not a medically validated measurement."
    )

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("Live Webcam Feed")
        frame_placeholder = st.empty()
        btn_col1, btn_col2 = st.columns(2)
        start_clicked = btn_col1.button("▶ Start Monitoring", use_container_width=True)
        stop_clicked = btn_col2.button("■ Stop Monitoring", use_container_width=True)

    with right_col:
        st.subheader("Status")
        status_ph = st.empty()
        metrics_ph = st.empty()
        st.subheader("Fatigue")
        score_ph = st.empty()
        bar_ph = st.empty()
        level_ph = st.empty()
        rec_ph = st.empty()
        with st.expander("How is the fatigue score calculated?"):
            breakdown_ph = st.empty()

    if start_clicked:
        reset_session()
        st.session_state.running = True

    if stop_clicked:
        st.session_state.running = False

    if not st.session_state.running:
        frame_placeholder.info("Click 'Start Monitoring' to begin.")
        status_ph.metric("Current Eye Status", st.session_state.last_label)
        return

    # --- Monitoring loop ---
    # Note: this is a simple blocking loop, the way small Streamlit demos
    # commonly read a webcam. It runs until Stop is clicked or an error
    # occurs; use the Stop button rather than closing the tab.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error(
            "Could not open the webcam. Check that it's connected, not in use "
            "by another app, and that this browser/OS has webcam permission "
            "granted for your terminal/IDE. See the Troubleshooting notes."
        )
        st.session_state.running = False
        return

    model, device = load_model()
    face_mesh = load_face_mesh()

    detected_classes = model.classifier.out_features
    if detected_classes != len(CLASS_NAMES):
        st.warning(
            f"Checkpoint has {detected_classes} output classes, but "
            f"CLASS_NAMES has {len(CLASS_NAMES)} entries ({CLASS_NAMES}). "
            "Update CLASS_NAMES at the top of app.py to match, then restart."
        )

    try:
        while st.session_state.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                st.warning("Lost the webcam feed. Stopping.")
                break

            frame = cv2.flip(frame, 1)  # mirror, feels more natural
            eye_crop, pts = get_eye_crop(frame, face_mesh)

            display_frame = frame.copy()

            if eye_crop is not None and eye_crop.size > 0:
                st.session_state.no_face_frames = 0
                label, conf = predict_eye_state(eye_crop, model, device)
                st.session_state.last_label = label
                st.session_state.last_conf = conf

                is_closed_now = label == "CLOSED"
                now = time.time()
                st.session_state.history.append((now, is_closed_now))
                # drop old history outside the PERCLOS window
                while (
                    st.session_state.history
                    and now - st.session_state.history[0][0] > PERCLOS_WINDOW_SECONDS
                ):
                    st.session_state.history.popleft()

                # blink tracking: OPEN -> CLOSED starts a closure,
                # CLOSED -> OPEN ends it and (if long enough) counts a blink
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

                # draw a simple box around the detected eye for feedback
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    color = (0, 0, 255) if is_closed_now else (0, 200, 0)
                    cv2.rectangle(
                        display_frame,
                        (min(xs) - 10, min(ys) - 10),
                        (max(xs) + 10, max(ys) + 10),
                        color,
                        2,
                    )
            else:
                st.session_state.no_face_frames += 1

            frame_placeholder.image(
                cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

            # --- update right-side panel ---
            session_seconds = time.time() - st.session_state.session_start
            minutes = int(session_seconds // 60)
            seconds = int(session_seconds % 60)
            blink_rate = (
                st.session_state.blink_count / max(session_seconds / 60, 0.05)
            )

            if st.session_state.no_face_frames > 15:
                status_ph.warning("No face/eye detected - adjust position or lighting.")
            else:
                status_ph.metric(
                    "Current Eye Status",
                    st.session_state.last_label,
                    f"{st.session_state.last_conf * 100:.0f}% confidence",
                )

            with metrics_ph.container():
                m1, m2 = st.columns(2)
                m1.metric("Blink Count", st.session_state.blink_count)
                m2.metric("Blink Rate", f"{blink_rate:.1f}/min")
                m3, m4 = st.columns(2)
                m3.metric("Session Duration", f"{minutes:02d}:{seconds:02d}")

            score, breakdown = compute_fatigue_score(
                st.session_state.history,
                st.session_state.blink_durations,
                session_seconds,
            )
            level, emoji = fatigue_level(score)

            score_ph.markdown(f"## Fatigue Score: {score}/100")
            bar_ph.progress(score / 100)
            level_ph.markdown(f"**Status: {emoji} {level}**")
            rec_ph.info(recommendation_for(level))
            breakdown_ph.table(breakdown)

            # small sleep keeps CPU usage sane and gives Streamlit a chance
            # to process the Stop button click
            time.sleep(0.03)

    finally:
        cap.release()


if __name__ == "__main__":
    main()
