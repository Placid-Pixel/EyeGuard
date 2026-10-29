"""
EyeGuard - Healthcare Database & Session Storage Manager
--------------------------------------------------------
Handles persistent SQLite storage for:
1. Monitoring sessions (fatigue score, blink rate, duration, PERCLOS)
2. Eye-health symptom check-ins (dryness, headache, blurred vision, discomfort, focus)
3. Before-vs-After break comparison logs
Includes sample historical data generation for realistic analytics right out of the box.
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "eyeguard_health.db")


def get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the necessary database tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Monitoring Sessions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            timestamp DATETIME,
            duration_seconds REAL,
            fatigue_score INTEGER,
            blink_rate REAL,
            avg_blink_duration REAL,
            perclos REAL,
            notes TEXT
        )
    """)

    # 2. Eye-Health Symptom Check-Ins
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS symptoms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp DATETIME,
            dryness INTEGER,
            headache INTEGER,
            blurred_vision INTEGER,
            discomfort INTEGER,
            focus_difficulty INTEGER,
            trigger_context TEXT,
            notes TEXT
        )
    """)

    # 3. Before-Break vs After-Break Comparisons
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS break_comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_start DATETIME,
            timestamp_end DATETIME,
            duration_seconds REAL,
            break_type TEXT,
            before_fatigue_score INTEGER,
            after_fatigue_score INTEGER,
            before_symptoms_json TEXT,
            after_symptoms_json TEXT,
            user_feedback TEXT
        )
    """)

    conn.commit()
    conn.close()
    seed_sample_data_if_empty()


def seed_sample_data_if_empty(force=False):
    """
    Populate realistic historical records for the last 10 days if DB is empty.
    Allows trend charts to be immediately rich, informative, and visual.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM sessions")
    session_count = cursor.fetchone()[0]

    if session_count > 0 and not force:
        conn.close()
        return

    if force:
        cursor.execute("DELETE FROM sessions")
        cursor.execute("DELETE FROM symptoms")
        cursor.execute("DELETE FROM break_comparisons")
        conn.commit()

    base_time = datetime.now() - timedelta(days=9)
    rng = np.random.RandomState(42)

    sample_sessions = []
    sample_symptoms = []

    # Generate 2 to 3 sessions per day over the last 9 days
    for day_offset in range(10):
        current_day = base_time + timedelta(days=day_offset)
        # Morning session: light fatigue
        morning_time = current_day.replace(hour=9, minute=rng.randint(15, 45), second=0)
        m_dur = rng.uniform(1200, 2400) # 20-40 min
        m_blink_rate = round(float(rng.uniform(15.5, 19.5)), 1)
        m_avg_dur = round(float(rng.uniform(0.18, 0.28)), 2)
        m_perclos = round(float(rng.uniform(2.5, 5.2)), 1)
        m_fatigue = int(round(rng.uniform(18, 34)))
        sess_id_m = f"SESS_{morning_time.strftime('%Y%m%d_%H%M%S')}"

        sample_sessions.append((
            sess_id_m, morning_time.strftime("%Y-%m-%d %H:%M:%S"), m_dur,
            m_fatigue, m_blink_rate, m_avg_dur, m_perclos, "Morning work block"
        ))

        # Checkin associated with morning
        sample_symptoms.append((
            sess_id_m, morning_time.strftime("%Y-%m-%d %H:%M:%S"),
            int(rng.choice([0, 1])), int(rng.choice([0, 1])), 0, int(rng.choice([0, 1])), 0,
            "morning_checkin", "Feeling well rested"
        ))

        # Afternoon / Evening session: higher fatigue, prolonged screen time
        afternoon_time = current_day.replace(hour=16, minute=int(rng.randint(10, 50)), second=0)
        a_dur = float(rng.uniform(2100, 3900)) # 35-65 min
        # Some days have elevated fatigue spikes (e.g. days 3, 6, 8)
        is_hard_day = day_offset in [3, 6, 8]
        if is_hard_day:
            a_blink_rate = round(float(rng.uniform(8.2, 11.5)), 1) # Low blink rate / staring
            a_avg_dur = round(float(rng.uniform(0.48, 0.65)), 2)   # Long closures
            a_perclos = round(float(rng.uniform(12.5, 18.0)), 1)   # High PERCLOS
            a_fatigue = int(round(rng.uniform(72, 88)))            # Elevated fatigue!
            notes = "Intensive coding & documentation; severe eye strain"
            sym_dryness = int(rng.choice([3, 4]))
            sym_headache = int(rng.choice([2, 3, 4]))
            sym_blur = int(rng.choice([2, 3]))
            sym_discomfort = int(rng.choice([3, 4]))
            sym_focus = int(rng.choice([3, 4]))
        else:
            a_blink_rate = round(float(rng.uniform(12.0, 15.0)), 1)
            a_avg_dur = round(float(rng.uniform(0.28, 0.38)), 2)
            a_perclos = round(float(rng.uniform(6.0, 9.5)), 1)
            a_fatigue = int(round(rng.uniform(42, 62)))
            notes = "Standard afternoon session"
            sym_dryness = int(rng.choice([1, 2]))
            sym_headache = int(rng.choice([1, 2]))
            sym_blur = int(rng.choice([0, 1]))
            sym_discomfort = int(rng.choice([1, 2]))
            sym_focus = int(rng.choice([1, 2]))

        sess_id_a = f"SESS_{afternoon_time.strftime('%Y%m%d_%H%M%S')}"
        sample_sessions.append((
            sess_id_a, afternoon_time.strftime("%Y-%m-%d %H:%M:%S"), a_dur,
            a_fatigue, a_blink_rate, a_avg_dur, a_perclos, notes
        ))

        sample_symptoms.append((
            sess_id_a, afternoon_time.strftime("%Y-%m-%d %H:%M:%S"),
            sym_dryness, sym_headache, sym_blur, sym_discomfort, sym_focus,
            "session_end", "Logged after extended screen usage"
        ))

    cursor.executemany("""
        INSERT INTO sessions (
            session_id, timestamp, duration_seconds, fatigue_score,
            blink_rate, avg_blink_duration, perclos, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_sessions)

    cursor.executemany("""
        INSERT INTO symptoms (
            session_id, timestamp, dryness, headache, blurred_vision,
            discomfort, focus_difficulty, trigger_context, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_symptoms)

    # Sample Break Comparisons (Before vs After)
    sample_breaks = [
        (
            (datetime.now() - timedelta(days=5, hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
            (datetime.now() - timedelta(days=5, hours=3, minutes=50)).strftime("%Y-%m-%d %H:%M:%S"),
            600, "20-20-20 Rule + Ocular Palming",
            78, 42,
            json.dumps({"dryness": 4, "headache": 3, "blurred_vision": 2, "discomfort": 4, "focus_difficulty": 3}),
            json.dumps({"dryness": 2, "headache": 1, "blurred_vision": 1, "discomfort": 2, "focus_difficulty": 1}),
            "Felt significantly refreshed after looking out the window and resting eyes."
        ),
        (
            (datetime.now() - timedelta(days=3, hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
            (datetime.now() - timedelta(days=3, hours=4, minutes=53)).strftime("%Y-%m-%d %H:%M:%S"),
            420, "Eye Rolling & Deep Blinking",
            68, 38,
            json.dumps({"dryness": 3, "headache": 2, "blurred_vision": 2, "discomfort": 3, "focus_difficulty": 2}),
            json.dumps({"dryness": 1, "headache": 1, "blurred_vision": 0, "discomfort": 1, "focus_difficulty": 1}),
            "Deliberate blinking eased irritation."
        ),
        (
            (datetime.now() - timedelta(days=1, hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
            (datetime.now() - timedelta(days=1, hours=2, minutes=48)).strftime("%Y-%m-%d %H:%M:%S"),
            720, "Walk, Hydration & Screen Off",
            84, 46,
            json.dumps({"dryness": 4, "headache": 4, "blurred_vision": 3, "discomfort": 4, "focus_difficulty": 4}),
            json.dumps({"dryness": 2, "headache": 2, "blurred_vision": 1, "discomfort": 2, "focus_difficulty": 2}),
            "Walking away from desk helped relieve tension and headache."
        ),
    ]

    cursor.executemany("""
        INSERT INTO break_comparisons (
            timestamp_start, timestamp_end, duration_seconds, break_type,
            before_fatigue_score, after_fatigue_score,
            before_symptoms_json, after_symptoms_json, user_feedback
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_breaks)

    conn.commit()
    conn.close()


def save_session(duration_seconds, fatigue_score, blink_rate, avg_blink_duration, perclos, notes=""):
    """Record a completed monitoring session."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_id = f"SESS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cursor.execute("""
        INSERT INTO sessions (
            session_id, timestamp, duration_seconds, fatigue_score,
            blink_rate, avg_blink_duration, perclos, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id, now_str, round(duration_seconds, 1), int(fatigue_score),
        round(blink_rate, 1), round(avg_blink_duration, 2), round(perclos, 1), notes
    ))
    conn.commit()
    conn.close()
    return session_id


def save_symptom_checkin(dryness, headache, blurred_vision, discomfort, focus_difficulty,
                         trigger_context="manual_checkin", notes="", session_id=None):
    """Save user self-reported symptoms (0-5 scale)."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO symptoms (
            session_id, timestamp, dryness, headache, blurred_vision,
            discomfort, focus_difficulty, trigger_context, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id, now_str, int(dryness), int(headache), int(blurred_vision),
        int(discomfort), int(focus_difficulty), trigger_context, notes
    ))
    conn.commit()
    conn.close()


def save_break_comparison(timestamp_start, timestamp_end, duration_seconds,
                          before_score, after_score, before_symptoms, after_symptoms,
                          break_type="Ergonomic Break", user_feedback=""):
    """Save before-and-after break comparison data."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO break_comparisons (
            timestamp_start, timestamp_end, duration_seconds, break_type,
            before_fatigue_score, after_fatigue_score,
            before_symptoms_json, after_symptoms_json, user_feedback
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp_start, timestamp_end, duration_seconds, break_type,
        int(before_score), int(after_score),
        json.dumps(before_symptoms), json.dumps(after_symptoms),
        user_feedback
    ))
    conn.commit()
    conn.close()


def get_sessions_df(limit=100):
    """Fetch sessions as pandas DataFrame."""
    init_db()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM sessions ORDER BY timestamp ASC", conn)
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col in ["duration_seconds", "fatigue_score", "blink_rate", "avg_blink_duration", "perclos"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_symptoms_df(limit=100):
    """Fetch symptoms as pandas DataFrame."""
    init_db()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM symptoms ORDER BY timestamp ASC", conn)
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col in ["dryness", "headache", "blurred_vision", "discomfort", "focus_difficulty"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_breaks_df():
    """Fetch break comparisons as pandas DataFrame."""
    init_db()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM break_comparisons ORDER BY timestamp_start DESC", conn)
    conn.close()
    if not df.empty:
        df["timestamp_start"] = pd.to_datetime(df["timestamp_start"])
        df["timestamp_end"] = pd.to_datetime(df["timestamp_end"])
        for col in ["duration_seconds", "before_fatigue_score", "after_fatigue_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_health_summary_stats():
    """Compute high-level summary metrics for executive display and reports."""
    sessions_df = get_sessions_df()
    symptoms_df = get_symptoms_df()
    breaks_df = get_breaks_df()

    if sessions_df.empty:
        return {
            "total_sessions": 0,
            "avg_fatigue": 0,
            "max_fatigue": 0,
            "elevated_sessions_count": 0,
            "avg_blink_rate": 0,
            "avg_perclos": 0,
            "total_symptom_reports": 0,
            "avg_dryness": 0,
            "avg_headache": 0,
            "avg_discomfort": 0,
            "break_count": 0,
            "avg_break_improvement": 0,
        }

    elevated = sessions_df[sessions_df["fatigue_score"] >= 60]
    avg_fatigue = round(float(sessions_df["fatigue_score"].mean()), 1)
    max_fatigue = int(sessions_df["fatigue_score"].max())
    avg_blink = round(float(sessions_df["blink_rate"].mean()), 1)
    avg_perclos = round(float(sessions_df["perclos"].mean()), 1)

    avg_dry = round(float(symptoms_df["dryness"].mean()), 1) if not symptoms_df.empty else 0
    avg_head = round(float(symptoms_df["headache"].mean()), 1) if not symptoms_df.empty else 0
    avg_disc = round(float(symptoms_df["discomfort"].mean()), 1) if not symptoms_df.empty else 0

    avg_break_imp = 0
    if not breaks_df.empty:
        improvements = breaks_df["before_fatigue_score"] - breaks_df["after_fatigue_score"]
        avg_break_imp = round(float(improvements.mean()), 1)

    return {
        "total_sessions": len(sessions_df),
        "avg_fatigue": avg_fatigue,
        "max_fatigue": max_fatigue,
        "elevated_sessions_count": len(elevated),
        "avg_blink_rate": avg_blink,
        "avg_perclos": avg_perclos,
        "total_symptom_reports": len(symptoms_df),
        "avg_dryness": avg_dry,
        "avg_headache": avg_head,
        "avg_discomfort": avg_disc,
        "break_count": len(breaks_df),
        "avg_break_improvement": avg_break_imp,
    }
