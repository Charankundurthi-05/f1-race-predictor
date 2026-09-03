from pathlib import Path
import subprocess
import sys
import json

import pandas as pd
import streamlit as st


# ================================================================
# PATHS
# ================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTION_DIR = DATA_DIR / "predictions"


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# SESSION STATE
# ================================================================

if "prediction_started" not in st.session_state:
    st.session_state.prediction_started = False


# ================================================================
# HELPERS
# ================================================================

def load_csv(path):
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_json(path):
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def clean_value(value):
    if pd.isna(value):
        return "—"
    return str(value)


def format_stage(stage):
    mapping = {
        "pre_practice": "Pre-Practice",
        "fp1": "After FP1",
        "fp2": "After FP2",
        "fp3": "After FP3",
        "qualifying": "After Qualifying",
    }

    return mapping.get(
        str(stage).lower(),
        str(stage).replace("_", " ").title(),
    )


def format_countdown(seconds):
    try:
        seconds = int(max(0, seconds))
    except Exception:
        return "—"

    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60

    return f"{days}d {hours}h {minutes}m"


# ================================================================
# CSS
# ================================================================

st.markdown(
    """
<style>
.hero {
    padding: 32px;
    border-radius: 18px;
    margin-bottom: 22px;
    background: linear-gradient(135deg, rgba(220, 0, 0, 0.95), rgba(80, 0, 0, 0.95));
    color: white;
}

.hero-label {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.9;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    margin-top: 8px;
}

.hero-subtitle {
    font-size: 17px;
    margin-top: 8px;
    opacity: 0.9;
}

.status-card {
    padding: 20px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.10);
    min-height: 100px;
}

.status-label {
    font-size: 13px;
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.status-value {
    font-size: 20px;
    font-weight: 750;
    margin-top: 8px;
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 15px;
}

.podium-card {
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.10);
}

.podium-position {
    font-size: 15px;
    font-weight: 700;
    opacity: 0.7;
}

.podium-driver {
    font-size: 21px;
    font-weight: 800;
    margin-top: 7px;
}

.podium-team {
    font-size: 14px;
    opacity: 0.7;
    margin-top: 4px;
}

.empty-prediction {
    padding: 38px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.10);
    margin-top: 25px;
    margin-bottom: 25px;
}

.empty-title {
    font-size: 25px;
    font-weight: 800;
}

.empty-text {
    font-size: 16px;
    opacity: 0.7;
    margin-top: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ================================================================
# WEEKEND STATE
# ================================================================

weekend_state = load_json(
    PREDICTION_DIR / "weekend_state.json"
)

if weekend_state is None:
    season = 2026
    round_number = 13
    display_race_name = "Italian Grand Prix"
    circuit_name = "Monza"
    weekend_format = "normal"
    current_stage = "pre_practice"
    countdown = 0
    race_date = ""
else:
    season = weekend_state.get("season", 2026)
    round_number = weekend_state.get("round", 13)

    display_race_name = (
        weekend_state.get("race_name")
        or weekend_state.get("official_name")
        or "Italian Grand Prix"
    )

    circuit_name = (
        weekend_state.get("circuit_name")
        or weekend_state.get("circuit")
        or "Monza"
    )

    weekend_format = weekend_state.get(
        "weekend_format",
        "normal",
    )

    current_stage = weekend_state.get(
        "current_stage",
        "pre_practice",
    )

    countdown = weekend_state.get(
        "countdown_seconds",
        0,
    )

    race_date = weekend_state.get(
        "race_date",
        "",
    )


# ================================================================
# HERO
# ================================================================

st.markdown(
    f"""
<div class="hero">
<div class="hero-label">ROUND {round_number} • {str(weekend_format).upper()} WEEKEND</div>
<div class="hero-title">{display_race_name}</div>
<div class="hero-subtitle">{circuit_name} • {str(race_date)[:10] if race_date else "—"}</div>
</div>
""",
    unsafe_allow_html=True,
)


# ================================================================
# STATUS CARDS
# ================================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
<div class="status-card">
<div class="status-label">Circuit</div>
<div class="status-value">{circuit_name}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
<div class="status-card">
<div class="status-label">Weekend</div>
<div class="status-value">{str(weekend_format).upper()}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
<div class="status-card">
<div class="status-label">Prediction Stage</div>
<div class="status-value">{format_stage(current_stage)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
<div class="status-card">
<div class="status-label">Race Countdown</div>
<div class="status-value">{format_countdown(countdown)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ================================================================
# PREDICTION BUTTON
# ================================================================

st.write("")

if st.button(
    "Lights out and away we go 🟢🟢🟢🟢🟢",
    use_container_width=True,
):
    st.session_state.prediction_started = True

    with st.spinner("Updating race prediction..."):

        pipeline = (
            ROOT
            / "src"
            / "prediction"
            / "run_live_pipeline.py"
        )

        result = subprocess.run(
            [
                sys.executable,
                str(pipeline),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )

    if result.returncode == 0:

        st.success(
            "Prediction updated successfully."
        )

        st.cache_data.clear()

        st.rerun()

    else:

        st.error(
            "Prediction pipeline failed."
        )

        if result.stdout:
            st.code(result.stdout[-5000:])

        if result.stderr:
            st.code(result.stderr[-5000:])


# ================================================================
# WAITING SCREEN
# ================================================================

if not st.session_state.prediction_started:

    st.markdown(
        """
<div class="empty-prediction">
<div class="empty-title">🏎️ Ready for Lights Out</div>
<div class="empty-text">Click the button above to generate the current race prediction.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.stop()


# ================================================================
# CURRENT PREDICTION
# ================================================================

prediction_path = (
    PREDICTION_DIR
    / "current_prediction_with_points.csv"
)

if not prediction_path.exists():
    prediction_path = (
        PREDICTION_DIR
        / "current_prediction.csv"
    )

prediction = load_csv(prediction_path)

if prediction is None or prediction.empty:

    st.warning(
        "No prediction is currently available."
    )

    st.stop()


if "predicted_position" in prediction.columns:

    prediction = (
        prediction
        .sort_values("predicted_position")
        .reset_index(drop=True)
    )


# ================================================================
# PODIUM
# ================================================================

st.markdown(
    '<div class="section-title">🔮 Predicted Podium</div>',
    unsafe_allow_html=True,
)


if len(prediction) >= 3:

    p1 = prediction.iloc[0]
    p2 = prediction.iloc[1]
    p3 = prediction.iloc[2]

    def driver_name(row):
        return clean_value(
            row.get("driver_name", "Unknown")
        )

    def team_name(row):
        return clean_value(
            row.get("team_name", "—")
        )

    col1, col2, col3 = st.columns(
        [1, 1.15, 1]
    )

    # ============================================================
    # P2
    # ============================================================

    with col1:
        st.markdown(
            f"""
<div class="podium-card" style="margin-top:35px;">
<div class="podium-position">🥈 P2</div>
<div class="podium-driver">{driver_name(p2)}</div>
<div class="podium-team">{team_name(p2)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ============================================================
    # P1
    # ============================================================

    with col2:
        st.markdown(
            f"""
<div class="podium-card" style="min-height:180px;">
<div class="podium-position">🥇 P1</div>
<div class="podium-driver">{driver_name(p1)}</div>
<div class="podium-team">{team_name(p1)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ============================================================
    # P3
    # ============================================================

    with col3:
        st.markdown(
            f"""
<div class="podium-card" style="margin-top:35px;">
<div class="podium-position">🥉 P3</div>
<div class="podium-driver">{driver_name(p3)}</div>
<div class="podium-team">{team_name(p3)}</div>
</div>
""",
            unsafe_allow_html=True,
        )


# ================================================================
# FULL GRID
# ================================================================

st.markdown(
    '<div class="section-title">🏁 Predicted Full Grid</div>',
    unsafe_allow_html=True,
)

grid_columns = [
    "predicted_position",
    "driver_name",
    "team_name",
    "predicted_race_points",
    "predicted_total_points",
]

available_grid_columns = [
    col
    for col in grid_columns
    if col in prediction.columns
]

grid = prediction[
    available_grid_columns
].copy()

rename_grid = {
    "predicted_position": "Position",
    "driver_name": "Driver",
    "team_name": "Team",
    "predicted_race_points": "Race Points",
    "predicted_total_points": "Total Points",
}

grid = grid.rename(
    columns=rename_grid
)

st.dataframe(
    grid,
    use_container_width=True,
    hide_index=True,
)


# ================================================================
# RACE WEEKEND PROGRESSION
# ================================================================

progression_path = (
    PREDICTION_DIR
    / "current_race_progression.csv"
)

progression = load_csv(progression_path)


if progression is not None and not progression.empty:

    st.markdown(
        '<div class="section-title">📈 Race Weekend Progression</div>',
        unsafe_allow_html=True,
    )

    progression_columns = [
        "driver_name",
        "no_practice_prediction",
        "fp1_position",
        "fp1_prediction",
        "fp2_position",
        "fp2_prediction",
        "fp3_position",
        "fp3_prediction",
        "qualifying_position",
        "final_prediction",
        "race_position",
        "comparison",
    ]

    available_progression_columns = [
        col
        for col in progression_columns
        if col in progression.columns
    ]

    progression_display = progression[
        available_progression_columns
    ].copy()

    rename_progression = {
        "driver_name": "Driver",
        "no_practice_prediction": "No Practice",
        "fp1_position": "FP1 Result",
        "fp1_prediction": "Prediction After FP1",
        "fp2_position": "FP2 Result",
        "fp2_prediction": "Prediction After FP2",
        "fp3_position": "FP3 Result",
        "fp3_prediction": "Prediction After FP3",
        "qualifying_position": "Qualifying Result",
        "final_prediction": "Final Prediction",
        "race_position": "Race Result",
        "comparison": "Comparison",
    }

    progression_display = progression_display.rename(
        columns=rename_progression
    )

    st.dataframe(
        progression_display,
        use_container_width=True,
        hide_index=True,
    )


# ================================================================
# CHAMPIONSHIP
# ================================================================

st.markdown(
    '<div class="section-title">🏆 Championship</div>',
    unsafe_allow_html=True,
)

driver_standings = load_csv(
    PROCESSED_DIR
    / "driver_info_2026_with_photos.csv"
)

constructor_standings = load_csv(
    PROCESSED_DIR
    / "constructor_info_2026.csv"
)


tab1, tab2 = st.tabs(
    ["Drivers", "Constructors"]
)


with tab1:

    if (
        driver_standings is not None
        and not driver_standings.empty
    ):

        driver_display = (
            driver_standings.copy()
        )

        preferred = [
            "position",
            "driver_name",
            "team_name",
            "points",
            "wins",
        ]

        cols = [
            c
            for c in preferred
            if c in driver_display.columns
        ]

        if cols:
            driver_display = driver_display[cols]

        st.dataframe(
            driver_display,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "2026 driver standings are not available."
        )


with tab2:

    if (
        constructor_standings is not None
        and not constructor_standings.empty
    ):

        constructor_display = (
            constructor_standings.copy()
        )

        preferred = [
            "position",
            "constructor_name",
            "points",
            "wins",
        ]

        cols = [
            c
            for c in preferred
            if c in constructor_display.columns
        ]

        if cols:
            constructor_display = (
                constructor_display[cols]
            )

        st.dataframe(
            constructor_display,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "2026 constructor standings are not available."
        )