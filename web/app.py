from pathlib import Path
import json

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = ROOT / "data" / "predictions"
PROCESSED_DIR = ROOT / "data" / "processed"


st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ================================================================
# TEAM COLORS
# ================================================================

TEAM_COLORS = {
    "McLaren": "#FF8000",
    "Ferrari": "#E80020",
    "Mercedes": "#27F4D2",
    "Red Bull": "#3671C6",
    "RB F1 Team": "#6692FF",
    "Alpine F1 Team": "#2293D1",
    "Aston Martin": "#229971",
    "Williams": "#64C4FF",
    "Haas F1 Team": "#B6BABD",
    "Audi": "#D0D0D0",
    "Cadillac F1 Team": "#B8B8B8",
}


# ================================================================
# LOADERS
# ================================================================

@st.cache_data(ttl=10)
def load_csv(path):

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(ttl=10)
def load_json(path):

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def team_color(team):

    team = str(team)

    for name, color in TEAM_COLORS.items():

        if name.lower() in team.lower():
            return color

    return "#888888"


def format_stage(stage):

    stages = {
        "pre_practice": "PRE-PRACTICE",
        "fp1": "FP1",
        "fp2": "FP2",
        "fp3": "FP3",
        "qualifying": "QUALIFYING",
        "sprint_qualifying": "SPRINT QUALIFYING",
        "sprint": "SPRINT",
    }

    return stages.get(
        str(stage).lower(),
        str(stage).upper(),
    )


def format_countdown(seconds):

    if seconds is None:
        return "—"

    try:
        seconds = max(0, int(seconds))
    except Exception:
        return "—"

    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60

    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m"

    seconds = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ================================================================
# CSS
# ================================================================

st.markdown(
    """
<style>

.stApp {
    background: var(--background-color);
    color: var(--text-color);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    padding: 30px;
    border-radius: 20px;
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 22px;
}

.hero-label {
    color: var(--text-color) !important;
    opacity: 0.58;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    color: var(--text-color) !important;
    font-size: 40px;
    font-weight: 850;
    margin-top: 7px;
}

.hero-subtitle {
    color: var(--text-color) !important;
    opacity: 0.65;
    font-size: 15px;
    margin-top: 9px;
}


/* ============================================================
   STATUS
   ============================================================ */

.status-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 15px;
    padding: 18px;
    min-height: 105px;
}

.status-label {
    color: var(--text-color) !important;
    opacity: 0.55;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.status-value {
    color: var(--text-color) !important;
    font-size: 21px;
    font-weight: 800;
    margin-top: 8px;
}


/* ============================================================
   SECTION
   ============================================================ */

.section-title {
    color: var(--text-color) !important;
    font-size: 27px;
    font-weight: 850;
    margin-top: 32px;
    margin-bottom: 16px;
}


/* ============================================================
   BUTTON
   ============================================================ */

.stButton > button {
    color: var(--text-color) !important;
    background: var(--secondary-background-color) !important;
    border: 1px solid rgba(128,128,128,0.30) !important;
    font-weight: 750 !important;
    border-radius: 10px !important;
}

.stButton > button:hover {
    color: var(--text-color) !important;
    background: var(--background-color) !important;
}


/* ============================================================
   PODIUM
   ============================================================ */

.podium-wrapper {
    width: 100%;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    gap: 20px;
    margin: 20px 0 45px 0;
}

.podium-slot {
    width: 31%;
}

.podium-slot.p1 {
    margin-bottom: 60px;
}

.podium-card {
    min-height: 235px;
    padding: 22px 15px;
    border-radius: 18px;
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.25);
    text-align: center;
}

.podium-position {
    color: var(--text-color) !important;
    opacity: 0.55;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
}

.podium-medal {
    font-size: 38px;
    margin: 6px 0;
}

.podium-driver {
    color: var(--text-color) !important;
    font-size: 21px;
    font-weight: 850;
}

.podium-team {
    color: var(--text-color) !important;
    opacity: 0.58;
    font-size: 14px;
    margin-top: 5px;
}

.podium-points {
    color: var(--text-color) !important;
    font-size: 13px;
    font-weight: 700;
    margin-top: 15px;
}


/* ============================================================
   GRID
   ============================================================ */

.driver-row {
    display: flex;
    align-items: center;
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 12px;
    margin-bottom: 7px;
    padding: 12px 15px;
}

.driver-pos {
    width: 45px;
    color: var(--text-color) !important;
    opacity: 0.55;
    font-size: 17px;
    font-weight: 800;
}

.team-bar {
    width: 4px;
    height: 38px;
    border-radius: 4px;
    margin-right: 12px;
}

.driver-name {
    color: var(--text-color) !important;
    font-size: 15px;
    font-weight: 750;
}

.driver-team {
    color: var(--text-color) !important;
    opacity: 0.55;
    font-size: 11px;
    margin-top: 2px;
}

.driver-points {
    margin-left: auto;
    text-align: right;
    color: var(--text-color) !important;
    font-weight: 750;
}

.small-muted {
    color: var(--text-color) !important;
    opacity: 0.50;
    font-size: 11px;
}


/* ============================================================
   PROGRESSION TABLE
   ============================================================ */

.progression-wrapper {
    width: 100%;
    overflow-x: auto;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 10px;
}

.progression-table {
    width: 100%;
    min-width: 1250px;
    border-collapse: collapse;
    font-size: 13px;
}

.progression-table th {
    padding: 13px 10px;
    background: rgba(128,128,128,0.15);
    color: var(--text-color) !important;
    font-weight: 800;
    text-align: center;
    border-bottom: 1px solid rgba(128,128,128,0.25);
    white-space: nowrap;
}

.progression-table th.driver-header {
    text-align: left;
}

.progression-table td {
    padding: 11px 10px;
    color: var(--text-color) !important;
    text-align: center;
    border-bottom: 1px solid rgba(128,128,128,0.12);
    white-space: nowrap;
}

.progression-table td.driver-cell {
    text-align: left;
    font-weight: 750;
}

.progression-table td.team-cell {
    text-align: left;
    opacity: 0.60;
}

.progression-table tr:nth-child(even) td {
    background: rgba(128,128,128,0.045);
}

.progression-table tr:last-child td {
    border-bottom: none;
}

.actual-cell {
    font-weight: 650;
}

.prediction-cell {
    font-weight: 850;
}

.final-cell {
    font-weight: 900;
}

.diff-good {
    font-weight: 850;
}

.diff-bad {
    font-weight: 850;
}

.waiting {
    opacity: 0.35;
}


/* ============================================================
   TABS
   ============================================================ */

button[data-baseweb="tab"] {
    color: var(--text-color) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--text-color) !important;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 800px) {

    .hero-title {
        font-size: 31px;
    }

    .podium-wrapper {
        gap: 7px;
    }

    .podium-card {
        min-height: 210px;
        padding: 17px 7px;
    }

    .podium-driver {
        font-size: 16px;
    }

    .podium-team {
        font-size: 11px;
    }

    .podium-medal {
        font-size: 28px;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ================================================================
# LOAD WEEKEND STATE
# ================================================================

state = load_json(
    PREDICTION_DIR / "weekend_state.json"
)

if state:

    season = state.get(
        "season",
        2026,
    )

    round_number = state.get(
        "round",
        "—",
    )

    race_name = state.get(
        "race_name",
        "Upcoming Grand Prix",
    )

    circuit_name = state.get(
        "circuit_name",
        state.get(
            "circuit_id",
            "—",
        ),
    )

    race_date = state.get(
        "race_date",
        "",
    )

    weekend_format = state.get(
        "weekend_format",
        "normal",
    )

    current_stage = state.get(
        "current_stage",
        "pre_practice",
    )

    countdown = state.get(
        "countdown_seconds",
    )

else:

    season = 2026
    round_number = "—"
    race_name = "Upcoming Grand Prix"
    circuit_name = "—"
    race_date = ""
    weekend_format = "normal"
    current_stage = "pre_practice"
    countdown = None


# ================================================================
# HEADER
# ================================================================

display_race_name = str(
    race_name
).replace(
    "Formula 1 Pirelli ",
    "",
)


st.markdown(
    f"""
<div class="hero">
<div class="hero-label">🏎️ MACHINE LEARNING • FORMULA 1</div>
<div class="hero-title">F1 Race Predictor</div>
<div class="hero-subtitle">Stage-aware Formula 1 race prediction using historical performance, practice, qualifying, circuit, regulation, sprint and weather features.</div>
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    f"""
<div class="hero">
<div class="hero-label">ROUND {round_number} • {str(weekend_format).upper()} WEEKEND</div>
<div class="hero-title">{display_race_name}</div>
<div class="hero-subtitle">{circuit_name} • {race_date[:10] if race_date else "—"}</div>
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
    use_container_width=False,
):

    with st.spinner(
        "Updating race prediction..."
    ):

        import subprocess
        import sys

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
            st.code(
                result.stdout[-5000:]
            )

        if result.stderr:
            st.code(
                result.stderr[-5000:]
            )


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


prediction = load_csv(
    prediction_path
)


if prediction is not None and not prediction.empty:

    if "predicted_position" in prediction.columns:

        prediction = prediction.sort_values(
            "predicted_position"
        ).reset_index(drop=True)


    # ============================================================
    # PODIUM
    # ============================================================

    st.markdown(
        '<div class="section-title">🔮 Predicted Podium</div>',
        unsafe_allow_html=True,
    )


    if len(prediction) >= 3:

        p1 = prediction.iloc[0]
        p2 = prediction.iloc[1]
        p3 = prediction.iloc[2]


        def get_driver(row):

            return row.get(
                "driver_name",
                "—",
            )


        def get_team(row):

            return row.get(
                "team_name",
                "—",
            )


        def get_points(row):

            try:

                return int(
                    row.get(
                        "predicted_race_points",
                        0,
                    )
                )

            except Exception:

                return 0


        p1_driver = get_driver(p1)
        p1_team = get_team(p1)
        p1_points = get_points(p1)

        p2_driver = get_driver(p2)
        p2_team = get_team(p2)
        p2_points = get_points(p2)

        p3_driver = get_driver(p3)
        p3_team = get_team(p3)
        p3_points = get_points(p3)


        p1_color = team_color(p1_team)
        p2_color = team_color(p2_team)
        p3_color = team_color(p3_team)


        st.markdown(
            f"""
<div class="podium-wrapper">
<div class="podium-slot p2">
<div class="podium-card" style="border-top:5px solid {p2_color};">
<div class="podium-position">P2</div>
<div class="podium-medal">🥈</div>
<div class="podium-driver">{p2_driver}</div>
<div class="podium-team">{p2_team}</div>
<div class="podium-points">{p2_points} predicted race points</div>
</div>
</div>

<div class="podium-slot p1">
<div class="podium-card" style="border-top:5px solid {p1_color};">
<div class="podium-position">P1</div>
<div class="podium-medal">🥇</div>
<div class="podium-driver">{p1_driver}</div>
<div class="podium-team">{p1_team}</div>
<div class="podium-points">{p1_points} predicted race points</div>
</div>
</div>

<div class="podium-slot p3">
<div class="podium-card" style="border-top:5px solid {p3_color};">
<div class="podium-position">P3</div>
<div class="podium-medal">🥉</div>
<div class="podium-driver">{p3_driver}</div>
<div class="podium-team">{p3_team}</div>
<div class="podium-points">{p3_points} predicted race points</div>
</div>
</div>
</div>
""",
            unsafe_allow_html=True,
        )


    # ============================================================
    # FULL GRID
    # ================================================================

    st.markdown(
        '<div class="section-title">🏁 Predicted Full Grid</div>',
        unsafe_allow_html=True,
    )

    if prediction is not None and not prediction.empty:

        grid_table = prediction.copy()

        grid_columns = [
            "predicted_position",
            "driver_name",
            "team_name",
            "predicted_race_points",
            "predicted_total_points",
        ]

        grid_columns = [
            column
            for column in grid_columns
            if column in grid_table.columns
        ]

        grid_table = grid_table[grid_columns].copy()

        grid_table = grid_table.rename(
            columns={
                "predicted_position": "Pos",
                "driver_name": "Driver",
                "team_name": "Team",
                "predicted_race_points": "Race Points",
                "predicted_total_points": "Total Points",
            }
        )

        if "Pos" in grid_table.columns:
            grid_table["Pos"] = pd.to_numeric(
                grid_table["Pos"],
                errors="coerce",
            ).astype("Int64")

        for column in ["Race Points", "Total Points"]:
            if column in grid_table.columns:
                grid_table[column] = pd.to_numeric(
                    grid_table[column],
                    errors="coerce",
                ).fillna(0)

        if "Pos" in grid_table.columns:
            grid_table = grid_table.sort_values("Pos")

        st.dataframe(
            grid_table,
            use_container_width=True,
            hide_index=True,
        )


# WEEKEND PROGRESSION
# ================================================================

st.markdown(
    '<div class="section-title">📈 Race Weekend Prediction Progression</div>',
    unsafe_allow_html=True,
)

progression_path = (
    PREDICTION_DIR
    / "current_race_progression.csv"
)

progression = load_csv(
    progression_path
)

if progression is None or progression.empty:

    st.info(
        "Race progression data is not available yet."
    )

else:

    display = progression.copy()

    position_columns = [
        "no_practice_prediction",
        "fp1_result",
        "prediction_after_fp1",
        "fp2_result",
        "prediction_after_fp2",
        "fp3_result",
        "prediction_after_fp3",
        "qualifying_result",
        "final_prediction",
        "race_result",
        "prediction_difference",
    ]

    for column in position_columns:
        if column in display.columns:
            display[column] = pd.to_numeric(
                display[column],
                errors="coerce",
            ).astype("Int64")

    progression_columns = [
        "driver_name",
        "team_name",
        "no_practice_prediction",
        "fp1_result",
        "prediction_after_fp1",
        "fp2_result",
        "prediction_after_fp2",
        "fp3_result",
        "prediction_after_fp3",
        "qualifying_result",
        "final_prediction",
        "race_result",
        "prediction_difference",
    ]

    progression_columns = [
        column
        for column in progression_columns
        if column in display.columns
    ]

    display = display[progression_columns].copy()

    display = display.rename(
        columns={
            "driver_name": "Driver",
            "team_name": "Team",
            "no_practice_prediction": "No Practice",
            "fp1_result": "FP1 Result",
            "prediction_after_fp1": "Prediction After FP1",
            "fp2_result": "FP2 Result",
            "prediction_after_fp2": "Prediction After FP2",
            "fp3_result": "FP3 Result",
            "prediction_after_fp3": "Prediction After FP3",
            "qualifying_result": "Qualifying Result",
            "final_prediction": "Final Prediction",
            "race_result": "Race Result",
            "prediction_difference": "Comparison",
        }
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Actual session results are shown separately from predictions. "
        "Comparison = Race position − Final qualifying-stage prediction."
    )


st.markdown(
    '<div class="section-title">📈 Race Weekend Prediction Progression</div>',
    unsafe_allow_html=True,
)


progression_path = (
    PREDICTION_DIR
    / "current_race_progression.csv"
)

progression = load_csv(
    progression_path
)


if progression is None or progression.empty:

    st.info(
        "Race progression data is not available yet."
    )

else:

    display = progression.copy()


    # ------------------------------------------------------------
    # FORMAT POSITIONS
    # ------------------------------------------------------------

    position_columns = [
        "no_practice_prediction",
        "fp1_result",
        "prediction_after_fp1",
        "fp2_result",
        "prediction_after_fp2",
        "fp3_result",
        "prediction_after_fp3",
        "qualifying_result",
        "final_prediction",
        "race_result",
        "prediction_difference",
    ]


    for column in position_columns:

        if column in display.columns:

            display[column] = pd.to_numeric(
                display[column],
                errors="coerce",
            )


    def display_position(value):

        if pd.isna(value):
            return '<span class="waiting">—</span>'

        return f"<strong>P{int(value)}</strong>"


    html = """
<div class="progression-wrapper">
<table class="progression-table">
<thead>
<tr>
<th class="driver-header">Driver</th>
<th class="driver-header">Team</th>
<th>No Practice</th>
<th>FP1</th>
<th>After FP1</th>
<th>FP2</th>
<th>After FP2</th>
<th>FP3</th>
<th>After FP3</th>
<th>Qualifying</th>
<th>Final Prediction</th>
<th>Race</th>
<th>Δ</th>
</tr>
</thead>
<tbody>
"""


    for _, row in display.iterrows():

        driver = row.get(
            "driver_name",
            "—",
        )

        team = row.get(
            "team_name",
            "—",
        )


        no_practice = display_position(
            row.get(
                "no_practice_prediction",
                pd.NA,
            )
        )

        fp1 = display_position(
            row.get(
                "fp1_result",
                pd.NA,
            )
        )

        after_fp1 = display_position(
            row.get(
                "prediction_after_fp1",
                pd.NA,
            )
        )

        fp2 = display_position(
            row.get(
                "fp2_result",
                pd.NA,
            )
        )

        after_fp2 = display_position(
            row.get(
                "prediction_after_fp2",
                pd.NA,
            )
        )

        fp3 = display_position(
            row.get(
                "fp3_result",
                pd.NA,
            )
        )

        after_fp3 = display_position(
            row.get(
                "prediction_after_fp3",
                pd.NA,
            )
        )

        qualifying = display_position(
            row.get(
                "qualifying_result",
                pd.NA,
            )
        )

        final_prediction = display_position(
            row.get(
                "final_prediction",
                pd.NA,
            )
        )

        race = display_position(
            row.get(
                "race_result",
                pd.NA,
            )
        )


        difference_value = row.get(
            "prediction_difference",
            pd.NA,
        )


        if pd.isna(difference_value):

            difference = (
                '<span class="waiting">—</span>'
            )

        else:

            difference_value = int(
                difference_value
            )

            if difference_value == 0:

                difference = (
                    '<strong>0</strong>'
                )

            elif difference_value > 0:

                difference = (
                    f'<strong>+{difference_value}</strong>'
                )

            else:

                difference = (
                    f'<strong>{difference_value}</strong>'
                )


        html += f"""
<tr>
<td class="driver-cell">{driver}</td>
<td class="team-cell">{team}</td>
<td>{no_practice}</td>
<td class="actual-cell">{fp1}</td>
<td class="prediction-cell">{after_fp1}</td>
<td class="actual-cell">{fp2}</td>
<td class="prediction-cell">{after_fp2}</td>
<td class="actual-cell">{fp3}</td>
<td class="prediction-cell">{after_fp3}</td>
<td class="actual-cell">{qualifying}</td>
<td class="prediction-cell final-cell">{final_prediction}</td>
<td class="actual-cell">{race}</td>
<td>{difference}</td>
</tr>
"""


    html += """
</tbody>
</table>
</div>
"""


    st.markdown(
        html,
        unsafe_allow_html=True,
    )


    st.caption(
        "Actual session results are shown separately from predictions. "
        "Δ = Race position − Final qualifying-stage prediction."
    )


# ================================================================
# CHAMPIONSHIP
# ================================================================

driver_standings = load_csv(
    PROCESSED_DIR
    / "driver_info_2026_with_photos.csv"
)

constructor_standings = load_csv(
    PROCESSED_DIR
    / "constructor_info_2026.csv"
)


st.markdown(
    '<div class="section-title">🏆 2026 Championship</div>',
    unsafe_allow_html=True,
)


tab1, tab2 = st.tabs(
    [
        "Drivers",
        "Constructors",
    ]
)


with tab1:

    if driver_standings is not None:

        columns = [
            "position",
            "driver_name",
            "constructor",
            "points",
            "wins",
        ]

        columns = [
            column
            for column in columns
            if column in driver_standings.columns
        ]

        table = driver_standings[
            columns
        ].copy()

        table = table.rename(
            columns={
                "position": "Pos",
                "driver_name": "Driver",
                "constructor": "Team",
                "points": "Points",
                "wins": "Wins",
            }
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Driver standings unavailable."
        )


with tab2:

    if constructor_standings is not None:

        columns = [
            "position",
            "constructor_name",
            "points",
            "wins",
        ]

        columns = [
            column
            for column in columns
            if column in constructor_standings.columns
        ]

        table = constructor_standings[
            columns
        ].copy()

        table = table.rename(
            columns={
                "position": "Pos",
                "constructor_name": "Team",
                "points": "Points",
                "wins": "Wins",
            }
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Constructor standings unavailable."
        )


st.divider()

st.caption(
    "F1 Race Predictor • Historical data from 2014 onward • "
    "Leakage-safe stage-aware machine learning"
)