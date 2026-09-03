import streamlit as st
import pandas as pd
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RACE_FILE = PROJECT_ROOT / "data" / "raw" / "race_results.csv"
CAREER_FILE = PROJECT_ROOT / "data" / "processed" / "constructor_career_stats.csv"


st.set_page_config(
    page_title="Teams | F1 Race Predictor",
    page_icon="🏁",
    layout="wide"
)


# =========================================================
# STYLING
# =========================================================

st.markdown(
    dedent("""
    <style>

    .team-hero {
        padding: 28px 30px;
        border-radius: 18px;
        margin-bottom: 22px;
        border: 1px solid rgba(128,128,128,0.25);
        background: linear-gradient(
            135deg,
            rgba(120,120,120,0.14),
            rgba(120,120,120,0.04)
        );
    }

    .team-hero h1 {
        margin: 0;
        font-size: 42px;
        font-weight: 800;
    }

    .team-hero p {
        margin: 7px 0 0 0;
        font-size: 16px;
        opacity: 0.72;
    }

    .section-title {
        font-size: 24px;
        font-weight: 750;
        margin-top: 28px;
        margin-bottom: 14px;
    }

    .team-info {
        padding: 18px 22px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.10);
        margin: 18px 0 24px 0;
    }

    .team-info-name {
        font-size: 22px;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .team-info-sub {
        font-size: 15px;
        opacity: 0.72;
    }

    [data-baseweb="select"] > div {
        background-color: rgba(128,128,128,0.10) !important;
        border-color: rgba(128,128,128,0.25) !important;
    }

    .team-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        overflow: hidden;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        font-size: 14px;
    }

    .team-table th {
        padding: 12px 14px;
        text-align: left;
        font-weight: 700;
        background: rgba(128,128,128,0.16);
        border-bottom: 1px solid rgba(128,128,128,0.25);
    }

    .team-table td {
        padding: 11px 14px;
        border-bottom: 1px solid rgba(128,128,128,0.12);
    }

    .team-table tr:nth-child(even) td {
        background: rgba(128,128,128,0.045);
    }

    .team-table tr:last-child td {
        border-bottom: none;
    }

    </style>
    """),
    unsafe_allow_html=True
)


# =========================================================
# LOAD RACE DATA
# =========================================================

if not RACE_FILE.exists():
    st.error("race_results.csv not found.")
    st.stop()


races = pd.read_csv(RACE_FILE)

races["season"] = pd.to_numeric(
    races["season"],
    errors="coerce"
)

races["round"] = pd.to_numeric(
    races["round"],
    errors="coerce"
)

races["finish_position"] = pd.to_numeric(
    races["finish_position"],
    errors="coerce"
)

races["grid_position"] = pd.to_numeric(
    races["grid_position"],
    errors="coerce"
)

races["points"] = pd.to_numeric(
    races["points"],
    errors="coerce"
).fillna(0)


races_2026 = races[
    races["season"] == 2026
].copy()


if races_2026.empty:
    st.error("No 2026 race data found.")
    st.stop()


# =========================================================
# CURRENT 2026 TEAMS
# =========================================================

current_teams = (
    races_2026[
        ["team_id", "team_name"]
    ]
    .dropna()
    .drop_duplicates("team_id")
    .sort_values("team_name")
    .reset_index(drop=True)
)


# =========================================================
# 2026 TEAM STANDINGS
# =========================================================

team_standings = (
    races_2026
    .groupby(
        ["team_id", "team_name"],
        as_index=False
    )["points"]
    .sum()
    .sort_values(
        ["points", "team_name"],
        ascending=[False, True]
    )
    .reset_index(drop=True)
)

team_standings["championship_position"] = range(
    1,
    len(team_standings) + 1
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    dedent("""
    <div class="team-hero">
        <h1>Teams</h1>
        <p>
            Current 2026 constructors and all-time Formula 1 team statistics
        </p>
    </div>
    """),
    unsafe_allow_html=True
)


# =========================================================
# MODE
# =========================================================

mode = st.radio(
    "Team database",
    [
        "Current 2026 Teams",
        "All-Time Teams"
    ],
    horizontal=True
)


# =========================================================
# CURRENT 2026
# =========================================================

if mode == "Current 2026 Teams":

    team_names = current_teams[
        "team_name"
    ].tolist()

    selected_team = st.selectbox(
        "Select team",
        team_names,
        key="current_team"
    )

    selected = current_teams[
        current_teams["team_name"] == selected_team
    ].iloc[0]

    team_id = selected["team_id"]


    # =====================================================
    # TEAM RESULTS
    # =====================================================

    team_results = races_2026[
        races_2026["team_id"] == team_id
    ].copy()

    team_results = team_results.sort_values("round")


    # =====================================================
    # TEAM STATS
    # =====================================================

    points = team_results["points"].sum()

    wins = int(
        (team_results["finish_position"] == 1).sum()
    )

    podiums = int(
        team_results["finish_position"]
        .between(1, 3)
        .sum()
    )

    poles = int(
        (team_results["grid_position"] == 1).sum()
    )

    races_completed = (
        team_results["round"]
        .nunique()
    )


    # =====================================================
    # CHAMPIONSHIP POSITION
    # =====================================================

    standing_match = team_standings[
        team_standings["team_id"] == team_id
    ]

    if standing_match.empty:
        championship_position = "—"
    else:
        championship_position = (
            f"P{int(standing_match.iloc[0]['championship_position'])}"
        )


    # =====================================================
    # TEAM HEADER
    # =====================================================

    st.markdown(
        dedent(f"""
        <div class="team-info">
            <div class="team-info-name">{selected_team}</div>
            <div class="team-info-sub">
                2026 Formula 1 constructor
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )


    # =====================================================
    # TEAM STATISTICS
    # =====================================================

    st.markdown(
        '<div class="section-title">2026 Season Statistics</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Points",
        f"{points:,.0f}"
    )

    c2.metric(
        "Wins",
        f"{wins:,}"
    )

    c3.metric(
        "Podiums",
        f"{podiums:,}"
    )

    c4.metric(
        "Poles",
        f"{poles:,}"
    )


    c5, c6 = st.columns(2)

    c5.metric(
        "Championship",
        championship_position
    )

    c6.metric(
        "Rounds",
        f"{races_completed:,}"
    )


    # =====================================================
    # 2026 DRIVERS
    # =====================================================

    st.markdown(
        '<div class="section-title">2026 Drivers</div>',
        unsafe_allow_html=True
    )


    team_drivers = (
        team_results[
            ["driver_id", "driver_name"]
        ]
        .drop_duplicates()
        .sort_values("driver_name")
    )


    st.markdown(
        team_drivers[
            ["driver_name"]
        ]
        .rename(
            columns={
                "driver_name": "Driver"
            }
        )
        .to_html(
            index=False,
            classes="team-table",
            escape=True
        ),
        unsafe_allow_html=True
    )


    # =====================================================
    # RACE-BY-RACE TEAM RECORD
    # =====================================================

    st.markdown(
        '<div class="section-title">2026 Race-by-Race Record</div>',
        unsafe_allow_html=True
    )


    race_summary = []

    for round_number in sorted(
        team_results["round"].dropna().unique()
    ):

        round_results = team_results[
            team_results["round"] == round_number
        ]

        race_name = round_results.iloc[0]["race_name"]

        round_points = round_results["points"].sum()

        best_finish_values = (
            round_results["finish_position"]
            .dropna()
        )

        if best_finish_values.empty:
            best_finish = "—"
        else:
            best_finish = int(
                best_finish_values.min()
            )

        race_summary.append(
            {
                "Round": int(round_number),
                "Race": race_name,
                "Best Finish": best_finish,
                "Points": round_points
            }
        )


    race_table = pd.DataFrame(race_summary)


    st.markdown(
        race_table.to_html(
            index=False,
            classes="team-table",
            escape=True
        ),
        unsafe_allow_html=True
    )


# =========================================================
# ALL-TIME TEAMS
# =========================================================

else:

    if not CAREER_FILE.exists():
        st.error(
            "constructor_career_stats.csv not found."
        )
        st.stop()


    career = pd.read_csv(CAREER_FILE)


    # -----------------------------------------------------
    # Find constructor name
    # -----------------------------------------------------

    name_column = None

    for column in [
        "constructor_name",
        "team_name",
        "constructorName",
        "name"
    ]:
        if column in career.columns:
            name_column = column
            break


    if name_column is None:
        st.error(
            "Constructor name column not found."
        )
        st.stop()


    all_time_teams = (
        career[name_column]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )


    selected_team = st.selectbox(
        "Select team",
        all_time_teams,
        key="all_time_team"
    )


    selected_rows = career[
        career[name_column].astype(str)
        == selected_team
    ]


    if selected_rows.empty:
        st.warning(
            f"No career statistics found for {selected_team}."
        )
        st.stop()


    row = selected_rows.iloc[0]


    def get_value(columns, default=0):

        for column in columns:

            if column in row.index:

                value = row[column]

                if pd.notna(value):
                    return value

        return default


    wins = get_value(
        [
            "total_race_wins",
            "race_wins",
            "wins"
        ]
    )

    podiums = get_value(
        [
            "total_podiums",
            "podiums"
        ]
    )

    championships = get_value(
        [
            "total_championship_wins",
            "championship_wins",
            "championships"
        ]
    )

    starts = get_value(
        [
            "total_race_starts",
            "race_starts",
            "starts"
        ]
    )

    points = get_value(
        [
            "total_points",
            "points"
        ]
    )


    # =====================================================
    # HEADER
    # =====================================================

    st.markdown(
        dedent(f"""
        <div class="team-info">
            <div class="team-info-name">{selected_team}</div>
            <div class="team-info-sub">
                Formula 1 constructor career profile
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )


    # =====================================================
    # ALL-TIME STATISTICS
    # =====================================================

    st.markdown(
        '<div class="section-title">All-Time Statistics</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Wins",
        f"{float(wins):,.0f}"
    )

    c2.metric(
        "Podiums",
        f"{float(podiums):,.0f}"
    )

    c3.metric(
        "Championships",
        f"{float(championships):,.0f}"
    )

    c4.metric(
        "Points",
        f"{float(points):,.0f}"
    )


    c5 = st.columns(1)[0]

    c5.metric(
        "Race Starts",
        f"{float(starts):,.0f}"
    )


    # =====================================================
    # ALL-TIME TEAM TABLE
    # =====================================================

    st.markdown(
        '<div class="section-title">All-Time Team Statistics</div>',
        unsafe_allow_html=True
    )


    display = career.copy()


    column_map = {}

    for column, label in [
        ("constructor_name", "Team"),
        ("team_name", "Team"),
        ("total_race_wins", "Wins"),
        ("race_wins", "Wins"),
        ("wins", "Wins"),
        ("total_podiums", "Podiums"),
        ("podiums", "Podiums"),
        ("total_championship_wins", "Championships"),
        ("championship_wins", "Championships"),
        ("championships", "Championships"),
        ("total_points", "Points"),
        ("points", "Points")
    ]:

        if column in display.columns:

            if label not in column_map.values():
                column_map[column] = label


    table_columns = list(column_map.keys())


    if table_columns:

        table = display[
            table_columns
        ].rename(
            columns=column_map
        )

        st.markdown(
            table.to_html(
                index=False,
                classes="team-table",
                escape=True
            ),
            unsafe_allow_html=True
        )