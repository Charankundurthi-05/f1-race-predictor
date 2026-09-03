import streamlit as st
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Drivers | F1 Race Predictor",
    page_icon="🏎️",
    layout="wide"
)


# ============================================================
# THEME / UI
# ============================================================

st.markdown(
    """
    <style>

    .hero {
        padding: 32px 40px;
        border-radius: 0 0 24px 24px;
        margin-bottom: 34px;
        background: linear-gradient(
            135deg,
            #f0f0f0,
            #fafafa
        );
        border: 1px solid #d8d8d8;
    }

    .hero h1 {
        margin: 0;
        font-size: 50px;
        font-weight: 800;
        color: #30313d;
    }

    .hero p {
        margin-top: 20px;
        margin-bottom: 0;
        font-size: 19px;
        color: #6d707a;
    }

    .driver-card {
        padding: 25px 30px;
        border-radius: 20px;
        background: #f0f0f0;
        border: 1px solid #d5d5d5;
        margin-top: 22px;
        margin-bottom: 34px;
    }

    .driver-name {
        font-size: 29px;
        font-weight: 800;
        color: #30313d;
    }

    .driver-team {
        margin-top: 8px;
        font-size: 17px;
        color: #70737d;
    }

    .section-title {
        font-size: 30px;
        font-weight: 800;
        color: var(--text-color);
        margin-top: 34px;
        margin-bottom: 20px;
    }

    .stat-card {
        padding: 22px 26px;
        border-radius: 18px;
        background: #eeeeee;
        border: 1px solid #d5d5d5;
        min-height: 115px;
        margin-bottom: 4px;
    }

    .stat-label {
        font-size: 15px;
        color: #777a84;
        margin-bottom: 11px;
    }

    .stat-value {
        font-size: 31px;
        font-weight: 800;
        color: #30313d;
    }

    .table-container {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid #d5d5d5;
        background: #eeeeee;
    }

    .info-box {
        padding: 20px 24px;
        border-radius: 18px;
        background: #eeeeee;
        border: 1px solid #d5d5d5;
        margin-bottom: 24px;
    }

    .info-title {
        font-size: 15px;
        color: #777a84;
    }

    .info-value {
        font-size: 25px;
        font-weight: 800;
        color: #30313d;
        margin-top: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================

def load_csv(filename):
    possible_paths = [
        PROCESSED_DIR / filename,
        RAW_DIR / filename
    ]

    for path in possible_paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                return None

    return None


def find_column(df, candidates):

    if df is None:
        return None

    lookup = {
        str(column).lower().strip(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.lower().strip()

        if key in lookup:
            return lookup[key]

    return None


def normalize_name(value):

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    replacements = {
        "ï": "i",
        "ü": "u",
        "ö": "o",
        "é": "e",
        "á": "a",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ā": "a",
        "ē": "e",
        "ī": "i",
        "ō": "o",
        "ū": "u"
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return value


def number(value, decimals=0):

    if value is None or pd.isna(value):
        return "—"

    try:

        value = float(value)

        if decimals == 0:
            return f"{int(value):,}"

        return f"{value:,.{decimals}f}"

    except Exception:
        return str(value)


def get_value(row, aliases, default=0):

    if row is None:
        return default

    if isinstance(row, dict):

        for alias in aliases:

            if alias in row:
                value = row[alias]

                if pd.notna(value):
                    return value

        return default

    for alias in aliases:

        if alias in row.index:

            value = row[alias]

            if pd.notna(value):
                return value

    return default


# ============================================================
# LOAD DATA
# ============================================================

race_results = load_csv("race_results.csv")
career_stats = load_csv("driver_career_stats.csv")
season_stats = load_csv("driver_season_stats.csv")


# ============================================================
# BUILD CURRENT 2026 DRIVER LIST
# ============================================================

current_drivers = pd.DataFrame()

if race_results is not None:

    season_col = find_column(
        race_results,
        ["season", "year"]
    )

    driver_col = find_column(
        race_results,
        ["driver_name", "driverName", "full_name"]
    )

    team_col = find_column(
        race_results,
        ["team_name", "constructor_name", "constructorName"]
    )

    round_col = find_column(
        race_results,
        ["round", "race_round"]
    )

    if season_col and driver_col:

        current_2026 = race_results[
            race_results[season_col].astype(str) == "2026"
        ].copy()

        if not current_2026.empty:

            if round_col:

                current_2026["_round"] = pd.to_numeric(
                    current_2026[round_col],
                    errors="coerce"
                )

            else:

                current_2026["_round"] = 0

            current_2026 = current_2026.sort_values("_round")

            rows = []

            for driver_name, group in current_2026.groupby(
                driver_col,
                dropna=True
            ):

                latest = group.iloc[-1]

                team = "—"

                if team_col:

                    value = latest.get(team_col)

                    if pd.notna(value):
                        team = str(value)

                rows.append(
                    {
                        "driver_name": str(driver_name),
                        "team_name": team
                    }
                )

            current_drivers = pd.DataFrame(rows)


# ============================================================
# FALLBACK CURRENT DRIVERS FROM SEASON DATA
# ============================================================

if current_drivers.empty and season_stats is not None:

    season_col = find_column(
        season_stats,
        ["season", "year"]
    )

    driver_col = find_column(
        season_stats,
        ["driver_name", "driverName", "full_name"]
    )

    team_col = find_column(
        season_stats,
        ["constructor_name", "team_name"]
    )

    if season_col and driver_col:

        temp = season_stats[
            season_stats[season_col].astype(str) == "2026"
        ].copy()

        rows = []

        for driver_name, group in temp.groupby(
            driver_col,
            dropna=True
        ):

            latest = group.iloc[-1]

            team = "—"

            if team_col and pd.notna(latest.get(team_col)):
                team = str(latest.get(team_col))

            rows.append(
                {
                    "driver_name": str(driver_name),
                    "team_name": team
                }
            )

        current_drivers = pd.DataFrame(rows)


current_drivers = current_drivers.sort_values(
    "driver_name"
).reset_index(drop=True)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Drivers</h1>
        <p>2026 Formula 1 driver standings, statistics and race results</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODE SELECTOR
# ============================================================

mode = st.radio(
    "Driver database",
    [
        "Current 2026 Drivers",
        "All-Time Drivers"
    ],
    horizontal=True
)


# ============================================================
# CURRENT 2026
# ============================================================

if mode == "Current 2026 Drivers":

    if current_drivers.empty:

        st.error("No 2026 driver data found.")

        st.stop()


    selected_driver = st.selectbox(
        "Select driver",
        current_drivers["driver_name"].tolist()
    )


    selected_driver_normalized = normalize_name(
        selected_driver
    )


    selected_driver_row = current_drivers[
        current_drivers["driver_name"] == selected_driver
    ].iloc[0]


    team = selected_driver_row["team_name"]


    # --------------------------------------------------------
    # DRIVER HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="driver-card">
            <div class="driver-name">{selected_driver}</div>
            <div class="driver-team">
                2026 Team: {team}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # 2026 DRIVER RACE DATA
    # --------------------------------------------------------

    driver_races = pd.DataFrame()

    if race_results is not None:

        season_col = find_column(
            race_results,
            ["season", "year"]
        )

        driver_col = find_column(
            race_results,
            ["driver_name", "driverName", "full_name"]
        )

        if season_col and driver_col:

            driver_races = race_results[
                (race_results[season_col].astype(str) == "2026")
                &
                (
                    race_results[driver_col]
                    .map(normalize_name)
                    == selected_driver_normalized
                )
            ].copy()


    # --------------------------------------------------------
    # SEASON STATS
    # --------------------------------------------------------

    stats = None

    if season_stats is not None:

        season_col = find_column(
            season_stats,
            ["season", "year"]
        )

        driver_col = find_column(
            season_stats,
            ["driver_name", "driverName", "full_name"]
        )

        if season_col and driver_col:

            temp = season_stats[
                season_stats[season_col].astype(str) == "2026"
            ].copy()

            temp["_driver"] = temp[
                driver_col
            ].map(normalize_name)

            matches = temp[
                temp["_driver"] == selected_driver_normalized
            ]

            if not matches.empty:
                stats = matches.iloc[-1]


    # --------------------------------------------------------
    # CALCULATE FALLBACK STATS
    # --------------------------------------------------------

    if stats is None:

        starts = len(driver_races)

        finish_col = find_column(
            driver_races,
            ["finish_position", "position"]
        )

        points_col = find_column(
            driver_races,
            ["points"]
        )

        if finish_col:

            finish = pd.to_numeric(
                driver_races[finish_col],
                errors="coerce"
            )

        else:

            finish = pd.Series(dtype=float)


        if points_col:

            race_points = pd.to_numeric(
                driver_races[points_col],
                errors="coerce"
            ).fillna(0)

        else:

            race_points = pd.Series(dtype=float)


        wins = int((finish == 1).sum())

        podiums = int(
            ((finish >= 1) & (finish <= 3)).sum()
        )

        best_finish = finish.min() if not finish.empty else None

        points = race_points.sum()

        championships = 0

        poles = 0

        races = starts

    else:

        starts = get_value(
            stats,
            [
                "totalRaceStarts",
                "starts",
                "race_starts",
                "total_race_starts"
            ],
            len(driver_races)
        )

        wins = get_value(
            stats,
            [
                "totalRaceWins",
                "wins",
                "race_wins",
                "total_race_wins"
            ]
        )

        podiums = get_value(
            stats,
            [
                "totalPodiums",
                "podiums",
                "total_podiums"
            ]
        )

        poles = get_value(
            stats,
            [
                "totalPolePositions",
                "poles",
                "pole_positions"
            ]
        )

        points = get_value(
            stats,
            [
                "totalPoints",
                "points",
                "season_points"
            ]
        )

        championships = get_value(
            stats,
            [
                "championships",
                "championship",
                "world_championships"
            ]
        )

        best_finish = get_value(
            stats,
            [
                "bestRaceResult",
                "best_finish",
                "best_finish_position"
            ],
            None
        )

        races = get_value(
            stats,
            [
                "totalRaceEntries",
                "races",
                "race_entries"
            ],
            starts
        )


    # --------------------------------------------------------
    # CURRENT CHAMPIONSHIP POSITION
    # --------------------------------------------------------

    championship_position = None

    if season_stats is not None:

        season_col = find_column(
            season_stats,
            ["season", "year"]
        )

        driver_col = find_column(
            season_stats,
            ["driver_name", "driverName", "full_name"]
        )

        position_col = find_column(
            season_stats,
            [
                "championship_position",
                "position",
                "driver_championship_position"
            ]
        )

        if season_col and driver_col and position_col:

            temp = season_stats[
                season_stats[season_col].astype(str) == "2026"
            ].copy()

            temp["_driver"] = temp[
                driver_col
            ].map(normalize_name)

            match = temp[
                temp["_driver"] == selected_driver_normalized
            ]

            if not match.empty:

                championship_position = match.iloc[-1][
                    position_col
                ]


    # If season statistics don't have position,
    # calculate it from available 2026 points.

    if championship_position is None:

        if season_stats is not None:

            season_col = find_column(
                season_stats,
                ["season", "year"]
            )

            driver_col = find_column(
                season_stats,
                ["driver_name", "driverName", "full_name"]
            )

            points_col = find_column(
                season_stats,
                ["totalPoints", "points", "season_points"]
            )

            if season_col and driver_col and points_col:

                temp = season_stats[
                    season_stats[season_col].astype(str) == "2026"
                ].copy()

                temp["_points"] = pd.to_numeric(
                    temp[points_col],
                    errors="coerce"
                ).fillna(0)

                temp = temp.sort_values(
                    "_points",
                    ascending=False
                ).reset_index(drop=True)

                temp["_driver"] = temp[
                    driver_col
                ].map(normalize_name)

                match = temp[
                    temp["_driver"] == selected_driver_normalized
                ]

                if not match.empty:

                    championship_position = (
                        match.index[0] + 1
                    )


    # --------------------------------------------------------
    # 2026 SEASON STATISTICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">2026 Season Statistics</div>',
        unsafe_allow_html=True
    )


    first_row = [
        ("Starts", starts),
        ("Wins", wins),
        ("Podiums", podiums),
        ("Poles", poles)
    ]


    columns = st.columns(4)

    for column, (label, value) in zip(
        columns,
        first_row
    ):

        with column:

            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">
                        {number(value)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    second_row = [
        (
            "Points",
            points
        ),
        (
            "Championship",
            f"P{int(float(championship_position))}"
            if championship_position is not None
            and not pd.isna(championship_position)
            else "—"
        ),
        (
            "Best Finish",
            best_finish if best_finish is not None else "—"
        ),
        (
            "Races",
            races
        )
    ]


    columns = st.columns(4)

    for column, (label, value) in zip(
        columns,
        second_row
    ):

        with column:

            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">
                        {number(value)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # RACE-BY-RACE RECORD
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">2026 Race-by-Race Record</div>',
        unsafe_allow_html=True
    )


    if not driver_races.empty:

        round_col = find_column(
            driver_races,
            ["round", "race_round"]
        )

        race_col = find_column(
            driver_races,
            ["race_name", "raceName"]
        )

        grid_col = find_column(
            driver_races,
            ["grid_position", "grid"]
        )

        finish_col = find_column(
            driver_races,
            ["finish_position", "position"]
        )

        points_col = find_column(
            driver_races,
            ["points"]
        )

        status_col = find_column(
            driver_races,
            ["status"]
        )


        race_table = pd.DataFrame()


        if round_col:
            race_table["Round"] = driver_races[round_col]

        if race_col:
            race_table["Race"] = driver_races[race_col]

        if grid_col:
            race_table["Grid"] = driver_races[grid_col]

        if finish_col:
            race_table["Finish"] = driver_races[finish_col]

        if points_col:
            race_table["Points"] = driver_races[points_col]

        if status_col:
            race_table["Status"] = driver_races[status_col]


        if round_col:

            race_table = race_table.sort_values(
                "Round"
            )


        # Clean numeric columns

        for column in ["Round", "Grid", "Finish"]:

            if column in race_table.columns:

                race_table[column] = pd.to_numeric(
                    race_table[column],
                    errors="coerce"
                ).astype("Int64")


        if "Points" in race_table.columns:

            race_table["Points"] = pd.to_numeric(
                race_table["Points"],
                errors="coerce"
            ).fillna(0)


        st.dataframe(
            race_table,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No completed 2026 race results are available for this driver yet."
        )


# ============================================================
# ALL-TIME DRIVERS
# ============================================================

else:

    if career_stats is None or career_stats.empty:

        st.error(
            "All-time driver statistics could not be found."
        )

        st.stop()


    driver_col = find_column(
        career_stats,
        [
            "driver_name",
            "driverName",
            "full_name",
            "name"
        ]
    )


    if driver_col is None:

        st.error(
            "No driver name column found in the career statistics."
        )

        st.stop()


    all_drivers = sorted(
        career_stats[driver_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


    selected_driver = st.selectbox(
        "Select driver",
        all_drivers
    )


    selected = career_stats[
        career_stats[driver_col].astype(str)
        == selected_driver
    ]


    if selected.empty:

        st.warning(
            f"No statistics found for {selected_driver}."
        )

        st.stop()


    stats = selected.iloc[0]


    # --------------------------------------------------------
    # ALL-TIME HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="driver-card">
            <div class="driver-name">{selected_driver}</div>
            <div class="driver-team">
                All-Time Formula 1 Driver
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # CAREER STATISTICS
    # --------------------------------------------------------

    starts = get_value(
        stats,
        [
            "totalRaceStarts",
            "starts",
            "race_starts",
            "total_race_starts"
        ]
    )

    wins = get_value(
        stats,
        [
            "totalRaceWins",
            "wins",
            "race_wins",
            "total_race_wins"
        ]
    )

    podiums = get_value(
        stats,
        [
            "totalPodiums",
            "podiums",
            "total_podiums"
        ]
    )

    poles = get_value(
        stats,
        [
            "totalPolePositions",
            "poles",
            "pole_positions",
            "total_poles"
        ]
    )

    fastest_laps = get_value(
        stats,
        [
            "totalFastestLaps",
            "fastest_laps",
            "fastestLaps"
        ]
    )

    points = get_value(
        stats,
        [
            "totalPoints",
            "points",
            "career_points"
        ]
    )

    championships = get_value(
        stats,
        [
            "championships",
            "championship",
            "world_championships",
            "titles"
        ]
    )

    best_finish = get_value(
        stats,
        [
            "bestRaceResult",
            "best_finish",
            "best_finish_position"
        ],
        None
    )


    st.markdown(
        '<div class="section-title">All-Time Career Statistics</div>',
        unsafe_allow_html=True
    )


    first_row = [
        ("Starts", starts),
        ("Wins", wins),
        ("Podiums", podiums),
        ("Poles", poles)
    ]


    columns = st.columns(4)

    for column, (label, value) in zip(
        columns,
        first_row
    ):

        with column:

            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">
                        {number(value)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    second_row = [
        ("Fastest Laps", fastest_laps),
        ("Points", points),
        ("Championships", championships),
        (
            "Best Finish",
            best_finish if best_finish is not None else "—"
        )
    ]


    columns = st.columns(4)

    for column, (label, value) in zip(
        columns,
        second_row
    ):

        with column:

            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">
                        {number(value)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # ALL-TIME SEASON-BY-SEASON
    # --------------------------------------------------------

    if season_stats is not None and not season_stats.empty:

        season_col = find_column(
            season_stats,
            ["season", "year"]
        )

        season_driver_col = find_column(
            season_stats,
            [
                "driver_name",
                "driverName",
                "full_name",
                "name"
            ]
        )


        if season_col and season_driver_col:

            temp = season_stats.copy()

            temp["_driver"] = temp[
                season_driver_col
            ].map(normalize_name)

            driver_seasons = temp[
                temp["_driver"]
                == normalize_name(selected_driver)
            ].copy()


            if not driver_seasons.empty:

                st.markdown(
                    '<div class="section-title">Season-by-Season Record</div>',
                    unsafe_allow_html=True
                )


                columns_to_show = []


                preferred = [
                    season_col,
                    "constructor_name",
                    "team_name",
                    "totalRaceStarts",
                    "starts",
                    "totalRaceWins",
                    "wins",
                    "totalPodiums",
                    "podiums",
                    "totalPoints",
                    "points",
                    "bestRaceResult",
                    "best_finish"
                ]


                for column in preferred:

                    if column in driver_seasons.columns:

                        if column not in columns_to_show:
                            columns_to_show.append(column)


                if columns_to_show:

                    table = driver_seasons[
                        columns_to_show
                    ].copy()


                    rename_map = {
                        season_col: "Season",
                        "constructor_name": "Team",
                        "team_name": "Team",
                        "totalRaceStarts": "Starts",
                        "starts": "Starts",
                        "totalRaceWins": "Wins",
                        "wins": "Wins",
                        "totalPodiums": "Podiums",
                        "podiums": "Podiums",
                        "totalPoints": "Points",
                        "points": "Points",
                        "bestRaceResult": "Best Finish",
                        "best_finish": "Best Finish"
                    }


                    table = table.rename(
                        columns=rename_map
                    )


                    table = table.loc[
                        :,
                        ~table.columns.duplicated()
                    ]


                    if "Season" in table.columns:

                        table["Season"] = pd.to_numeric(
                            table["Season"],
                            errors="coerce"
                        )

                        table = table.sort_values(
                            "Season",
                            ascending=False
                        )


                    st.dataframe(
                        table,
                        use_container_width=True,
                        hide_index=True
                    )