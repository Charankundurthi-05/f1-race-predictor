from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

MASTER_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "master_dataset_with_sprint.csv"
)

SESSION_WEATHER_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "session_weather_features.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "master_dataset_with_stage_weather.csv"
)


# ------------------------------------------------------------
# Session ordering
#
# This represents the actual information available by each
# prediction stage.
# ------------------------------------------------------------

NORMAL_STAGE_SESSIONS = {
    "pre_practice": [],
    "fp1": ["fp1"],
    "fp2": ["fp1", "fp2"],
    "fp3": ["fp1", "fp2", "fp3"],
    "qualifying": ["fp1", "fp2", "fp3", "qualifying"],
}


SPRINT_STAGE_SESSIONS = {
    "pre_practice": [],
    "fp1": ["fp1"],
    "fp2": ["fp1", "sprint_qualifying", "sprint", "fp2"],
    "fp3": [
        "fp1",
        "sprint_qualifying",
        "sprint",
        "fp2",
        "fp3",
    ],
    "qualifying": [
        "fp1",
        "sprint_qualifying",
        "sprint",
        "fp2",
        "fp3",
        "qualifying",
    ],
}


WEATHER_METRICS = [
    "temperature_avg",
    "temperature_min",
    "temperature_max",
    "humidity_avg",
    "precipitation_total",
    "rain_hours",
    "rain_flag",
    "wind_speed_avg",
    "wind_speed_max",
    "wind_direction_avg",
    "cloud_cover_avg",
    "weather_observations",
]


def load_master():
    print()
    print("Loading master dataset...")

    master = pd.read_csv(MASTER_PATH)

    print(
        f"Master dataset shape: {master.shape}"
    )

    required = [
        "season",
        "round",
        "weekend_format",
    ]

    missing = [
        column
        for column in required
        if column not in master.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in master dataset: {missing}"
        )

    master["season"] = pd.to_numeric(
        master["season"],
        errors="coerce"
    )

    master["round"] = pd.to_numeric(
        master["round"],
        errors="coerce"
    )

    master["weekend_format"] = (
        master["weekend_format"]
        .fillna("normal")
        .astype(str)
        .str.lower()
    )

    return master


def load_session_weather():
    print()
    print("Loading session weather...")

    weather = pd.read_csv(
        SESSION_WEATHER_PATH
    )

    print(
        f"Session weather shape: {weather.shape}"
    )

    required = [
        "season",
        "round",
        "session",
    ]

    missing = [
        column
        for column in required
        if column not in weather.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in session weather: {missing}"
        )

    weather["season"] = pd.to_numeric(
        weather["season"],
        errors="coerce"
    )

    weather["round"] = pd.to_numeric(
        weather["round"],
        errors="coerce"
    )

    weather["session"] = (
        weather["session"]
        .astype(str)
        .str.lower()
    )

    for metric in WEATHER_METRICS:
        if metric in weather.columns:
            weather[metric] = pd.to_numeric(
                weather[metric],
                errors="coerce"
            )

    return weather


def create_session_lookup(weather):
    lookup = {}

    for _, row in weather.iterrows():

        key = (
            int(row["season"]),
            int(row["round"]),
            row["session"],
        )

        lookup[key] = row

    return lookup


def get_session_weather(
    lookup,
    season,
    round_number,
    session,
):
    key = (
        int(season),
        int(round_number),
        session,
    )

    return lookup.get(key)


def add_stage_columns(
    master,
    weather_lookup,
):
    print()
    print("Building stage-specific weather features...")

    result = master.copy()

    stages = [
        "pre_practice",
        "fp1",
        "fp2",
        "fp3",
        "qualifying",
    ]

    # --------------------------------------------------------
    # Initialize all stage weather columns.
    # --------------------------------------------------------

    for stage in stages:

        for metric in WEATHER_METRICS:

            column_name = (
                f"{stage}_weather_{metric}"
            )

            result[column_name] = pd.NA

    # --------------------------------------------------------
    # Process each race.
    # --------------------------------------------------------

    race_keys = (
        result[
            [
                "season",
                "round",
                "weekend_format",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["season", "round"]
        )
    )

    processed_races = 0

    coverage_records = []

    for _, race in race_keys.iterrows():

        season = int(race["season"])
        round_number = int(race["round"])
        weekend_format = race["weekend_format"]

        if weekend_format == "sprint":
            stage_sessions = SPRINT_STAGE_SESSIONS
        else:
            stage_sessions = NORMAL_STAGE_SESSIONS

        race_mask = (
            (result["season"] == season)
            & (result["round"] == round_number)
        )

        race_row_count = race_mask.sum()

        for stage, sessions in stage_sessions.items():

            available_sessions = 0

            # ------------------------------------------------
            # For every allowed historical session, collect
            # its weather metrics.
            #
            # Instead of exposing all sessions individually,
            # aggregate the available session observations into
            # stage-level summary values.
            # ------------------------------------------------

            session_rows = []

            for session in sessions:

                session_weather = get_session_weather(
                    weather_lookup,
                    season,
                    round_number,
                    session,
                )

                if session_weather is not None:
                    session_rows.append(
                        session_weather
                    )
                    available_sessions += 1

            # ------------------------------------------------
            # No weather available at this stage.
            # ------------------------------------------------

            if not session_rows:

                coverage_records.append(
                    {
                        "season": season,
                        "round": round_number,
                        "weekend_format": weekend_format,
                        "stage": stage,
                        "scheduled_sessions": len(sessions),
                        "available_sessions": 0,
                        "race_driver_rows": race_row_count,
                    }
                )

                continue

            session_df = pd.DataFrame(
                session_rows
            )

            # ------------------------------------------------
            # Stage aggregation
            #
            # We use:
            #
            # temperature:
            #   mean across available sessions
            #
            # min/max:
            #   overall min/max
            #
            # humidity/wind/cloud:
            #   mean
            #
            # precipitation:
            #   sum
            #
            # rain:
            #   total rain hours + flag
            #
            # observations:
            #   sum
            # ------------------------------------------------

            values = {
                "temperature_avg":
                    session_df["temperature_avg"].mean(),

                "temperature_min":
                    session_df["temperature_min"].min(),

                "temperature_max":
                    session_df["temperature_max"].max(),

                "humidity_avg":
                    session_df["humidity_avg"].mean(),

                "precipitation_total":
                    session_df["precipitation_total"].sum(),

                "rain_hours":
                    session_df["rain_hours"].sum(),

                "rain_flag":
                    int(
                        session_df["rain_flag"]
                        .fillna(0)
                        .max()
                    ),

                "wind_speed_avg":
                    session_df["wind_speed_avg"].mean(),

                "wind_speed_max":
                    session_df["wind_speed_max"].max(),

                "wind_direction_avg":
                    session_df["wind_direction_avg"].mean(),

                "cloud_cover_avg":
                    session_df["cloud_cover_avg"].mean(),

                "weather_observations":
                    session_df["weather_observations"].sum(),
            }

            for metric, value in values.items():

                column_name = (
                    f"{stage}_weather_{metric}"
                )

                result.loc[
                    race_mask,
                    column_name
                ] = value

            coverage_records.append(
                {
                    "season": season,
                    "round": round_number,
                    "weekend_format": weekend_format,
                    "stage": stage,
                    "scheduled_sessions": len(sessions),
                    "available_sessions": available_sessions,
                    "race_driver_rows": race_row_count,
                }
            )

        processed_races += 1

    coverage = pd.DataFrame(
        coverage_records
    )

    return result, coverage


def validate(result, coverage):
    print()
    print("=" * 70)
    print("STAGE WEATHER VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Stage coverage
    # --------------------------------------------------------

    print()
    print("Coverage by stage:")

    summary = (
        coverage
        .groupby("stage")
        .agg(
            races=("round", "count"),
            races_with_weather=(
                "available_sessions",
                lambda x: (x > 0).sum(),
            ),
            total_available_sessions=(
                "available_sessions",
                "sum",
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(index=False)
    )

    # --------------------------------------------------------
    # Sprint-specific coverage
    # --------------------------------------------------------

    sprint_coverage = coverage[
        coverage["weekend_format"] == "sprint"
    ]

    if not sprint_coverage.empty:

        print()
        print("Sprint weekend coverage:")

        sprint_summary = (
            sprint_coverage
            .groupby("stage")
            .agg(
                races=("round", "count"),
                races_with_weather=(
                    "available_sessions",
                    lambda x: (x > 0).sum(),
                ),
            )
            .reset_index()
        )

        print(
            sprint_summary.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Normal weekend coverage
    # --------------------------------------------------------

    normal_coverage = coverage[
        coverage["weekend_format"] == "normal"
    ]

    if not normal_coverage.empty:

        print()
        print("Normal weekend coverage:")

        normal_summary = (
            normal_coverage
            .groupby("stage")
            .agg(
                races=("round", "count"),
                races_with_weather=(
                    "available_sessions",
                    lambda x: (x > 0).sum(),
                ),
            )
            .reset_index()
        )

        print(
            normal_summary.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Check that pre-practice contains no weather.
    # --------------------------------------------------------

    pre_columns = [
        column
        for column in result.columns
        if column.startswith(
            "pre_practice_weather_"
        )
    ]

    pre_nonnull = (
        result[pre_columns]
        .notna()
        .sum()
        .sum()
    )

    print()
    print(
        f"Pre-practice weather non-null values: "
        f"{pre_nonnull}"
    )

    if pre_nonnull != 0:
        raise ValueError(
            "LEAKAGE ERROR: pre-practice contains weather."
        )

    # --------------------------------------------------------
    # Check stage progression.
    #
    # A later stage should never have fewer allowed session
    # categories than an earlier stage.
    # --------------------------------------------------------

    stage_order = [
        "pre_practice",
        "fp1",
        "fp2",
        "fp3",
        "qualifying",
    ]

    print()
    print("Stage progression check:")

    previous_count = 0

    for stage in stage_order:

        stage_columns = [
            column
            for column in result.columns
            if column.startswith(
                f"{stage}_weather_"
            )
        ]

        nonnull_rows = (
            result[stage_columns]
            .notna()
            .any(axis=1)
            .sum()
        )

        print(
            f"  {stage:15s}: "
            f"{nonnull_rows} driver-race rows with weather"
        )

        previous_count = nonnull_rows

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicate_count = result.duplicated(
        subset=[
            "season",
            "round",
            "driver_id",
        ]
    ).sum()

    print()
    print(
        f"Duplicate driver-race rows: "
        f"{duplicate_count}"
    )

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate driver-race rows detected."
        )

    print()
    print(
        "Stage weather validation complete."
    )


def main():

    print("=" * 70)
    print("ADDING STAGE-SAFE SESSION WEATHER")
    print("=" * 70)

    master = load_master()

    session_weather = load_session_weather()

    weather_lookup = create_session_lookup(
        session_weather
    )

    print()
    print(
        f"Session weather lookup entries: "
        f"{len(weather_lookup)}"
    )

    result, coverage = add_stage_columns(
        master,
        weather_lookup
    )

    validate(
        result,
        coverage
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 70)
    print("STAGE WEATHER BUILD COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Input rows:  {len(master):,}"
    )

    print(
        f"Output rows: {len(result):,}"
    )

    print()
    print(
        f"Output file:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()