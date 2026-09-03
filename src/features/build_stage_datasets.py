import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset_with_stage_weather.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


STAGES = {
    "pre_practice": {
        "practice_sessions": [],
        "qualifying": False,
    },
    "fp1": {
        "practice_sessions": ["fp1"],
        "qualifying": False,
    },
    "fp2": {
        "practice_sessions": ["fp1", "fp2"],
        "qualifying": False,
    },
    "fp3": {
        "practice_sessions": ["fp1", "fp2", "fp3"],
        "qualifying": False,
    },
    "qualifying": {
        "practice_sessions": ["fp1", "fp2", "fp3"],
        "qualifying": True,
    },
}


PRACTICE_COLUMNS = [
    "fp1_position",
    "fp2_position",
    "fp3_position",
]


CIRCUIT_FEATURES = [
    "circuit_type",
    "circuit_direction",
    "circuit_latitude",
    "circuit_longitude",
    "circuit_length_km",
    "circuit_turns",
    "circuit_total_races_held",
]


REGULATION_FEATURES = [
    "regulation_era",
    "regulation_era_code",
]


HISTORICAL_FEATURES = [
    "driver_previous_finish",
    "driver_avg_finish_last_3",
    "driver_avg_finish_last_5",
    "driver_previous_points",
    "driver_avg_points_last_5",
    "team_avg_finish_last_5",
    "team_avg_points_last_5",
    "driver_circuit_avg_finish",
    "driver_circuit_races",
]


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


SPRINT_QUALIFYING_COLUMNS = [
    "sprint_qualifying_position",
    "sprint_qualifying_q1_ms",
    "sprint_qualifying_q2_ms",
    "sprint_qualifying_q3_ms",
    "sprint_qualifying_gap_ms",
    "sprint_qualifying_laps",
]


SPRINT_RACE_COLUMNS = [
    "sprint_finish_position",
    "sprint_grid_position",
    "sprint_positions_gained",
    "sprint_points",
    "sprint_laps",
]


def is_sprint_weekend(df):
    if "weekend_format" not in df.columns:
        return pd.Series(False, index=df.index)

    return (
        df["weekend_format"]
        .astype(str)
        .str.lower()
        .eq("sprint")
    )


def build_historical_features(df):
    df = df.sort_values(
        ["season", "race_date", "round", "driver_name"]
    ).copy()

    # ------------------------------------------------------------
    # Driver historical features
    #
    # Every feature is shifted by one race, so the current race
    # result can never influence the prediction for that race.
    # ------------------------------------------------------------

    driver_group = df.groupby(
        "driver_name",
        group_keys=False,
    )

    df["driver_previous_finish"] = (
        driver_group["finish_position"]
        .shift(1)
    )

    df["driver_avg_finish_last_3"] = (
        driver_group["finish_position"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    window=3,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    df["driver_avg_finish_last_5"] = (
        driver_group["finish_position"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    window=5,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    df["driver_previous_points"] = (
        driver_group["points"]
        .shift(1)
    )

    df["driver_avg_points_last_5"] = (
        driver_group["points"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    window=5,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    # ------------------------------------------------------------
    # Team historical features
    #
    # IMPORTANT:
    # First aggregate each team's completed race.
    # Then shift the race-level team history.
    #
    # This prevents one driver in the current race from affecting
    # another driver's team-history feature.
    # ------------------------------------------------------------

    race_team = (
        df.groupby(
            [
                "season",
                "race_date",
                "round",
                "team_name",
            ],
            as_index=False,
        )
        .agg(
            team_race_avg_finish=(
                "finish_position",
                "mean",
            ),
            team_race_total_points=(
                "points",
                "sum",
            ),
        )
        .sort_values(
            [
                "season",
                "race_date",
                "round",
                "team_name",
            ]
        )
    )

    team_group = race_team.groupby(
        "team_name",
        group_keys=False,
    )

    race_team["team_avg_finish_last_5"] = (
        team_group["team_race_avg_finish"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    window=5,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    race_team["team_avg_points_last_5"] = (
        team_group["team_race_total_points"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    window=5,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    team_history = race_team[
        [
            "season",
            "race_date",
            "round",
            "team_name",
            "team_avg_finish_last_5",
            "team_avg_points_last_5",
        ]
    ]

    df = df.merge(
        team_history,
        on=[
            "season",
            "race_date",
            "round",
            "team_name",
        ],
        how="left",
        validate="many_to_one",
    )

    # ------------------------------------------------------------
    # Driver-at-circuit historical features
    # ------------------------------------------------------------

    circuit_group = df.groupby(
        ["driver_name", "circuit_id"],
        group_keys=False,
    )

    df["driver_circuit_avg_finish"] = (
        circuit_group["finish_position"]
        .transform(
            lambda x: (
                x.shift(1)
                .expanding(
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    df["driver_circuit_races"] = (
        circuit_group.cumcount()
    )

    return df


def add_practice_features(df, sessions):
    df = df.copy()

    for column in PRACTICE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    if not sessions:
        df["fp1_position"] = pd.NA
        df["fp2_position"] = pd.NA
        df["fp3_position"] = pd.NA
        df["practice_avg_position"] = pd.NA
        df["practice_best_position"] = pd.NA
        df["practice_sessions_available"] = pd.NA
        df["practice_available"] = pd.NA
        return df

    selected_columns = [
        f"{session}_position"
        for session in sessions
    ]

    numeric_practice = (
        df[selected_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    df["practice_avg_position"] = (
        numeric_practice.mean(
            axis=1,
            skipna=True,
        )
    )

    df["practice_best_position"] = (
        numeric_practice.min(
            axis=1,
            skipna=True,
        )
    )

    df["practice_sessions_available"] = (
        numeric_practice.notna().sum(axis=1)
    )

    df["practice_available"] = (
        df["practice_sessions_available"] > 0
    )

    unavailable_sessions = [
        session
        for session in ["fp1", "fp2", "fp3"]
        if session not in sessions
    ]

    for session in unavailable_sessions:
        df[f"{session}_position"] = pd.NA

    return df


def add_qualifying_features(df, qualifying_stage):
    df = df.copy()

    if not qualifying_stage:
        df["qualifying_position"] = pd.NA
        df["q1"] = pd.NA
        df["q2"] = pd.NA
        df["q3"] = pd.NA
        df["qualifying_available"] = pd.NA
        df["grid_position"] = pd.NA
    else:
        df["qualifying_available"] = (
            df["qualifying_position"].notna()
        )

    return df


def add_sprint_features(df, stage_name):
    df = df.copy()

    sprint_mask = is_sprint_weekend(df)

    for column in SPRINT_QUALIFYING_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    for column in SPRINT_RACE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    # Sprint information is intentionally unavailable for all
    # stages before qualifying in our current five-stage model.

    if stage_name != "qualifying":
        for column in (
            SPRINT_QUALIFYING_COLUMNS
            + SPRINT_RACE_COLUMNS
        ):
            df[column] = pd.NA

    # Normal weekends can never have sprint information.

    for column in (
        SPRINT_QUALIFYING_COLUMNS
        + SPRINT_RACE_COLUMNS
    ):
        df.loc[~sprint_mask, column] = pd.NA

    df["sprint_qualifying_available"] = (
        sprint_mask
        & df["sprint_qualifying_position"].notna()
    )

    df["sprint_race_available"] = (
        sprint_mask
        & df["sprint_finish_position"].notna()
        if stage_name == "qualifying"
        else False
    )

    return df


def add_stage_weather_features(df, stage_name):
    df = df.copy()

    stages = [
        "pre_practice",
        "fp1",
        "fp2",
        "fp3",
        "qualifying",
    ]

    current_index = stages.index(stage_name)

    for source_stage in stages:
        for metric in WEATHER_METRICS:

            source_column = (
                f"{source_stage}_weather_{metric}"
            )

            if source_column not in df.columns:
                raise ValueError(
                    f"Missing stage weather column: "
                    f"{source_column}"
                )

            output_column = (
                f"weather_{source_stage}_{metric}"
            )

            source_index = stages.index(
                source_stage
            )

            if source_index <= current_index:
                df[output_column] = (
                    df[source_column]
                )
            else:
                df[output_column] = pd.NA

    return df


def remove_source_weather_columns(df):
    df = df.copy()

    stages = [
        "pre_practice",
        "fp1",
        "fp2",
        "fp3",
        "qualifying",
    ]

    source_columns = [
        f"{stage}_weather_{metric}"
        for stage in stages
        for metric in WEATHER_METRICS
    ]

    existing = [
        column
        for column in source_columns
        if column in df.columns
    ]

    if existing:
        df = df.drop(
            columns=existing
        )

    return df


def build_stage(df, stage_name, configuration):
    stage_df = df.copy()

    stage_df = add_practice_features(
        stage_df,
        configuration["practice_sessions"],
    )

    stage_df = add_qualifying_features(
        stage_df,
        configuration["qualifying"],
    )

    stage_df = add_sprint_features(
        stage_df,
        stage_name,
    )

    stage_df = add_stage_weather_features(
        stage_df,
        stage_name,
    )

    stage_df = remove_source_weather_columns(
        stage_df
    )

    return stage_df


def validate_historical_features(df, stage_name):
    print(
        "\nHistorical leakage validation:"
    )

    # Recalculate the expected team history independently
    # from completed races only.

    race_team = (
        df.groupby(
            [
                "season",
                "race_date",
                "round",
                "team_name",
            ],
            as_index=False,
        )
        .agg(
            team_race_avg_finish=(
                "finish_position",
                "mean",
            ),
            team_race_total_points=(
                "points",
                "sum",
            ),
        )
        .sort_values(
            [
                "season",
                "race_date",
                "round",
                "team_name",
            ]
        )
    )

    team_group = race_team.groupby(
        "team_name",
        group_keys=False,
    )

    expected_finish = (
        team_group["team_race_avg_finish"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    5,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    expected_points = (
        team_group["team_race_total_points"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    5,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    expected = race_team[
        [
            "season",
            "race_date",
            "round",
            "team_name",
        ]
    ].copy()

    expected[
        "expected_team_avg_finish_last_5"
    ] = expected_finish

    expected[
        "expected_team_avg_points_last_5"
    ] = expected_points

    comparison = df.merge(
        expected,
        on=[
            "season",
            "race_date",
            "round",
            "team_name",
        ],
        how="left",
        validate="many_to_one",
    )

    finish_match = np_allclose(
        comparison["team_avg_finish_last_5"],
        comparison["expected_team_avg_finish_last_5"],
    )

    points_match = np_allclose(
        comparison["team_avg_points_last_5"],
        comparison["expected_team_avg_points_last_5"],
    )

    if not finish_match or not points_match:
        raise ValueError(
            f"HISTORICAL LEAKAGE ERROR: "
            f"{stage_name} team historical features "
            f"do not match completed-race-only calculation."
        )

    print(
        "Team historical leakage check PASS."
    )


def np_allclose(left, right):
    left_values = pd.to_numeric(
        left,
        errors="coerce",
    ).to_numpy(dtype=float)

    right_values = pd.to_numeric(
        right,
        errors="coerce",
    ).to_numpy(dtype=float)

    both_nan = (
        pd.isna(left_values)
        & pd.isna(right_values)
    )

    equal = both_nan.copy()

    valid = ~both_nan

    equal[valid] = (
        abs(
            left_values[valid]
            - right_values[valid]
        )
        < 1e-10
    )

    return bool(equal.all())


def validate_weather_stage(df, stage_name):
    print(
        "\nWeather feature availability:"
    )

    for column in df.columns:

        if not column.startswith("weather_"):
            continue

        available = df[column].notna().sum()

        if available > 0:
            print(
                f"{column:55s} "
                f"{available:5d} / {len(df):5d}"
            )

    stage_order = [
        "pre_practice",
        "fp1",
        "fp2",
        "fp3",
        "qualifying",
    ]

    current_index = stage_order.index(
        stage_name
    )

    for future_stage in stage_order[
        current_index + 1:
    ]:

        future_columns = [
            column
            for column in df.columns
            if column.startswith(
                f"weather_{future_stage}_"
            )
        ]

        future_values = (
            df[future_columns]
            .notna()
            .sum()
            .sum()
        )

        if future_values != 0:
            raise ValueError(
                f"WEATHER LEAKAGE ERROR: "
                f"{stage_name} contains "
                f"{future_stage} weather."
            )

    print(
        f"Weather leakage check PASS: "
        f"{stage_name}"
    )


def validate_sprint_stage(df, stage_name):
    print(
        "\nSprint feature availability:"
    )

    sprint_mask = is_sprint_weekend(df)

    for column in (
        SPRINT_QUALIFYING_COLUMNS
        + SPRINT_RACE_COLUMNS
    ):

        available = df[column].notna().sum()

        print(
            f"{column:35s} "
            f"{available:5d} / {len(df):5d}"
        )

    normal_values = (
        df.loc[
            ~sprint_mask,
            SPRINT_QUALIFYING_COLUMNS
            + SPRINT_RACE_COLUMNS,
        ]
        .notna()
        .sum()
        .sum()
    )

    if normal_values != 0:
        raise ValueError(
            "SPRINT LEAKAGE ERROR: "
            "normal weekend contains sprint data."
        )

    if stage_name != "qualifying":

        early_values = (
            df[
                SPRINT_QUALIFYING_COLUMNS
                + SPRINT_RACE_COLUMNS
            ]
            .notna()
            .sum()
            .sum()
        )

        if early_values != 0:
            raise ValueError(
                f"SPRINT LEAKAGE ERROR: "
                f"{stage_name} contains future sprint data."
            )

    print(
        f"Sprint leakage check PASS: "
        f"{stage_name}"
    )


def validate_stage(df, stage_name):
    print(
        f"\n{'-' * 80}"
    )

    print(
        f"VALIDATING {stage_name.upper()}"
    )

    print(
        f"{'-' * 80}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        "\nCircuit feature availability:"
    )

    for column in CIRCUIT_FEATURES:

        available = df[column].notna().sum()

        print(
            f"{column:30s} "
            f"{available:5d} / {len(df):5d}"
        )

    print(
        "\nRegulation feature availability:"
    )

    for column in REGULATION_FEATURES:

        available = df[column].notna().sum()

        print(
            f"{column:30s} "
            f"{available:5d} / {len(df):5d}"
        )

    print(
        "\nPractice feature availability:"
    )

    for column in [
        "fp1_position",
        "fp2_position",
        "fp3_position",
        "practice_avg_position",
        "practice_best_position",
        "practice_sessions_available",
    ]:

        available = df[column].notna().sum()

        print(
            f"{column:30s} "
            f"{available:5d} / {len(df):5d}"
        )

    print(
        "\nQualifying feature availability:"
    )

    for column in [
        "qualifying_position",
        "q1",
        "q2",
        "q3",
        "qualifying_available",
        "grid_position",
    ]:

        available = df[column].notna().sum()

        print(
            f"{column:30s} "
            f"{available:5d} / {len(df):5d}"
        )

    print(
        "\nHistorical feature availability:"
    )

    for column in HISTORICAL_FEATURES:

        available = df[column].notna().sum()

        print(
            f"{column:30s} "
            f"{available:5d} / {len(df):5d}"
        )

    validate_historical_features(
        df,
        stage_name,
    )

    validate_sprint_stage(
        df,
        stage_name,
    )

    validate_weather_stage(
        df,
        stage_name,
    )

    duplicate_count = df.duplicated(
        subset=[
            "season",
            "round",
            "driver_name",
        ]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{duplicate_count} duplicate "
            "race-driver rows found."
        )

    print(
        "No duplicate race-driver rows."
    )

    missing_circuit = df[
        CIRCUIT_FEATURES
    ].isna().any(axis=1).sum()

    if missing_circuit > 0:
        raise ValueError(
            f"{missing_circuit} rows have "
            "missing circuit features."
        )

    print(
        "All rows have circuit features."
    )

    missing_regulation = df[
        REGULATION_FEATURES
    ].isna().any(axis=1).sum()

    if missing_regulation > 0:
        raise ValueError(
            f"{missing_regulation} rows have "
            "missing regulation features."
        )

    print(
        "All rows have regulation features."
    )

    raw_weather_columns = [
        column
        for column in df.columns
        if any(
            column.startswith(
                f"{stage}_weather_"
            )
            for stage in [
                "pre_practice",
                "fp1",
                "fp2",
                "fp3",
                "qualifying",
            ]
        )
    ]

    if raw_weather_columns:
        raise ValueError(
            "Raw source weather columns remain."
        )

    print(
        "Raw source weather columns removed."
    )

    print(
        f"\n{stage_name.upper()} validation complete."
    )


def main():
    print(
        "=" * 80
    )

    print(
        "BUILDING LEAKAGE-SAFE STAGE DATASETS"
    )

    print(
        "CIRCUIT + REGULATION + WEATHER + SPRINT"
    )

    print(
        "=" * 80
    )

    print(
        f"\nReading: {INPUT_PATH}"
    )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Input shape: {df.shape}"
    )

    required_columns = [
        "season",
        "round",
        "race_date",
        "driver_name",
        "team_name",
        "circuit_id",
        "finish_position",
        "points",
        "weekend_format",
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Missing required columns: "
            f"{missing_required}"
        )

    for column in CIRCUIT_FEATURES:
        if column not in df.columns:
            raise ValueError(
                f"Missing circuit feature: {column}"
            )

    for column in REGULATION_FEATURES:
        if column not in df.columns:
            raise ValueError(
                f"Missing regulation feature: {column}"
            )

    for stage in [
        "pre_practice",
        "fp1",
        "fp2",
        "fp3",
        "qualifying",
    ]:
        for metric in WEATHER_METRICS:
            column = (
                f"{stage}_weather_{metric}"
            )

            if column not in df.columns:
                raise ValueError(
                    f"Missing stage weather column: "
                    f"{column}"
                )

    for column in (
        SPRINT_QUALIFYING_COLUMNS
        + SPRINT_RACE_COLUMNS
    ):
        if column not in df.columns:
            raise ValueError(
                f"Missing sprint column: {column}"
            )

    print(
        "\nWeekend format distribution:"
    )

    print(
        df["weekend_format"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        "\nBuilding leakage-safe historical features..."
    )

    df = build_historical_features(
        df
    )

    print(
        "Historical features created."
    )

    for stage_name, configuration in STAGES.items():

        print(
            f"\nBuilding {stage_name} stage..."
        )

        stage_df = build_stage(
            df,
            stage_name,
            configuration,
        )

        output_path = (
            OUTPUT_DIR
            / f"stage_{stage_name}.csv"
        )

        stage_df.to_csv(
            output_path,
            index=False,
        )

        validate_stage(
            stage_df,
            stage_name,
        )

        print(
            f"\nSaved: {output_path}"
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "ALL STAGE DATASETS BUILT SUCCESSFULLY"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()