from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"


STAGES = [
    "pre_practice",
    "fp1",
    "fp2",
    "fp3",
    "qualifying",
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


PRACTICE_FEATURES = [
    "fp1_position",
    "fp2_position",
    "fp3_position",
    "practice_avg_position",
    "practice_best_position",
    "practice_sessions_available",
    "practice_available",
]


QUALIFYING_FEATURES = [
    "qualifying_position",
    "q1",
    "q2",
    "q3",
    "qualifying_available",
    "grid_position",
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


TARGET_COLUMN = "finish_position"


RAW_RESULT_COLUMNS = [
    "points",
    "position_text",
    "laps",
    "status",
    "fastest_lap_rank",
    "fastest_lap_time",
    "fastest_lap_speed",
    "fastest_lap_speed_unit",
]


def check_feature_availability(df):
    print("\nFeature availability:")

    all_features = (
        HISTORICAL_FEATURES
        + CIRCUIT_FEATURES
        + PRACTICE_FEATURES
        + QUALIFYING_FEATURES
    )

    missing_features = []

    for column in all_features:

        if column not in df.columns:
            print(
                f"{column:35s} MISSING COLUMN"
            )
            missing_features.append(column)
            continue

        available = df[column].notna().sum()

        percentage = (
            available / len(df) * 100
            if len(df) > 0
            else 0
        )

        print(
            f"{column:35s} "
            f"{available:5,} / {len(df):5,} "
            f"({percentage:6.2f}%)"
        )

    return len(missing_features) == 0


def check_raw_columns(df, stage):
    print("\nRaw result columns retained for reference:")

    if TARGET_COLUMN in df.columns:
        print(
            f"{TARGET_COLUMN:35s} TARGET "
            f"({df[TARGET_COLUMN].notna().sum():,} values)"
        )

    for column in RAW_RESULT_COLUMNS:

        if column in df.columns:
            non_null = df[column].notna().sum()

            print(
                f"{column:35s} "
                f"REFERENCE DATA "
                f"({non_null:,} values)"
            )

    if "grid_position" in df.columns:

        non_null = df["grid_position"].notna().sum()

        if stage == "qualifying":
            print(
                f"{'grid_position':35s} "
                f"ALLOWED AT QUALIFYING "
                f"({non_null:,} values)"
            )
        else:
            if non_null == 0:
                print(
                    f"{'grid_position':35s} "
                    "OK: empty before qualifying"
                )
            else:
                print(
                    f"{'grid_position':35s} "
                    f"ERROR: {non_null:,} values before qualifying"
                )
                return False

    return True


def check_historical_features(df):
    print("\nHistorical feature sanity check:")

    passed = True

    for column in HISTORICAL_FEATURES:

        if column not in df.columns:
            continue

        values = df[column].dropna()

        if values.empty:
            print(
                f"{column:35s} no values"
            )
            continue

        print(
            f"{column:35s} "
            f"min={values.min():.3f} "
            f"max={values.max():.3f} "
            f"mean={values.mean():.3f}"
        )

        if column.endswith("finish"):
            if values.min() < 1:
                print(
                    f"WARNING: {column} contains values below 1"
                )
                passed = False

    return passed


def check_stage_information(df, stage):
    print("\nStage-specific information:")

    passed = True

    if stage == "pre_practice":

        for column in PRACTICE_FEATURES:

            if column in df.columns:

                if df[column].notna().sum() != 0:

                    print(
                        f"ERROR: {column} contains "
                        "practice data before practice"
                    )

                    passed = False

        for column in QUALIFYING_FEATURES:

            if column in df.columns:

                if df[column].notna().sum() != 0:

                    print(
                        f"ERROR: {column} contains "
                        "qualifying data before qualifying"
                    )

                    passed = False

        if passed:
            print(
                "Pre-practice future-session check: PASS"
            )

    elif stage == "fp1":

        for column in [
            "fp2_position",
            "fp3_position",
        ]:

            if column in df.columns:

                if df[column].notna().sum() != 0:

                    print(
                        f"ERROR: {column} contains future data"
                    )

                    passed = False

        for column in QUALIFYING_FEATURES:

            if column in df.columns:

                if df[column].notna().sum() != 0:

                    print(
                        f"ERROR: {column} contains "
                        "future qualifying data"
                    )

                    passed = False

        if passed:
            print(
                "FP1 future-session check: PASS"
            )

    elif stage == "fp2":

        if "fp3_position" in df.columns:

            if df["fp3_position"].notna().sum() != 0:

                print(
                    "ERROR: FP2 contains future FP3 data"
                )

                passed = False

        for column in QUALIFYING_FEATURES:

            if column in df.columns:

                if df[column].notna().sum() != 0:

                    print(
                        f"ERROR: {column} contains "
                        "future qualifying data"
                    )

                    passed = False

        if passed:
            print(
                "FP2 future-session check: PASS"
            )

    elif stage == "fp3":

        for column in QUALIFYING_FEATURES:

            if column in df.columns:

                if df[column].notna().sum() != 0:

                    print(
                        f"ERROR: {column} contains "
                        "future qualifying data"
                    )

                    passed = False

        if passed:
            print(
                "FP3 future-session check: PASS"
            )

    elif stage == "qualifying":

        if "qualifying_position" in df.columns:

            print(
                "Qualifying position available: "
                f"{df['qualifying_position'].notna().sum():,}"
            )

        if "grid_position" in df.columns:

            print(
                "Grid position available: "
                f"{df['grid_position'].notna().sum():,}"
            )

        print(
            "Qualifying-stage information check: PASS"
        )

    return passed


def check_duplicates(df):
    duplicate_count = df.duplicated(
        subset=[
            "season",
            "round",
            "driver_name",
        ]
    ).sum()

    if duplicate_count > 0:

        print(
            f"\nERROR: {duplicate_count} duplicate "
            "race-driver rows found."
        )

        return False

    print(
        "\nNo duplicate race-driver rows."
    )

    return True


def check_circuit_features(df):
    print("\nCircuit feature availability:")

    passed = True

    for column in CIRCUIT_FEATURES:

        if column not in df.columns:

            print(
                f"{column:35s} MISSING COLUMN"
            )

            passed = False
            continue

        available = df[column].notna().sum()

        percentage = (
            available / len(df) * 100
        )

        print(
            f"{column:35s} "
            f"{available:5,} / {len(df):5,} "
            f"({percentage:6.2f}%)"
        )

        if available != len(df):

            passed = False

    return passed


def check_file(stage):

    path = DATA_DIR / f"stage_{stage}.csv"

    print("\n" + "=" * 80)
    print(stage.upper())
    print("=" * 80)

    if not path.exists():

        print(
            f"ERROR: {path} does not exist"
        )

        return False

    df = pd.read_csv(path)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print("\nBasic information:")

    print(
        f"Seasons: "
        f"{df['season'].min()} - "
        f"{df['season'].max()}"
    )

    print(
        f"Drivers: "
        f"{df['driver_id'].nunique():,}"
    )

    print(
        f"Teams: "
        f"{df['team_id'].nunique():,}"
    )

    print(
        f"Races: "
        f"{df[['season', 'round']].drop_duplicates().shape[0]:,}"
    )

    features_pass = check_feature_availability(df)

    circuit_pass = check_circuit_features(df)

    raw_columns_pass = check_raw_columns(
        df,
        stage
    )

    historical_pass = check_historical_features(
        df
    )

    stage_pass = check_stage_information(
        df,
        stage
    )

    duplicate_pass = check_duplicates(
        df
    )

    return all(
        [
            features_pass,
            circuit_pass,
            raw_columns_pass,
            historical_pass,
            stage_pass,
            duplicate_pass,
        ]
    )


def main():

    print("=" * 80)
    print("F1 FEATURE PIPELINE AUDIT")
    print("=" * 80)

    results = {}

    for stage in STAGES:

        try:

            results[stage] = check_file(
                stage
            )

        except Exception as error:

            results[stage] = False

            print(
                f"\nERROR in {stage}:"
            )

            print(error)

    print("\n")
    print("=" * 80)
    print("FINAL AUDIT RESULT")
    print("=" * 80)

    for stage, passed in results.items():

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{stage:20s} {status}"
        )

    if all(results.values()):

        print(
            "\nALL CURRENT FEATURE PIPELINES PASSED."
        )

        print(
            "Circuit features are integrated "
            "and leakage checks passed."
        )

    else:

        print(
            "\nSOME FEATURE PIPELINES FAILED."
        )

        print(
            "Fix the reported issue before "
            "adding new features."
        )


if __name__ == "__main__":
    main()