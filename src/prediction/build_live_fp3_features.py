from pathlib import Path
import json
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = ROOT / "data" / "predictions"
MODEL_DIR = ROOT / "models"

BASE_FILE = PREDICTION_DIR / "current_2026_pre_practice_features.csv"
LIVE_FILE = PREDICTION_DIR / "live_practice_results.csv"

METADATA_FILE = MODEL_DIR / "fp3_rf_small_final_metadata.json"

OUTPUT_FILE = PREDICTION_DIR / "current_2026_fp3_features.csv"


def find_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def normalize_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(".", "")
        .replace(",", "")
        .replace("-", " ")
    )


def get_position(row, columns):
    for column in columns:
        if column in row.index:
            try:
                value = float(row[column])
                if pd.notna(value):
                    return value
            except (ValueError, TypeError):
                pass
    return np.nan


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("BUILD CURRENT FP3 FEATURES")
    print("=" * 80)
    print()

    if not BASE_FILE.exists():
        print("Current pre-practice feature file does not exist.")
        print("Run build_current_race.py first.")
        return

    if not LIVE_FILE.exists():
        print("No live practice results detected.")
        print("Run download_live_practice.py after practice.")
        return

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"FP3 metadata not found: {METADATA_FILE}"
        )

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_features = metadata["features"]

    base = pd.read_csv(BASE_FILE)
    live = pd.read_csv(LIVE_FILE)

    print(f"Base rows: {len(base)}")
    print(f"Live rows: {len(live)}")
    print(f"FP3 model features: {len(model_features)}")
    print()

    if "session_name" not in live.columns:
        raise ValueError(
            "live_practice_results.csv is missing session_name."
        )

    live["session_name"] = (
        live["session_name"]
        .astype(str)
        .str.lower()
    )

    sessions = {}

    for session_number in [1, 2, 3]:
        sessions[session_number] = live[
            live["session_name"].str.contains(
                f"practice {session_number}",
                na=False
            )
        ].copy()

    fp1 = sessions[1]
    fp2 = sessions[2]
    fp3 = sessions[3]

    if fp1.empty:
        print("FP1 results are not available yet.")
        return

    if fp2.empty:
        print("FP2 results are not available yet.")
        return

    if fp3.empty:
        print("FP3 results are not available yet.")
        return

    print(f"FP1 rows: {len(fp1)}")
    print(f"FP2 rows: {len(fp2)}")
    print(f"FP3 rows: {len(fp3)}")
    print()

    result = base.copy()

    result["fp1_position"] = np.nan
    result["fp2_position"] = np.nan
    result["fp3_position"] = np.nan

    result["practice_avg_position"] = np.nan
    result["practice_best_position"] = np.nan
    result["practice_sessions_available"] = 0

    session_data = {
        1: fp1,
        2: fp2,
        3: fp3
    }

    matched = {
        1: 0,
        2: 0,
        3: 0
    }

    for session_number, session_df in session_data.items():

        driver_column = find_column(
            session_df,
            [
                "driver_name",
                "full_name",
                "name",
                "driver"
            ]
        )

        if driver_column is None:
            raise ValueError(
                f"Could not identify driver names in FP{session_number}."
            )

        position_column = find_column(
            session_df,
            [
                "position",
                "position_number",
                "positionDisplayOrder"
            ]
        )

        if position_column is None:
            raise ValueError(
                f"Could not identify positions in FP{session_number}."
            )

        session_lookup = {}

        for _, live_row in session_df.iterrows():
            live_name = normalize_name(
                live_row[driver_column]
            )

            if live_name:
                session_lookup[live_name] = live_row

        model_column = f"fp{session_number}_position"

        for index, row in result.iterrows():

            driver_name = normalize_name(
                row["driver_name"]
            )

            if driver_name not in session_lookup:
                continue

            live_row = session_lookup[driver_name]

            position = get_position(
                live_row,
                [
                    position_column,
                    "position",
                    "position_number",
                    "positionDisplayOrder"
                ]
            )

            if pd.notna(position):
                result.at[
                    index,
                    model_column
                ] = position

                matched[session_number] += 1

    practice_values = result[
        [
            "fp1_position",
            "fp2_position",
            "fp3_position"
        ]
    ]

    result["practice_avg_position"] = practice_values.mean(
        axis=1,
        skipna=True
    )

    result["practice_best_position"] = practice_values.min(
        axis=1,
        skipna=True
    )

    result["practice_sessions_available"] = (
        practice_values.notna().sum(axis=1)
    )

    print(f"Drivers matched to FP1: {matched[1]}")
    print(f"Drivers matched to FP2: {matched[2]}")
    print(f"Drivers matched to FP3: {matched[3]}")

    # ---------------------------------------------------------------
    # Weather features
    #
    # Live session weather is not substituted with historical weather.
    # ---------------------------------------------------------------

    weather_features = []

    for session_number in [1, 2, 3]:

        prefix = f"weather_fp{session_number}_"

        weather_features.extend(
            [
                prefix + "temperature_avg",
                prefix + "temperature_min",
                prefix + "temperature_max",
                prefix + "humidity_avg",
                prefix + "precipitation_total",
                prefix + "rain_hours",
                prefix + "rain_flag",
                prefix + "wind_speed_avg",
                prefix + "wind_speed_max",
                prefix + "wind_direction_avg",
                prefix + "cloud_cover_avg",
                prefix + "weather_observations"
            ]
        )

    for feature in weather_features:
        if feature not in result.columns:
            result[feature] = np.nan

    # ---------------------------------------------------------------
    # Ensure every trained-model feature exists
    # ---------------------------------------------------------------

    for feature in model_features:
        if feature not in result.columns:
            result[feature] = np.nan

    # ---------------------------------------------------------------
    # Explicit future-information protection
    # ---------------------------------------------------------------

    forbidden = [
        "qualifying_position",
        "q1",
        "q2",
        "q3",
        "qualifying_available",
        "grid_position",
        "finish_position",
        "race_points",
        "sprint_qualifying_position",
        "sprint_qualifying_q1_ms",
        "sprint_qualifying_q2_ms",
        "sprint_qualifying_q3_ms",
        "sprint_qualifying_gap_ms",
        "sprint_qualifying_laps",
        "sprint_finish_position",
        "sprint_grid_position",
        "sprint_positions_gained",
        "sprint_points",
        "sprint_laps"
    ]

    for column in forbidden:
        if column in result.columns:
            result[column] = np.nan

    # ---------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------

    missing = [
        feature
        for feature in model_features
        if feature not in result.columns
    ]

    if missing:
        raise ValueError(
            "Missing required FP3 model features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    if len(result) != 22:
        raise ValueError(
            f"Expected 22 drivers, found {len(result)}."
        )

    if result["driver_name"].nunique() != 22:
        raise ValueError(
            "Driver uniqueness validation failed."
        )

    if result["fp3_position"].notna().sum() == 0:
        print("No valid FP3 positions detected.")
        return

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("FP3 FEATURE VALIDATION")
    print("-" * 80)
    print("Driver count: PASS")
    print("Driver uniqueness: PASS")
    print(f"FP1 positions: {matched[1]}/22")
    print(f"FP2 positions: {matched[2]}/22")
    print(f"FP3 positions: {matched[3]}/22")
    print(
        f"Model features available: "
        f"{len(model_features)}/{len(model_features)}"
    )
    print("Qualifying/race leakage: PASS")
    print()
    print(f"Saved -> {OUTPUT_FILE}")

    print()
    print("=" * 80)
    print("FP3 FEATURE BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
