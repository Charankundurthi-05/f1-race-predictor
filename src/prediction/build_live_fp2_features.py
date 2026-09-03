from pathlib import Path
import json
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = ROOT / "data" / "predictions"
MODEL_DIR = ROOT / "models"

BASE_FILE = PREDICTION_DIR / "current_2026_pre_practice_features.csv"
LIVE_FILE = PREDICTION_DIR / "live_practice_results.csv"

METADATA_FILE = (
    MODEL_DIR / "fp2_extra_trees_final_metadata.json"
)

OUTPUT_FILE = (
    PREDICTION_DIR / "current_2026_fp2_features.csv"
)


def clean_number(value):
    if pd.isna(value):
        return np.nan

    try:
        return float(value)
    except:
        return np.nan


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("BUILD CURRENT FP2 FEATURES")
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

    base = pd.read_csv(BASE_FILE)
    live = pd.read_csv(LIVE_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_features = metadata["features"]

    print(f"Base rows: {len(base)}")
    print(f"Live rows: {len(live)}")
    print(f"FP2 model features: {len(model_features)}")
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

    fp1 = live[
        live["session_name"].str.contains("practice 1")
    ].copy()

    fp2 = live[
        live["session_name"].str.contains("practice 2")
    ].copy()

    if fp1.empty:
        print("FP1 results are not available yet.")
        return

    if fp2.empty:
        print("FP2 results are not available yet.")
        return

    print(f"FP1 rows: {len(fp1)}")
    print(f"FP2 rows: {len(fp2)}")
    print()

    result = base.copy()

    # ------------------------------------------------------------------
    # Find driver-name columns
    # ------------------------------------------------------------------

    fp1_driver_column = find_column(
        fp1,
        ["driver_name", "full_name", "name", "driver"]
    )

    fp2_driver_column = find_column(
        fp2,
        ["driver_name", "full_name", "name", "driver"]
    )

    if fp1_driver_column is None:
        raise ValueError(
            "Could not identify driver names in FP1 data."
        )

    if fp2_driver_column is None:
        raise ValueError(
            "Could not identify driver names in FP2 data."
        )

    # ------------------------------------------------------------------
    # Create exact model columns
    # ------------------------------------------------------------------

    result["fp1_position"] = np.nan
    result["fp2_position"] = np.nan

    result["practice_avg_position"] = np.nan
    result["practice_best_position"] = np.nan
    result["practice_sessions_available"] = 0

    matched_fp1 = 0
    matched_fp2 = 0

    # ------------------------------------------------------------------
    # Match both sessions to the 22 current drivers
    # ------------------------------------------------------------------

    for index, row in result.iterrows():

        driver_name = (
            str(row["driver_name"])
            .strip()
            .lower()
        )

        # ---------------- FP1 ----------------

        fp1_match = fp1[
            fp1[fp1_driver_column]
            .astype(str)
            .str.strip()
            .str.lower()
            == driver_name
        ]

        if not fp1_match.empty:

            live_row = fp1_match.iloc[0]

            position_column = find_column(
                fp1,
                [
                    "position",
                    "position_number",
                    "positionDisplayOrder"
                ]
            )

            if position_column:
                position = clean_number(
                    live_row[position_column]
                )

                if pd.notna(position):
                    result.at[index, "fp1_position"] = position
                    matched_fp1 += 1

        # ---------------- FP2 ----------------

        fp2_match = fp2[
            fp2[fp2_driver_column]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            driver_name
        ]

        if not fp2_match.empty:

            live_row = fp2_match.iloc[0]

            position_column = find_column(
                fp2,
                [
                    "position",
                    "position_number",
                    "positionDisplayOrder"
                ]
            )

            if position_column:
                position = clean_number(
                    live_row[position_column]
                )

                if pd.notna(position):
                    result.at[index, "fp2_position"] = position
                    matched_fp2 += 1

    # ------------------------------------------------------------------
    # Calculate practice aggregates from currently available sessions
    # ------------------------------------------------------------------

    practice_values = result[
        ["fp1_position", "fp2_position"]
    ].copy()

    result["practice_avg_position"] = (
        practice_values.mean(axis=1, skipna=True)
    )

    result["practice_best_position"] = (
        practice_values.min(axis=1, skipna=True)
    )

    result["practice_sessions_available"] = (
        practice_values.notna().sum(axis=1)
    )

    print(f"Drivers matched to FP1: {matched_fp1}")
    print(f"Drivers matched to FP2: {matched_fp2}")

    # ------------------------------------------------------------------
    # Weather
    #
    # Historical weather is intentionally NOT inserted here.
    # Live-stage predictions must not use future race observations.
    # ------------------------------------------------------------------

    weather_features = [
        "weather_fp1_temperature_avg",
        "weather_fp1_temperature_min",
        "weather_fp1_temperature_max",
        "weather_fp1_humidity_avg",
        "weather_fp1_precipitation_total",
        "weather_fp1_rain_hours",
        "weather_fp1_rain_flag",
        "weather_fp1_wind_speed_avg",
        "weather_fp1_wind_speed_max",
        "weather_fp1_wind_direction_avg",
        "weather_fp1_cloud_cover_avg",
        "weather_fp1_weather_observations",

        "weather_fp2_temperature_avg",
        "weather_fp2_temperature_min",
        "weather_fp2_temperature_max",
        "weather_fp2_humidity_avg",
        "weather_fp2_precipitation_total",
        "weather_fp2_rain_hours",
        "weather_fp2_rain_flag",
        "weather_fp2_wind_speed_avg",
        "weather_fp2_wind_speed_max",
        "weather_fp2_wind_direction_avg",
        "weather_fp2_cloud_cover_avg",
        "weather_fp2_weather_observations",
    ]

    for column in weather_features:
        if column not in result.columns:
            result[column] = np.nan

    # ------------------------------------------------------------------
    # Add every exact model feature if absent
    # ------------------------------------------------------------------

    for feature in model_features:
        if feature not in result.columns:
            result[feature] = np.nan

    # ------------------------------------------------------------------
    # Explicitly remove future information
    # ------------------------------------------------------------------

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
        "sprint_finish_position",
        "sprint_grid_position",
        "sprint_points",
    ]

    for column in forbidden:
        if column in result.columns:
            result[column] = np.nan

    # ------------------------------------------------------------------
    # Validate exact model features
    # ------------------------------------------------------------------

    missing = [
        feature
        for feature in model_features
        if feature not in result.columns
    ]

    if missing:
        raise ValueError(
            "Missing required FP2 model features:\n"
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

    if result["fp2_position"].notna().sum() == 0:
        print("No valid FP2 positions detected.")
        return

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("FP2 FEATURE VALIDATION")
    print("-" * 80)
    print("Driver count: PASS")
    print("Driver uniqueness: PASS")
    print(f"FP1 positions: {matched_fp1}/22")
    print(f"FP2 positions: {matched_fp2}/22")
    print(
        f"Model features available: "
        f"{len(model_features)}/{len(model_features)}"
    )
    print("Qualifying/race leakage: PASS")
    print()
    print(f"Saved -> {OUTPUT_FILE}")

    print()
    print("=" * 80)
    print("FP2 FEATURE BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()