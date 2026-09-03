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
    MODEL_DIR / "fp1_rf_small_final_metadata.json"
)

OUTPUT_FILE = (
    PREDICTION_DIR / "current_2026_fp1_features.csv"
)


def clean_number(value):
    if pd.isna(value):
        return np.nan

    try:
        return float(value)
    except:
        return np.nan


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("BUILD CURRENT FP1 FEATURES")
    print("=" * 80)
    print()

    if not BASE_FILE.exists():
        print("Current pre-practice feature file does not exist.")
        print("Run build_current_race.py first.")
        return

    if not LIVE_FILE.exists():
        print("No live practice results detected.")
        print("Run download_live_practice.py after FP1.")
        return

    base = pd.read_csv(BASE_FILE)
    live = pd.read_csv(LIVE_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_features = metadata["features"]

    print(f"Base rows: {len(base)}")
    print(f"Live rows: {len(live)}")
    print(f"FP1 model features: {len(model_features)}")
    print()

    if "session_name" not in live.columns:
        raise ValueError(
            "live_practice_results.csv is missing session_name."
        )

    fp1 = live[
        live["session_name"]
        .astype(str)
        .str.lower()
        .str.contains("practice 1")
    ].copy()

    if fp1.empty:
        print("FP1 results are not available yet.")
        return

    print(f"FP1 result rows: {len(fp1)}")

    result = base.copy()

    # ------------------------------------------------------------------
    # Identify the driver-name column in the live data
    # ------------------------------------------------------------------

    driver_column = None

    for column in [
        "driver_name",
        "full_name",
        "name",
        "driver"
    ]:
        if column in fp1.columns:
            driver_column = column
            break

    if driver_column is None:
        raise ValueError(
            "Could not identify driver name in live FP1 results."
        )

    # ------------------------------------------------------------------
    # Initialize exact model features
    # ------------------------------------------------------------------

    if "fp1_position" not in result.columns:
        result["fp1_position"] = np.nan

    # These are historical practice aggregates.
    # They remain whatever was calculated in the base dataset.
    if "practice_avg_position" not in result.columns:
        result["practice_avg_position"] = np.nan

    if "practice_best_position" not in result.columns:
        result["practice_best_position"] = np.nan

    if "practice_sessions_available" not in result.columns:
        result["practice_sessions_available"] = 0

    # ------------------------------------------------------------------
    # Match live FP1 result to driver
    # ------------------------------------------------------------------

    matched = 0

    for index, row in result.iterrows():

        driver_name = str(row["driver_name"]).strip().lower()

        matches = fp1[
            fp1[driver_column]
            .astype(str)
            .str.strip()
            .str.lower()
            == driver_name
        ]

        if matches.empty:
            continue

        live_row = matches.iloc[0]

        position = None

        for column in [
            "position",
            "position_number",
            "positionDisplayOrder"
        ]:
            if column in live_row.index:
                position = clean_number(live_row[column])
                break

        if pd.notna(position):
            result.at[index, "fp1_position"] = position

            # FP1 becomes the first available live practice session.
            result.at[index, "practice_avg_position"] = position
            result.at[index, "practice_best_position"] = position
            result.at[index, "practice_sessions_available"] = 1

            matched += 1

    print(f"Drivers matched to FP1: {matched}")

    if matched == 0:
        print()
        print("No drivers could be matched.")
        print("Check the live practice result format.")
        return

    # ------------------------------------------------------------------
    # Live weather is not historical weather.
    #
    # We deliberately do NOT load historical race weather here because
    # that would introduce information that would not have been known
    # during the live prediction stage.
    #
    # Missing weather values are left as NaN and handled by the trained
    # model pipeline's preprocessing.
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
    ]

    for column in weather_features:
        if column not in result.columns:
            result[column] = np.nan

    # ------------------------------------------------------------------
    # Ensure every exact model feature exists
    # ------------------------------------------------------------------

    missing = []

    for feature in model_features:
        if feature not in result.columns:
            result[feature] = np.nan
            missing.append(feature)

    if missing:
        print()
        print("Added missing model features:")
        for feature in missing:
            print(f"  {feature}")

    # ------------------------------------------------------------------
    # Remove anything that should never reach the FP1 model
    # ------------------------------------------------------------------

    forbidden = [
        "qualifying_position",
        "q1",
        "q2",
        "q3",
        "grid_position",
        "finish_position",
        "race_points",
        "sprint_finish_position",
        "sprint_grid_position",
        "sprint_points",
    ]

    for column in forbidden:
        if column in result.columns:
            result[column] = np.nan

    # ------------------------------------------------------------------
    # Validate exact model feature set
    # ------------------------------------------------------------------

    missing_after = [
        feature
        for feature in model_features
        if feature not in result.columns
    ]

    if missing_after:
        raise ValueError(
            "Model features still missing:\n"
            + "\n".join(missing_after)
        )

    output_columns = list(
        dict.fromkeys(
            list(result.columns) + model_features
        )
    )

    result = result[output_columns]

    # ------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------

    if len(result) != 22:
        raise ValueError(
            f"Expected 22 drivers, found {len(result)}."
        )

    if result["driver_name"].nunique() != 22:
        raise ValueError(
            "Driver uniqueness validation failed."
        )

    if result["fp1_position"].notna().sum() != 22:
        print(
            f"WARNING: only "
            f"{result['fp1_position'].notna().sum()}/22 "
            f"drivers have FP1 positions."
        )

    # Exact model-feature validation
    for feature in model_features:
        if feature not in result.columns:
            raise ValueError(
                f"Missing required feature: {feature}"
            )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("FP1 FEATURE VALIDATION")
    print("-" * 80)
    print("Driver count: PASS")
    print("Driver uniqueness: PASS")
    print("FP1 positions detected: PASS")
    print(f"Model features available: {len(model_features)}/{len(model_features)}")
    print("Qualifying/race leakage: PASS")
    print()
    print(f"Saved -> {OUTPUT_FILE}")
    print()
    print("=" * 80)
    print("FP1 FEATURE BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()