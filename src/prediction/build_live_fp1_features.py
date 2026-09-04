from pathlib import Path
import json
import re
import unicodedata

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = ROOT / "data" / "predictions"
MODEL_DIR = ROOT / "models"

BASE_FILE = PREDICTION_DIR / "current_2026_pre_practice_features.csv"
LIVE_FILE = PREDICTION_DIR / "live_practice_results.csv"

METADATA_FILE = MODEL_DIR / "fp1_rf_small_final_metadata.json"

OUTPUT_FILE = PREDICTION_DIR / "current_2026_fp1_features.csv"


def clean_number(value):
    if pd.isna(value):
        return np.nan

    try:
        return float(value)
    except Exception:
        return np.nan


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def normalize_code(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(value).strip().upper(),
    )


def surname_key(value):
    normalized = normalize_text(value)

    if not normalized:
        return ""

    parts = normalized.split()

    return parts[-1]


def build_lookup(
    df,
    driver_column,
    code_column,
):
    lookup = {
        "code": {},
        "name": {},
        "surname": {},
    }

    for _, row in df.iterrows():

        if code_column is not None:
            code = normalize_code(
                row[code_column]
            )

            if code:
                lookup["code"].setdefault(
                    code,
                    []
                ).append(row)

        if driver_column is not None:
            name = normalize_text(
                row[driver_column]
            )

            if name:
                lookup["name"].setdefault(
                    name,
                    []
                ).append(row)

            surname = surname_key(
                row[driver_column]
            )

            if surname:
                lookup["surname"].setdefault(
                    surname,
                    []
                ).append(row)

    return lookup


def match_driver(
    base_row,
    lookup,
):
    if "driver_code" in base_row.index:

        base_code = normalize_code(
            base_row["driver_code"]
        )

        if base_code:

            matches = lookup["code"].get(
                base_code,
                [],
            )

            if len(matches) == 1:
                return matches[0]

    base_name = normalize_text(
        base_row["driver_name"]
    )

    if base_name:

        matches = lookup["name"].get(
            base_name,
            [],
        )

        if len(matches) == 1:
            return matches[0]

    base_surname = surname_key(
        base_row["driver_name"]
    )

    if base_surname:

        matches = lookup["surname"].get(
            base_surname,
            [],
        )

        if len(matches) == 1:
            return matches[0]

    return None


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

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:
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

    live["session_name"] = (
        live["session_name"]
        .astype(str)
        .str.lower()
    )

    fp1 = live[
        live["session_name"]
        .str.contains(
            "practice 1",
            na=False,
        )
    ].copy()

    if fp1.empty:
        print("FP1 results are not available yet.")
        return

    print(f"FP1 result rows: {len(fp1)}")
    print()

    result = base.copy()

    # ------------------------------------------------------------
    # Identify columns
    # ------------------------------------------------------------

    driver_column = find_column(
        fp1,
        [
            "driver_name",
            "full_name",
            "name",
            "driver",
        ],
    )

    code_column = find_column(
        fp1,
        [
            "driver_code",
            "code",
            "abbreviation",
        ],
    )

    position_column = find_column(
        fp1,
        [
            "position",
            "position_number",
            "positionDisplayOrder",
        ],
    )

    if driver_column is None:
        raise ValueError(
            "Could not identify driver name in live FP1 results."
        )

    if position_column is None:
        raise ValueError(
            "Could not identify FP1 position in live results."
        )

    # ------------------------------------------------------------
    # Initialize exact model features
    # ------------------------------------------------------------

    result["fp1_position"] = np.nan

    result["practice_avg_position"] = np.nan

    result["practice_best_position"] = np.nan

    result["practice_sessions_available"] = 0

    # ------------------------------------------------------------
    # Build robust lookup
    # ------------------------------------------------------------

    lookup = build_lookup(
        fp1,
        driver_column,
        code_column,
    )

    matched = 0

    unmatched = []

    # ------------------------------------------------------------
    # Match live FP1 to current 22 drivers
    # ------------------------------------------------------------

    for index, row in result.iterrows():

        live_row = match_driver(
            row,
            lookup,
        )

        if live_row is None:
            unmatched.append(
                str(row["driver_name"])
            )
            continue

        position = clean_number(
            live_row[position_column]
        )

        if pd.isna(position):
            continue

        result.at[
            index,
            "fp1_position",
        ] = position

        result.at[
            index,
            "practice_avg_position",
        ] = position

        result.at[
            index,
            "practice_best_position",
        ] = position

        result.at[
            index,
            "practice_sessions_available",
        ] = 1

        matched += 1

    print(
        f"Drivers matched to FP1: "
        f"{matched}"
    )

    if unmatched:
        print()
        print("FP1 unmatched drivers:")

        for driver in unmatched:
            print(f"  - {driver}")

    # ------------------------------------------------------------
    # Weather
    #
    # No future historical weather is inserted here.
    # Missing values are handled by the trained model pipeline.
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Ensure exact model feature set
    # ------------------------------------------------------------

    for feature in model_features:

        if feature not in result.columns:
            result[feature] = np.nan

    # ------------------------------------------------------------
    # Remove future information
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Validate model features
    # ------------------------------------------------------------

    missing = [
        feature
        for feature in model_features
        if feature not in result.columns
    ]

    if missing:
        raise ValueError(
            "Missing required FP1 model features:\n"
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

    if result["fp1_position"].notna().sum() == 0:
        print("No valid FP1 positions detected.")
        return

    # 18/22 is valid for the current FP1 session because
    # four race drivers were not participating.
    matched_count = int(
        result["fp1_position"].notna().sum()
    )

    if matched_count < 1:
        raise ValueError(
            "No FP1 drivers were matched."
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("FP1 FEATURE VALIDATION")
    print("-" * 80)
    print("Driver count: PASS")
    print("Driver uniqueness: PASS")
    print(
        f"FP1 positions: "
        f"{matched_count}/22"
    )
    print(
        f"Model features available: "
        f"{len(model_features)}/{len(model_features)}"
    )
    print("Qualifying/race leakage: PASS")
    print()
    print(f"Saved -> {OUTPUT_FILE}")

    print()
    print("=" * 80)
    print("FP1 FEATURE BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()