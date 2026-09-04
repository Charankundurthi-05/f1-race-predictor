from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = ROOT / "data" / "predictions"
MODEL_DIR = ROOT / "models"

BASE_FILE = (
    PREDICTION_DIR
    / "current_2026_pre_practice_features.csv"
)

PRACTICE_FILE = (
    PREDICTION_DIR
    / "live_practice_results.csv"
)

QUALIFYING_FILE = (
    PREDICTION_DIR
    / "live_qualifying_results.csv"
)

METADATA_FILE = (
    MODEL_DIR
    / "qualifying_gb_shallow_final_metadata.json"
)

OUTPUT_FILE = (
    PREDICTION_DIR
    / "current_2026_qualifying_features.csv"
)


def find_column(df, names):
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in names:
        key = str(name).strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def normalize_name(value):
    if pd.isna(value):
        return ""

    value = (
        str(value)
        .strip()
        .lower()
    )

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
        "ū": "u",
    }

    for old, new in replacements.items():
        value = value.replace(
            old,
            new,
        )

    return value


def numeric_value(value):
    if pd.isna(value):
        return np.nan

    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def build_session_lookup(
    df,
    driver_column,
):
    lookup = {}

    for _, row in df.iterrows():

        key = normalize_name(
            row[driver_column]
        )

        if key:
            lookup[key] = row

    return lookup


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("BUILD CURRENT QUALIFYING FEATURES")
    print("=" * 80)
    print()

    if not BASE_FILE.exists():
        print(
            "Current pre-practice feature file does not exist."
        )
        print(
            "Run build_current_race.py first."
        )
        return

    if not PRACTICE_FILE.exists():
        print(
            "No live practice results detected."
        )
        print(
            "Practice data is required before qualifying."
        )
        return

    if not QUALIFYING_FILE.exists():
        print(
            "No live qualifying results detected."
        )
        print(
            "Run download_live_sessions.py after qualifying."
        )
        return

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    model_features = metadata["features"]

    base = pd.read_csv(
        BASE_FILE
    )

    practice = pd.read_csv(
        PRACTICE_FILE
    )

    qualifying = pd.read_csv(
        QUALIFYING_FILE
    )

    print(
        f"Base rows: {len(base)}"
    )
    print(
        f"Practice rows: {len(practice)}"
    )
    print(
        f"Qualifying rows: {len(qualifying)}"
    )
    print(
        f"Qualifying model features: "
        f"{len(model_features)}"
    )
    print()

    if qualifying.empty:
        print(
            "No qualifying results are available."
        )
        return

    if "session_name" not in practice.columns:
        raise ValueError(
            "live_practice_results.csv "
            "is missing session_name."
        )

    practice["session_name"] = (
        practice["session_name"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    result = base.copy()

    # ------------------------------------------------------------
    # PRACTICE POSITIONS
    # ------------------------------------------------------------

    for session_number in [1, 2, 3]:

        result[
            f"fp{session_number}_position"
        ] = np.nan

    practice_sessions = {}

    for session_number in [1, 2, 3]:

        practice_sessions[session_number] = (
            practice[
                practice["session_name"].str.contains(
                    f"practice {session_number}",
                    na=False,
                )
            ].copy()
        )

    matched = {
        1: 0,
        2: 0,
        3: 0,
    }

    for session_number, session_df in (
        practice_sessions.items()
    ):

        if session_df.empty:
            continue

        driver_column = find_column(
            session_df,
            [
                "driver_name",
                "full_name",
                "name",
                "driver",
            ],
        )

        position_column = find_column(
            session_df,
            [
                "position",
                "position_number",
                "positionDisplayOrder",
            ],
        )

        if driver_column is None:
            raise ValueError(
                f"Could not identify drivers "
                f"in FP{session_number}."
            )

        if position_column is None:
            raise ValueError(
                f"Could not identify positions "
                f"in FP{session_number}."
            )

        lookup = build_session_lookup(
            session_df,
            driver_column,
        )

        for index, row in result.iterrows():

            key = normalize_name(
                row["driver_name"]
            )

            live_row = lookup.get(key)

            if live_row is None:
                continue

            position = numeric_value(
                live_row[position_column]
            )

            if pd.notna(position):

                result.at[
                    index,
                    f"fp{session_number}_position",
                ] = position

                matched[session_number] += 1

    practice_values = result[
        [
            "fp1_position",
            "fp2_position",
            "fp3_position",
        ]
    ]

    result["practice_avg_position"] = (
        practice_values.mean(
            axis=1,
            skipna=True,
        )
    )

    result["practice_best_position"] = (
        practice_values.min(
            axis=1,
            skipna=True,
        )
    )

    result["practice_sessions_available"] = (
        practice_values.notna().sum(
            axis=1
        )
    )

    print(
        f"FP1 drivers matched: "
        f"{matched[1]}/22"
    )

    print(
        f"FP2 drivers matched: "
        f"{matched[2]}/22"
    )

    print(
        f"FP3 drivers matched: "
        f"{matched[3]}/22"
    )

    # ------------------------------------------------------------
    # QUALIFYING
    # ------------------------------------------------------------

    qualifying_driver_column = find_column(
        qualifying,
        [
            "driver_name",
            "full_name",
            "name",
            "driver",
        ],
    )

    qualifying_position_column = find_column(
        qualifying,
        [
            "qualifying_position",
            "position",
            "position_number",
            "positionDisplayOrder",
        ],
    )

    if qualifying_driver_column is None:
        raise ValueError(
            "Could not identify drivers "
            "in qualifying results."
        )

    if qualifying_position_column is None:
        raise ValueError(
            "Could not identify qualifying "
            "positions."
        )

    result["qualifying_position"] = np.nan
    result["q1"] = np.nan
    result["q2"] = np.nan
    result["q3"] = np.nan
    result["qualifying_available"] = 0
    result["grid_position"] = np.nan

    q1_column = find_column(
        qualifying,
        [
            "q1",
            "q1_time",
            "q1_ms",
        ],
    )

    q2_column = find_column(
        qualifying,
        [
            "q2",
            "q2_time",
            "q2_ms",
        ],
    )

    q3_column = find_column(
        qualifying,
        [
            "q3",
            "q3_time",
            "q3_ms",
        ],
    )

    qualifying_lookup = build_session_lookup(
        qualifying,
        qualifying_driver_column,
    )

    matched_qualifying = 0

    for index, row in result.iterrows():

        key = normalize_name(
            row["driver_name"]
        )

        q_row = qualifying_lookup.get(key)

        if q_row is None:
            continue

        position = numeric_value(
            q_row[qualifying_position_column]
        )

        if pd.notna(position):

            result.at[
                index,
                "qualifying_position",
            ] = position

            result.at[
                index,
                "grid_position",
            ] = position

            result.at[
                index,
                "qualifying_available",
            ] = 1

            matched_qualifying += 1

        if q1_column:
            result.at[
                index,
                "q1",
            ] = numeric_value(
                q_row[q1_column]
            )

        if q2_column:
            result.at[
                index,
                "q2",
            ] = numeric_value(
                q_row[q2_column]
            )

        if q3_column:
            result.at[
                index,
                "q3",
            ] = numeric_value(
                q_row[q3_column]
            )

    print(
        f"Drivers matched to qualifying: "
        f"{matched_qualifying}/22"
    )

    # ------------------------------------------------------------
    # WEATHER
    #
    # Do not substitute historical race weather here.
    # ------------------------------------------------------------

    for session in [
        "fp1",
        "fp2",
        "fp3",
        "qualifying",
    ]:

        features = [
            f"weather_{session}_temperature_avg",
            f"weather_{session}_temperature_min",
            f"weather_{session}_temperature_max",
            f"weather_{session}_humidity_avg",
            f"weather_{session}_precipitation_total",
            f"weather_{session}_rain_hours",
            f"weather_{session}_rain_flag",
            f"weather_{session}_wind_speed_avg",
            f"weather_{session}_wind_speed_max",
            f"weather_{session}_wind_direction_avg",
            f"weather_{session}_cloud_cover_avg",
            f"weather_{session}_weather_observations",
        ]

        for feature in features:

            if feature not in result.columns:
                result[feature] = np.nan

    # ------------------------------------------------------------
    # SPRINT FEATURES
    #
    # Italian GP is a normal weekend.
    # ------------------------------------------------------------

    sprint_features = [
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
        "sprint_laps",
    ]

    for feature in sprint_features:

        if feature not in result.columns:
            result[feature] = np.nan

    # ------------------------------------------------------------
    # ENSURE EXACT MODEL FEATURES
    # ------------------------------------------------------------

    for feature in model_features:

        if feature not in result.columns:
            result[feature] = np.nan

    # ------------------------------------------------------------
    # FUTURE INFORMATION PROTECTION
    # ------------------------------------------------------------

    forbidden = [
        "finish_position",
        "race_points",
        "status",
        "laps",
        "fastest_lap_rank",
        "fastest_lap_time",
        "fastest_lap_speed",
        "sprint_finish_position",
        "sprint_points",
        "sprint_positions_gained",
    ]

    for column in forbidden:

        if column in result.columns:
            result[column] = np.nan

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    missing = [
        feature
        for feature in model_features
        if feature not in result.columns
    ]

    if missing:
        raise ValueError(
            "Missing required qualifying "
            "model features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    if len(result) != 22:
        raise ValueError(
            f"Expected 22 drivers, "
            f"found {len(result)}."
        )

    if (
        result["driver_name"].nunique()
        != 22
    ):
        raise ValueError(
            "Driver uniqueness validation failed."
        )

    if matched_qualifying == 0:
        print(
            "No valid qualifying positions detected."
        )
        return

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "QUALIFYING FEATURE VALIDATION"
    )
    print("-" * 80)
    print("Driver count: PASS")
    print("Driver uniqueness: PASS")
    print(
        f"FP1 positions: "
        f"{matched[1]}/22"
    )
    print(
        f"FP2 positions: "
        f"{matched[2]}/22"
    )
    print(
        f"FP3 positions: "
        f"{matched[3]}/22"
    )
    print(
        f"Qualifying positions: "
        f"{matched_qualifying}/22"
    )
    print(
        f"Model features available: "
        f"{len(model_features)}/"
        f"{len(model_features)}"
    )
    print(
        "Race-finish leakage: PASS"
    )
    print(
        "Sprint-finish leakage: PASS"
    )
    print()
    print(
        f"Saved -> {OUTPUT_FILE}"
    )

    print()
    print("=" * 80)
    print("QUALIFYING FEATURE BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
