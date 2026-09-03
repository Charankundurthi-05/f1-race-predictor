from pathlib import Path
import json
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = ROOT / "data" / "predictions"
MODEL_DIR = ROOT / "models"

BASE_FILE = (
    PREDICTION_DIR /
    "current_2026_pre_practice_features.csv"
)

LIVE_PRACTICE_FILE = (
    PREDICTION_DIR /
    "live_practice_results.csv"
)

LIVE_SQ_FILE = (
    PREDICTION_DIR /
    "live_sprint_qualifying_results.csv"
)

OUTPUT_FILE = (
    PREDICTION_DIR /
    "current_2026_sprint_features.csv"
)

METADATA_FILE = (
    MODEL_DIR /
    "sprint_gb_shallow_final_metadata.json"
)


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("BUILD LIVE SPRINT FEATURES")
    print("=" * 80)
    print()

    if not BASE_FILE.exists():
        print("Base pre-practice feature file not found.")
        print(
            "Run build_current_race.py first."
        )
        return

    if not LIVE_SQ_FILE.exists():
        print(
            "No live Sprint Qualifying results detected."
        )
        print(
            "Run the live Sprint Qualifying downloader "
            "after the session."
        )
        return

    if not METADATA_FILE.exists():
        print(
            "Sprint model metadata not found."
        )
        return

    base = pd.read_csv(BASE_FILE)
    sq = pd.read_csv(LIVE_SQ_FILE)

    if sq.empty:
        print(
            "Sprint Qualifying file exists "
            "but contains no results."
        )
        return

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    required_features = metadata.get(
        "features",
        []
    )

    print(
        f"Base rows: {len(base)}"
    )

    print(
        f"Sprint Qualifying rows: {len(sq)}"
    )

    # ---------------------------------------------------------------
    # Driver mapping
    # ---------------------------------------------------------------

    if "driver_name" not in sq.columns:
        print(
            "Sprint Qualifying file has no driver_name column."
        )
        return

    if "driver_name" not in base.columns:
        print(
            "Base feature file has no driver_name column."
        )
        return

    sq["driver_name"] = (
        sq["driver_name"]
        .astype(str)
        .str.strip()
    )

    base["driver_name"] = (
        base["driver_name"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------------
    # Start with current race
    # ---------------------------------------------------------------

    df = base.copy()

    # ---------------------------------------------------------------
    # Sprint Qualifying position
    # ---------------------------------------------------------------

    position_column = None

    for candidate in [
        "sprint_qualifying_position",
        "qualifying_position",
        "position"
    ]:
        if candidate in sq.columns:
            position_column = candidate
            break

    if position_column is None:
        print(
            "Could not find Sprint Qualifying position."
        )
        print(
            "Available columns:"
        )

        for column in sq.columns:
            print(
                f"  - {column}"
            )

        return

    sq_position = sq[
        ["driver_name", position_column]
    ].copy()

    sq_position = (
        sq_position
        .drop_duplicates(
            subset=["driver_name"],
            keep="last"
        )
    )

    sq_position["sprint_qualifying_position"] = pd.to_numeric(
        sq_position[position_column],
        errors="coerce"
    )

    sq_position = sq_position[
        [
            "driver_name",
            "sprint_qualifying_position"
        ]
    ]

    df = df.drop(
        columns=[
            "sprint_qualifying_position"
        ],
        errors="ignore"
    )

    df = df.merge(
        sq_position,
        on="driver_name",
        how="left"
    )

    # ---------------------------------------------------------------
    # Sprint Qualifying timing features
    # ---------------------------------------------------------------

    source_mapping = {
        "sprint_qualifying_q1_ms": [
            "sprint_qualifying_q1_ms",
            "q1_ms",
            "q1"
        ],
        "sprint_qualifying_q2_ms": [
            "sprint_qualifying_q2_ms",
            "q2_ms",
            "q2"
        ],
        "sprint_qualifying_q3_ms": [
            "sprint_qualifying_q3_ms",
            "q3_ms",
            "q3"
        ],
        "sprint_qualifying_gap_ms": [
            "sprint_qualifying_gap_ms",
            "gap_ms",
            "gap"
        ],
        "sprint_qualifying_laps": [
            "sprint_qualifying_laps",
            "laps"
        ]
    }

    for target, candidates in source_mapping.items():

        source = next(
            (
                column
                for column in candidates
                if column in sq.columns
            ),
            None
        )

        df = df.drop(
            columns=[target],
            errors="ignore"
        )

        if source is not None:

            values = sq[
                ["driver_name", source]
            ].copy()

            values[source] = pd.to_numeric(
                values[source],
                errors="coerce"
            )

            values = (
                values
                .drop_duplicates(
                    subset=["driver_name"],
                    keep="last"
                )
                .rename(
                    columns={
                        source: target
                    }
                )
            )

            df = df.merge(
                values[
                    [
                        "driver_name",
                        target
                    ]
                ],
                on="driver_name",
                how="left"
            )

        else:

            df[target] = np.nan

    # ---------------------------------------------------------------
    # Sprint qualifying availability
    # ---------------------------------------------------------------

    df["sprint_qualifying_available"] = (
        df["sprint_qualifying_position"]
        .notna()
        .astype(int)
    )

    # ---------------------------------------------------------------
    # Practice data
    # ---------------------------------------------------------------

    if LIVE_PRACTICE_FILE.exists():

        practice = pd.read_csv(
            LIVE_PRACTICE_FILE
        )

        if not practice.empty:

            practice["driver_name"] = (
                practice["driver_name"]
                .astype(str)
                .str.strip()
            )

            position_column = None

            for candidate in [
                "position",
                "position_number",
                "practice_position"
            ]:
                if candidate in practice.columns:
                    position_column = candidate
                    break

            if position_column is not None:

                practice[position_column] = pd.to_numeric(
                    practice[position_column],
                    errors="coerce"
                )

                practice_summary = (
                    practice
                    .groupby("driver_name")[
                        position_column
                    ]
                    .agg(
                        practice_avg_position="mean",
                        practice_best_position="min",
                        practice_sessions_available="count"
                    )
                    .reset_index()
                )

                df = df.drop(
                    columns=[
                        "practice_avg_position",
                        "practice_best_position",
                        "practice_sessions_available"
                    ],
                    errors="ignore"
                )

                df = df.merge(
                    practice_summary,
                    on="driver_name",
                    how="left"
                )

    # ---------------------------------------------------------------
    # Preserve sprint weekend flag
    # ---------------------------------------------------------------

    if "weekend_format" in df.columns:

        df["weekend_format"] = "sprint"

    # ---------------------------------------------------------------
    # Sprint outcome leakage protection
    # ---------------------------------------------------------------

    forbidden = [
        "sprint_finish_position",
        "sprint_grid_position",
        "sprint_positions_gained",
        "sprint_points",
        "sprint_laps",
        "sprint_pit_stops",
        "sprint_race_available"
    ]

    for column in forbidden:

        if column in df.columns:

            if column == "sprint_race_available":
                df[column] = 0
            else:
                df[column] = np.nan

    # ---------------------------------------------------------------
    # Ensure all model features exist
    # ---------------------------------------------------------------

    for feature in required_features:

        if feature not in df.columns:

            df[feature] = np.nan

    # ---------------------------------------------------------------
    # Driver validation
    # ---------------------------------------------------------------

    unique_drivers = df["driver_name"].nunique()

    print()
    print(
        f"Current drivers: {unique_drivers}"
    )

    if unique_drivers != 22:

        print(
            "DRIVER COUNT CHECK: FAIL"
        )

        print(
            "Expected 22 drivers."
        )

        return

    print(
        "Driver count check: PASS"
    )

    sq_count = (
        df["sprint_qualifying_position"]
        .notna()
        .sum()
    )

    print(
        f"Sprint Qualifying positions: "
        f"{sq_count}/22"
    )

    if sq_count != 22:

        print(
            "Sprint Qualifying completeness: WAITING"
        )

        print(
            "Feature file will not be generated "
            "until all 22 drivers are available."
        )

        return

    print(
        "Sprint Qualifying completeness: PASS"
    )

    # ---------------------------------------------------------------
    # Exact model feature validation
    # ---------------------------------------------------------------

    missing = [
        feature
        for feature in required_features
        if feature not in df.columns
    ]

    if missing:

        print(
            "MODEL FEATURE CHECK: FAIL"
        )

        for feature in missing:
            print(
                f"  - {feature}"
            )

        return

    print(
        f"Model feature check: PASS "
        f"({len(required_features)} features)"
    )

    # ---------------------------------------------------------------
    # Leakage validation
    # ---------------------------------------------------------------

    leakage_found = []

    for column in forbidden:

        if column not in df.columns:
            continue

        if column == "sprint_race_available":
            if (df[column] != 0).any():
                leakage_found.append(column)

        else:
            if df[column].notna().any():
                leakage_found.append(column)

    if leakage_found:

        print(
            "SPRINT LEAKAGE CHECK: FAIL"
        )

        for column in leakage_found:
            print(
                f"  - {column}"
            )

        return

    print(
        "Sprint race leakage check: PASS"
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"Saved -> {OUTPUT_FILE}"
    )

    print()
    print("=" * 80)
    print("LIVE SPRINT FEATURE BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()