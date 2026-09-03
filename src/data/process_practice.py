from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "practice_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "practice_features.csv"


def main():
    print("=" * 70)
    print("PROCESSING PRACTICE SESSION DATA")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("\nLoading practice data...")

    practice = pd.read_csv(INPUT_PATH)

    print(f"Raw practice rows: {len(practice):,}")

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    required_columns = [
        "season",
        "round",
        "race_name",
        "date",
        "circuit_id",
        "circuit_name",
        "session_type",
        "driver_id",
        "driver_code",
        "driver_name",
        "team_id",
        "team_name",
        "position",
        "laps_completed",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in practice.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # ---------------------------------------------------------
    # Keep only FP1 / FP2 / FP3
    # ---------------------------------------------------------

    practice = practice[
        practice["session_type"].isin(
            ["FP1", "FP2", "FP3"]
        )
    ].copy()

    print(
        f"FP1/FP2/FP3 rows: {len(practice):,}"
    )

    # ---------------------------------------------------------
    # Convert numeric fields
    # ---------------------------------------------------------

    numeric_columns = [
        "season",
        "round",
        "position",
        "laps_completed",
        "fastest_lap_time_ms",
        "gap_to_leader_ms",
        "interval_to_previous_ms",
    ]

    for column in numeric_columns:
        if column in practice.columns:
            practice[column] = pd.to_numeric(
                practice[column],
                errors="coerce"
            )

    # ---------------------------------------------------------
    # Remove invalid session rows
    # ---------------------------------------------------------

    before = len(practice)

    practice = practice.dropna(
        subset=[
            "season",
            "round",
            "driver_id",
            "session_type",
            "position",
        ]
    )

    print(
        f"Rows removed because of missing core fields: "
        f"{before - len(practice):,}"
    )

    # ---------------------------------------------------------
    # Remove duplicate driver/session records
    # ---------------------------------------------------------

    duplicate_count = practice.duplicated(
        subset=[
            "season",
            "round",
            "driver_id",
            "session_type",
        ]
    ).sum()

    print(
        f"Duplicate driver/session rows: "
        f"{duplicate_count:,}"
    )

    practice = practice.drop_duplicates(
        subset=[
            "season",
            "round",
            "driver_id",
            "session_type",
        ],
        keep="first"
    )

    # ---------------------------------------------------------
    # Session feature creation
    # ---------------------------------------------------------

    print("\nCreating session-level features...")

    session_columns = [
        "season",
        "round",
        "race_name",
        "date",
        "circuit_id",
        "circuit_name",
        "driver_id",
        "driver_code",
        "driver_name",
        "team_id",
        "team_name",
        "session_type",
        "position",
        "laps_completed",
        "fastest_lap_time_ms",
        "gap_to_leader_ms",
        "interval_to_previous_ms",
    ]

    session_columns = [
        column
        for column in session_columns
        if column in practice.columns
    ]

    session_data = practice[session_columns].copy()

    # ---------------------------------------------------------
    # Pivot sessions
    # ---------------------------------------------------------

    print("Pivoting FP1 / FP2 / FP3 into driver-race rows...")

    identity_columns = [
        "season",
        "round",
        "race_name",
        "date",
        "circuit_id",
        "circuit_name",
        "driver_id",
        "driver_code",
        "driver_name",
        "team_id",
        "team_name",
    ]

    # One row per driver/race.
    identity = (
        session_data[
            identity_columns
        ]
        .drop_duplicates(
            subset=[
                "season",
                "round",
                "driver_id",
            ]
        )
    )

    feature_columns = [
        "position",
        "laps_completed",
        "fastest_lap_time_ms",
        "gap_to_leader_ms",
        "interval_to_previous_ms",
    ]

    feature_frames = []

    for session in ["FP1", "FP2", "FP3"]:

        session_frame = session_data[
            session_data["session_type"] == session
        ].copy()

        if session_frame.empty:
            continue

        session_frame = session_frame[
            [
                "season",
                "round",
                "driver_id",
            ]
            + [
                column
                for column in feature_columns
                if column in session_frame.columns
            ]
        ]

        rename_map = {}

        for column in feature_columns:
            if column in session_frame.columns:
                rename_map[column] = (
                    f"{session.lower()}_{column}"
                )

        session_frame = session_frame.rename(
            columns=rename_map
        )

        feature_frames.append(
            session_frame
        )

    # ---------------------------------------------------------
    # Merge all sessions
    # ---------------------------------------------------------

    features = identity.copy()

    for frame in feature_frames:

        features = features.merge(
            frame,
            on=[
                "season",
                "round",
                "driver_id",
            ],
            how="left"
        )

    # ---------------------------------------------------------
    # Session availability flags
    # ---------------------------------------------------------

    for session in ["fp1", "fp2", "fp3"]:

        position_column = f"{session}_position"

        features[f"{session}_available"] = (
            features[position_column]
            .notna()
            .astype(int)
        )

    # ---------------------------------------------------------
    # Relative pace features
    # ---------------------------------------------------------

    # Gap to session leader is already supplied by F1DB.
    # Convert milliseconds to seconds for easier interpretation.

    for session in ["fp1", "fp2", "fp3"]:

        gap_column = (
            f"{session}_gap_to_leader_ms"
        )

        if gap_column in features.columns:

            features[
                f"{session}_gap_to_leader_seconds"
            ] = (
                features[gap_column] / 1000.0
            )

    # ---------------------------------------------------------
    # Session consistency features
    # ---------------------------------------------------------

    position_columns = [
        "fp1_position",
        "fp2_position",
        "fp3_position",
    ]

    existing_position_columns = [
        column
        for column in position_columns
        if column in features.columns
    ]

    if existing_position_columns:

        features["practice_avg_position"] = (
            features[
                existing_position_columns
            ]
            .mean(axis=1)
        )

        features["practice_best_position"] = (
            features[
                existing_position_columns
            ]
            .min(axis=1)
        )

        features["practice_worst_position"] = (
            features[
                existing_position_columns
            ]
            .max(axis=1)
        )

        features["practice_position_std"] = (
            features[
                existing_position_columns
            ]
            .std(axis=1)
        )

    # ---------------------------------------------------------
    # Pace consistency
    # ---------------------------------------------------------

    gap_columns = [
        "fp1_gap_to_leader_seconds",
        "fp2_gap_to_leader_seconds",
        "fp3_gap_to_leader_seconds",
    ]

    existing_gap_columns = [
        column
        for column in gap_columns
        if column in features.columns
    ]

    if existing_gap_columns:

        features["practice_avg_gap_seconds"] = (
            features[
                existing_gap_columns
            ]
            .mean(axis=1)
        )

        features["practice_best_gap_seconds"] = (
            features[
                existing_gap_columns
            ]
            .min(axis=1)
        )

    # ---------------------------------------------------------
    # Practice trend
    # ---------------------------------------------------------

    if (
        "fp1_position" in features.columns
        and "fp3_position" in features.columns
    ):

        features["fp1_to_fp3_position_change"] = (
            features["fp1_position"]
            - features["fp3_position"]
        )

    if (
        "fp1_gap_to_leader_seconds" in features.columns
        and "fp3_gap_to_leader_seconds" in features.columns
    ):

        features["fp1_to_fp3_gap_change_seconds"] = (
            features["fp1_gap_to_leader_seconds"]
            - features["fp3_gap_to_leader_seconds"]
        )

    # ---------------------------------------------------------
    # Total practice laps
    # ---------------------------------------------------------

    lap_columns = [
        "fp1_laps_completed",
        "fp2_laps_completed",
        "fp3_laps_completed",
    ]

    existing_lap_columns = [
        column
        for column in lap_columns
        if column in features.columns
    ]

    if existing_lap_columns:

        features["practice_total_laps"] = (
            features[
                existing_lap_columns
            ]
            .sum(axis=1, min_count=1)
        )

    # ---------------------------------------------------------
    # Count available practice sessions
    # ---------------------------------------------------------

    availability_columns = [
        "fp1_available",
        "fp2_available",
        "fp3_available",
    ]

    features["practice_sessions_available"] = (
        features[
            availability_columns
        ]
        .sum(axis=1)
    )

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------

    features = features.sort_values(
        [
            "season",
            "round",
            "driver_name",
        ]
    )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRACTICE FEATURE VALIDATION")
    print("=" * 70)

    print(
        f"\nFinal rows: {len(features):,}"
    )

    print(
        f"Unique races: "
        f"{features[['season', 'round']].drop_duplicates().shape[0]:,}"
    )

    print(
        f"Unique drivers: "
        f"{features['driver_id'].nunique():,}"
    )

    print("\nSession availability:")

    for column in availability_columns:

        if column in features.columns:

            available = features[column].sum()

            print(
                f"  {column}: {int(available):,}"
            )

    print("\nMissing core identity fields:")

    identity_check = [
        "season",
        "round",
        "race_name",
        "circuit_name",
        "driver_id",
        "driver_name",
        "driver_code",
        "team_id",
        "team_name",
    ]

    print(
        features[
            identity_check
        ]
        .isna()
        .sum()
        .to_string()
    )

    # ---------------------------------------------------------
    # Show columns
    # ---------------------------------------------------------

    print("\nFeature columns:")

    for column in features.columns:
        print(f"  {column}")

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    features.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("PRACTICE PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )

    # ---------------------------------------------------------
    # Display sample
    # ---------------------------------------------------------

    print("\nSample:")

    sample_columns = [
        "season",
        "round",
        "race_name",
        "driver_name",
        "team_name",
        "fp1_position",
        "fp2_position",
        "fp3_position",
        "practice_avg_position",
        "practice_best_position",
        "practice_sessions_available",
    ]

    sample_columns = [
        column
        for column in sample_columns
        if column in features.columns
    ]

    print(
        features[
            sample_columns
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()