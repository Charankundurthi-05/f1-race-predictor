from pathlib import Path
import io
import zipfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRACTICE_PATH = PROJECT_ROOT / "data" / "raw" / "practice_results.csv"
RACE_PATH = PROJECT_ROOT / "data" / "raw" / "race_results.csv"
F1DB_ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "f1db-csv.zip"

OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "practice_results.csv"


def load_csv_from_zip(zip_file, filename):
    return pd.read_csv(
        io.BytesIO(
            zip_file.read(filename)
        )
    )


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find {description}. "
        f"Available columns: {df.columns.tolist()}"
    )


def main():
    print("=" * 70)
    print("ENRICHING F1DB PRACTICE DATA")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load practice data
    # ---------------------------------------------------------

    print("\nLoading practice data...")

    practice = pd.read_csv(PRACTICE_PATH)

    print(
        f"Practice rows: {len(practice):,}"
    )

    # ---------------------------------------------------------
    # Load race results
    # ---------------------------------------------------------

    print("\nLoading race results...")

    race = pd.read_csv(RACE_PATH)

    print(
        f"Race rows: {len(race):,}"
    )

    # ---------------------------------------------------------
    # Load F1DB master tables
    # ---------------------------------------------------------

    print("\nLoading F1DB driver and constructor tables...")

    with zipfile.ZipFile(F1DB_ZIP_PATH, "r") as z:
        drivers = load_csv_from_zip(
            z,
            "f1db-drivers.csv"
        )

        constructors = load_csv_from_zip(
            z,
            "f1db-constructors.csv"
        )

    print(
        f"F1DB drivers: {len(drivers):,}"
    )

    print(
        f"F1DB constructors: {len(constructors):,}"
    )

    # ---------------------------------------------------------
    # Detect driver columns
    # ---------------------------------------------------------

    driver_id_column = find_column(
        drivers,
        [
            "id",
            "driverId",
        ],
        "driver ID"
    )

    driver_name_column = find_column(
        drivers,
        [
            "fullName",
            "name",
        ],
        "driver full name"
    )

    driver_code_column = find_column(
        drivers,
        [
            "abbreviation",
            "code",
            "shortCode",
        ],
        "driver abbreviation/code"
    )

    # ---------------------------------------------------------
    # Detect constructor columns
    # ---------------------------------------------------------

    constructor_id_column = find_column(
        constructors,
        [
            "id",
            "constructorId",
        ],
        "constructor ID"
    )

    constructor_name_column = find_column(
        constructors,
        [
            "name",
            "fullName",
        ],
        "constructor name"
    )

    print("\nDetected F1DB columns:")

    print(
        f"  Driver ID: {driver_id_column}"
    )

    print(
        f"  Driver name: {driver_name_column}"
    )

    print(
        f"  Driver code: {driver_code_column}"
    )

    print(
        f"  Constructor ID: {constructor_id_column}"
    )

    print(
        f"  Constructor name: {constructor_name_column}"
    )

    # ---------------------------------------------------------
    # Normalize IDs
    # ---------------------------------------------------------

    practice["driver_id"] = (
        practice["driver_id"]
        .astype(str)
        .str.strip()
    )

    practice["team_id"] = (
        practice["team_id"]
        .astype(str)
        .str.strip()
    )

    drivers[driver_id_column] = (
        drivers[driver_id_column]
        .astype(str)
        .str.strip()
    )

    constructors[constructor_id_column] = (
        constructors[constructor_id_column]
        .astype(str)
        .str.strip()
    )

    race["season"] = pd.to_numeric(
        race["season"],
        errors="coerce"
    )

    race["round"] = pd.to_numeric(
        race["round"],
        errors="coerce"
    )

    practice["season"] = pd.to_numeric(
        practice["season"],
        errors="coerce"
    )

    practice["round"] = pd.to_numeric(
        practice["round"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Driver mapping
    # ---------------------------------------------------------

    print("\nMapping F1DB driver information...")

    driver_map = drivers[
        [
            driver_id_column,
            driver_name_column,
            driver_code_column,
        ]
    ].copy()

    driver_map.columns = [
        "driver_id",
        "driver_name",
        "driver_code",
    ]

    driver_map = driver_map.drop_duplicates(
        subset=["driver_id"]
    )

    practice = practice.drop(
        columns=[
            "driver_name",
            "driver_code",
        ],
        errors="ignore"
    )

    practice = practice.merge(
        driver_map,
        on="driver_id",
        how="left"
    )

    # ---------------------------------------------------------
    # Constructor/team mapping
    # ---------------------------------------------------------

    print("\nMapping F1DB constructor information...")

    constructor_map = constructors[
        [
            constructor_id_column,
            constructor_name_column,
        ]
    ].copy()

    constructor_map.columns = [
        "team_id",
        "team_name",
    ]

    constructor_map = constructor_map.drop_duplicates(
        subset=["team_id"]
    )

    practice = practice.drop(
        columns=[
            "team_name",
        ],
        errors="ignore"
    )

    practice = practice.merge(
        constructor_map,
        on="team_id",
        how="left"
    )

    # ---------------------------------------------------------
    # Race information
    # ---------------------------------------------------------

    print("\nMapping race and circuit information...")

    race_info = (
        race[
            [
                "season",
                "round",
                "race_name",
                "race_date",
                "circuit_id",
                "circuit_name",
            ]
        ]
        .drop_duplicates(
            subset=[
                "season",
                "round",
            ]
        )
    )

    practice = practice.drop(
        columns=[
            "race_name",
            "date",
            "circuit_id",
            "circuit_name",
        ],
        errors="ignore"
    )

    practice = practice.merge(
        race_info,
        on=[
            "season",
            "round",
        ],
        how="left"
    )

    practice = practice.rename(
        columns={
            "race_date": "date"
        }
    )

    # ---------------------------------------------------------
    # Standardize column names
    # ---------------------------------------------------------

    rename_map = {
        "driverNumber": "driver_number",
        "positionDisplayOrder": "position",
        "positionNumber": "position_number",
        "positionText": "position_text",
        "laps": "laps_completed",
        "time": "fastest_lap_time",
        "timeMillis": "fastest_lap_time_ms",
        "gap": "gap_to_leader",
        "gapMillis": "gap_to_leader_ms",
        "interval": "interval_to_previous",
        "intervalMillis": "interval_to_previous_ms",
    }

    practice = practice.rename(
        columns={
            old: new
            for old, new in rename_map.items()
            if old in practice.columns
        }
    )

    # ---------------------------------------------------------
    # Select final columns
    # ---------------------------------------------------------

    desired_columns = [
        "season",
        "round",
        "raceId",
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
        "driver_number",
        "position",
        "position_number",
        "position_text",
        "laps_completed",
        "fastest_lap_time",
        "fastest_lap_time_ms",
        "gap_to_leader",
        "gap_to_leader_ms",
        "interval_to_previous",
        "interval_to_previous_ms",
    ]

    desired_columns = [
        column
        for column in desired_columns
        if column in practice.columns
    ]

    practice = practice[desired_columns]

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------

    practice = practice.sort_values(
        [
            "season",
            "round",
            "session_type",
            "position",
        ],
        na_position="last"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    missing_driver_names = (
        practice["driver_name"]
        .isna()
        .sum()
    )

    missing_driver_codes = (
        practice["driver_code"]
        .isna()
        .sum()
    )

    missing_team_names = (
        practice["team_name"]
        .isna()
        .sum()
    )

    missing_races = (
        practice["race_name"]
        .isna()
        .sum()
    )

    missing_circuits = (
        practice["circuit_name"]
        .isna()
        .sum()
    )

    missing_driver_ids = (
        practice["driver_id"]
        .isna()
        .sum()
    )

    missing_team_ids = (
        practice["team_id"]
        .isna()
        .sum()
    )

    print(
        f"\nMissing driver IDs: {missing_driver_ids:,}"
    )

    print(
        f"Missing driver names: {missing_driver_names:,}"
    )

    print(
        f"Missing driver codes: {missing_driver_codes:,}"
    )

    print(
        f"Missing team IDs: {missing_team_ids:,}"
    )

    print(
        f"Missing team names: {missing_team_names:,}"
    )

    print(
        f"Missing race names: {missing_races:,}"
    )

    print(
        f"Missing circuit names: {missing_circuits:,}"
    )

    # ---------------------------------------------------------
    # Unresolved driver IDs
    # ---------------------------------------------------------

    if missing_driver_names > 0:
        print("\nUnresolved driver IDs:")

        unresolved = (
            practice[
                practice["driver_name"].isna()
            ]["driver_id"]
            .drop_duplicates()
            .tolist()
        )

        for driver_id in unresolved[:30]:
            print(f"  {driver_id}")

        if len(unresolved) > 30:
            print(
                f"  ... and "
                f"{len(unresolved) - 30} more"
            )

    # ---------------------------------------------------------
    # Unresolved team IDs
    # ---------------------------------------------------------

    if missing_team_names > 0:
        print("\nUnresolved team IDs:")

        unresolved = (
            practice[
                practice["team_name"].isna()
            ]["team_id"]
            .drop_duplicates()
            .tolist()
        )

        for team_id in unresolved[:30]:
            print(f"  {team_id}")

        if len(unresolved) > 30:
            print(
                f"  ... and "
                f"{len(unresolved) - 30} more"
            )

    # ---------------------------------------------------------
    # Season counts
    # ---------------------------------------------------------

    print("\nRows by season:")

    season_counts = (
        practice["season"]
        .value_counts()
        .sort_index()
    )

    for season, count in season_counts.items():
        print(
            f"  {int(season)}: {count:,}"
        )

    # ---------------------------------------------------------
    # Session counts
    # ---------------------------------------------------------

    print("\nRows by session:")

    session_counts = (
        practice["session_type"]
        .value_counts()
        .sort_index()
    )

    for session, count in session_counts.items():
        print(
            f"  {session}: {count:,}"
        )

    # ---------------------------------------------------------
    # Final statistics
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL DATA CHECK")
    print("=" * 70)

    print(
        f"\nTotal rows: {len(practice):,}"
    )

    print(
        f"Unique races: "
        f"{practice[['season', 'round']].drop_duplicates().shape[0]:,}"
    )

    print(
        f"Unique drivers: "
        f"{practice['driver_id'].nunique():,}"
    )

    print(
        f"Unique teams: "
        f"{practice['team_id'].nunique():,}"
    )

    # ---------------------------------------------------------
    # Sample records
    # ---------------------------------------------------------

    print("\nSample records:")

    sample_columns = [
        "season",
        "round",
        "race_name",
        "session_type",
        "driver_name",
        "driver_code",
        "team_name",
        "position",
        "laps_completed",
    ]

    print(
        practice[sample_columns]
        .head(10)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    practice.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("ENRICHMENT COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()