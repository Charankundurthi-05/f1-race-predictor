import io
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "f1db-csv.zip"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "practice_results.csv"

START_YEAR = 2014
END_YEAR = 2026


PRACTICE_FILES = {
    "FP1": "f1db-races-free-practice-1-results.csv",
    "FP2": "f1db-races-free-practice-2-results.csv",
    "FP3": "f1db-races-free-practice-3-results.csv",
}


def load_csv_from_zip(zip_file, filename):
    print(f"Loading {filename}...")
    data = zip_file.read(filename)
    return pd.read_csv(io.BytesIO(data))


def find_file(zip_file, keywords):
    matches = []

    for name in zip_file.namelist():
        lower_name = name.lower()

        if all(keyword.lower() in lower_name for keyword in keywords):
            matches.append(name)

    return matches


def clean_position(value):
    if pd.isna(value):
        return pd.NA

    try:
        return int(float(value))
    except (ValueError, TypeError):
        return pd.NA


def clean_laps(value):
    if pd.isna(value):
        return pd.NA

    try:
        return int(float(value))
    except (ValueError, TypeError):
        return pd.NA


def main():
    print("=" * 70)
    print("F1DB HISTORICAL PRACTICE IMPORT")
    print("=" * 70)

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"F1DB ZIP file not found:\n{ZIP_PATH}"
        )

    print(f"\nSource: {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        available_files = set(z.namelist())

        print("\nChecking required files...")

        missing = []

        for session_type, filename in PRACTICE_FILES.items():
            if filename in available_files:
                print(f"  OK  {session_type}: {filename}")
            else:
                print(f"  MISSING  {session_type}: {filename}")
                missing.append(filename)

        if missing:
            raise FileNotFoundError(
                "Required F1DB practice files are missing:\n"
                + "\n".join(missing)
            )

        print("\nLoading race metadata...")
        races = load_csv_from_zip(z, "f1db-races.csv")

        print(f"Race metadata rows: {len(races):,}")

        races = races[
            (races["year"] >= START_YEAR)
            & (races["year"] <= END_YEAR)
        ].copy()

        race_columns = [
            "id",
            "year",
            "round",
            "date",
            "officialName",
            "grandPrixId",
            "circuitId",
        ]

        available_race_columns = [
            column
            for column in race_columns
            if column in races.columns
        ]

        races = races[available_race_columns].copy()

        races = races.rename(
            columns={
                "id": "raceId",
                "officialName": "race_name",
                "grandPrixId": "grand_prix_id",
                "circuitId": "circuit_id",
            }
        )

        races["raceId"] = races["raceId"].astype(str)

        print(f"Historical races available: {len(races):,}")

        all_sessions = []

        for session_type, filename in PRACTICE_FILES.items():
            df = load_csv_from_zip(z, filename)

            print(
                f"  {session_type} total rows: {len(df):,}"
            )

            df["year"] = pd.to_numeric(
                df["year"],
                errors="coerce"
            )

            df = df[
                (df["year"] >= START_YEAR)
                & (df["year"] <= END_YEAR)
            ].copy()

            print(
                f"  {session_type} 2014-2026 rows: "
                f"{len(df):,}"
            )

            df["session_type"] = session_type

            all_sessions.append(df)

        practice = pd.concat(
            all_sessions,
            ignore_index=True
        )

        print(
            f"\nCombined practice rows: "
            f"{len(practice):,}"
        )

        practice["raceId"] = practice["raceId"].astype(str)

        print("\nJoining race metadata...")

        practice = practice.merge(
            races,
            on="raceId",
            how="left",
            suffixes=("", "_race")
        )

        missing_races = practice["race_name"].isna().sum()

        print(
            f"Rows without race metadata: "
            f"{missing_races:,}"
        )

        practice["position"] = practice[
            "positionNumber"
        ].apply(clean_position)

        practice["laps_completed"] = practice[
            "laps"
        ].apply(clean_laps)

        practice["fastest_lap_time"] = practice[
            "time"
        ]

        practice["fastest_lap_time_ms"] = pd.to_numeric(
            practice["timeMillis"],
            errors="coerce"
        )

        practice["gap_to_leader"] = pd.to_numeric(
            practice["gapMillis"],
            errors="coerce"
        )

        practice["interval_to_previous"] = pd.to_numeric(
            practice["intervalMillis"],
            errors="coerce"
        )

        practice["driver_id"] = practice[
            "driverId"
        ].astype(str)

        practice["team_id"] = practice[
            "constructorId"
        ].astype(str)

        practice["driver_code"] = practice[
            "driverId"
        ].astype(str)

        practice["driver_name"] = practice[
            "driverId"
        ].astype(str)

        practice["team_name"] = practice[
            "constructorId"
        ].astype(str)

        practice["round"] = pd.to_numeric(
            practice["round"],
            errors="coerce"
        )

        practice["season"] = practice["year"].astype(int)

        practice["driver_number"] = practice[
            "driverNumber"
        ]

        output_columns = [
            "season",
            "round",
            "raceId",
            "race_name",
            "date",
            "circuit_id",
            "grand_prix_id",
            "session_type",
            "driver_id",
            "driver_code",
            "driver_name",
            "team_id",
            "team_name",
            "driver_number",
            "position",
            "laps_completed",
            "fastest_lap_time",
            "fastest_lap_time_ms",
            "gap_to_leader",
            "interval_to_previous",
        ]

        output_columns = [
            column
            for column in output_columns
            if column in practice.columns
        ]

        practice = practice[output_columns].copy()

        practice = practice.sort_values(
            [
                "season",
                "round",
                "session_type",
                "position",
            ],
            na_position="last"
        )

        practice = practice.drop_duplicates(
            subset=[
                "season",
                "round",
                "session_type",
                "driver_id",
            ],
            keep="first"
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        practice.to_csv(
            OUTPUT_PATH,
            index=False
        )

    print("\n" + "=" * 70)
    print("IMPORT COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )

    print(
        f"\nTotal rows: {len(practice):,}"
    )

    print("\nRows by season:")

    season_counts = (
        practice
        .groupby("season")
        .size()
        .sort_index()
    )

    for season, count in season_counts.items():
        print(f"  {season}: {count:,}")

    print("\nRows by session:")

    session_counts = (
        practice
        .groupby("session_type")
        .size()
        .sort_index()
    )

    for session, count in session_counts.items():
        print(f"  {session}: {count:,}")

    print("\nRows by season and session:")

    coverage = pd.crosstab(
        practice["season"],
        practice["session_type"]
    )

    print(coverage.to_string())

    print("\nUnique races:")
    print(
        practice["raceId"].nunique()
    )

    print("\nUnique drivers:")
    print(
        practice["driver_id"].nunique()
    )


if __name__ == "__main__":
    main()