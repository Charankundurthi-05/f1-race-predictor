from pathlib import Path
import pandas as pd
import numpy as np
import re
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RACE_FILE = PROJECT_ROOT / "data" / "raw" / "race_results.csv"
QUALIFYING_FILE = PROJECT_ROOT / "data" / "raw" / "qualifying_results.csv"
PRACTICE_FILE = PROJECT_ROOT / "data" / "processed" / "practice_features.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "master_dataset.csv"


def normalize_code(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().upper()

    if value in {"NAN", "NONE", "NULL"}:
        return ""

    return value


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def create_driver_key(row):
    code = normalize_code(row.get("driver_code"))

    if code:
        return code

    return normalize_text(row.get("driver_name"))


def clean_position(value):
    if pd.isna(value):
        return np.nan

    try:
        number = float(value)

        if number <= 0:
            return np.nan

        return number

    except (ValueError, TypeError):
        return np.nan


def main():

    print("=" * 70)
    print("BUILDING MASTER F1 DATASET")
    print("=" * 70)

    print("\nLoading race results...")
    race = pd.read_csv(RACE_FILE)

    print(f"Race rows: {len(race):,}")

    print("\nLoading qualifying results...")
    qualifying = pd.read_csv(QUALIFYING_FILE)

    print(f"Qualifying rows: {len(qualifying):,}")

    print("\nLoading practice features...")
    practice = pd.read_csv(PRACTICE_FILE)

    print(f"Practice rows: {len(practice):,}")

    # ---------------------------------------------------------
    # BASIC CLEANING
    # ---------------------------------------------------------

    print("\nCleaning identifiers...")

    for dataframe in [race, qualifying, practice]:

        dataframe["season"] = pd.to_numeric(
            dataframe["season"],
            errors="coerce"
        ).astype("Int64")

        dataframe["round"] = pd.to_numeric(
            dataframe["round"],
            errors="coerce"
        ).astype("Int64")

        dataframe["driver_code"] = dataframe["driver_code"].apply(
            normalize_code
        )

    # ---------------------------------------------------------
    # CREATE CANONICAL DRIVER KEY
    # ---------------------------------------------------------

    print("Creating canonical driver keys...")

    race["driver_key"] = race.apply(
        create_driver_key,
        axis=1
    )

    qualifying["driver_key"] = qualifying.apply(
        create_driver_key,
        axis=1
    )

    practice["driver_key"] = practice.apply(
        create_driver_key,
        axis=1
    )

    # ---------------------------------------------------------
    # CLEAN TARGET
    # ---------------------------------------------------------

    race["finish_position"] = race["finish_position"].apply(
        clean_position
    )

    race["grid_position"] = race["grid_position"].apply(
        clean_position
    )

    race["points"] = pd.to_numeric(
        race["points"],
        errors="coerce"
    )

    qualifying["qualifying_position"] = qualifying[
        "qualifying_position"
    ].apply(clean_position)

    # ---------------------------------------------------------
    # REMOVE UNNECESSARY RACE COLUMNS
    # ---------------------------------------------------------

    race_columns = [
        "season",
        "round",
        "race_name",
        "race_date",
        "circuit_id",
        "circuit_name",
        "driver_id",
        "driver_code",
        "driver_name",
        "team_id",
        "team_name",
        "number",
        "grid_position",
        "finish_position",
        "position_text",
        "points",
        "laps",
        "status",
        "fastest_lap_rank",
        "fastest_lap_time",
        "fastest_lap_speed",
        "fastest_lap_speed_unit",
        "driver_key"
    ]

    race = race[race_columns].copy()

    # ---------------------------------------------------------
    # QUALIFYING DATA
    # ---------------------------------------------------------

    qualifying_columns = [
        "season",
        "round",
        "driver_key",
        "qualifying_position",
        "q1",
        "q2",
        "q3"
    ]

    qualifying = qualifying[qualifying_columns].copy()

    # Make sure qualifying has one row per driver/race.
    qualifying = qualifying.drop_duplicates(
        subset=["season", "round", "driver_key"],
        keep="first"
    )

    # ---------------------------------------------------------
    # PRACTICE DATA
    # ---------------------------------------------------------

    practice_columns = [
        "season",
        "round",
        "driver_key",
        "fp1_position",
        "fp2_position",
        "fp3_position",
        "practice_avg_position",
        "practice_best_position",
        "practice_sessions_available"
    ]

    available_practice_columns = [
        column
        for column in practice_columns
        if column in practice.columns
    ]

    practice = practice[available_practice_columns].copy()

    practice = practice.drop_duplicates(
        subset=["season", "round", "driver_key"],
        keep="first"
    )

    # ---------------------------------------------------------
    # MERGE QUALIFYING
    # ---------------------------------------------------------

    print("\nMerging qualifying data...")

    master = race.merge(
        qualifying,
        on=["season", "round", "driver_key"],
        how="left",
        indicator="qualifying_match"
    )

    qualifying_matches = (
        master["qualifying_match"] == "both"
    ).sum()

    print(
        f"Qualifying matches: "
        f"{qualifying_matches:,} / {len(master):,}"
    )

    master["qualifying_available"] = (
        master["qualifying_match"] == "both"
    ).astype(int)

    master.drop(
        columns=["qualifying_match"],
        inplace=True
    )

    # ---------------------------------------------------------
    # MERGE PRACTICE
    # ---------------------------------------------------------

    print("\nMerging practice data...")

    master = master.merge(
        practice,
        on=["season", "round", "driver_key"],
        how="left",
        indicator="practice_match"
    )

    practice_matches = (
        master["practice_match"] == "both"
    ).sum()

    print(
        f"Practice matches: "
        f"{practice_matches:,} / {len(master):,}"
    )

    master["practice_available"] = (
        master["practice_match"] == "both"
    ).astype(int)

    master.drop(
        columns=["practice_match"],
        inplace=True
    )

    # ---------------------------------------------------------
    # SORT
    # ---------------------------------------------------------

    master = master.sort_values(
        by=[
            "season",
            "round",
            "finish_position",
            "driver_name"
        ],
        na_position="last"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MASTER DATASET VALIDATION")
    print("=" * 70)

    duplicate_count = master.duplicated(
        subset=["season", "round", "driver_key"]
    ).sum()

    print(f"\nTotal rows: {len(master):,}")
    print(f"Total columns: {len(master.columns)}")
    print(f"Duplicate driver-race rows: {duplicate_count}")

    print("\nMissing target values:")
    print(
        f"finish_position: "
        f"{master['finish_position'].isna().sum():,}"
    )

    print("\nQualifying availability:")
    print(
        master["qualifying_available"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nPractice availability:")
    print(
        master["practice_available"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nPractice session availability:")
    if "practice_sessions_available" in master.columns:
        print(
            master["practice_sessions_available"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

    print("\nRows by season:")
    print(
        master.groupby("season")
        .size()
        .to_string()
    )

    print("\nRows by season with practice:")
    practice_by_season = (
        master.groupby("season")["practice_available"]
        .sum()
    )

    print(practice_by_season.to_string())

    print("\nRows by season with qualifying:")
    qualifying_by_season = (
        master.groupby("season")["qualifying_available"]
        .sum()
    )

    print(qualifying_by_season.to_string())

    # ---------------------------------------------------------
    # DRIVER VALIDATION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DRIVER VALIDATION")
    print("=" * 70)

    print("\nUnique drivers:")
    print(master["driver_key"].nunique())

    print("\nDriver sample:")

    driver_sample = (
        master[
            [
                "driver_key",
                "driver_name",
                "driver_code"
            ]
        ]
        .drop_duplicates()
        .head(20)
    )

    print(
        driver_sample.to_string(index=False)
    )

    # ---------------------------------------------------------
    # TEAM VALIDATION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEAM VALIDATION")
    print("=" * 70)

    print(
        f"\nUnique teams: "
        f"{master['team_name'].nunique()}"
    )

    print(
        master[
            [
                "team_id",
                "team_name"
            ]
        ]
        .drop_duplicates()
        .head(30)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # 2026 VALIDATION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("2026 VALIDATION")
    print("=" * 70)

    data_2026 = master[
        master["season"] == 2026
    ]

    print(
        f"\n2026 rows: {len(data_2026):,}"
    )

    if len(data_2026) > 0:

        print(
            "\n2026 sample:"
        )

        print(
            data_2026[
                [
                    "round",
                    "race_name",
                    "driver_name",
                    "team_name",
                    "finish_position",
                    "qualifying_position",
                    "fp1_position",
                    "fp2_position",
                    "fp3_position"
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # CHECK FOR LEAKAGE-CANDIDATE COLUMNS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("LEAKAGE SAFETY CHECK")
    print("=" * 70)

    print(
        "\nIMPORTANT: race outcome columns are retained only "
        "as targets / post-race information."
    )

    print(
        "The future modelling scripts will NOT use these "
        "as prediction features:"
    )

    leakage_columns = [
        "finish_position",
        "points",
        "status",
        "laps",
        "fastest_lap_rank",
        "fastest_lap_time",
        "fastest_lap_speed"
    ]

    for column in leakage_columns:
        if column in master.columns:
            print(f"  - {column}")

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    master.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("MASTER DATASET COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print(
        f"\nFinal shape: "
        f"{master.shape[0]:,} rows × "
        f"{master.shape[1]} columns"
    )


if __name__ == "__main__":
    main()