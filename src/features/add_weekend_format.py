import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset_with_regulation.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset_with_weekend_format.csv"
)

F1DB_ZIP = PROJECT_ROOT / "data" / "raw" / "f1db-csv.zip"


def main():
    print("=" * 70)
    print("ADDING WEEKEND FORMAT")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    if not F1DB_ZIP.exists():
        raise FileNotFoundError(
            f"F1DB ZIP not found: {F1DB_ZIP}"
        )

    print(f"Loading master dataset: {INPUT_FILE}")

    master = pd.read_csv(INPUT_FILE)

    print(f"Master dataset shape: {master.shape}")

    required_master_columns = [
        "season",
        "round",
        "driver_id",
    ]

    for column in required_master_columns:
        if column not in master.columns:
            raise ValueError(
                f"Missing required master column: {column}"
            )

    print("Loading F1DB race schedule...")

    with zipfile.ZipFile(F1DB_ZIP) as z:
        races = pd.read_csv(
            z.open("f1db-races.csv")
        )

    races = races[
        races["year"].between(2014, 2026)
    ].copy()

    required_schedule_columns = [
        "year",
        "round",
        "sprintQualifyingDate",
        "sprintRaceDate",
    ]

    for column in required_schedule_columns:
        if column not in races.columns:
            raise ValueError(
                f"Missing required F1DB schedule column: {column}"
            )

    print(
        f"F1DB scheduled races in 2014-2026: "
        f"{len(races)}"
    )

    schedule = races[
        [
            "year",
            "round",
            "sprintQualifyingDate",
            "sprintRaceDate",
        ]
    ].copy()

    schedule["weekend_format"] = "normal"

    sprint_mask = (
        schedule["sprintQualifyingDate"].notna()
        | schedule["sprintRaceDate"].notna()
    )

    schedule.loc[
        sprint_mask,
        "weekend_format"
    ] = "sprint"

    schedule = schedule.rename(
        columns={
            "year": "season",
        }
    )

    key_columns = [
        "season",
        "round",
    ]

    duplicate_schedule_keys = schedule.duplicated(
        subset=key_columns,
        keep=False,
    )

    if duplicate_schedule_keys.any():
        duplicates = schedule.loc[
            duplicate_schedule_keys,
            key_columns,
        ]

        raise ValueError(
            "Duplicate season/round combinations "
            "found in F1DB schedule:\n"
            f"{duplicates.to_string(index=False)}"
        )

    schedule_races = schedule[
        key_columns
    ].drop_duplicates()

    schedule_sprint_count = (
        schedule["weekend_format"]
        .eq("sprint")
        .sum()
    )

    schedule_normal_count = (
        schedule["weekend_format"]
        .eq("normal")
        .sum()
    )

    print()
    print("Schedule weekend format distribution:")
    print(
        f"  normal: {schedule_normal_count}"
    )
    print(
        f"  sprint: {schedule_sprint_count}"
    )

    master_races = master[
        key_columns
    ].drop_duplicates()

    print()
    print("Master dataset race coverage:")
    print(
        f"  races present: {len(master_races)}"
    )

    missing_from_master = (
        schedule_races.merge(
            master_races,
            on=key_columns,
            how="left",
            indicator=True,
        )
    )

    missing_from_master = missing_from_master[
        missing_from_master["_merge"]
        == "left_only"
    ].drop(
        columns=["_merge"]
    )

    print(
        f"  scheduled races not yet in master: "
        f"{len(missing_from_master)}"
    )

    if not missing_from_master.empty:
        print()
        print(
            "These scheduled races are not yet in "
            "the completed-results master:"
        )

        print(
            missing_from_master
            .sort_values(key_columns)
            .to_string(index=False)
        )

    master_not_in_schedule = (
        master_races.merge(
            schedule_races,
            on=key_columns,
            how="left",
            indicator=True,
        )
    )

    master_not_in_schedule = master_not_in_schedule[
        master_not_in_schedule["_merge"]
        == "left_only"
    ].drop(
        columns=["_merge"]
    )

    if not master_not_in_schedule.empty:
        print()
        print(
            "ERROR: Master contains races that are "
            "not present in F1DB schedule:"
        )

        print(
            master_not_in_schedule
            .sort_values(key_columns)
            .to_string(index=False)
        )

        raise ValueError(
            "Master contains races missing from "
            "the F1DB schedule."
        )

    master = master.drop(
        columns=[
            "weekend_format",
        ],
        errors="ignore",
    )

    master = master.merge(
        schedule[
            [
                "season",
                "round",
                "weekend_format",
            ]
        ],
        on=key_columns,
        how="left",
        validate="many_to_one",
    )

    missing_format = (
        master["weekend_format"]
        .isna()
        .sum()
    )

    if missing_format > 0:
        missing_races = (
            master.loc[
                master["weekend_format"].isna(),
                key_columns,
            ]
            .drop_duplicates()
            .sort_values(key_columns)
        )

        print()
        print(
            "ERROR: Missing weekend format "
            "for master races:"
        )

        print(
            missing_races.to_string(index=False)
        )

        raise ValueError(
            f"{missing_format} master rows have "
            "no weekend_format."
        )

    master["weekend_format"] = (
        master["weekend_format"]
        .astype("category")
    )

    print()
    print(
        f"Final dataset shape: {master.shape}"
    )

    final_races = master[
        [
            "season",
            "round",
            "weekend_format",
        ]
    ].drop_duplicates()

    final_sprint_count = (
        final_races["weekend_format"]
        .eq("sprint")
        .sum()
    )

    final_normal_count = (
        final_races["weekend_format"]
        .eq("normal")
        .sum()
    )

    print()
    print("Final weekend format distribution:")
    print(
        f"  normal: {final_normal_count}"
    )
    print(
        f"  sprint: {final_sprint_count}"
    )

    duplicate_rows = master.duplicated(
        subset=[
            "season",
            "round",
            "driver_id",
        ],
        keep=False,
    ).sum()

    print()
    print("Validation:")
    print(
        f"  Unique races in master: "
        f"{len(final_races)}"
    )

    print(
        f"  Sprint races in master: "
        f"{final_sprint_count}"
    )

    print(
        f"  Normal races in master: "
        f"{final_normal_count}"
    )

    print(
        f"  Duplicate race-driver rows: "
        f"{duplicate_rows}"
    )

    if duplicate_rows != 0:
        raise ValueError(
            f"Found {duplicate_rows} duplicate "
            "race-driver rows."
        )

    if final_sprint_count > schedule_sprint_count:
        raise ValueError(
            "Master contains more sprint races "
            "than the schedule."
        )

    if (
        final_sprint_count
        + final_normal_count
        != len(final_races)
    ):
        raise ValueError(
            "Weekend format counts do not match "
            "the number of master races."
        )

    print()
    print("Saving:")
    print(OUTPUT_FILE)

    master.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("WEEKEND FORMAT FEATURE ADDED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()