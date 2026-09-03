from pathlib import Path
import zipfile
import pandas as pd
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset_with_weekend_format.csv"
)

ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "f1db-csv.zip"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset_with_sprint.csv"
)


def load_f1db_csv(zip_file, filename):
    with zipfile.ZipFile(zip_file) as z:
        return pd.read_csv(z.open(filename))


def clean_name(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    value = value.replace("jr.", "")
    value = value.replace("jr", "")

    value = " ".join(value.split())

    return value


def build_driver_mapping(master, f1db_drivers):
    master_drivers = (
        master[
            ["driver_id", "driver_name"]
        ]
        .dropna(
            subset=[
                "driver_id",
                "driver_name"
            ]
        )
        .drop_duplicates()
        .copy()
    )

    f1db_drivers = f1db_drivers[
        [
            "id",
            "firstName",
            "lastName"
        ]
    ].copy()

    f1db_drivers["driver_name"] = (
        f1db_drivers["firstName"]
        .fillna("")
        .astype(str)
        .str.strip()
        + " "
        + f1db_drivers["lastName"]
        .fillna("")
        .astype(str)
        .str.strip()
    ).str.strip()

    master_drivers["name_key"] = (
        master_drivers["driver_name"]
        .apply(clean_name)
    )

    f1db_drivers["name_key"] = (
        f1db_drivers["driver_name"]
        .apply(clean_name)
    )

    alias_map = {
        "kimi antonelli": "andrea kimi antonelli",
        "carlos sainz jr": "carlos sainz",
        "carlos sainz": "carlos sainz",
    }

    master_drivers["name_key"] = (
        master_drivers["name_key"]
        .replace(alias_map)
    )

    f1db_drivers["name_key"] = (
        f1db_drivers["name_key"]
        .replace(alias_map)
    )

    master_name_counts = (
        master_drivers
        .groupby("name_key")["driver_id"]
        .nunique()
    )

    f1db_name_counts = (
        f1db_drivers
        .groupby("name_key")["id"]
        .nunique()
    )

    duplicate_master_names = (
        master_name_counts[
            master_name_counts > 1
        ]
    )

    duplicate_f1db_names = (
        f1db_name_counts[
            f1db_name_counts > 1
        ]
    )

    if len(duplicate_master_names) > 0:
        raise ValueError(
            "Master dataset contains driver names "
            "mapped to multiple driver IDs:\n"
            + duplicate_master_names.to_string()
        )

    if len(duplicate_f1db_names) > 0:
        print(
            "\nWARNING: Ambiguous F1DB driver names found:"
        )

        print(
            duplicate_f1db_names.to_string()
        )

        print(
            "\nThese ambiguous names will be excluded "
            "from automatic mapping."
        )

        f1db_drivers = f1db_drivers[
            ~f1db_drivers["name_key"].isin(
                duplicate_f1db_names.index
            )
        ]

    mapping = f1db_drivers.merge(
        master_drivers[
            [
                "driver_id",
                "name_key"
            ]
        ],
        on="name_key",
        how="inner"
    )

    mapping = mapping[
        [
            "id",
            "driver_id",
            "driver_name"
        ]
    ].drop_duplicates()

    return mapping


def prepare_sprint_qualifying(
    sprint_qualifying,
    driver_mapping
):
    sq = sprint_qualifying.copy()

    sq = sq.merge(
        driver_mapping[
            [
                "id",
                "driver_id"
            ]
        ],
        left_on="driverId",
        right_on="id",
        how="left"
    )

    sq["season"] = pd.to_numeric(
        sq["year"],
        errors="coerce"
    )

    sq["round"] = pd.to_numeric(
        sq["round"],
        errors="coerce"
    )

    sq["sprint_qualifying_position"] = (
        pd.to_numeric(
            sq["positionNumber"],
            errors="coerce"
        )
    )

    sq["sprint_qualifying_q1_ms"] = (
        pd.to_numeric(
            sq["q1Millis"],
            errors="coerce"
        )
    )

    sq["sprint_qualifying_q2_ms"] = (
        pd.to_numeric(
            sq["q2Millis"],
            errors="coerce"
        )
    )

    sq["sprint_qualifying_q3_ms"] = (
        pd.to_numeric(
            sq["q3Millis"],
            errors="coerce"
        )
    )

    sq["sprint_qualifying_gap_ms"] = (
        pd.to_numeric(
            sq["gapMillis"],
            errors="coerce"
        )
    )

    sq["sprint_qualifying_laps"] = (
        pd.to_numeric(
            sq["laps"],
            errors="coerce"
        )
    )

    result = sq[
        [
            "season",
            "round",
            "driver_id",
            "sprint_qualifying_position",
            "sprint_qualifying_q1_ms",
            "sprint_qualifying_q2_ms",
            "sprint_qualifying_q3_ms",
            "sprint_qualifying_gap_ms",
            "sprint_qualifying_laps"
        ]
    ].copy()

    return result


def prepare_sprint_race(
    sprint_race,
    driver_mapping
):
    sr = sprint_race.copy()

    sr = sr.merge(
        driver_mapping[
            [
                "id",
                "driver_id"
            ]
        ],
        left_on="driverId",
        right_on="id",
        how="left"
    )

    sr["season"] = pd.to_numeric(
        sr["year"],
        errors="coerce"
    )

    sr["round"] = pd.to_numeric(
        sr["round"],
        errors="coerce"
    )

    sr["sprint_finish_position"] = (
        pd.to_numeric(
            sr["positionNumber"],
            errors="coerce"
        )
    )

    sr["sprint_grid_position"] = (
        pd.to_numeric(
            sr["gridPositionNumber"],
            errors="coerce"
        )
    )

    sr["sprint_positions_gained"] = (
        pd.to_numeric(
            sr["positionsGained"],
            errors="coerce"
        )
    )

    sr["sprint_points"] = (
        pd.to_numeric(
            sr["points"],
            errors="coerce"
        )
    )

    sr["sprint_laps"] = (
        pd.to_numeric(
            sr["laps"],
            errors="coerce"
        )
    )

    sr["sprint_pit_stops"] = (
        pd.to_numeric(
            sr["pitStops"],
            errors="coerce"
        )
    )

    result = sr[
        [
            "season",
            "round",
            "driver_id",
            "sprint_finish_position",
            "sprint_grid_position",
            "sprint_positions_gained",
            "sprint_points",
            "sprint_laps",
            "sprint_pit_stops"
        ]
    ].copy()

    return result


def validate_unique(
    dataframe,
    keys,
    label
):
    valid_rows = dataframe.dropna(
        subset=keys
    )

    duplicates = valid_rows[
        valid_rows.duplicated(
            keys,
            keep=False
        )
    ]

    if len(duplicates) > 0:
        print(
            f"\nERROR: Duplicate rows found in {label}:"
        )

        print(
            duplicates[
                keys
            ]
            .sort_values(keys)
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            f"{label} contains duplicate "
            "race-driver rows."
        )


def main():
    print("=" * 70)
    print("ADDING SPRINT FEATURES")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    if not ZIP_FILE.exists():
        raise FileNotFoundError(
            f"F1DB ZIP not found:\n{ZIP_FILE}"
        )

    print(
        f"\nInput:  {INPUT_FILE}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    master = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nMaster dataset shape: "
        f"{master.shape}"
    )

    if "weekend_format" not in master.columns:
        raise ValueError(
            "weekend_format column is missing "
            "from the master dataset."
        )

    print("\nWeekend format:")

    print(
        master[
            "weekend_format"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    master["season"] = pd.to_numeric(
        master["season"],
        errors="coerce"
    )

    master["round"] = pd.to_numeric(
        master["round"],
        errors="coerce"
    )

    master["driver_id"] = (
        master["driver_id"]
        .astype(str)
    )

    f1db_drivers = load_f1db_csv(
        ZIP_FILE,
        "f1db-drivers.csv"
    )

    sprint_qualifying = load_f1db_csv(
        ZIP_FILE,
        "f1db-races-sprint-qualifying-results.csv"
    )

    sprint_race = load_f1db_csv(
        ZIP_FILE,
        "f1db-races-sprint-race-results.csv"
    )

    print(
        f"\nF1DB Sprint Qualifying rows: "
        f"{len(sprint_qualifying)}"
    )

    print(
        f"F1DB Sprint Race rows:       "
        f"{len(sprint_race)}"
    )

    driver_mapping = build_driver_mapping(
        master,
        f1db_drivers
    )

    print(
        f"\nDriver mappings created: "
        f"{len(driver_mapping)}"
    )

    print("\nImportant mappings:")

    important_ids = [
        "carlos-sainz-jr",
        "kimi-antonelli",
        "nyck-de-vries"
    ]

    important_mapping = driver_mapping[
        driver_mapping[
            "id"
        ].isin(important_ids)
    ]

    if len(important_mapping) > 0:
        print(
            important_mapping
            .to_string(index=False)
        )

    sq = prepare_sprint_qualifying(
        sprint_qualifying,
        driver_mapping
    )

    sr = prepare_sprint_race(
        sprint_race,
        driver_mapping
    )

    unmatched_sq = (
        sq["driver_id"]
        .isna()
        .sum()
    )

    unmatched_sr = (
        sr["driver_id"]
        .isna()
        .sum()
    )

    print(
        f"\nUnmatched Sprint Qualifying rows: "
        f"{unmatched_sq}"
    )

    print(
        f"Unmatched Sprint Race rows:       "
        f"{unmatched_sr}"
    )

    sq = sq.dropna(
        subset=["driver_id"]
    ).copy()

    sr = sr.dropna(
        subset=["driver_id"]
    ).copy()

    sq["driver_id"] = (
        sq["driver_id"]
        .astype(str)
    )

    sr["driver_id"] = (
        sr["driver_id"]
        .astype(str)
    )

    validate_unique(
        sq,
        [
            "season",
            "round",
            "driver_id"
        ],
        "F1DB Sprint Qualifying"
    )

    validate_unique(
        sr,
        [
            "season",
            "round",
            "driver_id"
        ],
        "F1DB Sprint Race"
    )

    # --------------------------------------------------------------
    # CRITICAL PROTECTION:
    # Keep only race-driver combinations that our master dataset
    # explicitly identifies as Sprint weekends.
    #
    # This prevents F1DB Sprint records from being merged into
    # normal-weekend rows that happen to share season/round/driver.
    # --------------------------------------------------------------

    sprint_keys = (
        master[
            master["weekend_format"].eq("sprint")
        ][
            [
                "season",
                "round",
                "driver_id"
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    print(
        "\nMaster Sprint race-driver keys:"
    )

    print(
        f"{len(sprint_keys)}"
    )

    sq_before_filter = len(sq)

    sq = sq.merge(
        sprint_keys,
        on=[
            "season",
            "round",
            "driver_id"
        ],
        how="inner"
    )

    sr_before_filter = len(sr)

    sr = sr.merge(
        sprint_keys,
        on=[
            "season",
            "round",
            "driver_id"
        ],
        how="inner"
    )

    print(
        "\nSprint source filtering:"
    )

    print(
        f"Sprint Qualifying before: "
        f"{sq_before_filter}"
    )

    print(
        f"Sprint Qualifying after:  "
        f"{len(sq)}"
    )

    print(
        f"Sprint Race before:       "
        f"{sr_before_filter}"
    )

    print(
        f"Sprint Race after:        "
        f"{len(sr)}"
    )

    validate_unique(
        sq,
        [
            "season",
            "round",
            "driver_id"
        ],
        "Filtered Sprint Qualifying"
    )

    validate_unique(
        sr,
        [
            "season",
            "round",
            "driver_id"
        ],
        "Filtered Sprint Race"
    )

    sprint_columns = [
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
        "sprint_pit_stops"
    ]

    existing_columns = [
        column
        for column in sprint_columns
        if column in master.columns
    ]

    if existing_columns:
        print(
            "\nRemoving existing Sprint columns:"
        )

        print(
            existing_columns
        )

        master = master.drop(
            columns=existing_columns
        )

    master = master.merge(
        sq,
        on=[
            "season",
            "round",
            "driver_id"
        ],
        how="left",
        validate="one_to_one"
    )

    master = master.merge(
        sr,
        on=[
            "season",
            "round",
            "driver_id"
        ],
        how="left",
        validate="one_to_one"
    )

    validate_unique(
        master,
        [
            "season",
            "round",
            "driver_id"
        ],
        "Final master dataset"
    )

    sprint_mask = (
        master[
            "weekend_format"
        ]
        .eq("sprint")
    )

    normal_mask = (
        master[
            "weekend_format"
        ]
        .eq("normal")
    )

    print(
        "\nSprint feature availability:"
    )

    for column in sprint_columns:
        sprint_available = (
            master.loc[
                sprint_mask,
                column
            ]
            .notna()
            .sum()
        )

        normal_available = (
            master.loc[
                normal_mask,
                column
            ]
            .notna()
            .sum()
        )

        print(
            f"{column:35s} "
            f"sprint={sprint_available:4d} "
            f"normal={normal_available:4d}"
        )

    normal_nonnull = (
        master.loc[
            normal_mask,
            sprint_columns
        ]
        .notna()
        .sum()
        .sum()
    )

    if normal_nonnull != 0:
        print(
            "\nERROR: Sprint features found on "
            "normal weekends."
        )

        offending = master.loc[
            normal_mask,
            [
                "season",
                "round",
                "driver_id",
                "weekend_format"
            ]
            + sprint_columns
        ]

        offending = offending[
            offending[
                sprint_columns
            ]
            .notna()
            .any(axis=1)
        ]

        print(
            offending
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "Sprint features unexpectedly contain "
            "values on normal weekends."
        )

    print(
        "\nNormal weekend validation: PASS"
    )

    print(
        "\nSprint race coverage by season:"
    )

    sprint_coverage = (
        master.loc[sprint_mask]
        .groupby("season")
        .agg(
            races=(
                "round",
                "nunique"
            ),
            driver_rows=(
                "driver_id",
                "count"
            ),
            sprint_qualifying_results=(
                "sprint_qualifying_position",
                "count"
            ),
            sprint_race_results=(
                "sprint_finish_position",
                "count"
            )
        )
        .reset_index()
    )

    print(
        sprint_coverage
        .to_string(index=False)
    )

    print(
        "\nFinal dataset shape:",
        master.shape
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    master.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved successfully:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SPRINT FEATURE BUILD COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()