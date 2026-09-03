from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "predictions"

INPUT_FILE = DATA_DIR / "stage_pre_practice.csv"
OUTPUT_FILE = (
    OUTPUT_DIR
    / "current_2026_pre_practice_features.csv"
)

CURRENT_SEASON = 2026
CURRENT_ROUND = 13
CURRENT_RACE_NAME = "Italian Grand Prix"
CURRENT_RACE_DATE = "2026-09-06"


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    return pd.read_csv(INPUT_FILE)


def find_monza_rows(df):
    circuit_names = (
        df["circuit_name"]
        .astype(str)
        .str.lower()
    )

    mask = (
        circuit_names.str.contains(
            "monza",
            na=False
        )
        |
        df["circuit_id"]
        .astype(str)
        .str.lower()
        .eq("monza")
    )

    monza = df[mask].copy()

    if monza.empty:
        raise ValueError(
            "No historical Monza rows found."
        )

    return monza


def get_latest_2026_rows(df):
    season = df[
        df["season"] == CURRENT_SEASON
    ].copy()

    if season.empty:
        raise ValueError(
            "No 2026 rows found."
        )

    latest_round = season["round"].max()

    latest = season[
        season["round"] == latest_round
    ].copy()

    if latest.empty:
        raise ValueError(
            "Could not find latest 2026 rows."
        )

    print(
        f"Latest completed 2026 round in dataset: "
        f"{latest_round}"
    )

    print(
        f"Drivers in latest round: "
        f"{len(latest)}"
    )

    return latest


def get_monza_circuit_features(monza):
    columns = [
        "circuit_id",
        "circuit_name",
        "circuit_type",
        "circuit_direction",
        "circuit_latitude",
        "circuit_longitude",
        "circuit_length_km",
        "circuit_turns",
        "circuit_total_races_held",
    ]

    latest = (
        monza
        .sort_values(
            [
                "season",
                "round"
            ]
        )
        .iloc[-1]
    )

    return {
        column: latest[column]
        for column in columns
    }


def get_driver_monza_history(monza):
    required_columns = [
        "driver_id",
        "season",
        "round",
        "driver_circuit_avg_finish",
        "driver_circuit_races",
    ]

    missing = [
        column
        for column in required_columns
        if column not in monza.columns
    ]

    if missing:
        raise ValueError(
            "Missing Monza history columns: "
            + ", ".join(missing)
        )

    history = (
        monza[
            required_columns
        ]
        .sort_values(
            [
                "driver_id",
                "season",
                "round"
            ]
        )
        .groupby(
            "driver_id",
            as_index=False
        )
        .tail(1)
        .copy()
    )

    return history[
        [
            "driver_id",
            "driver_circuit_avg_finish",
            "driver_circuit_races",
        ]
    ].drop_duplicates(
        "driver_id"
    )


def build_current_race(
    latest_2026,
    monza
):
    form_columns = [
        "driver_previous_finish",
        "driver_avg_finish_last_3",
        "driver_avg_finish_last_5",
        "driver_previous_points",
        "driver_avg_points_last_5",
        "team_avg_finish_last_5",
        "team_avg_points_last_5",
    ]

    missing = [
        column
        for column in form_columns
        if column not in latest_2026.columns
    ]

    if missing:
        raise ValueError(
            "Missing current-form columns: "
            + ", ".join(missing)
        )

    circuit_features = (
        get_monza_circuit_features(
            monza
        )
    )

    driver_history = (
        get_driver_monza_history(
            monza
        )
    )

    current = latest_2026.copy()

    current = current.drop(
        columns=[
            "circuit_id",
            "circuit_name",
            "circuit_type",
            "circuit_direction",
            "circuit_latitude",
            "circuit_longitude",
            "circuit_length_km",
            "circuit_turns",
            "circuit_total_races_held",
            "driver_circuit_avg_finish",
            "driver_circuit_races",
        ],
        errors="ignore"
    )

    current = current.merge(
        driver_history,
        on="driver_id",
        how="left"
    )

    for column, value in circuit_features.items():
        current[column] = value

    current["season"] = CURRENT_SEASON
    current["round"] = CURRENT_ROUND
    current["race_name"] = CURRENT_RACE_NAME
    current["race_date"] = CURRENT_RACE_DATE

    current["regulation_era"] = "new_2026"
    current["regulation_era_code"] = 3

    current["weekend_format"] = "normal"

    current["grid_position"] = pd.NA
    current["finish_position"] = pd.NA

    practice_columns = [
        column
        for column in current.columns
        if (
            column.startswith("fp1_")
            or column.startswith("fp2_")
            or column.startswith("fp3_")
            or column.startswith("practice_")
        )
    ]

    qualifying_columns = [
        column
        for column in current.columns
        if (
            column.startswith("qualifying")
            or column.startswith("q1")
            or column.startswith("q2")
            or column.startswith("q3")
        )
    ]

    for column in practice_columns:
        current[column] = pd.NA

    for column in qualifying_columns:
        current[column] = pd.NA

    missing_history = (
        current["driver_circuit_avg_finish"]
        .isna()
    )

    print()
    print(
        f"Drivers without historical Monza "
        f"performance: {int(missing_history.sum())}"
    )

    if missing_history.any():
        for name in current.loc[
            missing_history,
            "driver_name"
        ]:
            print(
                f"  - {name}"
            )

    return current


def validate(result):
    print()
    print("=" * 80)
    print("VALIDATING CURRENT 2026 RACE FEATURES")
    print("=" * 80)

    print(
        f"Season: {result['season'].iloc[0]}"
    )

    print(
        f"Round: {result['round'].iloc[0]}"
    )

    print(
        f"Race: {result['race_name'].iloc[0]}"
    )

    print(
        f"Circuit: {result['circuit_name'].iloc[0]}"
    )

    print(
        f"Regulation era: "
        f"{result['regulation_era'].iloc[0]}"
    )

    print(
        f"Rows: {len(result)}"
    )

    if len(result) != 22:
        raise ValueError(
            f"Expected 22 current drivers, "
            f"found {len(result)}."
        )

    if result["driver_id"].duplicated().any():
        raise ValueError(
            "Duplicate drivers detected."
        )

    if result["driver_name"].isna().any():
        raise ValueError(
            "Missing driver names."
        )

    if (
        result["regulation_era"]
        != "new_2026"
    ).any():
        raise ValueError(
            "Incorrect regulation era."
        )

    if (
        result["regulation_era_code"]
        != 3
    ).any():
        raise ValueError(
            "Incorrect regulation era code."
        )

    leakage_columns = [
        "qualifying_position",
        "q1",
        "q2",
        "q3",
        "grid_position",
        "finish_position",
    ]

    for column in leakage_columns:
        if column not in result.columns:
            continue

        if result[column].notna().any():
            raise ValueError(
                f"Leakage detected in {column}"
            )

    print()
    print(
        "Qualifying/race leakage: PASS"
    )

    print(
        "Driver uniqueness: PASS"
    )

    print(
        "Driver count: PASS"
    )

    print(
        "Regulation era: PASS"
    )


def main():
    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("BUILD CURRENT 2026 RACE FEATURES")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    latest_2026 = (
        get_latest_2026_rows(df)
    )

    monza = find_monza_rows(df)

    print(
        f"Historical Monza rows: "
        f"{len(monza)}"
    )

    result = build_current_race(
        latest_2026,
        monza
    )

    validate(result)

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"Saved -> {OUTPUT_FILE}"
    )

    print()
    print("=" * 80)
    print("CURRENT RACE FEATURE BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()