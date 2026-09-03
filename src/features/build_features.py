from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

RACE_FILE = ROOT / "data" / "raw" / "race_results.csv"
QUALIFYING_FILE = ROOT / "data" / "raw" / "qualifying_results.csv"
PRACTICE_FILE = ROOT / "data" / "processed" / "practice_results.csv"

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "features.csv"


def load_data():
    race = pd.read_csv(RACE_FILE)
    qualifying = pd.read_csv(QUALIFYING_FILE)
    practice = pd.read_csv(PRACTICE_FILE)

    return race, qualifying, practice


def clean_numeric_columns(df):
    numeric_columns = [
        "position",
        "finish_position",
        "grid_position",
        "qualifying_position",
        "points",
        "laps",
        "fastest_lap_rank",
        "q1",
        "q2",
        "q3",
        "fastest_lap_time",
        "average_lap_time",
        "top_speed",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def build_race_base(race, qualifying):
    race = clean_numeric_columns(race)
    qualifying = clean_numeric_columns(qualifying)

    race = race.copy()
    qualifying = qualifying.copy()

    base_columns = [
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
        "points",
        "laps",
        "status",
    ]

    base_columns = [c for c in base_columns if c in race.columns]

    base = race[base_columns].copy()

    qualifying_columns = [
        "season",
        "round",
        "driver_id",
        "qualifying_position",
        "q1",
        "q2",
        "q3",
    ]

    qualifying_columns = [
        c for c in qualifying_columns if c in qualifying.columns
    ]

    qualifying_small = qualifying[qualifying_columns].copy()

    merge_columns = ["season", "round", "driver_id"]

    base = base.merge(
        qualifying_small,
        on=merge_columns,
        how="left",
        suffixes=("", "_qualifying")
    )

    return base


def add_historical_features(df):
    df = df.sort_values(
        ["driver_id", "season", "round"]
    ).copy()

    driver_group = df.groupby("driver_id", group_keys=False)

    df["driver_previous_finish"] = (
        driver_group["finish_position"]
        .shift(1)
    )

    df["driver_avg_finish_last_3"] = (
        driver_group["finish_position"]
        .transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
    )

    df["driver_avg_finish_last_5"] = (
        driver_group["finish_position"]
        .transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
    )

    df["driver_previous_points"] = (
        driver_group["points"]
        .shift(1)
    )

    df["driver_avg_points_last_5"] = (
        driver_group["points"]
        .transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
    )

    team_group = df.groupby("team_id", group_keys=False)

    df["team_avg_finish_last_5"] = (
        team_group["finish_position"]
        .transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
    )

    df["team_avg_points_last_5"] = (
        team_group["points"]
        .transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
    )

    return df


def add_circuit_features(df):
    df = df.sort_values(
        ["driver_id", "circuit_id", "season", "round"]
    ).copy()

    circuit_group = df.groupby(
        ["driver_id", "circuit_id"],
        group_keys=False
    )

    df["driver_circuit_avg_finish"] = (
        circuit_group["finish_position"]
        .transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
    )

    df["driver_circuit_races"] = (
        circuit_group["finish_position"]
        .transform(
            lambda x: x.shift(1).expanding().count()
        )
    )

    return df


def prepare_practice(practice):
    practice = clean_numeric_columns(practice)

    practice = practice.copy()

    pivot = practice.pivot_table(
        index=[
            "season",
            "round",
            "race_name",
            "driver_id",
            "driver_name"
        ],
        columns="session_type",
        values=[
            "position",
            "fastest_lap_rank",
            "fastest_lap_time",
            "average_lap_time",
            "top_speed",
            "laps_completed"
        ],
        aggfunc="first"
    )

    pivot.columns = [
        f"{session.lower()}_{metric}"
        for metric, session in pivot.columns
    ]

    pivot = pivot.reset_index()

    return pivot


def add_practice_features(df, practice):
    practice_features = prepare_practice(practice)

    merge_columns = [
        "season",
        "round",
        "driver_id"
    ]

    df = df.merge(
        practice_features,
        on=merge_columns,
        how="left"
    )

    return df


def add_qualifying_derived_features(df):
    if "qualifying_position" in df.columns:
        df["qualifying_position"] = pd.to_numeric(
            df["qualifying_position"],
            errors="coerce"
        )

        df["qualifying_position_valid"] = (
            df["qualifying_position"].notna().astype(int)
        )

    return df


def remove_future_information(df):
    forbidden = [
        "finish_position",
        "points",
        "laps",
        "status"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in forbidden
    ]

    return df[feature_columns]


def main():
    print("Loading datasets...")

    race, qualifying, practice = load_data()

    print(f"Race data: {race.shape}")
    print(f"Qualifying data: {qualifying.shape}")
    print(f"Practice data: {practice.shape}")

    print("\nBuilding race base...")

    df = build_race_base(
        race,
        qualifying
    )

    print(f"After race + qualifying merge: {df.shape}")

    print("\nAdding historical driver/team features...")

    df = add_historical_features(df)

    print("\nAdding circuit-specific features...")

    df = add_circuit_features(df)

    print("\nAdding practice features...")

    df = add_practice_features(
        df,
        practice
    )

    print("\nAdding qualifying features...")

    df = add_qualifying_derived_features(df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved feature dataset to:")
    print(OUTPUT_FILE)

    print("\nFinal shape:")
    print(df.shape)

    print("\nColumns:")

    for column in df.columns:
        print(column)

    print("\nSample:")

    display_columns = [
        "season",
        "round",
        "race_name",
        "driver_name",
        "team_name",
        "qualifying_position",
        "fp1_position",
        "fp2_position",
        "fp3_position",
        "driver_avg_finish_last_3",
        "driver_avg_finish_last_5",
        "team_avg_finish_last_5",
        "driver_circuit_avg_finish",
        "finish_position"
    ]

    display_columns = [
        c for c in display_columns
        if c in df.columns
    ]

    print(
        df[display_columns]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()