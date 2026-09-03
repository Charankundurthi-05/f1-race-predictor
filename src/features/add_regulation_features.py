from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset_with_circuit.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset_with_regulation.csv"
)


def assign_regulation_era(season):
    if 2014 <= season <= 2021:
        return "turbo_hybrid"

    if 2022 <= season <= 2025:
        return "ground_effect"

    if season >= 2026:
        return "new_2026"

    return "unknown"


def assign_regulation_era_code(era):
    codes = {
        "turbo_hybrid": 1,
        "ground_effect": 2,
        "new_2026": 3,
        "unknown": 0,
    }

    return codes.get(era, 0)


def main():

    print("=" * 80)
    print("ADDING REGULATION ERA FEATURES")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{INPUT_PATH}"
        )

    print(f"\nReading: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print(f"Input shape: {df.shape}")

    if "season" not in df.columns:
        raise ValueError(
            "Dataset does not contain the 'season' column."
        )

    df["season"] = pd.to_numeric(
        df["season"],
        errors="coerce",
    )

    if df["season"].isna().any():
        raise ValueError(
            "Season column contains missing or invalid values."
        )

    df["regulation_era"] = df["season"].apply(
        assign_regulation_era
    )

    df["regulation_era_code"] = df["regulation_era"].apply(
        assign_regulation_era_code
    )

    print("\nRegulation era distribution:")

    print(
        df["regulation_era"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSeason → regulation era:")

    season_era = (
        df[
            [
                "season",
                "regulation_era",
                "regulation_era_code",
            ]
        ]
        .drop_duplicates()
        .sort_values("season")
    )

    print(
        season_era.to_string(index=False)
    )

    print("\nFeature availability:")

    for column in [
        "regulation_era",
        "regulation_era_code",
    ]:

        available = df[column].notna().sum()
        total = len(df)

        percentage = (
            available / total * 100
            if total > 0
            else 0
        )

        print(
            f"{column:30s} "
            f"{available:5d} / {total:5d} "
            f"({percentage:6.2f}%)"
        )

    if (
        df["regulation_era"].eq("unknown").any()
    ):
        unknown_seasons = sorted(
            df.loc[
                df["regulation_era"] == "unknown",
                "season",
            ].unique()
        )

        raise ValueError(
            f"Unknown regulation era for seasons: "
            f"{unknown_seasons}"
        )

    print("\nChecking regulation-era consistency...")

    consistency = (
        df.groupby("season", dropna=False)["regulation_era"]
        .nunique()
    )

    inconsistent = consistency[
        consistency > 1
    ]

    if not inconsistent.empty:
        raise ValueError(
            "A season has multiple regulation eras:\n"
            f"{inconsistent.to_string()}"
        )

    print(
        "Every season maps to exactly one regulation era."
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print(
        f"Final shape: {df.shape}"
    )

    print("\n" + "=" * 80)
    print("REGULATION ERA FEATURES ADDED")
    print("=" * 80)


if __name__ == "__main__":
    main()