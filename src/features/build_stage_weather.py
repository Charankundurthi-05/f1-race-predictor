from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset_with_sprint.csv"
)

WEATHER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stage_weather_features.csv"
)


def load_data():
    print("=" * 70)
    print("BUILDING STAGE-SAFE WEATHER FEATURES")
    print("=" * 70)

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Master dataset not found:\n{MASTER_FILE}"
        )

    if not WEATHER_FILE.exists():
        raise FileNotFoundError(
            f"Weather feature file not found:\n{WEATHER_FILE}"
        )

    master = pd.read_csv(MASTER_FILE)
    weather = pd.read_csv(WEATHER_FILE)

    print(
        f"\nMaster dataset shape: {master.shape}"
    )

    print(
        f"Weather dataset shape: {weather.shape}"
    )

    return master, weather


def prepare_keys(master, weather):
    for dataframe in [master, weather]:
        dataframe["season"] = pd.to_numeric(
            dataframe["season"],
            errors="coerce"
        )

        dataframe["round"] = pd.to_numeric(
            dataframe["round"],
            errors="coerce"
        )

        dataframe["circuit_id"] = (
            dataframe["circuit_id"]
            .astype(str)
        )

    return master, weather


def validate_weather_keys(weather):
    duplicates = weather[
        weather.duplicated(
            [
                "season",
                "round",
                "circuit_id"
            ],
            keep=False
        )
    ]

    if len(duplicates) > 0:
        print(
            "\nERROR: Duplicate weather rows found:"
        )

        print(
            duplicates[
                [
                    "season",
                    "round",
                    "circuit_id"
                ]
            ]
            .sort_values(
                [
                    "season",
                    "round",
                    "circuit_id"
                ]
            )
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "Weather dataset contains duplicate "
            "race keys."
        )


def get_weather_columns(weather):
    excluded_columns = {
        "season",
        "round",
        "race_name",
        "race_date",
        "circuit_id",
        "circuit_name",
        "weather_observations"
    }

    return [
        column
        for column in weather.columns
        if column not in excluded_columns
    ]


def build_stage_weather(master, weather):
    weather_columns = get_weather_columns(
        weather
    )

    print(
        f"\nWeather feature columns available: "
        f"{len(weather_columns)}"
    )

    print(
        "\nWeather columns:"
    )

    for column in weather_columns:
        print(
            f"  {column}"
        )

    # --------------------------------------------------------------
    # IMPORTANT:
    #
    # We intentionally separate weather information by when it
    # becomes available.
    #
    # Historical/known weather aggregates:
    # - weekend_*       -> NOT safe for early prediction
    # - fp1_period_*    -> safe only after FP1
    # - race_day_*      -> NOT safe for early prediction
    #
    # Since the current weather file contains historical observed
    # weather, we do NOT pretend that future weather was known.
    #
    # For this stage we only expose FP1-period observations after
    # FP1. The remaining observed-weather columns are retained in
    # the source weather file but never attached to early stages.
    # --------------------------------------------------------------

    fp1_columns = [
        column
        for column in weather_columns
        if column.startswith("fp1_period_")
    ]

    print(
        f"\nFP1-safe weather columns: "
        f"{len(fp1_columns)}"
    )

    # Rename them so their meaning is explicit.
    rename_map = {
        column: f"weather_{column}"
        for column in fp1_columns
    }

    weather_stage = weather[
        [
            "season",
            "round",
            "circuit_id"
        ] + fp1_columns
    ].copy()

    weather_stage = weather_stage.rename(
        columns=rename_map
    )

    return weather_stage


def main():
    master, weather = load_data()

    master, weather = prepare_keys(
        master,
        weather
    )

    validate_weather_keys(
        weather
    )

    weather_stage = build_stage_weather(
        master,
        weather
    )

    output_columns = [
        "season",
        "round",
        "circuit_id"
    ] + [
        column
        for column in weather_stage.columns
        if column not in [
            "season",
            "round",
            "circuit_id"
        ]
    ]

    weather_stage = weather_stage[
        output_columns
    ]

    print(
        "\nStage weather dataset shape:"
        f" {weather_stage.shape}"
    )

    print(
        "\nWeather stage preview:"
    )

    print(
        weather_stage
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nWeather missingness:"
    )

    missingness = (
        weather_stage
        .isna()
        .sum()
    )

    print(
        missingness.to_string()
    )

    # Verify that every weather row represents a race
    # represented in the master dataset.
    master_keys = master[
        [
            "season",
            "round",
            "circuit_id"
        ]
    ].drop_duplicates()

    weather_keys = weather_stage[
        [
            "season",
            "round",
            "circuit_id"
        ]
    ]

    unmatched = weather_stage.merge(
        master_keys,
        on=[
            "season",
            "round",
            "circuit_id"
        ],
        how="left",
        indicator=True
    )

    unmatched = unmatched[
        unmatched["_merge"] == "left_only"
    ]

    print(
        f"\nWeather races not present in master: "
        f"{len(unmatched)}"
    )

    if len(unmatched) > 0:
        raise ValueError(
            "Weather contains races that are not "
            "present in the master dataset."
        )

    # Verify no duplicate weather keys.
    if weather_stage.duplicated(
        [
            "season",
            "round",
            "circuit_id"
        ]
    ).any():
        raise ValueError(
            "Duplicate race keys remain in "
            "stage weather dataset."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    weather_stage.to_csv(
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
        "STAGE-SAFE WEATHER BUILD COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()