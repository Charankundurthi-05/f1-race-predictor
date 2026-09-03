from pathlib import Path
import json
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


BEST_MODELS = {
    "pre_practice": "gb_shallow",
    "fp1": "rf_small",
    "fp2": "extra_trees",
    "fp3": "rf_small",
    "qualifying": "gb_shallow",
}


STAGES = [
    "pre_practice",
    "fp1",
    "fp2",
    "fp3",
    "qualifying",
]


def load_model(stage):
    model_name = BEST_MODELS[stage]

    model_path = (
        MODEL_DIR
        / f"{stage}_{model_name}_final_model.joblib"
    )

    metadata_path = (
        MODEL_DIR
        / f"{stage}_{model_name}_final_metadata.json"
    )

    model = joblib.load(model_path)

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    return model, metadata


def generate_predictions(stage):
    print()
    print("=" * 80)
    print(f"GENERATING {stage.upper()} PREDICTIONS")
    print("=" * 80)

    data_path = (
        DATA_DIR
        / f"stage_{stage}.csv"
    )

    df = pd.read_csv(data_path)

    if (
        stage != "pre_practice"
        and "stage_available" in df.columns
    ):
        df = df[
            df["stage_available"] == 1
        ].copy()

    model, metadata = load_model(stage)

    features = metadata["features"]

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{stage}: missing features {missing}"
        )

    X = df[features].copy()

    df["model_prediction"] = model.predict(X)

    result_groups = []

    for _, race in df.groupby(
        [
            "season",
            "round",
            "race_date",
        ],
        sort=True
    ):
        race = race.copy()

        race = race.sort_values(
            [
                "model_prediction",
                "driver_name",
            ]
        ).reset_index(drop=True)

        race["predicted_position"] = (
            range(1, len(race) + 1)
        )

        result_groups.append(race)

    result = pd.concat(
        result_groups,
        ignore_index=True
    )

    result = result.sort_values(
        [
            "season",
            "round",
            "predicted_position",
        ]
    )

    output_columns = [
        "season",
        "round",
        "race_name",
        "race_date",
        "circuit_name",
        "driver_name",
        "driver_code",
        "team_name",
        "model_prediction",
        "predicted_position",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in result.columns
    ]

    result = result[output_columns]

    output_path = (
        OUTPUT_DIR
        / f"{stage}_live_predictions.csv"
    )

    result.to_csv(
        output_path,
        index=False
    )

    print(
        f"Races: "
        f"{result[['season', 'round']].drop_duplicates().shape[0]}"
    )

    print(
        f"Drivers: {len(result)}"
    )

    print(
        f"Saved: {output_path}"
    )


def main():
    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("PREDICTION PIPELINE")
    print("=" * 80)

    for stage in STAGES:
        generate_predictions(stage)

    print()
    print("=" * 80)
    print("PREDICTION PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()