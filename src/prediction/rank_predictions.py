from pathlib import Path
import joblib
import pandas as pd
import numpy as np


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


def rank_predictions(df):
    df = df.copy()

    df = df.sort_values(
        ["predicted_finish", "driver_name"]
    ).reset_index(drop=True)

    df["predicted_position"] = (
        np.arange(len(df)) + 1
    )

    return df


def main():
    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("FULL GRID PREDICTION ENGINE")
    print("=" * 80)

    for stage in STAGES:
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

        model_name = BEST_MODELS[stage]

        model_path = (
            MODEL_DIR
            / f"{stage}_{model_name}_final_model.joblib"
        )

        print(
            f"Loading model: {model_path.name}"
        )

        model = joblib.load(model_path)

        metadata_path = (
            MODEL_DIR
            / f"{stage}_{model_name}_final_metadata.json"
        )

        metadata = pd.read_json(
            metadata_path,
            typ="series"
        )

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

        df["predicted_finish"] = (
            model.predict(X)
        )

        result_groups = []

        for _, race in df.groupby(
            ["season", "round", "race_date"],
            sort=True
        ):
            ranked = rank_predictions(race)
            result_groups.append(ranked)

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
            "predicted_finish",
            "predicted_position",
        ]

        output_columns = [
            c for c in output_columns
            if c in result.columns
        ]

        output = result[
            output_columns
        ].copy()

        output_path = (
            OUTPUT_DIR
            / f"{stage}_predictions.csv"
        )

        output.to_csv(
            output_path,
            index=False
        )

        duplicate_check = (
            output.groupby(
                [
                    "season",
                    "round",
                    "race_date",
                ]
            )["predicted_position"]
            .apply(
                lambda x:
                x.duplicated().any()
            )
            .any()
        )

        if duplicate_check:
            raise ValueError(
                f"{stage}: duplicate predicted positions detected"
            )

        print(
            f"Races: "
            f"{output[['season', 'round']].drop_duplicates().shape[0]}"
        )

        print(
            f"Drivers: {len(output)}"
        )

        print(
            f"Saved: {output_path}"
        )

    print()
    print("=" * 80)
    print("FULL GRID PREDICTION ENGINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()