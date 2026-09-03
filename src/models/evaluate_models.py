from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

STAGES = ["pre_practice", "fp1", "fp2", "fp3", "qualifying"]
TARGET = "finish_position"


def load_model(stage, model_name):
    return joblib.load(
        MODEL_DIR / f"{stage}_{model_name}_model.joblib"
    )


def load_metadata(stage, model_name):
    with open(
        MODEL_DIR / f"{stage}_{model_name}_metadata.json",
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_stage_data(stage):
    return pd.read_csv(
        DATA_DIR / f"stage_{stage}.csv"
    )


def predict_with_saved_model(model, metadata, df):
    features = metadata["features"]

    missing = [
        c for c in features
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    return model.predict(df[features].copy())


def race_key_columns(df):
    if all(
        c in df.columns
        for c in ["season", "round", "race_date"]
    ):
        return ["season", "round", "race_date"]

    return ["season", "round"]


def calculate_race_metrics(race_df):
    race_df = race_df[
        race_df[TARGET].notna()
    ].copy()

    if race_df.empty:
        return None

    race_df = race_df.sort_values(
        ["predicted_finish", "driver_name"]
    ).reset_index(drop=True)

    race_df["predicted_rank"] = (
        race_df.index + 1
    )

    actual_order = (
        race_df.sort_values(
            [TARGET, "driver_name"]
        )
        .reset_index(drop=True)
    )

    predicted_order = race_df.copy()

    predicted_names = predicted_order[
        "driver_name"
    ].tolist()

    actual_names = actual_order[
        "driver_name"
    ].tolist()

    grid_size = len(race_df)

    predicted_rank_map = {
        name: i + 1
        for i, name in enumerate(
            predicted_names
        )
    }

    actual_rank_map = {
        name: i + 1
        for i, name in enumerate(
            actual_names
        )
    }

    common_drivers = set(
        predicted_rank_map
    ) & set(actual_rank_map)

    rank_errors = [
        abs(
            predicted_rank_map[name]
            - actual_rank_map[name]
        )
        for name in common_drivers
    ]

    predicted_top3 = set(
        predicted_names[:3]
    )

    actual_top3 = set(
        actual_names[:3]
    )

    predicted_top5 = set(
        predicted_names[:5]
    )

    actual_top5 = set(
        actual_names[:5]
    )

    predicted_top10 = set(
        predicted_names[:10]
    )

    actual_top10 = set(
        actual_names[:10]
    )

    winner_correct = int(
        predicted_names[0]
        == actual_names[0]
    )

    top3_overlap = (
        len(
            predicted_top3
            & actual_top3
        )
        / min(3, grid_size)
    )

    top5_overlap = (
        len(
            predicted_top5
            & actual_top5
        )
        / min(5, grid_size)
    )

    top10_overlap = (
        len(
            predicted_top10
            & actual_top10
        )
        / min(10, grid_size)
    )

    podium_exact = int(
        predicted_top3
        == actual_top3
    )

    exact_position_accuracy = (
        np.mean(
            [
                predicted_rank_map[name]
                == actual_rank_map[name]
                for name in common_drivers
            ]
        )
        if common_drivers
        else 0.0
    )

    predicted_ranks = [
        predicted_rank_map[name]
        for name in common_drivers
    ]

    actual_ranks = [
        actual_rank_map[name]
        for name in common_drivers
    ]

    spearman = pd.Series(
        predicted_ranks
    ).corr(
        pd.Series(actual_ranks),
        method="spearman",
    )

    raw_mae = np.mean(
        np.abs(
            race_df["predicted_finish"]
            - race_df[TARGET]
        )
    )

    return {
        "mae_raw_prediction": float(
            raw_mae
        ),
        "race_position_mae": float(
            np.mean(rank_errors)
        ),
        "winner_correct": winner_correct,
        "top3_overlap": float(
            top3_overlap
        ),
        "top5_overlap": float(
            top5_overlap
        ),
        "top10_overlap": float(
            top10_overlap
        ),
        "podium_exact": podium_exact,
        "exact_position_accuracy": float(
            exact_position_accuracy
        ),
        "spearman_rank_correlation": (
            float(spearman)
            if pd.notna(spearman)
            else 0.0
        ),
        "grid_size": grid_size,
    }


def calculate_overall_metrics(predictions_df):
    key_columns = race_key_columns(
        predictions_df
    )

    race_results = []

    for keys, race_df in predictions_df.groupby(
        key_columns,
        sort=True,
    ):
        metrics = calculate_race_metrics(
            race_df
        )

        if metrics is None:
            continue

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(
            zip(key_columns, keys)
        )

        row.update(metrics)

        race_results.append(row)

    race_metrics = pd.DataFrame(
        race_results
    )

    if race_metrics.empty:
        raise ValueError(
            "No valid race-level results found."
        )

    overall = {
        "races": len(race_metrics),
        "mae_raw_prediction":
            race_metrics[
                "mae_raw_prediction"
            ].mean(),
        "race_position_mae":
            race_metrics[
                "race_position_mae"
            ].mean(),
        "winner_accuracy":
            race_metrics[
                "winner_correct"
            ].mean(),
        "top3_hit_rate":
            race_metrics[
                "top3_overlap"
            ].mean(),
        "top5_hit_rate":
            race_metrics[
                "top5_overlap"
            ].mean(),
        "top10_hit_rate":
            race_metrics[
                "top10_overlap"
            ].mean(),
        "exact_podium_rate":
            race_metrics[
                "podium_exact"
            ].mean(),
        "exact_position_accuracy":
            race_metrics[
                "exact_position_accuracy"
            ].mean(),
        "average_spearman":
            race_metrics[
                "spearman_rank_correlation"
            ].mean(),
        "average_grid_size":
            race_metrics[
                "grid_size"
            ].mean(),
    }

    return overall, race_metrics


def evaluate_stage(
    stage,
    model_name,
    season,
):
    print()
    print("=" * 80)
    print(
        f"EVALUATING {stage.upper()} | "
        f"{model_name.upper()} | {season}"
    )
    print("=" * 80)

    model = load_model(
        stage,
        model_name,
    )

    metadata = load_metadata(
        stage,
        model_name,
    )

    df = load_stage_data(stage)

    df = df[
        df["season"] == season
    ].copy()

    if (
        "stage_available" in df.columns
        and stage != "pre_practice"
    ):
        df = df[
            df["stage_available"] == 1
        ].copy()

    df = df[
        df[TARGET].notna()
    ].copy()

    print(f"Rows: {len(df)}")

    df["predicted_finish"] = (
        predict_with_saved_model(
            model,
            metadata,
            df,
        )
    )

    overall, race_metrics = (
        calculate_overall_metrics(df)
    )

    print()
    print("RACE-LEVEL METRICS")
    print("-" * 80)

    print(
        f"Races:                 "
        f"{overall['races']}"
    )
    print(
        f"Raw prediction MAE:    "
        f"{overall['mae_raw_prediction']:.3f}"
    )
    print(
        f"Race position MAE:     "
        f"{overall['race_position_mae']:.3f}"
    )
    print(
        f"Winner accuracy:       "
        f"{overall['winner_accuracy']:.1%}"
    )
    print(
        f"Top-3 hit rate:        "
        f"{overall['top3_hit_rate']:.1%}"
    )
    print(
        f"Top-5 hit rate:        "
        f"{overall['top5_hit_rate']:.1%}"
    )
    print(
        f"Top-10 hit rate:       "
        f"{overall['top10_hit_rate']:.1%}"
    )
    print(
        f"Exact podium rate:     "
        f"{overall['exact_podium_rate']:.1%}"
    )
    print(
        f"Exact position rate:   "
        f"{overall['exact_position_accuracy']:.1%}"
    )
    print(
        f"Average Spearman:      "
        f"{overall['average_spearman']:.3f}"
    )
    print(
        f"Average grid size:     "
        f"{overall['average_grid_size']:.1f}"
    )

    race_metrics.to_csv(
        MODEL_DIR
        / (
            f"{stage}_{model_name}_"
            f"{season}_race_metrics.csv"
        ),
        index=False,
    )

    df.to_csv(
        MODEL_DIR
        / (
            f"{stage}_{model_name}_"
            f"{season}_evaluated_predictions.csv"
        ),
        index=False,
    )

    return overall


def main():
    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("CORRECTED RACE-LEVEL MODEL EVALUATION")
    print("=" * 80)

    all_results = []

    for stage in STAGES:
        for model_name in [
            "random_forest",
            "gradient_boosting",
        ]:
            for season in [2025, 2026]:

                metrics = evaluate_stage(
                    stage,
                    model_name,
                    season,
                )

                row = {
                    "stage": stage,
                    "model": model_name,
                    "season": season,
                }

                row.update(metrics)
                all_results.append(row)

    results_df = pd.DataFrame(
        all_results
    )

    output_path = (
        MODEL_DIR
        / "corrected_race_level_evaluation.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL RACE-LEVEL COMPARISON")
    print("=" * 80)

    columns = [
        "stage",
        "model",
        "season",
        "race_position_mae",
        "winner_accuracy",
        "top3_hit_rate",
        "top5_hit_rate",
        "top10_hit_rate",
        "exact_podium_rate",
        "exact_position_accuracy",
        "average_spearman",
    ]

    print(
        results_df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print(
        f"Saved comparison: {output_path}"
    )

    print()
    print("=" * 80)
    print("RACE-LEVEL EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()