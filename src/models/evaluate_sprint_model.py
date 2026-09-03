from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
PREDICTION_DIR = ROOT / "data" / "predictions"


def evaluate(path, year):
    df = pd.read_csv(path)

    actual = df["sprint_finish_position"]
    predicted = df["predicted_sprint_finish"]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    race_results = []

    for (season, round_number), group in df.groupby(
        ["season", "round"]
    ):
        group = group.copy()

        actual_order = (
            group.sort_values("sprint_finish_position")
            ["driver_name"]
            .tolist()
        )

        predicted_order = (
            group.sort_values("predicted_sprint_finish")
            ["driver_name"]
            .tolist()
        )

        actual_top3 = set(actual_order[:3])
        actual_top5 = set(actual_order[:5])
        actual_top10 = set(actual_order[:10])

        predicted_top3 = set(predicted_order[:3])
        predicted_top5 = set(predicted_order[:5])
        predicted_top10 = set(predicted_order[:10])

        winner_correct = (
            predicted_order[0] == actual_order[0]
        )

        top3_accuracy = len(
            actual_top3 & predicted_top3
        ) / 3

        top5_accuracy = len(
            actual_top5 & predicted_top5
        ) / 5

        top10_accuracy = len(
            actual_top10 & predicted_top10
        ) / 10

        exact_positions = (
            sum(
                a == p
                for a, p in zip(
                    actual_order,
                    predicted_order
                )
            )
            / len(actual_order)
        )

        correlation = spearmanr(
            group["sprint_finish_position"],
            group["predicted_sprint_finish"]
        ).statistic

        race_results.append(
            {
                "season": season,
                "round": round_number,
                "winner_correct": winner_correct,
                "top3_accuracy": top3_accuracy,
                "top5_accuracy": top5_accuracy,
                "top10_accuracy": top10_accuracy,
                "exact_position_accuracy": exact_positions,
                "spearman": correlation,
            }
        )

    race_df = pd.DataFrame(race_results)

    print()
    print("=" * 80)
    print(f"SPRINT MODEL EVALUATION — {year}")
    print("=" * 80)

    print()
    print(f"Rows: {len(df)}")
    print(f"Sprint races: {len(race_df)}")

    print()
    print("POSITION METRICS")
    print("-" * 80)
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    print()
    print("RACE-LEVEL METRICS")
    print("-" * 80)

    print(
        f"Winner accuracy: "
        f"{race_df['winner_correct'].mean() * 100:.1f}%"
    )

    print(
        f"Top-3 accuracy: "
        f"{race_df['top3_accuracy'].mean() * 100:.1f}%"
    )

    print(
        f"Top-5 accuracy: "
        f"{race_df['top5_accuracy'].mean() * 100:.1f}%"
    )

    print(
        f"Top-10 accuracy: "
        f"{race_df['top10_accuracy'].mean() * 100:.1f}%"
    )

    print(
        f"Exact position accuracy: "
        f"{race_df['exact_position_accuracy'].mean() * 100:.1f}%"
    )

    print(
        f"Spearman correlation: "
        f"{race_df['spearman'].mean():.3f}"
    )

    return race_df


def main():

    validation_file = (
        PREDICTION_DIR /
        "sprint_2025_validation_predictions.csv"
    )

    test_file = (
        PREDICTION_DIR /
        "sprint_2026_test_predictions.csv"
    )

    if not validation_file.exists():
        raise FileNotFoundError(
            f"Missing: {validation_file}"
        )

    if not test_file.exists():
        raise FileNotFoundError(
            f"Missing: {test_file}"
        )

    validation_results = evaluate(
        validation_file,
        2025
    )

    test_results = evaluate(
        test_file,
        2026
    )

    print()
    print("=" * 80)
    print("SPRINT MODEL EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()