from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PREDICTION_DIR = ROOT / "data" / "predictions"


RACE_POINTS = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}

SPRINT_POINTS = {
    1: 8,
    2: 7,
    3: 6,
    4: 5,
    5: 4,
    6: 3,
    7: 2,
    8: 1,
}


def race_points(position):
    return RACE_POINTS.get(int(position), 0)


def sprint_points(position):
    return SPRINT_POINTS.get(int(position), 0)


def calculate_file(input_path, output_path):
    df = pd.read_csv(input_path)

    if "predicted_position" not in df.columns:
        raise ValueError(
            f"'predicted_position' not found in {input_path}"
        )

    df["predicted_race_points"] = (
        df["predicted_position"].apply(race_points)
    )

    if "predicted_sprint_position" in df.columns:
        df["predicted_sprint_points"] = (
            df["predicted_sprint_position"].apply(sprint_points)
        )
    else:
        df["predicted_sprint_points"] = 0

    df["predicted_total_points"] = (
        df["predicted_race_points"]
        + df["predicted_sprint_points"]
    )

    df.to_csv(output_path, index=False)

    print(f"Rows: {len(df)}")
    print(f"Saved: {output_path}")


def calculate_current_prediction():
    input_path = PREDICTION_DIR / "current_prediction.csv"

    if not input_path.exists():
        print("Current prediction does not exist.")
        print("Run generate_current_prediction.py first.")
        return

    df = pd.read_csv(input_path)

    df["predicted_race_points"] = (
        df["predicted_position"].apply(race_points)
    )

    df["predicted_sprint_points"] = 0

    df["predicted_total_points"] = (
        df["predicted_race_points"]
        + df["predicted_sprint_points"]
    )

    output_path = (
        PREDICTION_DIR / "current_prediction_with_points.csv"
    )

    df.to_csv(output_path, index=False)

    print()
    print("CURRENT RACE PREDICTED POINTS")
    print("-" * 80)

    display_columns = [
        "predicted_position",
        "driver_name",
        "team_name",
        "predicted_race_points",
        "predicted_total_points",
    ]

    print(
        df[display_columns].to_string(index=False)
    )

    print()
    print(f"Saved -> {output_path}")


def main():
    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("POINTS CALCULATION ENGINE")
    print("=" * 80)

    stage_files = {
        "PRE_PRACTICE": (
            "pre_practice_predictions.csv",
            "pre_practice_predictions_with_points.csv",
        ),
        "FP1": (
            "fp1_predictions.csv",
            "fp1_predictions_with_points.csv",
        ),
        "FP2": (
            "fp2_predictions.csv",
            "fp2_predictions_with_points.csv",
        ),
        "FP3": (
            "fp3_predictions.csv",
            "fp3_predictions_with_points.csv",
        ),
        "QUALIFYING": (
            "qualifying_predictions.csv",
            "qualifying_predictions_with_points.csv",
        ),
    }

    for stage, (input_file, output_file) in stage_files.items():
        input_path = PREDICTION_DIR / input_file
        output_path = PREDICTION_DIR / output_file

        if not input_path.exists():
            print()
            print(f"Skipping {stage}: file not found.")
            continue

        print()
        print("=" * 80)
        print(f"CALCULATING {stage} POINTS")
        print("=" * 80)

        calculate_file(input_path, output_path)

    print()
    print("=" * 80)
    print("CALCULATING CURRENT 2026 RACE POINTS")
    print("=" * 80)

    calculate_current_prediction()

    print()
    print("=" * 80)
    print("POINTS CALCULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()