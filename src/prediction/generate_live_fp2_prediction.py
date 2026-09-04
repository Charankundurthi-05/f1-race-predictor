from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = ROOT / "data" / "predictions"
MODEL_DIR = ROOT / "models"

INPUT_FILE = PREDICTION_DIR / "current_2026_fp2_features.csv"

MODEL_FILE = MODEL_DIR / "fp2_extra_trees_final_model.joblib"
METADATA_FILE = MODEL_DIR / "fp2_extra_trees_final_metadata.json"

OUTPUT_FILE = PREDICTION_DIR / "current_2026_fp2_prediction.csv"
CURRENT_OUTPUT_FILE = PREDICTION_DIR / "current_prediction.csv"


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


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("CURRENT FP2 RACE PREDICTION")
    print("=" * 80)
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"FP2 feature file does not exist:\n{INPUT_FILE}\n"
            "Run build_live_fp2_features.py first."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"FP2 model does not exist:\n{MODEL_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"FP2 model metadata does not exist:\n{METADATA_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_features = metadata["features"]

    print(f"Input rows: {len(df)}")
    print(f"Model features: {len(model_features)}")
    print()

    # ------------------------------------------------------------------
    # Validate current race
    # ------------------------------------------------------------------

    if len(df) != 22:
        raise ValueError(
            f"Expected 22 drivers, found {len(df)}."
        )

    if "driver_name" not in df.columns:
        raise ValueError(
            "driver_name column is missing."
        )

    if df["driver_name"].nunique() != 22:
        raise ValueError(
            "Driver uniqueness validation failed."
        )

    # IMPORTANT:
    # The FP2 feature builder creates fp2_position.
    # Do not use practice_2_position here.
    if "fp2_position" not in df.columns:
        raise ValueError(
            "FP2 model feature column 'fp2_position' is missing."
        )

    if df["fp2_position"].notna().sum() != 22:
        raise ValueError(
            "FP2 practice positions are incomplete. "
            f"Found {df['fp2_position'].notna().sum()}/22."
        )

    # ------------------------------------------------------------------
    # Validate model features
    # ------------------------------------------------------------------

    missing_features = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required FP2 model features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    # Exact training feature order
    X = df[model_features].copy()

    print("FP2 practice position validation: PASS")
    print(
        f"FP2 positions available: "
        f"{df['fp2_position'].notna().sum()}/22"
    )
    print()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    model = joblib.load(MODEL_FILE)

    # ------------------------------------------------------------------
    # Predict finishing position
    # ------------------------------------------------------------------

    predictions = model.predict(X)

    predictions = np.asarray(predictions, dtype=float)

    # Smaller predicted finish position = better result.
    df["predicted_finish_raw"] = predictions

    df = df.sort_values(
        "predicted_finish_raw",
        ascending=True
    ).reset_index(drop=True)

    df["predicted_position"] = (
        np.arange(1, len(df) + 1)
    )

    # ------------------------------------------------------------------
    # Calculate predicted race points
    # ------------------------------------------------------------------

    df["predicted_race_points"] = (
        df["predicted_position"]
        .map(RACE_POINTS)
        .fillna(0)
        .astype(int)
    )

    # Normal weekend currently being processed.
    df["predicted_sprint_points"] = 0

    df["predicted_total_points"] = (
        df["predicted_race_points"]
        + df["predicted_sprint_points"]
    )

    # ------------------------------------------------------------------
    # Save detailed FP2 prediction
    # ------------------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ------------------------------------------------------------------
    # Save the compact file consumed by the dashboard
    # ------------------------------------------------------------------

    current_columns = [
        column
        for column in [
            "season",
            "round",
            "race_name",
            "race_date",
            "circuit_name",
            "driver_name",
            "team_name",
            "driver_code",
            "fp1_position",
            "fp2_position",
            "predicted_position",
            "predicted_race_points",
            "predicted_sprint_points",
            "predicted_total_points",
            "predicted_finish_raw",
        ]
        if column in df.columns
    ]

    current_prediction = df[current_columns].copy()

    current_prediction.to_csv(
        CURRENT_OUTPUT_FILE,
        index=False
    )

    # ------------------------------------------------------------------
    # Display prediction
    # ------------------------------------------------------------------

    print("PREDICTED GRID AFTER FP2")
    print("-" * 80)

    for _, row in df.iterrows():

        driver = str(row["driver_name"])
        team = str(row["team_name"])

        position = int(row["predicted_position"])
        fp2_position = int(row["fp2_position"])
        points = int(row["predicted_race_points"])

        print(
            f"{position:2d}. "
            f"{driver:<30} "
            f"{team:<22} "
            f"FP2: {fp2_position:2d}  "
            f"Pts: {points:2d}"
        )

    print()
    print(f"Saved -> {OUTPUT_FILE}")
    print(f"Saved -> {CURRENT_OUTPUT_FILE}")
    print()
    print("=" * 80)
    print("CURRENT FP2 PREDICTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()