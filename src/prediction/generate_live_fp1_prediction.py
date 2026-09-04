from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = ROOT / "data" / "predictions"
MODEL_DIR = ROOT / "models"

MODEL_PATH = (
    MODEL_DIR
    / "fp1_rf_small_final_model.joblib"
)

METADATA_PATH = (
    MODEL_DIR
    / "fp1_rf_small_final_metadata.json"
)

INPUT_PATH = (
    PREDICTION_DIR
    / "current_2026_fp1_features.csv"
)

OUTPUT_PATH = (
    PREDICTION_DIR
    / "current_2026_fp1_prediction.csv"
)

STANDARD_OUTPUT = (
    PREDICTION_DIR
    / "current_prediction.csv"
)

POINTS_OUTPUT = (
    PREDICTION_DIR
    / "current_prediction_with_points.csv"
)


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
    print("CURRENT FP1 RACE PREDICTION")
    print("=" * 80)
    print()

    if not INPUT_PATH.exists():
        print("No live FP1 feature file exists yet.")
        print("Run build_live_fp1_features.py after FP1.")
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    features = metadata["features"]

    print(f"Input rows: {len(df)}")
    print(f"Model features: {len(features)}")
    print()

    # ------------------------------------------------------------
    # Validate structure
    # ------------------------------------------------------------

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
            "Driver uniqueness check failed."
        )

    # IMPORTANT:
    # The FP1 feature builder creates fp1_position.
    #
    # We do NOT require all 22 drivers to have an FP1 result.
    # Drivers who did not participate in FP1 retain NaN and are
    # handled by the trained preprocessing pipeline.
    if "fp1_position" not in df.columns:
        raise ValueError(
            "FP1 model feature 'fp1_position' is missing."
        )

    fp1_count = int(
        df["fp1_position"].notna().sum()
    )

    if fp1_count == 0:
        raise ValueError(
            "No FP1 results are available."
        )

    print(
        f"FP1 positions available: "
        f"{fp1_count}/22"
    )

    if fp1_count < 22:
        print(
            "Some drivers did not participate in FP1; "
            "missing FP1 values will be handled by the model."
        )

    # ------------------------------------------------------------
    # Validate exact features
    # ------------------------------------------------------------

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    model = joblib.load(MODEL_PATH)

    X = df[features].copy()

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------

    predictions = model.predict(X)

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    result = df.copy()

    # Add all prediction columns together to avoid unnecessary
    # DataFrame fragmentation warnings.
    result["predicted_finish_raw"] = predictions

    result = result.sort_values(
        "predicted_finish_raw",
        ascending=True,
    ).reset_index(
        drop=True
    )

    result["predicted_position"] = (
        np.arange(
            1,
            len(result) + 1,
        )
    )

    result["predicted_race_points"] = (
        result["predicted_position"]
        .map(RACE_POINTS)
        .fillna(0)
        .astype(int)
    )

    result["predicted_sprint_points"] = 0

    result["predicted_total_points"] = (
        result["predicted_race_points"]
        + result["predicted_sprint_points"]
    )

    # ------------------------------------------------------------
    # Display
    # ------------------------------------------------------------

    race = result.iloc[0]

    print()
    print(
        f"Race: {race['race_name']}"
    )
    print(
        f"Round: {int(race['round'])}"
    )
    print(
        f"Circuit: {race['circuit_name']}"
    )

    print()
    print("PREDICTED RACE GRID AFTER FP1")
    print("-" * 80)

    for _, row in result.iterrows():

        print(
            f"{int(row['predicted_position']):2d}. "
            f"{row['driver_name']:<30} "
            f"{row['team_name']:<22} "
            f"FP1: "
            f"{str(row['fp1_position']):>4}  "
            f"{int(row['predicted_race_points']):2d} pts"
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    result.to_csv(
        STANDARD_OUTPUT,
        index=False,
    )

    result.to_csv(
        POINTS_OUTPUT,
        index=False,
    )

    print()
    print(f"Saved -> {OUTPUT_PATH}")
    print(f"Standard output -> {STANDARD_OUTPUT}")
    print(f"Points output -> {POINTS_OUTPUT}")

    print()
    print("=" * 80)
    print("FP1 PREDICTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()