from pathlib import Path
import json
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models"
PREDICTION_DIR = ROOT / "data" / "predictions"

MODEL_PATHS = {
    "PRE_PRACTICE": MODEL_DIR / "pre_practice_gb_shallow_final_model.joblib",
    "FP1": MODEL_DIR / "fp1_rf_small_final_model.joblib",
    "FP2": MODEL_DIR / "fp2_extra_trees_final_model.joblib",
    "FP3": MODEL_DIR / "fp3_rf_small_final_model.joblib",
    "QUALIFYING": MODEL_DIR / "qualifying_gb_shallow_final_model.joblib",
}

METADATA_PATHS = {
    "PRE_PRACTICE": MODEL_DIR / "pre_practice_gb_shallow_final_metadata.json",
    "FP1": MODEL_DIR / "fp1_rf_small_final_metadata.json",
    "FP2": MODEL_DIR / "fp2_extra_trees_final_metadata.json",
    "FP3": MODEL_DIR / "fp3_rf_small_final_metadata.json",
    "QUALIFYING": MODEL_DIR / "qualifying_gb_shallow_final_metadata.json",
}

STAGE_DATASETS = {
    "PRE_PRACTICE": ROOT / "data" / "processed" / "stage_pre_practice.csv",
    "FP1": ROOT / "data" / "processed" / "stage_fp1.csv",
    "FP2": ROOT / "data" / "processed" / "stage_fp2.csv",
    "FP3": ROOT / "data" / "processed" / "stage_fp3.csv",
    "QUALIFYING": ROOT / "data" / "processed" / "stage_qualifying.csv",
}

CURRENT_PRE_PRACTICE = (
    ROOT / "data" / "predictions" / "current_2026_pre_practice_features.csv"
)


def get_current_stage():
    controller_status = PREDICTION_DIR / "live_controller_status.json"

    if controller_status.exists():
        try:
            with open(controller_status, "r", encoding="utf-8") as f:
                data = json.load(f)

            stage = str(data.get("current_stage", "PRE_PRACTICE")).upper()

            if stage in MODEL_PATHS:
                return stage

        except Exception:
            pass

    return "PRE_PRACTICE"


def load_metadata(stage):
    with open(METADATA_PATHS[stage], "r", encoding="utf-8") as f:
        return json.load(f)


def load_prediction_data(stage):
    if stage == "PRE_PRACTICE" and CURRENT_PRE_PRACTICE.exists():
        print("Using CURRENT 2026 race feature file.")
        print(f"Input -> {CURRENT_PRE_PRACTICE}")
        return pd.read_csv(CURRENT_PRE_PRACTICE)

    path = STAGE_DATASETS[stage]

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if stage == "PRE_PRACTICE":
        latest_season = df["season"].max()
        season_df = df[df["season"] == latest_season].copy()

        latest_round = season_df["round"].max()

        df = season_df[
            season_df["round"] == latest_round
        ].copy()

    return df


def main():
    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("CURRENT FULL-GRID PREDICTION")
    print("=" * 80)
    print()

    stage = get_current_stage()

    print(f"Current stage: {stage}")
    print()

    model_path = MODEL_PATHS[stage]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    metadata = load_metadata(stage)

    feature_columns = metadata["features"]

    df = load_prediction_data(stage)

    print()
    print(f"Generating prediction for: {stage}")
    print(f"Input rows: {len(df)}")
    print(f"Model features: {len(feature_columns)}")
    print()

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required model features:\n"
            + "\n".join(f"  - {x}" for x in missing_features)
        )

    X = df[feature_columns].copy()

    predictions = model.predict(X)

    result = df.copy()
    result["predicted_finish_raw"] = predictions

    result = result.sort_values(
        by="predicted_finish_raw",
        ascending=True
    ).reset_index(drop=True)

    result["predicted_position"] = range(1, len(result) + 1)

    print(
        f"Race: {result.iloc[0]['race_name']}"
    )
    print(
        f"Round: {int(result.iloc[0]['round'])}"
    )

    if "race_date" in result.columns:
        print(
            f"Date: {result.iloc[0]['race_date']}"
        )

    if "circuit_name" in result.columns:
        print(
            f"Circuit: {result.iloc[0]['circuit_name']}"
        )

    print()
    print("PREDICTED GRID")
    print("-" * 80)

    for _, row in result.iterrows():
        driver = row.get("driver_name", "Unknown")
        team = row.get("team_name", "Unknown")
        position = int(row["predicted_position"])

        print(
            f"{position:2d}. {driver:<28} {team}"
        )

    output_path = PREDICTION_DIR / "current_prediction.csv"

    result.to_csv(output_path, index=False)

    print()
    print(f"Saved -> {output_path}")
    print()
    print("=" * 80)
    print("CURRENT PREDICTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()