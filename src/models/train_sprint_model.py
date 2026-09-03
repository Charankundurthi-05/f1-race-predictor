from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "processed" / "stage_qualifying.csv"
MODEL_DIR = ROOT / "models"
PREDICTION_DIR = ROOT / "data" / "predictions"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "season",
    "round",
    "number",

    "driver_previous_finish",
    "driver_avg_finish_last_3",
    "driver_avg_finish_last_5",
    "driver_previous_points",
    "driver_avg_points_last_5",

    "team_avg_finish_last_5",
    "team_avg_points_last_5",

    "driver_circuit_avg_finish",
    "driver_circuit_races",

    "circuit_latitude",
    "circuit_longitude",
    "circuit_length_km",
    "circuit_turns",
    "circuit_total_races_held",

    "circuit_type",
    "circuit_direction",

    "regulation_era",

    "fp1_position",
    "fp2_position",
    "fp3_position",

    "practice_avg_position",
    "practice_best_position",
    "practice_sessions_available",

    "weather_fp1_temperature_avg",
    "weather_fp1_temperature_min",
    "weather_fp1_temperature_max",
    "weather_fp1_humidity_avg",
    "weather_fp1_precipitation_total",
    "weather_fp1_rain_hours",
    "weather_fp1_rain_flag",
    "weather_fp1_wind_speed_avg",
    "weather_fp1_wind_speed_max",
    "weather_fp1_wind_direction_avg",
    "weather_fp1_cloud_cover_avg",
    "weather_fp1_weather_observations",

    "weather_fp2_temperature_avg",
    "weather_fp2_temperature_min",
    "weather_fp2_temperature_max",
    "weather_fp2_humidity_avg",
    "weather_fp2_precipitation_total",
    "weather_fp2_rain_hours",
    "weather_fp2_rain_flag",
    "weather_fp2_wind_speed_avg",
    "weather_fp2_wind_speed_max",
    "weather_fp2_wind_direction_avg",
    "weather_fp2_cloud_cover_avg",
    "weather_fp2_weather_observations",

    "weather_fp3_temperature_avg",
    "weather_fp3_temperature_min",
    "weather_fp3_temperature_max",
    "weather_fp3_humidity_avg",
    "weather_fp3_precipitation_total",
    "weather_fp3_rain_hours",
    "weather_fp3_rain_flag",
    "weather_fp3_wind_speed_avg",
    "weather_fp3_wind_speed_max",
    "weather_fp3_wind_direction_avg",
    "weather_fp3_cloud_cover_avg",
    "weather_fp3_weather_observations",
]

TARGET = "sprint_finish_position"


CATEGORICAL_FEATURES = [
    "circuit_type",
    "circuit_direction",
    "regulation_era",
]

NUMERIC_FEATURES = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_FEATURES
]


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("TRAIN DEDICATED SPRINT MODEL")
    print("=" * 80)
    print()

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"Dataset rows: {len(df)}")
    print(f"Dataset columns: {len(df.columns)}")
    print()

    # ---------------------------------------------------------------
    # Sprint races only
    # ---------------------------------------------------------------

    if "sprint_race_available" not in df.columns:
        raise ValueError(
            "sprint_race_available column is missing."
        )

    sprint_df = df[
        df["sprint_race_available"] == 1
    ].copy()

    print(f"Sprint rows: {len(sprint_df)}")

    if sprint_df.empty:
        raise ValueError(
            "No sprint races found."
        )

    # ---------------------------------------------------------------
    # Target
    # ---------------------------------------------------------------

    if TARGET not in sprint_df.columns:
        raise ValueError(
            f"Target column missing: {TARGET}"
        )

    sprint_df = sprint_df[
        sprint_df[TARGET].notna()
    ].copy()

    print(
        f"Rows with sprint results: {len(sprint_df)}"
    )

    # ---------------------------------------------------------------
    # Feature validation
    # ---------------------------------------------------------------

    missing = [
        feature
        for feature in FEATURES
        if feature not in sprint_df.columns
    ]

    if missing:

        print()
        print("Missing features:")

        for feature in missing:
            print(f"  - {feature}")

        raise ValueError(
            "Required sprint model features are missing."
        )

    # ---------------------------------------------------------------
    # Explicit leakage protection
    # ---------------------------------------------------------------

    forbidden = [
        "sprint_qualifying_position",
        "sprint_qualifying_q1_ms",
        "sprint_qualifying_q2_ms",
        "sprint_qualifying_q3_ms",
        "sprint_qualifying_gap_ms",
        "sprint_qualifying_laps",
        "sprint_finish_position",
        "sprint_grid_position",
        "sprint_positions_gained",
        "sprint_points",
        "sprint_laps",
    ]

    leakage = [
        feature
        for feature in FEATURES
        if feature in forbidden
    ]

    if leakage:

        raise ValueError(
            "SPRINT LEAKAGE DETECTED:\n"
            + "\n".join(leakage)
        )

    print()
    print("SPRINT LEAKAGE CHECK: PASS")

    # ---------------------------------------------------------------
    # Historical split
    # ---------------------------------------------------------------

    train_df = sprint_df[
        sprint_df["season"] <= 2024
    ].copy()

    validation_df = sprint_df[
        sprint_df["season"] == 2025
    ].copy()

    test_df = sprint_df[
        sprint_df["season"] == 2026
    ].copy()

    print()
    print("DATA SPLIT")
    print("-" * 80)
    print(f"Training rows:   {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"2026 test rows:  {len(test_df)}")

    if train_df.empty:
        raise ValueError(
            "Training set is empty."
        )

    # ---------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=2,
                    min_samples_leaf=5,
                    random_state=42
                )
            )
        ]
    )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    print()
    print("Training sprint model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------

    validation_predictions = None
    test_predictions = None

    if not validation_df.empty:

        validation_predictions = model.predict(
            validation_df[FEATURES]
        )

    if not test_df.empty:

        test_predictions = model.predict(
            test_df[FEATURES]
        )

    # ---------------------------------------------------------------
    # Save model
    # ---------------------------------------------------------------

    model_path = (
        MODEL_DIR /
        "sprint_gb_shallow_final_model.joblib"
    )

    metadata_path = (
        MODEL_DIR /
        "sprint_gb_shallow_final_metadata.json"
    )

    joblib.dump(
        model,
        model_path
    )

    metadata = {
        "model_type": "GradientBoostingRegressor",
        "model_name": "sprint_gb_shallow",
        "target": TARGET,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "training_years": "2014-2024",
        "validation_year": 2025,
        "test_year": 2026,
        "sprint_only": True,
        "leakage_protection": True,
        "preprocessing": "median_imputation_numeric_onehot_categorical"
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # ---------------------------------------------------------------
    # Save validation predictions
    # ---------------------------------------------------------------

    if validation_predictions is not None:

        validation_output = validation_df[
            [
                "season",
                "round",
                "race_name",
                "driver_name",
                "team_name",
                TARGET
            ]
        ].copy()

        validation_output[
            "predicted_sprint_finish"
        ] = validation_predictions

        validation_output.to_csv(
            PREDICTION_DIR /
            "sprint_2025_validation_predictions.csv",
            index=False
        )

    # ---------------------------------------------------------------
    # Save 2026 test predictions
    # ---------------------------------------------------------------

    if test_predictions is not None:

        test_output = test_df[
            [
                "season",
                "round",
                "race_name",
                "driver_name",
                "team_name",
                TARGET
            ]
        ].copy()

        test_output[
            "predicted_sprint_finish"
        ] = test_predictions

        test_output.to_csv(
            PREDICTION_DIR /
            "sprint_2026_test_predictions.csv",
            index=False
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print()
    print("=" * 80)
    print("SPRINT MODEL TRAINING COMPLETE")
    print("=" * 80)
    print()

    print(f"Features: {len(FEATURES)}")
    print(f"Numeric features: {len(NUMERIC_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Target: {TARGET}")

    print()
    print(f"Model saved -> {model_path}")
    print(f"Metadata saved -> {metadata_path}")

    if validation_predictions is not None:
        print(
            "2025 predictions saved -> "
            f"{PREDICTION_DIR / 'sprint_2025_validation_predictions.csv'}"
        )

    if test_predictions is not None:
        print(
            "2026 predictions saved -> "
            f"{PREDICTION_DIR / 'sprint_2026_test_predictions.csv'}"
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()