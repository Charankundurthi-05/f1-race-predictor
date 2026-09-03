from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


STAGES = [
    "pre_practice",
    "fp1",
    "fp2",
    "fp3",
    "qualifying",
]

TARGET = "finish_position"


IDENTITY_FEATURES = [
    "driver_name",
    "team_name",
]


BASE_NUMERIC_FEATURES = [
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
]


CIRCUIT_FEATURES = [
    "circuit_latitude",
    "circuit_longitude",
    "circuit_length_km",
    "circuit_turns",
    "circuit_total_races_held",
]


CIRCUIT_CATEGORICAL_FEATURES = [
    "circuit_type",
    "circuit_direction",
]


REGULATION_FEATURES = [
    "regulation_era",
]


PRACTICE_FEATURES = {
    "pre_practice": [],
    "fp1": [
        "fp1_position",
        "practice_avg_position",
        "practice_best_position",
        "practice_sessions_available",
    ],
    "fp2": [
        "fp1_position",
        "fp2_position",
        "practice_avg_position",
        "practice_best_position",
        "practice_sessions_available",
    ],
    "fp3": [
        "fp1_position",
        "fp2_position",
        "fp3_position",
        "practice_avg_position",
        "practice_best_position",
        "practice_sessions_available",
    ],
    "qualifying": [
        "fp1_position",
        "fp2_position",
        "fp3_position",
        "practice_avg_position",
        "practice_best_position",
        "practice_sessions_available",
    ],
}


QUALIFYING_FEATURES = {
    "pre_practice": [],
    "fp1": [],
    "fp2": [],
    "fp3": [],
    "qualifying": [
        "qualifying_position",
        "q1",
        "q2",
        "q3",
        "qualifying_available",
        "grid_position",
    ],
}


WEATHER_FEATURES = {
    "pre_practice": [],

    "fp1": [
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
    ],

    "fp2": [
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
    ],

    "fp3": [
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
    ],

    "qualifying": [
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

        "weather_qualifying_temperature_avg",
        "weather_qualifying_temperature_min",
        "weather_qualifying_temperature_max",
        "weather_qualifying_humidity_avg",
        "weather_qualifying_precipitation_total",
        "weather_qualifying_rain_hours",
        "weather_qualifying_rain_flag",
        "weather_qualifying_wind_speed_avg",
        "weather_qualifying_wind_speed_max",
        "weather_qualifying_wind_direction_avg",
        "weather_qualifying_cloud_cover_avg",
        "weather_qualifying_weather_observations",
    ],
}


SPRINT_FEATURES = {
    "pre_practice": [],

    "fp1": [],

    "fp2": [],

    "fp3": [],

    "qualifying": [
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
    ],
}


META_COLUMNS = [
    "season",
    "round",
    "race_name",
    "race_date",
    "driver_name",
    "team_name",
    "stage_name",
    "stage_available",
    "finish_position",
]


def load_stage(stage):
    path = DATA_DIR / f"stage_{stage}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)

    print(f"Loaded {stage}: {df.shape}")

    return df


def get_allowed_features(stage):
    features = []

    features.extend(BASE_NUMERIC_FEATURES)
    features.extend(CIRCUIT_FEATURES)
    features.extend(CIRCUIT_CATEGORICAL_FEATURES)
    features.extend(REGULATION_FEATURES)

    features.extend(PRACTICE_FEATURES[stage])
    features.extend(QUALIFYING_FEATURES[stage])
    features.extend(WEATHER_FEATURES[stage])
    features.extend(SPRINT_FEATURES[stage])

    features.extend(IDENTITY_FEATURES)

    features = list(dict.fromkeys(features))

    return features


def validate_features(df, stage):
    allowed = get_allowed_features(stage)

    missing = [
        column
        for column in allowed
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{stage}: required features missing:\n{missing}"
        )

    leakage_keywords = [
        "finish_position",
        "points",
        "status",
        "position_text",
    ]

    suspicious = []

    for column in allowed:
        lower = column.lower()

        if column in BASE_NUMERIC_FEATURES:
            continue

        if (
            "finish_position" in lower
            and column not in SPRINT_FEATURES[stage]
        ):
            suspicious.append(column)

        if column == "points":
            suspicious.append(column)

        if column == "status":
            suspicious.append(column)

        if column == "position_text":
            suspicious.append(column)

    if suspicious:
        raise ValueError(
            f"{stage}: suspicious leakage features detected: {suspicious}"
        )

    return allowed


def prepare_dataset(df, stage):
    if "stage_available" in df.columns:
        if stage != "pre_practice":
            df = df[df["stage_available"] == 1].copy()
        else:
            df = df.copy()

    df = df[df[TARGET].notna()].copy()

    allowed = validate_features(df, stage)

    X = df[allowed].copy()
    y = df[TARGET].astype(float).copy()

    return df, X, y, allowed


def build_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        column
        for column in X.columns
        if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def build_models(preprocessor):
    random_forest = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    gradient_boosting = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.04,
                    max_depth=3,
                    min_samples_leaf=4,
                    loss="huber",
                    random_state=42,
                ),
            ),
        ]
    )

    return {
        "random_forest": random_forest,
        "gradient_boosting": gradient_boosting,
    }


def calculate_metrics(y_true, predictions):
    predictions = np.asarray(predictions)
    y_true = np.asarray(y_true)

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    predicted_position = np.rint(predictions)

    top3 = np.mean(
        (
            predicted_position <= 3
        )
        & (
            y_true <= 3
        )
    )

    top10 = np.mean(
        (
            predicted_position <= 10
        )
        & (
            y_true <= 10
        )
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "top3_overlap": float(top3),
        "top10_overlap": float(top10),
    }


def evaluate_model(
    model,
    X,
    y,
):
    predictions = model.predict(X)

    return calculate_metrics(
        y,
        predictions,
    ), predictions


def save_predictions(
    df,
    predictions,
    stage,
    model_name,
):
    output = df[
        [
            "season",
            "round",
            "race_name",
            "race_date",
            "driver_name",
            "team_name",
            TARGET,
        ]
    ].copy()

    output["predicted_finish"] = predictions

    output["predicted_finish_rounded"] = (
        np.rint(predictions).astype(int)
    )

    output["prediction_error"] = (
        output["predicted_finish"]
        - output[TARGET]
    )

    output = output.sort_values(
        [
            "season",
            "round",
            "predicted_finish",
        ]
    )

    path = (
        MODEL_DIR
        / f"{stage}_{model_name}_2026_predictions.csv"
    )

    output.to_csv(
        path,
        index=False,
    )

    return path


def save_model_metadata(
    model,
    stage,
    model_name,
    features,
    metrics_2025,
    metrics_2026,
):
    metadata = {
        "stage": stage,
        "model_name": model_name,
        "target": TARGET,
        "features": features,
        "feature_count": len(features),
        "train_until_season": 2025,
        "validation_season": 2025,
        "test_season": 2026,
        "metrics_2025": metrics_2025,
        "metrics_2026": metrics_2026,
    }

    path = (
        MODEL_DIR
        / f"{stage}_{model_name}_metadata.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    return path


def train_stage(stage):
    print()
    print("=" * 80)
    print(f"TRAINING ENRICHED MODELS: {stage.upper()}")
    print("=" * 80)

    df = load_stage(stage)

    df, X, y, features = prepare_dataset(
        df,
        stage,
    )

    print(f"Allowed features: {len(features)}")
    print(f"Training rows available: {len(df)}")

    print()
    print("Feature groups:")

    print(f"  Base/history: {len(BASE_NUMERIC_FEATURES)}")
    print(
        f"  Circuit: "
        f"{len(CIRCUIT_FEATURES) + len(CIRCUIT_CATEGORICAL_FEATURES)}"
    )
    print(f"  Regulation: {len(REGULATION_FEATURES)}")
    print(f"  Practice: {len(PRACTICE_FEATURES[stage])}")
    print(f"  Qualifying: {len(QUALIFYING_FEATURES[stage])}")
    print(f"  Weather: {len(WEATHER_FEATURES[stage])}")
    print(f"  Sprint: {len(SPRINT_FEATURES[stage])}")
    print(f"  Identity: {len(IDENTITY_FEATURES)}")

    train_mask = df["season"] <= 2024
    validation_mask = df["season"] == 2025
    test_mask = df["season"] == 2026

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_validation = X.loc[validation_mask]
    y_validation = y.loc[validation_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    print()
    print("Dataset split:")
    print(f"  Train:      {len(X_train)}")
    print(f"  Validation: {len(X_validation)}")
    print(f"  Test 2026:  {len(X_test)}")

    if len(X_train) == 0:
        raise ValueError(
            f"{stage}: no training data."
        )

    if len(X_validation) == 0:
        raise ValueError(
            f"{stage}: no 2025 validation data."
        )

    if len(X_test) == 0:
        raise ValueError(
            f"{stage}: no 2026 test data."
        )

    preprocessor = build_preprocessor(
        X_train
    )

    models = build_models(
        preprocessor
    )

    results = []

    for model_name, model in models.items():

        print()
        print("-" * 80)
        print(f"Training {model_name}...")
        print("-" * 80)

        model.fit(
            X_train,
            y_train,
        )

        metrics_2025, predictions_2025 = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        metrics_2026, predictions_2026 = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print()
        print("2025 validation:")
        print(
            f"  MAE:  {metrics_2025['mae']:.3f}"
        )
        print(
            f"  RMSE: {metrics_2025['rmse']:.3f}"
        )
        print(
            f"  R²:   {metrics_2025['r2']:.3f}"
        )
        print(
            f"  Top-3 overlap:  "
            f"{metrics_2025['top3_overlap']:.1%}"
        )
        print(
            f"  Top-10 overlap: "
            f"{metrics_2025['top10_overlap']:.1%}"
        )

        print()
        print("2026 test:")
        print(
            f"  MAE:  {metrics_2026['mae']:.3f}"
        )
        print(
            f"  RMSE: {metrics_2026['rmse']:.3f}"
        )
        print(
            f"  R²:   {metrics_2026['r2']:.3f}"
        )
        print(
            f"  Top-3 overlap:  "
            f"{metrics_2026['top3_overlap']:.1%}"
        )
        print(
            f"  Top-10 overlap: "
            f"{metrics_2026['top10_overlap']:.1%}"
        )

        model_path = (
            MODEL_DIR
            / f"{stage}_{model_name}_model.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        print()
        print(f"Saved model: {model_path}")

        prediction_path = save_predictions(
            df.loc[test_mask].copy(),
            predictions_2026,
            stage,
            model_name,
        )

        print(
            f"Saved predictions: {prediction_path}"
        )

        metadata_path = save_model_metadata(
            model,
            stage,
            model_name,
            features,
            metrics_2025,
            metrics_2026,
        )

        print(
            f"Saved metadata: {metadata_path}"
        )

        results.append(
            {
                "stage": stage,
                "model": model_name,

                "feature_count": len(features),

                "validation_mae": metrics_2025["mae"],
                "validation_rmse": metrics_2025["rmse"],
                "validation_r2": metrics_2025["r2"],
                "validation_top3": metrics_2025[
                    "top3_overlap"
                ],
                "validation_top10": metrics_2025[
                    "top10_overlap"
                ],

                "test_2026_mae": metrics_2026["mae"],
                "test_2026_rmse": metrics_2026["rmse"],
                "test_2026_r2": metrics_2026["r2"],
                "test_2026_top3": metrics_2026[
                    "top3_overlap"
                ],
                "test_2026_top10": metrics_2026[
                    "top10_overlap"
                ],
            }
        )

    return results


def main():
    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("ENRICHED MULTI-STAGE MODEL TRAINING")
    print("=" * 80)

    all_results = []

    for stage in STAGES:
        stage_results = train_stage(
            stage
        )

        all_results.extend(
            stage_results
        )

    results_df = pd.DataFrame(
        all_results
    )

    comparison_path = (
        MODEL_DIR
        / "enriched_model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    display_columns = [
        "stage",
        "model",
        "feature_count",
        "validation_mae",
        "validation_r2",
        "test_2026_mae",
        "test_2026_r2",
        "test_2026_top3",
        "test_2026_top10",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print(
        f"Saved comparison: {comparison_path}"
    )

    print()
    print("=" * 80)
    print("ENRICHED MODEL TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()