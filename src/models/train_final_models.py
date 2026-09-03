from pathlib import Path
import json
import warnings

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
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


BEST_MODELS = {
    "pre_practice": "gb_shallow",
    "fp1": "rf_small",
    "fp2": "extra_trees",
    "fp3": "rf_small",
    "qualifying": "gb_shallow",
}


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


IDENTITY_FEATURES = [
    "driver_name",
    "team_name",
]


CIRCUIT_FEATURES = [
    "circuit_latitude",
    "circuit_longitude",
    "circuit_length_km",
    "circuit_turns",
    "circuit_total_races_held",
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


def get_features(stage):
    features = (
        BASE_NUMERIC_FEATURES
        + CIRCUIT_FEATURES
        + REGULATION_FEATURES
        + PRACTICE_FEATURES[stage]
        + QUALIFYING_FEATURES[stage]
        + WEATHER_FEATURES[stage]
        + SPRINT_FEATURES[stage]
        + IDENTITY_FEATURES
    )

    return list(dict.fromkeys(features))


def build_preprocessor(X):
    numeric = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical = [
        c for c in X.columns
        if c not in numeric
    ]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    )
                ]),
                numeric,
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]),
                categorical,
            ),
        ]
    )


def build_estimator(name):
    if name == "rf_small":
        return RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

    if name == "rf_large":
        return RandomForestRegressor(
            n_estimators=700,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=42,
            n_jobs=-1,
        )

    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

    if name == "gb_standard":
        return GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=4,
            loss="huber",
            random_state=42,
        )

    if name == "gb_deeper":
        return GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=4,
            loss="huber",
            random_state=42,
        )

    if name == "gb_shallow":
        return GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=4,
            loss="huber",
            random_state=42,
        )

    raise ValueError(
        f"Unknown model: {name}"
    )


def main():
    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("FINAL MODEL TRAINING")
    print("=" * 80)

    for stage in STAGES:
        model_name = BEST_MODELS[stage]

        print()
        print("=" * 80)
        print(
            f"TRAINING {stage.upper()} "
            f"USING {model_name}"
        )
        print("=" * 80)

        path = DATA_DIR / f"stage_{stage}.csv"

        df = pd.read_csv(path)

        if (
            stage != "pre_practice"
            and "stage_available" in df.columns
        ):
            df = df[
                df["stage_available"] == 1
            ].copy()

        df = df[
            df[TARGET].notna()
        ].copy()

        features = get_features(stage)

        missing = [
            c for c in features
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{stage}: missing features {missing}"
            )

        X = df[features].copy()
        y = df[TARGET].astype(float)

        preprocessor = build_preprocessor(X)

        estimator = build_estimator(
            model_name
        )

        model = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                estimator
            ),
        ])

        model.fit(X, y)

        model_path = (
            MODEL_DIR
            / f"{stage}_{model_name}_final_model.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        metadata = {
            "stage": stage,
            "model": model_name,
            "target": TARGET,
            "training_seasons": "2014-2026",
            "feature_count": len(features),
            "features": features,
            "training_rows": len(df),
        }

        metadata_path = (
            MODEL_DIR
            / f"{stage}_{model_name}_final_metadata.json"
        )

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

        print(
            f"Training rows: {len(df)}"
        )

        print(
            f"Features: {len(features)}"
        )

        print(
            f"Model saved: {model_path}"
        )

        print(
            f"Metadata saved: {metadata_path}"
        )

    print()
    print("=" * 80)
    print("FINAL MODEL TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()