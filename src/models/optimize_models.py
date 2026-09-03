from pathlib import Path
import json
import warnings

import joblib
import numpy as np
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


def calculate_race_metrics(df):
    race_metrics = []

    for _, race in df.groupby(
        ["season", "round", "race_date"]
    ):
        race = race.copy()

        race = race.sort_values(
            ["predicted_finish", "driver_name"]
        ).reset_index(drop=True)

        predicted = race[
            "driver_name"
        ].tolist()

        actual = race.sort_values(
            ["finish_position", "driver_name"]
        )[
            "driver_name"
        ].tolist()

        if not predicted or not actual:
            continue

        predicted_rank = {
            driver: i + 1
            for i, driver in enumerate(predicted)
        }

        actual_rank = {
            driver: i + 1
            for i, driver in enumerate(actual)
        }

        common = set(predicted_rank) & set(actual_rank)

        rank_mae = np.mean([
            abs(
                predicted_rank[d]
                - actual_rank[d]
            )
            for d in common
        ])

        top3 = (
            len(
                set(predicted[:3])
                & set(actual[:3])
            )
            / min(3, len(actual))
        )

        top5 = (
            len(
                set(predicted[:5])
                & set(actual[:5])
            )
            / min(5, len(actual))
        )

        top10 = (
            len(
                set(predicted[:10])
                & set(actual[:10])
            )
            / min(10, len(actual))
        )

        winner = int(
            predicted[0] == actual[0]
        )

        race_metrics.append({
            "rank_mae": rank_mae,
            "top3": top3,
            "top5": top5,
            "top10": top10,
            "winner": winner,
        })

    metrics = pd.DataFrame(
        race_metrics
    )

    return {
        "rank_mae": metrics["rank_mae"].mean(),
        "top3": metrics["top3"].mean(),
        "top5": metrics["top5"].mean(),
        "top10": metrics["top10"].mean(),
        "winner": metrics["winner"].mean(),
    }


def build_candidate(
    preprocessor,
    name,
):
    if name == "rf_small":
        estimator = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

    elif name == "rf_large":
        estimator = RandomForestRegressor(
            n_estimators=700,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=42,
            n_jobs=-1,
        )

    elif name == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

    elif name == "gb_standard":
        estimator = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=4,
            loss="huber",
            random_state=42,
        )

    elif name == "gb_deeper":
        estimator = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=4,
            loss="huber",
            random_state=42,
        )

    elif name == "gb_shallow":
        estimator = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=4,
            loss="huber",
            random_state=42,
        )

    else:
        raise ValueError(
            f"Unknown candidate: {name}"
        )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def main():
    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("MODEL OPTIMIZATION")
    print("=" * 80)

    candidates = [
        "rf_small",
        "rf_large",
        "extra_trees",
        "gb_standard",
        "gb_deeper",
        "gb_shallow",
    ]

    all_results = []

    for stage in STAGES:

        print()
        print("=" * 80)
        print(f"OPTIMIZING {stage.upper()}")
        print("=" * 80)

        df = pd.read_csv(
            DATA_DIR / f"stage_{stage}.csv"
        )

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

        train_mask = df["season"] <= 2024
        validation_mask = df["season"] == 2025
        test_mask = df["season"] == 2026

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]

        X_validation = X.loc[
            validation_mask
        ]
        y_validation = y.loc[
            validation_mask
        ]

        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        preprocessor = build_preprocessor(
            X_train
        )

        stage_results = []

        for candidate in candidates:

            print()
            print(
                f"Training {candidate}..."
            )

            model = build_candidate(
                preprocessor,
                candidate,
            )

            model.fit(
                X_train,
                y_train,
            )

            validation_predictions = (
                model.predict(
                    X_validation
                )
            )

            validation_df = df.loc[
                validation_mask
            ].copy()

            validation_df[
                "predicted_finish"
            ] = validation_predictions

            validation_metrics = (
                calculate_race_metrics(
                    validation_df
                )
            )

            test_predictions = model.predict(
                X_test
            )

            test_df = df.loc[
                test_mask
            ].copy()

            test_df[
                "predicted_finish"
            ] = test_predictions

            test_metrics = (
                calculate_race_metrics(
                    test_df
                )
            )

            print(
                f"2025 rank MAE: "
                f"{validation_metrics['rank_mae']:.3f}"
            )

            print(
                f"2025 Top-3: "
                f"{validation_metrics['top3']:.1%}"
            )

            print(
                f"2025 Top-10: "
                f"{validation_metrics['top10']:.1%}"
            )

            print(
                f"2025 Winner: "
                f"{validation_metrics['winner']:.1%}"
            )

            print(
                f"2026 rank MAE: "
                f"{test_metrics['rank_mae']:.3f}"
            )

            print(
                f"2026 Top-3: "
                f"{test_metrics['top3']:.1%}"
            )

            print(
                f"2026 Top-10: "
                f"{test_metrics['top10']:.1%}"
            )

            stage_results.append({
                "stage": stage,
                "candidate": candidate,
                "validation_rank_mae":
                    validation_metrics[
                        "rank_mae"
                    ],
                "validation_top3":
                    validation_metrics[
                        "top3"
                    ],
                "validation_top5":
                    validation_metrics[
                        "top5"
                    ],
                "validation_top10":
                    validation_metrics[
                        "top10"
                    ],
                "validation_winner":
                    validation_metrics[
                        "winner"
                    ],
                "test_rank_mae":
                    test_metrics[
                        "rank_mae"
                    ],
                "test_top3":
                    test_metrics[
                        "top3"
                    ],
                "test_top5":
                    test_metrics[
                        "top5"
                    ],
                "test_top10":
                    test_metrics[
                        "top10"
                    ],
                "test_winner":
                    test_metrics[
                        "winner"
                    ],
            })

        stage_df = pd.DataFrame(
            stage_results
        )

        stage_df = stage_df.sort_values(
            [
                "validation_rank_mae",
                "validation_top10",
            ],
            ascending=[
                True,
                False,
            ],
        )

        best_candidate = stage_df.iloc[0][
            "candidate"
        ]

        print()
        print(
            f"BEST 2025 CANDIDATE: "
            f"{best_candidate}"
        )

        all_results.extend(
            stage_results
        )

    results_df = pd.DataFrame(
        all_results
    )

    output_path = (
        MODEL_DIR
        / "model_optimization_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)

    print(
        results_df.sort_values(
            [
                "stage",
                "validation_rank_mae",
            ]
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print(
        f"Saved: {output_path}"
    )

    print()
    print("=" * 80)
    print("MODEL OPTIMIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()