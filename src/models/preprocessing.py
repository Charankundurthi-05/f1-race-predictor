from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


STAGE_FILES = {
    "pre_practice": "stage_pre_practice.csv",
    "fp1": "stage_fp1.csv",
    "fp2": "stage_fp2.csv",
    "fp3": "stage_fp3.csv",
    "qualifying": "stage_qualifying.csv",
}


STAGE_FEATURES = {
    "pre_practice": [
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
    ],
    "fp1": [
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
        "fp1_position",
        "practice_avg_position",
        "practice_best_position",
        "practice_sessions_available",
    ],
    "fp2": [
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
        "fp1_position",
        "fp2_position",
        "practice_avg_position",
        "practice_best_position",
        "practice_sessions_available",
    ],
    "fp3": [
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
        "fp1_position",
        "fp2_position",
        "fp3_position",
        "practice_avg_position",
        "practice_best_position",
        "practice_sessions_available",
    ],
    "qualifying": [
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
        "fp1_position",
        "fp2_position",
        "fp3_position",
        "practice_avg_position",
        "practice_best_position",
        "practice_sessions_available",
        "qualifying_position",
        "q1",
        "q2",
        "q3",
        "grid_position",
    ],
}


def parse_lap_time(value):
    """
    Convert F1 lap-time strings into seconds.

    Examples:
        1:32.456 -> 92.456
        0:58.123 -> 58.123
        92.456   -> 92.456
        missing  -> NaN
    """

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if not text:
        return np.nan

    try:
        if ":" in text:
            parts = text.split(":")

            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])

                return minutes * 60 + seconds

        return float(text)

    except (ValueError, TypeError):
        return np.nan


def prepare_features(df, stage):
    """
    Prepare the feature matrix for one prediction stage.

    The target finish_position is deliberately excluded.
    """

    if stage not in STAGE_FEATURES:
        raise ValueError(f"Unknown stage: {stage}")

    missing = [
        column
        for column in STAGE_FEATURES[stage]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{stage}: missing required features: {missing}"
        )

    features = df[STAGE_FEATURES[stage]].copy()

    if stage == "qualifying":
        for column in ["q1", "q2", "q3"]:
            features[column] = features[column].apply(parse_lap_time)

    for column in features.columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    return features


def prepare_target(df):
    """
    Prepare the race finishing-position target.
    """

    if "finish_position" not in df.columns:
        raise ValueError(
            "Dataset does not contain finish_position target."
        )

    target = pd.to_numeric(
        df["finish_position"],
        errors="coerce",
    )

    return target


def chronological_split(
    df,
    train_end_season,
    validation_season=None,
    test_season=None,
):
    """
    Split data chronologically by F1 season.

    Example:

        train_end_season=2024
        validation_season=2025
        test_season=2026

    This prevents future races from entering earlier training data.
    """

    train = df[
        df["season"] <= train_end_season
    ].copy()

    validation = None
    test = None

    if validation_season is not None:
        validation = df[
            df["season"] == validation_season
        ].copy()

    if test_season is not None:
        test = df[
            df["season"] == test_season
        ].copy()

    return train, validation, test


def impute_features(
    X_train,
    X_validation=None,
    X_test=None,
):
    """
    Fit missing-value imputation ONLY on training data.

    This is important because using information from validation/test
    data during preprocessing would introduce leakage.
    """

    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)

    X_validation_imputed = None
    X_test_imputed = None

    if X_validation is not None:
        X_validation_imputed = imputer.transform(
            X_validation
        )

    if X_test is not None:
        X_test_imputed = imputer.transform(X_test)

    return (
        X_train_imputed,
        X_validation_imputed,
        X_test_imputed,
        imputer,
    )


def load_stage(stage):
    """
    Load one stage dataset.
    """

    if stage not in STAGE_FILES:
        raise ValueError(f"Unknown stage: {stage}")

    path = PROCESSED_DIR / STAGE_FILES[stage]

    if not path.exists():
        raise FileNotFoundError(
            f"Stage dataset not found: {path}"
        )

    return pd.read_csv(path)


def build_training_data(stage):
    """
    Load a stage dataset and create X and y.
    """

    df = load_stage(stage)

    X = prepare_features(df, stage)
    y = prepare_target(df)

    valid_target = y.notna()

    X = X.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True)

    return df.loc[valid_target].reset_index(drop=True), X, y


def validate_preprocessing(stage):
    """
    Run basic preprocessing checks.
    """

    df, X, y = build_training_data(stage)

    print(f"\n{stage.upper()}")
    print("=" * 70)

    print(f"Rows: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(f"Target rows: {len(y)}")

    print("\nFeature data types:")
    print(X.dtypes.to_string())

    print("\nMissing feature values:")
    print(X.isna().sum().to_string())

    print("\nTarget missing values:")
    print(y.isna().sum())

    if stage == "qualifying":
        print("\nQualifying lap times after conversion:")

        print(
            X[["q1", "q2", "q3"]]
            .describe()
            .to_string()
        )

    return True


if __name__ == "__main__":

    print("F1 MODEL PREPROCESSING VALIDATION")
    print("=" * 70)

    for stage in STAGE_FILES:
        validate_preprocessing(stage)

    print("\n" + "=" * 70)
    print("CHRONOLOGICAL SPLIT TEST")
    print("=" * 70)

    df = load_stage("fp3")

    train, validation, test = chronological_split(
        df,
        train_end_season=2024,
        validation_season=2025,
        test_season=2026,
    )

    print(f"Training seasons:  2014-2024")
    print(f"Training rows:     {len(train)}")
    print(f"Validation season: 2025")
    print(f"Validation rows:   {len(validation)}")
    print(f"Test season:       2026")
    print(f"Test rows:         {len(test)}")

    print("\nSplit verification:")

    print(
        f"Training max season: "
        f"{train['season'].max()}"
    )

    print(
        f"Validation seasons: "
        f"{sorted(validation['season'].unique())}"
    )

    print(
        f"Test seasons: "
        f"{sorted(test['season'].unique())}"
    )

    print("\nPREPROCESSING VALIDATION COMPLETE")