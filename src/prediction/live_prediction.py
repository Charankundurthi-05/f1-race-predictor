from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
PREDICTION_DIR = BASE_DIR / "data" / "predictions"


STAGES = [
    "pre_practice",
    "fp1",
    "fp2",
    "fp3",
    "qualifying",
]


MODEL_FILES = {
    "pre_practice": "pre_practice_gb_shallow_final_model.joblib",
    "fp1": "fp1_rf_small_final_model.joblib",
    "fp2": "fp2_extra_trees_final_model.joblib",
    "fp3": "fp3_rf_small_final_model.joblib",
    "qualifying": "qualifying_gb_shallow_final_model.joblib",
}


DATA_FILES = {
    "pre_practice": "stage_pre_practice.csv",
    "fp1": "stage_fp1.csv",
    "fp2": "stage_fp2.csv",
    "fp3": "stage_fp3.csv",
    "qualifying": "stage_qualifying.csv",
}


def load_model(stage):
    model_path = MODEL_DIR / MODEL_FILES[stage]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(model_path)


def load_model_metadata(stage):
    metadata_path = (
        MODEL_DIR
        / f"{MODEL_FILES[stage].replace('.joblib', '_metadata.json')}"
    )

    if not metadata_path.exists():
        return None

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_stage_data(stage):
    path = DATA_DIR / DATA_FILES[stage]

    if not path.exists():
        raise FileNotFoundError(
            f"Stage dataset not found: {path}"
        )

    return pd.read_csv(path)


def load_live_practice():
    path = RAW_DIR / "live_2026_practice.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)

    if df.empty:
        return None

    return df


def get_latest_practice_stage(live_practice):
    if live_practice is None:
        return None

    available = set(
        live_practice["session_name"]
        .dropna()
        .astype(str)
    )

    if "Practice 3" in available:
        return "fp3"

    if "Practice 2" in available:
        return "fp2"

    if "Practice 1" in available:
        return "fp1"

    return None


def convert_duration_to_seconds(value):
    if pd.isna(value):
        return np.nan

    value = str(value)

    try:
        parts = value.split(":")

        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])

            return minutes * 60 + seconds

        return float(value)

    except (ValueError, TypeError):
        return np.nan


def prepare_live_practice_features(
    live_practice,
    stage
):
    if live_practice is None:
        return None

    session_map = {
        "fp1": "Practice 1",
        "fp2": "Practice 2",
        "fp3": "Practice 3",
    }

    session_name = session_map[stage]

    current = live_practice[
        live_practice["session_name"]
        == session_name
    ].copy()

    if current.empty:
        return None

    current["practice_time_seconds"] = (
        current["duration"]
        .apply(convert_duration_to_seconds)
    )

    current = current.sort_values(
        "position"
    )

    current["practice_position"] = (
        pd.to_numeric(
            current["position"],
            errors="coerce"
        )
    )

    return current[
        [
            "driver_number",
            "driver_name",
            "driver_code",
            "team_name",
            "practice_position",
            "practice_time_seconds",
            "gap_to_leader",
            "number_of_laps",
        ]
    ]


def identify_stage_practice_columns(df):
    candidates = [
        column
        for column in df.columns
        if "practice" in column.lower()
        or "fp1" in column.lower()
        or "fp2" in column.lower()
        or "fp3" in column.lower()
    ]

    return candidates


def build_live_prediction(stage):
    print()
    print("=" * 80)
    print(f"LIVE PREDICTION — {stage.upper()}")
    print("=" * 80)

    base = load_stage_data(stage)

    model = load_model(stage)

    metadata = load_model_metadata(stage)

    live_practice = load_live_practice()

    if live_practice is None:
        print()
        print(
            "No live practice results are available."
        )
        print(
            "Using the pre-weekend trained dataset."
        )

        return None

    latest_stage = get_latest_practice_stage(
        live_practice
    )

    if latest_stage is None:
        print(
            "No usable practice session found."
        )
        return None

    print(
        f"Latest available practice: "
        f"{latest_stage.upper()}"
    )

    live_features = prepare_live_practice_features(
        live_practice,
        latest_stage
    )

    if live_features is None:
        print(
            "Live practice data could not "
            "be converted into features."
        )
        return None

    print(
        f"Live drivers: "
        f"{len(live_features)}"
    )

    if metadata is not None:
        feature_columns = metadata.get(
            "feature_columns"
        )
    else:
        feature_columns = None

    if not feature_columns:
        print(
            "Model metadata does not contain "
            "feature columns."
        )
        return None

    missing = [
        column
        for column in feature_columns
        if column not in base.columns
    ]

    if missing:
        print()
        print(
            "WARNING: Missing model features:"
        )

        for column in missing[:20]:
            print(
                f"  - {column}"
            )

        if len(missing) > 20:
            print(
                f"  ... and {len(missing) - 20} more"
            )

        print()
        print(
            "Live feature injection is not "
            "ready for this stage yet."
        )

        return None

    print()
    print(
        "Model feature validation: PASS"
    )

    return {
        "base": base,
        "model": model,
        "feature_columns": feature_columns,
        "live_features": live_features,
        "latest_stage": latest_stage,
    }


def main():
    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("LIVE PREDICTION ENGINE")
    print("=" * 80)

    live_practice = load_live_practice()

    if live_practice is None:
        print()
        print(
            "No live practice CSV exists yet."
        )

        print(
            "Run download_live_practice.py "
            "after a practice session."
        )

        return

    latest_stage = get_latest_practice_stage(
        live_practice
    )

    print()
    print(
        f"Latest available stage: "
        f"{latest_stage}"
    )

    result = build_live_prediction(
        latest_stage
    )

    if result is None:
        print()
        print(
            "Live prediction engine is waiting "
            "for usable session data."
        )
    else:
        print()
        print(
            "Live prediction engine is ready "
            "for feature injection."
        )

    print()
    print("=" * 80)
    print("LIVE PREDICTION CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()