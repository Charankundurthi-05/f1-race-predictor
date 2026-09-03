from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = ROOT / "data" / "predictions"
MODEL_DIR = ROOT / "models"


NORMAL_STAGES = [
    "PRE_PRACTICE",
    "FP1",
    "FP2",
    "FP3",
    "QUALIFYING",
]

SPRINT_STAGES = [
    "PRE_PRACTICE",
    "SPRINT_QUALIFYING",
    "SPRINT",
    "QUALIFYING",
]


MODEL_FILES = {
    "PRE_PRACTICE": "pre_practice_gb_shallow_final_model.joblib",
    "FP1": "fp1_rf_small_final_model.joblib",
    "FP2": "fp2_extra_trees_final_model.joblib",
    "FP3": "fp3_rf_small_final_model.joblib",
    "QUALIFYING": "qualifying_gb_shallow_final_model.joblib",
    "SPRINT": "sprint_gb_shallow_final_model.joblib",
}


METADATA_FILES = {
    "PRE_PRACTICE": "pre_practice_gb_shallow_final_metadata.json",
    "FP1": "fp1_rf_small_final_metadata.json",
    "FP2": "fp2_extra_trees_final_metadata.json",
    "FP3": "fp3_rf_small_final_metadata.json",
    "QUALIFYING": "qualifying_gb_shallow_final_metadata.json",
    "SPRINT": "sprint_gb_shallow_final_metadata.json",
}


FEATURE_FILES = {
    "PRE_PRACTICE": "current_2026_pre_practice_features.csv",
    "FP1": "current_2026_fp1_features.csv",
    "FP2": "current_2026_fp2_features.csv",
    "FP3": "current_2026_fp3_features.csv",
    "QUALIFYING": "current_2026_qualifying_features.csv",
    "SPRINT": "current_2026_sprint_features.csv",
}


def load_weekend_format():
    state_file = (
        ROOT
        / "data"
        / "predictions"
        / "weekend_state.json"
    )

    if state_file.exists():

        try:
            with open(
                state_file,
                "r",
                encoding="utf-8"
            ) as f:
                state = json.load(f)

            weekend_format = str(
                state.get(
                    "weekend_format",
                    "normal"
                )
            ).lower()

            if weekend_format in {
                "normal",
                "sprint"
            }:
                return weekend_format

        except Exception:
            pass

    # Fallback: inspect current prediction data.
    current_file = (
        PREDICTION_DIR
        / "current_2026_pre_practice_features.csv"
    )

    if current_file.exists():

        try:
            df = pd.read_csv(current_file)

            if "weekend_format" in df.columns:

                value = str(
                    df["weekend_format"]
                    .dropna()
                    .iloc[0]
                ).lower()

                if value in {
                    "normal",
                    "sprint"
                }:
                    return value

        except Exception:
            pass

    return "normal"


def load_practice_results():

    path = (
        PREDICTION_DIR
        / "live_practice_results.csv"
    )

    if not path.exists():
        return None

    try:

        df = pd.read_csv(path)

        if df.empty:
            return None

        return df

    except Exception:
        return None


def get_session_names(df):

    if df is None:
        return []

    if "session_name" in df.columns:

        return (
            df["session_name"]
            .astype(str)
            .str.upper()
            .tolist()
        )

    if "session" in df.columns:

        return (
            df["session"]
            .astype(str)
            .str.upper()
            .tolist()
        )

    return []


def practice_available(df, practice_number):

    if df is None:
        return False

    sessions = get_session_names(df)

    target = f"PRACTICE {practice_number}"

    return any(
        target in session
        for session in sessions
    )


def sprint_qualifying_available():

    path = (
        PREDICTION_DIR
        / "live_sprint_qualifying_results.csv"
    )

    if not path.exists():
        return False

    try:

        df = pd.read_csv(path)

        return not df.empty

    except Exception:
        return False


def sprint_results_available():

    path = (
        PREDICTION_DIR
        / "live_sprint_results.csv"
    )

    if not path.exists():
        return False

    try:

        df = pd.read_csv(path)

        return not df.empty

    except Exception:
        return False


def qualifying_available():

    path = (
        PREDICTION_DIR
        / "live_qualifying_results.csv"
    )

    if not path.exists():
        return False

    try:

        df = pd.read_csv(path)

        return not df.empty

    except Exception:
        return False


def determine_stage():

    weekend_format = load_weekend_format()

    practice_df = load_practice_results()

    # ---------------------------------------------------------------
    # SPRINT WEEKEND
    # ---------------------------------------------------------------

    if weekend_format == "sprint":

        if qualifying_available():
            return "QUALIFYING"

        if sprint_results_available():
            return "QUALIFYING"

        if sprint_qualifying_available():
            return "SPRINT"

        return "PRE_PRACTICE"

    # ---------------------------------------------------------------
    # NORMAL WEEKEND
    # ---------------------------------------------------------------

    if qualifying_available():
        return "QUALIFYING"

    if practice_available(
        practice_df,
        3
    ):
        return "FP3"

    if practice_available(
        practice_df,
        2
    ):
        return "FP2"

    if practice_available(
        practice_df,
        1
    ):
        return "FP1"

    return "PRE_PRACTICE"


def validate_model(stage):

    if stage not in MODEL_FILES:
        print(
            f"No model configured for stage: {stage}"
        )
        return False

    model_path = (
        MODEL_DIR
        / MODEL_FILES[stage]
    )

    metadata_path = (
        MODEL_DIR
        / METADATA_FILES[stage]
    )

    print(
        f"Validating {stage} model..."
    )

    model_ok = model_path.exists()
    metadata_ok = metadata_path.exists()

    print(
        f"  Model: "
        f"{'PASS' if model_ok else 'FAIL'}"
    )

    print(
        f"  Metadata: "
        f"{'PASS' if metadata_ok else 'FAIL'}"
    )

    if not model_ok or not metadata_ok:
        return False

    try:

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        features = metadata.get(
            "features",
            []
        )

        print(
            f"  Features: {len(features)}"
        )

        return len(features) > 0

    except Exception as exc:

        print(
            f"  Metadata error: {exc}"
        )

        return False


def check_dataset(stage):

    if stage not in FEATURE_FILES:
        return False

    dataset_path = (
        PREDICTION_DIR
        / FEATURE_FILES[stage]
    )

    if not dataset_path.exists():

        print()
        print(
            f"Waiting for {stage} feature dataset."
        )

        return False

    try:

        df = pd.read_csv(
            dataset_path
        )

        print(
            f"Current feature dataset rows: "
            f"{len(df)}"
        )

        if "driver_name" in df.columns:

            print(
                f"Unique drivers: "
                f"{df['driver_name'].nunique()}"
            )

        return True

    except Exception as exc:

        print(
            f"Dataset check warning: {exc}"
        )

        return False


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("LIVE PREDICTION CONTROLLER")
    print("=" * 80)
    print()

    weekend_format = load_weekend_format()

    print(
        f"Weekend format: "
        f"{weekend_format.upper()}"
    )

    print()

    stage = determine_stage()

    print(
        f"Current prediction stage: "
        f"{stage}"
    )

    print()

    # ---------------------------------------------------------------
    # Stage explanation
    # ---------------------------------------------------------------

    explanations = {
        "PRE_PRACTICE":
            "No required live session results detected. "
            "Using the pre-practice model.",

        "FP1":
            "FP1 results detected. "
            "Using the FP1 model.",

        "FP2":
            "FP1 + FP2 results detected. "
            "Using the FP2 model.",

        "FP3":
            "FP1 + FP2 + FP3 results detected. "
            "Using the FP3 model.",

        "SPRINT":
            "Sprint Qualifying results detected. "
            "Using the dedicated sprint model.",

        "QUALIFYING":
            "Qualifying results detected. "
            "Using the qualifying model.",
    }

    print(
        explanations.get(
            stage,
            "Using configured model."
        )
    )

    print()

    # ---------------------------------------------------------------
    # Model validation
    # ---------------------------------------------------------------

    if not validate_model(stage):

        print()
        print(
            "CONTROLLER STATUS: NOT READY"
        )

        return

    print()

    # ---------------------------------------------------------------
    # Dataset validation
    # ---------------------------------------------------------------

    dataset_ready = check_dataset(stage)

    # Pre-practice is expected to exist already.
    # Other stages may legitimately be waiting for session data.

    if not dataset_ready and stage != "PRE_PRACTICE":

        print()

        print(
            f"{stage} model is available, "
            "but its current feature dataset is not ready yet."
        )

        print()
        print(
            "CONTROLLER STATUS: WAITING"
        )

        return

    print()

    print(
        "CONTROLLER STATUS: READY"
    )

    print()
    print("=" * 80)
    print(
        "LIVE PREDICTION CONTROLLER CHECK COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()