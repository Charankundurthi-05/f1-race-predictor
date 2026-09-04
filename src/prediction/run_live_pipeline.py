from pathlib import Path
import json
import shutil
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PREDICTION_DIR = ROOT / "data" / "predictions"

LIVE_RAW_FILE = (
    RAW_DIR / "live_2026_practice.csv"
)

LIVE_PREDICTION_FILE = (
    PREDICTION_DIR / "live_practice_results.csv"
)

WEEKEND_STATE_FILE = (
    PREDICTION_DIR / "weekend_state.json"
)


# ================================================================
# HELPERS
# ================================================================

def run_script(script_relative_path):
    script_path = ROOT / script_relative_path

    print()
    print("=" * 80)
    print(f"RUNNING: {script_relative_path}")
    print("=" * 80)

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Script failed: {script_relative_path}"
        )

    return result


def load_weekend_state():
    if not WEEKEND_STATE_FILE.exists():
        return {}

    try:
        with open(
            WEEKEND_STATE_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)
    except Exception:
        return {}


def normalize_stage(stage):
    stage = str(stage).strip().lower()

    mapping = {
        "pre_practice": "PRE_PRACTICE",
        "fp1": "FP1",
        "fp2": "FP2",
        "fp3": "FP3",
        "qualifying": "QUALIFYING",
        "race": "RACE",
        "post_race": "RACE",
        "finished": "RACE",
    }

    return mapping.get(
        stage,
        stage.upper(),
    )


def copy_live_practice_file():

    if not LIVE_RAW_FILE.exists():

        print()
        print(
            "No live practice file was produced."
        )

        return False

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        LIVE_RAW_FILE,
        LIVE_PREDICTION_FILE,
    )

    print()
    print(
        "Live practice results synchronized:"
    )
    print(
        f"  RAW        -> {LIVE_RAW_FILE}"
    )
    print(
        f"  PREDICTION -> {LIVE_PREDICTION_FILE}"
    )

    return True


def detect_practice_stage():
    """
    Detect the latest practice session actually present
    in live_practice_results.csv.

    No results = PRE_PRACTICE
    Practice 1   = FP1
    Practice 2   = FP2
    Practice 3   = FP3
    """

    if not LIVE_PREDICTION_FILE.exists():
        return "PRE_PRACTICE"

    try:
        live = pd.read_csv(
            LIVE_PREDICTION_FILE
        )
    except Exception:
        return "PRE_PRACTICE"

    if live.empty:
        return "PRE_PRACTICE"

    if "session_name" not in live.columns:
        return "PRE_PRACTICE"

    sessions = (
        live["session_name"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if sessions.str.contains(
        "practice 3",
        na=False,
    ).any():

        return "FP3"

    if sessions.str.contains(
        "practice 2",
        na=False,
    ).any():

        return "FP2"

    if sessions.str.contains(
        "practice 1",
        na=False,
    ).any():

        return "FP1"

    return "PRE_PRACTICE"


def detect_current_stage():
    """
    Weekend-state stage is authoritative once the weekend
    progresses beyond practice.

    During practice, live practice results are also used
    so the system remains responsive even if the schedule
    clock has not advanced exactly as expected.
    """

    weekend_state = load_weekend_state()

    scheduled_stage = normalize_stage(
        weekend_state.get(
            "current_stage",
            "pre_practice",
        )
    )

    practice_stage = detect_practice_stage()

    stage_order = {
        "PRE_PRACTICE": 0,
        "FP1": 1,
        "FP2": 2,
        "FP3": 3,
        "QUALIFYING": 4,
        "RACE": 5,
    }

    scheduled_value = stage_order.get(
        scheduled_stage,
        0,
    )

    practice_value = stage_order.get(
        practice_stage,
        0,
    )

    # Use the later stage available.
    if practice_value > scheduled_value:
        return practice_stage

    return scheduled_stage


def run_progression():
    run_script(
        "src/prediction/build_race_progression.py"
    )


def run_practice_stack():

    # ------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------

    run_script(
        "src/prediction/generate_current_prediction.py"
    )

    # ------------------------------------------------------------
    # FP1
    # ------------------------------------------------------------

    run_script(
        "src/prediction/build_live_fp1_features.py"
    )

    run_script(
        "src/prediction/generate_live_fp1_prediction.py"
    )

    # ------------------------------------------------------------
    # FP2
    # ------------------------------------------------------------

    run_script(
        "src/prediction/build_live_fp2_features.py"
    )

    run_script(
        "src/prediction/generate_live_fp2_prediction.py"
    )


def run_fp3_stack():

    run_practice_stack()

    # ------------------------------------------------------------
    # FP3
    # ------------------------------------------------------------

    run_script(
        "src/prediction/build_live_fp3_features.py"
    )

    run_script(
        "src/prediction/generate_live_fp3_prediction.py"
    )


def run_qualifying_stack():

    # ------------------------------------------------------------
    # Rebuild all practice predictions so the progression
    # remains complete even when the app is opened later.
    # ------------------------------------------------------------

    run_fp3_stack()

    # ------------------------------------------------------------
    # Download qualifying / session results.
    # ------------------------------------------------------------

    run_script(
        "src/data/download_live_sessions.py"
    )

    # ------------------------------------------------------------
    # Build qualifying features.
    # ------------------------------------------------------------

    run_script(
        "src/prediction/build_live_qualifying_features.py"
    )

    # ------------------------------------------------------------
    # Generate final race prediction after qualifying.
    # ------------------------------------------------------------

    run_script(
        "src/prediction/generate_live_qualifying_prediction.py"
    )


def run_race_stack():

    # ------------------------------------------------------------
    # Keep all previous prediction stages available.
    # ------------------------------------------------------------

    run_qualifying_stack()

    # ------------------------------------------------------------
    # Refresh live race / qualifying session data.
    # ------------------------------------------------------------

    run_script(
        "src/data/download_live_sessions.py"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("LIVE PREDICTION PIPELINE")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. Update weekend state
    # ------------------------------------------------------------

    run_script(
        "src/racing/weekend_state.py"
    )

    # ------------------------------------------------------------
    # 2. Refresh live practice data
    # ------------------------------------------------------------

    run_script(
        "src/data/download_live_practice.py"
    )

    # ------------------------------------------------------------
    # 3. Synchronize raw -> prediction location
    # ------------------------------------------------------------

    copy_live_practice_file()

    # ------------------------------------------------------------
    # 4. Detect current stage
    # ------------------------------------------------------------

    stage = detect_current_stage()

    print()
    print("=" * 80)
    print(f"DETECTED STAGE: {stage}")
    print("=" * 80)

    # ============================================================
    # PRE-PRACTICE
    # ============================================================

    if stage == "PRE_PRACTICE":

        run_script(
            "src/prediction/generate_current_prediction.py"
        )

        run_script(
            "src/prediction/calculate_points.py"
        )

        run_progression()

    # ============================================================
    # FP1
    # ============================================================

    elif stage == "FP1":

        run_script(
            "src/prediction/generate_current_prediction.py"
        )

        run_script(
            "src/prediction/build_live_fp1_features.py"
        )

        run_script(
            "src/prediction/generate_live_fp1_prediction.py"
        )

        run_script(
            "src/prediction/calculate_points.py"
        )

        run_progression()

    # ============================================================
    # FP2
    # ============================================================

    elif stage == "FP2":

        run_practice_stack()

        run_script(
            "src/prediction/calculate_points.py"
        )

        run_progression()

    # ============================================================
    # FP3
    # ============================================================

    elif stage == "FP3":

        run_fp3_stack()

        run_script(
            "src/prediction/calculate_points.py"
        )

        run_progression()

    # ============================================================
    # QUALIFYING
    # ============================================================

    elif stage == "QUALIFYING":

        run_qualifying_stack()

        run_script(
            "src/prediction/calculate_points.py"
        )

        run_progression()

    # ============================================================
    # RACE
    # ============================================================

    elif stage == "RACE":

        run_race_stack()

        run_script(
            "src/prediction/calculate_points.py"
        )

        run_progression()

    # ============================================================
    # UNKNOWN
    # ============================================================

    else:

        raise RuntimeError(
            f"Unsupported live stage: {stage}"
        )

    # ------------------------------------------------------------
    # FINAL STATUS
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("LIVE PIPELINE COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Final detected stage: {stage}"
    )

    print(
        f"Progression file: "
        f"{PREDICTION_DIR / 'current_race_progression.csv'}"
    )

    print()


if __name__ == "__main__":
    main()