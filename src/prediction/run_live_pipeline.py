from pathlib import Path
import subprocess
import sys
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = ROOT / "data" / "predictions"


def run_script(script):

    print()
    print("=" * 80)
    print(f"RUNNING: {script}")
    print("=" * 80)
    print()

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / script)
        ],
        cwd=ROOT
    )

    return result.returncode == 0


def get_weekend_format():

    state_file = (
        PREDICTION_DIR /
        "weekend_state.json"
    )

    if state_file.exists():

        try:

            with open(
                state_file,
                "r",
                encoding="utf-8"
            ) as f:
                state = json.load(f)

            return str(
                state.get(
                    "weekend_format",
                    "normal"
                )
            ).lower()

        except Exception:
            pass

    return "normal"


def detect_stage():

    qualifying_file = (
        PREDICTION_DIR /
        "live_qualifying_results.csv"
    )

    sprint_qualifying_file = (
        PREDICTION_DIR /
        "live_sprint_qualifying_results.csv"
    )

    sprint_results_file = (
        PREDICTION_DIR /
        "live_sprint_results.csv"
    )

    practice_file = (
        PREDICTION_DIR /
        "live_practice_results.csv"
    )

    weekend_format = get_weekend_format()

    # ---------------------------------------------------------------
    # Sprint weekend
    # ---------------------------------------------------------------

    if weekend_format == "sprint":

        if qualifying_file.exists():

            try:
                if not pd.read_csv(
                    qualifying_file
                ).empty:
                    return "QUALIFYING"
            except Exception:
                pass

        if sprint_results_file.exists():

            try:
                if not pd.read_csv(
                    sprint_results_file
                ).empty:
                    return "QUALIFYING"
            except Exception:
                pass

        if sprint_qualifying_file.exists():

            try:
                if not pd.read_csv(
                    sprint_qualifying_file
                ).empty:
                    return "SPRINT"
            except Exception:
                pass

        return "PRE_PRACTICE"

    # ---------------------------------------------------------------
    # Normal weekend
    # ---------------------------------------------------------------

    if qualifying_file.exists():

        try:
            if not pd.read_csv(
                qualifying_file
            ).empty:
                return "QUALIFYING"
        except Exception:
            pass

    if practice_file.exists():

        try:

            df = pd.read_csv(
                practice_file
            )

            if not df.empty:

                if "session_name" in df.columns:

                    sessions = (
                        df["session_name"]
                        .astype(str)
                        .str.upper()
                    )

                elif "session" in df.columns:

                    sessions = (
                        df["session"]
                        .astype(str)
                        .str.upper()
                    )

                else:
                    sessions = pd.Series(
                        dtype=str
                    )

                if sessions.str.contains(
                    "PRACTICE 3"
                ).any():
                    return "FP3"

                if sessions.str.contains(
                    "PRACTICE 2"
                ).any():
                    return "FP2"

                if sessions.str.contains(
                    "PRACTICE 1"
                ).any():
                    return "FP1"

        except Exception:
            pass

    return "PRE_PRACTICE"


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("LIVE PREDICTION PIPELINE")
    print("=" * 80)

    # ---------------------------------------------------------------
    # Refresh weekend state
    # ---------------------------------------------------------------

    run_script(
        "src/racing/weekend_state.py"
    )

    # ---------------------------------------------------------------
    # Determine weekend type
    # ---------------------------------------------------------------

    weekend_format = get_weekend_format()

    print()
    print(
        f"Weekend format: "
        f"{weekend_format.upper()}"
    )

    # ---------------------------------------------------------------
    # Download normal practice data
    # ---------------------------------------------------------------

    run_script(
        "src/data/download_live_practice.py"
    )

    # ---------------------------------------------------------------
    # Sprint qualifying downloader
    # ---------------------------------------------------------------

    if weekend_format == "sprint":

        run_script(
            "src/data/download_live_sprint_qualifying.py"
        )

    # ---------------------------------------------------------------
    # Detect current stage
    # ---------------------------------------------------------------

    stage = detect_stage()

    print()
    print(
        "=" * 80
    )
    print(
        f"DETECTED STAGE: {stage}"
    )
    print(
        "=" * 80
    )

    # ---------------------------------------------------------------
    # PRE-PRACTICE
    # ---------------------------------------------------------------

    if stage == "PRE_PRACTICE":

        run_script(
            "src/prediction/generate_current_prediction.py"
        )

        run_script(
            "src/prediction/calculate_points.py"
        )

    # ---------------------------------------------------------------
    # FP1
    # ---------------------------------------------------------------

    elif stage == "FP1":

        run_script(
            "src/prediction/build_live_fp1_features.py"
        )

        run_script(
            "src/prediction/generate_live_fp1_prediction.py"
        )

    # ---------------------------------------------------------------
    # FP2
    # ---------------------------------------------------------------

    elif stage == "FP2":

        run_script(
            "src/prediction/build_live_fp2_features.py"
        )

        run_script(
            "src/prediction/generate_live_fp2_prediction.py"
        )

    # ---------------------------------------------------------------
    # FP3
    # ---------------------------------------------------------------

    elif stage == "FP3":

        run_script(
            "src/prediction/build_live_fp3_features.py"
        )

        run_script(
            "src/prediction/generate_live_fp3_prediction.py"
        )

    # ---------------------------------------------------------------
    # SPRINT
    # ---------------------------------------------------------------

    elif stage == "SPRINT":

        print()
        print(
            "SPRINT QUALIFYING DETECTED."
        )

        print(
            "Building live sprint features..."
        )

        run_script(
            "src/prediction/build_live_sprint_features.py"
        )

        run_script(
            "src/prediction/generate_live_sprint_prediction.py"
        )

    # ---------------------------------------------------------------
    # QUALIFYING
    # ---------------------------------------------------------------

    elif stage == "QUALIFYING":

        # On sprint weekends the sprint has already happened.
        if weekend_format == "sprint":

            sprint_file = (
                PREDICTION_DIR /
                "live_sprint_results.csv"
            )

            if sprint_file.exists():

                try:

                    sprint_df = pd.read_csv(
                        sprint_file
                    )

                    if not sprint_df.empty:

                        print()
                        print(
                            "Sprint results detected."
                        )

                        print(
                            "Sprint points can be "
                            "included in weekend totals."
                        )

                except Exception:
                    pass

        run_script(
            "src/prediction/build_live_qualifying_features.py"
        )

        run_script(
            "src/prediction/generate_live_qualifying_prediction.py"
        )

    print()
    print("=" * 80)
    print("LIVE PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()