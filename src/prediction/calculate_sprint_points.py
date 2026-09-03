from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = ROOT / "data" / "predictions"


SPRINT_POINTS = {
    1: 8,
    2: 7,
    3: 6,
    4: 5,
    5: 4,
    6: 3,
    7: 2,
    8: 1
}


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("SPRINT POINTS ENGINE")
    print("=" * 80)
    print()

    prediction_file = (
        PREDICTION_DIR /
        "current_prediction.csv"
    )

    if not prediction_file.exists():
        print("No current prediction exists.")
        print("Run the current prediction pipeline first.")
        return

    df = pd.read_csv(prediction_file)

    if "predicted_position" not in df.columns:
        raise ValueError(
            "predicted_position is missing."
        )

    if "weekend_format" in df.columns:

        sprint_weekend = (
            df["weekend_format"]
            .astype(str)
            .str.lower()
            .eq("sprint")
            .any()
        )

    else:
        sprint_weekend = False

    # ---------------------------------------------------------------
    # Normal weekend
    # ---------------------------------------------------------------

    if not sprint_weekend:

        df["predicted_sprint_position"] = pd.NA
        df["predicted_sprint_points"] = 0

        print("Current weekend: NORMAL")
        print("Sprint prediction: NOT APPLICABLE")

    # ---------------------------------------------------------------
    # Sprint weekend
    # ---------------------------------------------------------------

    else:

        print("Current weekend: SPRINT")

        # Until a dedicated sprint-position model is connected,
        # qualifying/predicted race order is used as the provisional
        # sprint order.
        #
        # This will be replaced by the dedicated sprint model when
        # sprint-weekend live features are implemented.

        df["predicted_sprint_position"] = (
            df["predicted_position"]
        )

        df["predicted_sprint_points"] = (
            df["predicted_sprint_position"]
            .map(SPRINT_POINTS)
            .fillna(0)
            .astype(int)
        )

    # ---------------------------------------------------------------
    # Total points
    # ---------------------------------------------------------------

    if "predicted_race_points" not in df.columns:

        race_points = {
            1: 25,
            2: 18,
            3: 15,
            4: 12,
            5: 10,
            6: 8,
            7: 6,
            8: 4,
            9: 2,
            10: 1
        }

        df["predicted_race_points"] = (
            df["predicted_position"]
            .map(race_points)
            .fillna(0)
            .astype(int)
        )

    df["predicted_total_points"] = (
        df["predicted_race_points"]
        + df["predicted_sprint_points"]
    )

    output_file = (
        PREDICTION_DIR /
        "current_prediction_with_sprint_points.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print()
    print("SPRINT POINTS")
    print("-" * 80)

    if sprint_weekend:

        for _, row in df.iterrows():

            print(
                f"{int(row['predicted_sprint_position']):2d}. "
                f"{row['driver_name']:<28} "
                f"{int(row['predicted_sprint_points']):2d} pts"
            )

    else:

        print(
            "No sprint points awarded on this weekend."
        )

    print()
    print(f"Saved -> {output_file}")

    print()
    print("=" * 80)
    print("SPRINT POINTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()