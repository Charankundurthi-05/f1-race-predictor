from pathlib import Path
import json
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models"
PREDICTION_DIR = ROOT / "data" / "predictions"

MODEL_FILE = (
    MODEL_DIR /
    "sprint_gb_shallow_final_model.joblib"
)

METADATA_FILE = (
    MODEL_DIR /
    "sprint_gb_shallow_final_metadata.json"
)

FEATURE_FILE = (
    PREDICTION_DIR /
    "current_2026_sprint_features.csv"
)

OUTPUT_FILE = (
    PREDICTION_DIR /
    "current_2026_sprint_prediction.csv"
)

CURRENT_OUTPUT_FILE = (
    PREDICTION_DIR /
    "current_prediction.csv"
)

CURRENT_POINTS_FILE = (
    PREDICTION_DIR /
    "current_prediction_with_sprint_points.csv"
)


SPRINT_POINTS = {
    1: 8,
    2: 7,
    3: 6,
    4: 5,
    5: 4,
    6: 3,
    7: 2,
    8: 1,
}


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("GENERATE LIVE SPRINT PREDICTION")
    print("=" * 80)
    print()

    if not FEATURE_FILE.exists():

        print(
            "No live sprint feature file exists."
        )

        print(
            "Run build_live_sprint_features.py "
            "after Sprint Qualifying."
        )

        return

    if not MODEL_FILE.exists():

        print(
            "Sprint model not found."
        )

        print(
            f"Expected -> {MODEL_FILE}"
        )

        return

    if not METADATA_FILE.exists():

        print(
            "Sprint model metadata not found."
        )

        return

    features_df = pd.read_csv(
        FEATURE_FILE
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    features = metadata.get(
        "features",
        []
    )

    if not features:

        print(
            "Sprint model metadata contains no features."
        )

        return

    missing = [
        feature
        for feature in features
        if feature not in features_df.columns
    ]

    if missing:

        print(
            "MODEL FEATURE CHECK: FAIL"
        )

        for feature in missing:
            print(
                f"  - {feature}"
            )

        return

    if "driver_name" not in features_df.columns:

        print(
            "driver_name column missing."
        )

        return

    if len(features_df) != 22:

        print(
            f"Driver count check: FAIL "
            f"({len(features_df)} rows)"
        )

        return

    if (
        features_df["driver_name"]
        .nunique()
        != 22
    ):

        print(
            "Driver uniqueness check: FAIL"
        )

        return

    print(
        f"Input rows: {len(features_df)}"
    )

    print(
        f"Model features: {len(features)}"
    )

    print()

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------

    model = joblib.load(
        MODEL_FILE
    )

    # ---------------------------------------------------------------
    # Predict sprint finishing position
    # ---------------------------------------------------------------

    predicted_finish = model.predict(
        features_df[features]
    )

    output = features_df[
        [
            column
            for column in [
                "season",
                "round",
                "race_name",
                "race_date",
                "circuit_name",
                "driver_name",
                "driver_code",
                "team_name",
                "sprint_qualifying_position"
            ]
            if column in features_df.columns
        ]
    ].copy()

    output[
        "predicted_sprint_finish_raw"
    ] = predicted_finish

    # Lower predicted finish = better position.
    output = output.sort_values(
        "predicted_sprint_finish_raw"
    ).reset_index(
        drop=True
    )

    output[
        "predicted_sprint_position"
    ] = (
        output.index + 1
    )

    output[
        "predicted_sprint_points"
    ] = (
        output[
            "predicted_sprint_position"
        ]
        .map(SPRINT_POINTS)
        .fillna(0)
        .astype(int)
    )

    # ---------------------------------------------------------------
    # Save dedicated sprint prediction
    # ---------------------------------------------------------------

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------------
    # Display prediction
    # ---------------------------------------------------------------

    print()
    print("=" * 80)
    print("PREDICTED SPRINT GRID")
    print("=" * 80)

    for _, row in output.iterrows():

        position = int(
            row["predicted_sprint_position"]
        )

        points = int(
            row["predicted_sprint_points"]
        )

        print(
            f"{position:2d}. "
            f"{row['driver_name']} "
            f"({row['team_name']}) "
            f"— {points} pts"
        )

    # ---------------------------------------------------------------
    # Update current prediction
    # ---------------------------------------------------------------

    current = output.copy()

    current[
        "predicted_position"
    ] = current[
        "predicted_sprint_position"
    ]

    current[
        "predicted_race_points"
    ] = (
        current["predicted_position"]
        .map({
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
        })
        .fillna(0)
        .astype(int)
    )

    current[
        "predicted_total_points"
    ] = (
        current["predicted_race_points"]
        + current["predicted_sprint_points"]
    )

    current.to_csv(
        CURRENT_OUTPUT_FILE,
        index=False
    )

    current.to_csv(
        CURRENT_POINTS_FILE,
        index=False
    )

    print()
    print(
        f"Saved -> {OUTPUT_FILE}"
    )

    print(
        f"Updated -> {CURRENT_OUTPUT_FILE}"
    )

    print(
        f"Updated -> {CURRENT_POINTS_FILE}"
    )

    print()
    print("=" * 80)
    print("LIVE SPRINT PREDICTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()