from pathlib import Path
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = ROOT / "data" / "predictions"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


# ================================================================
# HELPERS
# ================================================================

def load_csv(path):
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return None


def load_json(path):
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return None


def normalise_name(value):

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = (
        value
        .replace("’", "'")
        .replace("-", " ")
        .replace("_", " ")
    )

    value = re.sub(
        r"[^a-z0-9 ]",
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def find_column(df, candidates):

    if df is None:
        return None

    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        key = str(candidate).strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def clean_position(value):

    if pd.isna(value):
        return pd.NA

    try:

        number = float(value)

        if number <= 0:
            return pd.NA

        return int(number)

    except Exception:

        match = re.match(
            r"^(\d+)",
            str(value).strip(),
        )

        if match:
            return int(match.group(1))

    return pd.NA


def make_driver_key(driver_id=None, driver_name=None):

    if driver_id is not None and pd.notna(driver_id):

        value = normalise_name(driver_id)

        if value:
            return f"id:{value}"

    if driver_name is not None and pd.notna(driver_name):

        value = normalise_name(driver_name)

        if value:
            return f"name:{value}"

    return ""


# ================================================================
# PREDICTIONS
# ================================================================

def load_prediction_stage(
    filename,
    season,
    round_number,
):

    path = PREDICTION_DIR / filename

    df = load_csv(path)

    if df is None:
        return {}

    season_col = find_column(
        df,
        ["season", "year"],
    )

    round_col = find_column(
        df,
        ["round"],
    )

    position_col = find_column(
        df,
        [
            "predicted_position",
            "prediction_position",
        ],
    )

    driver_id_col = find_column(
        df,
        [
            "driver_id",
            "driverId",
        ],
    )

    driver_name_col = find_column(
        df,
        [
            "driver_name",
            "driverName",
            "full_name",
        ],
    )

    if position_col is None:
        return {}

    if season_col:

        df = df[
            pd.to_numeric(
                df[season_col],
                errors="coerce",
            )
            == season
        ]

    if round_col:

        df = df[
            pd.to_numeric(
                df[round_col],
                errors="coerce",
            )
            == round_number
        ]

    if df.empty:
        return {}

    result = {}

    for _, row in df.iterrows():

        driver_key = make_driver_key(
            row[driver_id_col]
            if driver_id_col
            else None,
            row[driver_name_col]
            if driver_name_col
            else None,
        )

        if not driver_key:
            continue

        position = clean_position(
            row[position_col]
        )

        if pd.notna(position):

            result[driver_key] = int(position)

    return result


# ================================================================
# CURRENT PREDICTION
# ================================================================

def load_current_prediction(
    season,
    round_number,
):

    files = [
        "current_prediction_with_points.csv",
        "current_prediction.csv",
    ]

    for filename in files:

        path = PREDICTION_DIR / filename

        df = load_csv(path)

        if df is None:
            continue

        season_col = find_column(
            df,
            ["season", "year"],
        )

        round_col = find_column(
            df,
            ["round"],
        )

        position_col = find_column(
            df,
            [
                "predicted_position",
                "prediction_position",
            ],
        )

        driver_id_col = find_column(
            df,
            [
                "driver_id",
                "driverId",
            ],
        )

        driver_name_col = find_column(
            df,
            [
                "driver_name",
                "driverName",
                "full_name",
            ],
        )

        team_col = find_column(
            df,
            [
                "team_name",
                "constructor_name",
                "constructor",
                "team",
            ],
        )

        if position_col is None:
            continue

        if season_col:

            filtered = df[
                pd.to_numeric(
                    df[season_col],
                    errors="coerce",
                )
                == season
            ]

        else:

            filtered = df.copy()

        if round_col:

            filtered = filtered[
                pd.to_numeric(
                    filtered[round_col],
                    errors="coerce",
                )
                == round_number
            ]

        if filtered.empty:
            continue

        result = {}

        for _, row in filtered.iterrows():

            driver_key = make_driver_key(
                row[driver_id_col]
                if driver_id_col
                else None,
                row[driver_name_col]
                if driver_name_col
                else None,
            )

            if not driver_key:
                continue

            position = clean_position(
                row[position_col]
            )

            if pd.isna(position):
                continue

            result[driver_key] = {
                "driver_name":
                    str(
                        row[driver_name_col]
                    )
                    if driver_name_col
                    else driver_key,
                "team_name":
                    str(
                        row[team_col]
                    )
                    if team_col
                    else "—",
                "position":
                    int(position),
            }

        if result:
            return result

    return {}


# ================================================================
# PRACTICE RESULTS
# ================================================================

def load_practice_results(
    season,
    round_number,
):

    files = [
        RAW_DIR / "practice_results.csv",
        PROCESSED_DIR / "practice_results.csv",
    ]

    df = None

    for path in files:

        df = load_csv(path)

        if df is not None:
            break

    if df is None:
        return {
            "fp1": {},
            "fp2": {},
            "fp3": {},
        }

    season_col = find_column(
        df,
        ["season", "year"],
    )

    round_col = find_column(
        df,
        ["round"],
    )

    session_col = find_column(
        df,
        [
            "session",
            "session_name",
            "sessionName",
            "practice_session",
        ],
    )

    position_col = find_column(
        df,
        [
            "position",
            "position_number",
            "positionNumber",
            "position_display_order",
            "positionDisplayOrder",
        ],
    )

    driver_id_col = find_column(
        df,
        [
            "driver_id",
            "driverId",
        ],
    )

    driver_name_col = find_column(
        df,
        [
            "driver_name",
            "driverName",
            "full_name",
        ],
    )

    result = {
        "fp1": {},
        "fp2": {},
        "fp3": {},
    }

    if position_col is None:
        return result

    if season_col:

        df = df[
            pd.to_numeric(
                df[season_col],
                errors="coerce",
            )
            == season
        ]

    if round_col:

        df = df[
            pd.to_numeric(
                df[round_col],
                errors="coerce",
            )
            == round_number
        ]

    if df.empty or session_col is None:
        return result

    for _, row in df.iterrows():

        session = normalise_name(
            row[session_col]
        )

        if (
            "practice 1" in session
            or session == "fp1"
        ):

            stage = "fp1"

        elif (
            "practice 2" in session
            or session == "fp2"
        ):

            stage = "fp2"

        elif (
            "practice 3" in session
            or session == "fp3"
        ):

            stage = "fp3"

        else:
            continue

        driver_key = make_driver_key(
            row[driver_id_col]
            if driver_id_col
            else None,
            row[driver_name_col]
            if driver_name_col
            else None,
        )

        if not driver_key:
            continue

        position = clean_position(
            row[position_col]
        )

        if pd.notna(position):

            result[stage][driver_key] = int(
                position
            )

    return result


# ================================================================
# QUALIFYING
# ================================================================

def load_qualifying_results(
    season,
    round_number,
):

    path = RAW_DIR / "qualifying_results.csv"

    df = load_csv(path)

    if df is None:
        return {}

    season_col = find_column(
        df,
        ["season", "year"],
    )

    round_col = find_column(
        df,
        ["round"],
    )

    position_col = find_column(
        df,
        [
            "qualifying_position",
            "position",
        ],
    )

    driver_id_col = find_column(
        df,
        [
            "driver_id",
            "driverId",
        ],
    )

    driver_name_col = find_column(
        df,
        [
            "driver_name",
            "driverName",
            "full_name",
        ],
    )

    if position_col is None:
        return {}

    if season_col:

        df = df[
            pd.to_numeric(
                df[season_col],
                errors="coerce",
            )
            == season
        ]

    if round_col:

        df = df[
            pd.to_numeric(
                df[round_col],
                errors="coerce",
            )
            == round_number
        ]

    result = {}

    for _, row in df.iterrows():

        driver_key = make_driver_key(
            row[driver_id_col]
            if driver_id_col
            else None,
            row[driver_name_col]
            if driver_name_col
            else None,
        )

        if not driver_key:
            continue

        position = clean_position(
            row[position_col]
        )

        if pd.notna(position):

            result[driver_key] = int(
                position
            )

    return result


# ================================================================
# RACE RESULT
# ================================================================

def load_race_results(
    season,
    round_number,
):

    path = RAW_DIR / "race_results.csv"

    df = load_csv(path)

    if df is None:
        return {}

    season_col = find_column(
        df,
        ["season", "year"],
    )

    round_col = find_column(
        df,
        ["round"],
    )

    position_col = find_column(
        df,
        [
            "finish_position",
            "position",
            "finishPosition",
        ],
    )

    driver_id_col = find_column(
        df,
        [
            "driver_id",
            "driverId",
        ],
    )

    driver_name_col = find_column(
        df,
        [
            "driver_name",
            "driverName",
            "full_name",
        ],
    )

    if position_col is None:
        return {}

    if season_col:

        df = df[
            pd.to_numeric(
                df[season_col],
                errors="coerce",
            )
            == season
        ]

    if round_col:

        df = df[
            pd.to_numeric(
                df[round_col],
                errors="coerce",
            )
            == round_number
        ]

    result = {}

    for _, row in df.iterrows():

        driver_key = make_driver_key(
            row[driver_id_col]
            if driver_id_col
            else None,
            row[driver_name_col]
            if driver_name_col
            else None,
        )

        if not driver_key:
            continue

        position = clean_position(
            row[position_col]
        )

        if pd.notna(position):

            result[driver_key] = int(
                position
            )

    return result


# ================================================================
# BUILD PROGRESSION
# ================================================================

def build_progression():

    state = load_json(
        PREDICTION_DIR / "weekend_state.json"
    )

    if state is None:

        print(
            "weekend_state.json not found."
        )

        return

    season = int(
        state.get(
            "season",
            2026,
        )
    )

    round_number = int(
        state.get(
            "round",
            1,
        )
    )

    race_name = state.get(
        "race_name",
        "Upcoming Grand Prix",
    )

    weekend_format = state.get(
        "weekend_format",
        "normal",
    )


    print("")
    print("F1 RACE PREDICTOR")
    print("BUILD RACE WEEKEND PROGRESSION")
    print("")
    print(
        f"Season: {season}"
    )
    print(
        f"Round: {round_number}"
    )
    print(
        f"Race: {race_name}"
    )
    print(
        f"Weekend: {str(weekend_format).upper()}"
    )
    print("")


    # ------------------------------------------------------------
    # BASE DRIVER LIST
    # ------------------------------------------------------------

    print(
        "Loading current prediction..."
    )

    current_prediction = load_current_prediction(
        season,
        round_number,
    )


    # ------------------------------------------------------------
    # STAGE PREDICTIONS
    # ------------------------------------------------------------

    print(
        "Loading prediction stages..."
    )

    predictions = {

        "no_practice":
            load_prediction_stage(
                "pre_practice_live_predictions.csv",
                season,
                round_number,
            ),

        "after_fp1":
            load_prediction_stage(
                "fp1_live_predictions.csv",
                season,
                round_number,
            ),

        "after_fp2":
            load_prediction_stage(
                "fp2_live_predictions.csv",
                season,
                round_number,
            ),

        "after_fp3":
            load_prediction_stage(
                "fp3_live_predictions.csv",
                season,
                round_number,
            ),

        "after_qualifying":
            load_prediction_stage(
                "qualifying_live_predictions.csv",
                season,
                round_number,
            ),
    }


    # ------------------------------------------------------------
    # ACTUAL RESULTS
    # ------------------------------------------------------------

    print(
        "Loading actual practice results..."
    )

    practice = load_practice_results(
        season,
        round_number,
    )

    print(
        "Loading qualifying result..."
    )

    qualifying = load_qualifying_results(
        season,
        round_number,
    )

    print(
        "Loading race result..."
    )

    race = load_race_results(
        season,
        round_number,
    )


    # ------------------------------------------------------------
    # DRIVER INFORMATION FROM CURRENT PREDICTION
    # ------------------------------------------------------------

    driver_information = {}

    for key, info in current_prediction.items():

        driver_information[key] = {
            "driver_name":
                info["driver_name"],
            "team_name":
                info["team_name"],
        }


    # ------------------------------------------------------------
    # ADD DRIVERS FROM EVERY AVAILABLE SOURCE
    # ------------------------------------------------------------

    all_keys = set(
        current_prediction.keys()
    )

    for prediction_dict in predictions.values():

        all_keys.update(
            prediction_dict.keys()
        )

    all_keys.update(
        practice["fp1"].keys()
    )

    all_keys.update(
        practice["fp2"].keys()
    )

    all_keys.update(
        practice["fp3"].keys()
    )

    all_keys.update(
        qualifying.keys()
    )

    all_keys.update(
        race.keys()
    )


    # ------------------------------------------------------------
    # BUILD ROWS
    # ------------------------------------------------------------

    rows = []

    for driver_key in all_keys:

        info = driver_information.get(
            driver_key,
            {
                "driver_name":
                    driver_key.replace(
                        "id:",
                        ""
                    ).replace(
                        "name:",
                        ""
                    ),
                "team_name":
                    "—",
            },
        )


        # --------------------------------------------------------
        # GET PREDICTIONS
        # --------------------------------------------------------

        no_practice = predictions[
            "no_practice"
        ].get(
            driver_key,
            pd.NA,
        )

        if pd.isna(no_practice):

            if driver_key in current_prediction:

                no_practice = (
                    current_prediction[
                        driver_key
                    ]["position"]
                )


        after_fp1 = predictions[
            "after_fp1"
        ].get(
            driver_key,
            pd.NA,
        )

        after_fp2 = predictions[
            "after_fp2"
        ].get(
            driver_key,
            pd.NA,
        )

        after_fp3 = predictions[
            "after_fp3"
        ].get(
            driver_key,
            pd.NA,
        )

        final_prediction = predictions[
            "after_qualifying"
        ].get(
            driver_key,
            pd.NA,
        )


        # --------------------------------------------------------
        # ACTUAL SESSION RESULTS
        # --------------------------------------------------------

        fp1_result = practice[
            "fp1"
        ].get(
            driver_key,
            pd.NA,
        )

        fp2_result = practice[
            "fp2"
        ].get(
            driver_key,
            pd.NA,
        )

        fp3_result = practice[
            "fp3"
        ].get(
            driver_key,
            pd.NA,
        )

        qualifying_result = qualifying.get(
            driver_key,
            pd.NA,
        )

        race_result = race.get(
            driver_key,
            pd.NA,
        )


        # --------------------------------------------------------
        # FINAL PREDICTION
        # --------------------------------------------------------

        # Until qualifying prediction exists,
        # keep the latest available prediction as the
        # temporary displayed prediction.

        if pd.isna(final_prediction):

            if pd.notna(after_fp3):

                final_prediction = after_fp3

            elif pd.notna(after_fp2):

                final_prediction = after_fp2

            elif pd.notna(after_fp1):

                final_prediction = after_fp1

            elif pd.notna(no_practice):

                final_prediction = no_practice


        # --------------------------------------------------------
        # DIFFERENCE
        # --------------------------------------------------------

        if (
            pd.notna(
                predictions[
                    "after_qualifying"
                ].get(
                    driver_key,
                    pd.NA,
                )
            )
            and pd.notna(race_result)
        ):

            difference = (
                int(race_result)
                -
                int(
                    predictions[
                        "after_qualifying"
                    ][driver_key]
                )
            )

        else:

            difference = pd.NA


        rows.append(
            {
                "driver_name":
                    info["driver_name"],

                "team_name":
                    info["team_name"],

                "no_practice_prediction":
                    no_practice,

                "fp1_result":
                    fp1_result,

                "prediction_after_fp1":
                    after_fp1,

                "fp2_result":
                    fp2_result,

                "prediction_after_fp2":
                    after_fp2,

                "fp3_result":
                    fp3_result,

                "prediction_after_fp3":
                    after_fp3,

                "qualifying_result":
                    qualifying_result,

                "final_prediction":
                    final_prediction,

                "race_result":
                    race_result,

                "prediction_difference":
                    difference,
            }
        )


    # ------------------------------------------------------------
    # CREATE DATAFRAME
    # ------------------------------------------------------------

    result = pd.DataFrame(
        rows,
        columns=[
            "driver_name",
            "team_name",
            "no_practice_prediction",
            "fp1_result",
            "prediction_after_fp1",
            "fp2_result",
            "prediction_after_fp2",
            "fp3_result",
            "prediction_after_fp3",
            "qualifying_result",
            "final_prediction",
            "race_result",
            "prediction_difference",
        ],
    )


    # ------------------------------------------------------------
    # SORT
    # ------------------------------------------------------------

    if not result.empty:

        result["_sort"] = pd.to_numeric(
            result["final_prediction"],
            errors="coerce",
        )

        result = result.sort_values(
            [
                "_sort",
                "driver_name",
            ],
            na_position="last",
        )

        result = result.drop(
            columns="_sort"
        )

    result.insert(
        0,
        "position",
        range(
            1,
            len(result) + 1,
        ),
    )


    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    output_path = (
        PREDICTION_DIR
        / "current_race_progression.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )


    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    print("")
    print(
        f"Drivers: {len(result)}"
    )

    print(
        f"No-practice predictions: "
        f"{result['no_practice_prediction'].notna().sum()}"
    )

    print(
        f"FP1 results: "
        f"{result['fp1_result'].notna().sum()}"
    )

    print(
        f"Predictions after FP1: "
        f"{result['prediction_after_fp1'].notna().sum()}"
    )

    print(
        f"FP2 results: "
        f"{result['fp2_result'].notna().sum()}"
    )

    print(
        f"Predictions after FP2: "
        f"{result['prediction_after_fp2'].notna().sum()}"
    )

    print(
        f"FP3 results: "
        f"{result['fp3_result'].notna().sum()}"
    )

    print(
        f"Predictions after FP3: "
        f"{result['prediction_after_fp3'].notna().sum()}"
    )

    print(
        f"Qualifying results: "
        f"{result['qualifying_result'].notna().sum()}"
    )

    print(
        f"Final predictions: "
        f"{result['final_prediction'].notna().sum()}"
    )

    print(
        f"Race results: "
        f"{result['race_result'].notna().sum()}"
    )

    print(
        f"Final prediction/race comparisons: "
        f"{result['prediction_difference'].notna().sum()}"
    )

    print("")
    print(
        f"Saved -> {output_path}"
    )
    print("")
    print(
        "RACE PROGRESSION COMPLETE"
    )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    build_progression()