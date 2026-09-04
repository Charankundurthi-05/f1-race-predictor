from pathlib import Path
import json
import re
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = ROOT / "data" / "predictions"
RAW_DIR = ROOT / "data" / "raw"


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


def normalise_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

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
    ).strip()

    return value


def normalise_code(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(value).strip().upper(),
    )


def surname_key(value):
    text = normalise_text(value)

    if not text:
        return ""

    return text.split()[-1]


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


def canonical_code(value):
    code = normalise_code(value)

    if code:
        return code

    return ""


# ================================================================
# CURRENT DRIVER NAME MATCHING
# ================================================================

def build_name_aliases(name):
    aliases = set()

    normalized = normalise_text(name)

    if normalized:
        aliases.add(normalized)

        parts = normalized.split()

        if len(parts) >= 2:
            aliases.add(
                " ".join(parts[::-1])
            )

        aliases.add(parts[-1])

    return aliases


def match_name(
    target_name,
    candidate_names,
):
    target_aliases = build_name_aliases(
        target_name
    )

    # Exact normalized/full-name match
    for name, row in candidate_names:

        aliases = build_name_aliases(name)

        if target_aliases.intersection(aliases):
            return row

    return None


# ================================================================
# PREDICTION FILE LOADER
# ================================================================

def load_prediction_file(
    filename,
    season,
    round_number,
):
    path = PREDICTION_DIR / filename

    df = load_csv(path)

    if df is None or df.empty:
        return {}

    season_col = find_column(
        df,
        [
            "season",
            "year",
        ],
    )

    round_col = find_column(
        df,
        [
            "round",
            "race_round",
        ],
    )

    position_col = find_column(
        df,
        [
            "predicted_position",
            "prediction_position",
        ],
    )

    driver_name_col = find_column(
        df,
        [
            "driver_name",
            "driverName",
            "full_name",
            "name",
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

    if (
        position_col is None
        or driver_name_col is None
    ):
        return {}

    filtered = df.copy()

    if season_col:

        temp = filtered[
            pd.to_numeric(
                filtered[season_col],
                errors="coerce",
            )
            == season
        ]

        if not temp.empty:
            filtered = temp

    if round_col:

        temp = filtered[
            pd.to_numeric(
                filtered[round_col],
                errors="coerce",
            )
            == round_number
        ]

        if not temp.empty:
            filtered = temp

    if filtered.empty:
        return {}

    # IMPORTANT:
    # For progression joins we deliberately use normalized
    # driver names rather than source-specific driver IDs/codes.
    rows = []

    for _, row in filtered.iterrows():

        position = clean_position(
            row[position_col]
        )

        if pd.isna(position):
            continue

        rows.append(
            (
                row[driver_name_col],
                {
                    "position": int(position),
                    "driver_name": str(
                        row[driver_name_col]
                    ),
                    "team_name": (
                        str(row[team_col])
                        if team_col
                        else "—"
                    ),
                },
            )
        )

    if not rows:
        return {}

    return {
        normalise_text(name): info
        for name, info in rows
        if normalise_text(name)
    }


# ================================================================
# PRE-PRACTICE
# ================================================================

def load_no_practice_prediction(
    season,
    round_number,
):
    candidates = [
        "pre_practice_live_predictions.csv",
        "pre_practice_predictions_with_points.csv",
        "pre_practice_predictions.csv",
    ]

    for filename in candidates:

        raw = load_prediction_file(
            filename,
            season,
            round_number,
        )

        if not raw:
            continue

        # Convert name-keyed source into a clean lookup.
        result = {}

        for source_name, info in raw.items():

            key = normalise_text(
                source_name
            )

            if key:
                result[key] = info

                # Also allow reversed names.
                parts = key.split()

                if len(parts) >= 2:
                    result[
                        " ".join(parts[::-1])
                    ] = info

                # Surname fallback where unique.
                if len(parts) >= 2:
                    surname = parts[-1]

                    existing = result.get(
                        f"__surname__{surname}"
                    )

                    if existing is None:
                        result[
                            f"__surname__{surname}"
                        ] = info
                    else:
                        result[
                            f"__surname__{surname}"
                        ] = None

        print(
            f"Loaded pre-practice prediction: "
            f"{filename}"
        )

        return result

    return {}


def find_pre_practice_for_driver(
    driver_name,
    prediction_lookup,
):
    normalized = normalise_text(
        driver_name
    )

    if normalized in prediction_lookup:

        return prediction_lookup[
            normalized
        ]

    parts = normalized.split()

    if len(parts) >= 2:

        reversed_name = " ".join(
            parts[::-1]
        )

        if reversed_name in prediction_lookup:

            return prediction_lookup[
                reversed_name
            ]

    if parts:

        surname = parts[-1]

        key = f"__surname__{surname}"

        if key in prediction_lookup:

            value = prediction_lookup[key]

            if value is not None:
                return value

    return None


# ================================================================
# CURRENT DRIVERS
# ================================================================

def load_current_drivers(
    season,
    round_number,
):
    candidates = [
        "current_prediction.csv",
        "current_prediction_with_points.csv",
        "current_2026_fp2_prediction.csv",
        "current_2026_fp1_prediction.csv",
    ]

    for filename in candidates:

        path = PREDICTION_DIR / filename

        df = load_csv(path)

        if df is None or df.empty:
            continue

        season_col = find_column(
            df,
            [
                "season",
                "year",
            ],
        )

        round_col = find_column(
            df,
            [
                "round",
                "race_round",
            ],
        )

        driver_name_col = find_column(
            df,
            [
                "driver_name",
                "driverName",
                "full_name",
                "name",
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

        if driver_name_col is None:
            continue

        filtered = df.copy()

        if season_col:

            temp = filtered[
                pd.to_numeric(
                    filtered[season_col],
                    errors="coerce",
                )
                == season
            ]

            if not temp.empty:
                filtered = temp

        if round_col:

            temp = filtered[
                pd.to_numeric(
                    filtered[round_col],
                    errors="coerce",
                )
                == round_number
            ]

            if not temp.empty:
                filtered = temp

        if filtered.empty:
            continue

        drivers = {}

        for _, row in filtered.iterrows():

            name = str(
                row[driver_name_col]
            )

            key = normalise_text(name)

            if not key:
                continue

            drivers[key] = {
                "driver_name": name,
                "team_name": (
                    str(row[team_col])
                    if team_col
                    else "—"
                ),
            }

        if len(drivers) == 22:
            return drivers

    return {}


# ================================================================
# LIVE PRACTICE RESULTS
# ================================================================

def load_live_practice_results(
    season,
):
    path = (
        PREDICTION_DIR
        / "live_practice_results.csv"
    )

    df = load_csv(path)

    if df is None or df.empty:

        return {
            "fp1": {},
            "fp2": {},
            "fp3": {},
        }

    season_col = find_column(
        df,
        [
            "season",
            "year",
        ],
    )

    session_col = find_column(
        df,
        [
            "session_name",
            "session",
            "sessionName",
        ],
    )

    position_col = find_column(
        df,
        [
            "position",
            "position_number",
            "positionNumber",
            "positionDisplayOrder",
        ],
    )

    driver_name_col = find_column(
        df,
        [
            "driver_name",
            "driverName",
            "full_name",
            "name",
        ],
    )

    result = {
        "fp1": {},
        "fp2": {},
        "fp3": {},
    }

    if (
        session_col is None
        or position_col is None
        or driver_name_col is None
    ):
        return result

    filtered = df.copy()

    if season_col:

        temp = filtered[
            pd.to_numeric(
                filtered[season_col],
                errors="coerce",
            )
            == season
        ]

        if not temp.empty:
            filtered = temp

    for _, row in filtered.iterrows():

        session = normalise_text(
            row[session_col]
        )

        if "practice 1" in session:
            stage = "fp1"

        elif "practice 2" in session:
            stage = "fp2"

        elif "practice 3" in session:
            stage = "fp3"

        else:
            continue

        position = clean_position(
            row[position_col]
        )

        if pd.isna(position):
            continue

        name = str(
            row[driver_name_col]
        )

        result[stage][
            normalise_text(name)
        ] = {
            "position": int(position),
            "driver_name": name,
        }

    return result


def find_practice_result_for_driver(
    driver_name,
    stage_results,
):
    normalized = normalise_text(
        driver_name
    )

    if normalized in stage_results:
        return stage_results[normalized]

    parts = normalized.split()

    if len(parts) >= 2:

        reversed_name = " ".join(
            parts[::-1]
        )

        if reversed_name in stage_results:
            return stage_results[
                reversed_name
            ]

    # Surname fallback
    if parts:

        surname = parts[-1]

        matches = [
            info
            for key, info in stage_results.items()
            if key.split()[-1] == surname
        ]

        if len(matches) == 1:
            return matches[0]

    return None


# ================================================================
# QUALIFYING RESULT
# ================================================================

def load_qualifying_result(
    season,
    round_number,
):
    path = (
        PREDICTION_DIR
        / "live_qualifying_results.csv"
    )

    df = load_csv(path)

    if df is None or df.empty:
        return {}

    season_col = find_column(
        df,
        [
            "season",
            "year",
        ],
    )

    round_col = find_column(
        df,
        [
            "round",
            "race_round",
        ],
    )

    position_col = find_column(
        df,
        [
            "qualifying_position",
            "position",
        ],
    )

    driver_name_col = find_column(
        df,
        [
            "driver_name",
            "driverName",
            "full_name",
            "name",
        ],
    )

    if (
        position_col is None
        or driver_name_col is None
    ):
        return {}

    filtered = df.copy()

    if season_col:

        temp = filtered[
            pd.to_numeric(
                filtered[season_col],
                errors="coerce",
            )
            == season
        ]

        if not temp.empty:
            filtered = temp

    if round_col:

        temp = filtered[
            pd.to_numeric(
                filtered[round_col],
                errors="coerce",
            )
            == round_number
        ]

        if not temp.empty:
            filtered = temp

    result = {}

    for _, row in filtered.iterrows():

        position = clean_position(
            row[position_col]
        )

        if pd.isna(position):
            continue

        result[
            normalise_text(
                row[driver_name_col]
            )
        ] = int(position)

    return result


# ================================================================
# RACE RESULT
# ================================================================

def load_race_result(
    season,
    round_number,
):
    path = (
        PREDICTION_DIR
        / "live_race_results.csv"
    )

    df = load_csv(path)

    if df is None or df.empty:
        return {}

    season_col = find_column(
        df,
        [
            "season",
            "year",
        ],
    )

    round_col = find_column(
        df,
        [
            "round",
            "race_round",
        ],
    )

    position_col = find_column(
        df,
        [
            "finish_position",
            "position",
            "finishPosition",
        ],
    )

    driver_name_col = find_column(
        df,
        [
            "driver_name",
            "driverName",
            "full_name",
            "name",
        ],
    )

    if (
        position_col is None
        or driver_name_col is None
    ):
        return {}

    filtered = df.copy()

    if season_col:

        temp = filtered[
            pd.to_numeric(
                filtered[season_col],
                errors="coerce",
            )
            == season
        ]

        if not temp.empty:
            filtered = temp

    if round_col:

        temp = filtered[
            pd.to_numeric(
                filtered[round_col],
                errors="coerce",
            )
            == round_number
        ]

        if not temp.empty:
            filtered = temp

    result = {}

    for _, row in filtered.iterrows():

        position = clean_position(
            row[position_col]
        )

        if pd.isna(position):
            continue

        result[
            normalise_text(
                row[driver_name_col]
            )
        ] = int(position)

    return result


# ================================================================
# CURRENT STAGE
# ================================================================

def get_current_stage(state):
    stage = state.get(
        "current_stage",
        state.get(
            "stage",
            "pre_practice",
        ),
    )

    stage = normalise_text(stage)

    if (
        stage in {
            "prepractice",
            "pre practice",
            "pre_practice",
            "before practice",
        }
    ):
        return "pre_practice"

    if (
        "practice 1" in stage
        or stage == "fp1"
    ):
        return "fp1"

    if (
        "practice 2" in stage
        or stage == "fp2"
    ):
        return "fp2"

    if (
        "practice 3" in stage
        or stage == "fp3"
    ):
        return "fp3"

    if "qualifying" in stage:
        return "qualifying"

    if (
        "race" in stage
        or "complete" in stage
    ):
        return "race"

    return "pre_practice"


def stage_number(stage):
    return {
        "pre_practice": 0,
        "fp1": 1,
        "fp2": 2,
        "fp3": 3,
        "qualifying": 4,
        "race": 5,
    }.get(
        stage,
        0,
    )


def stage_has_happened(
    current_stage,
    required_stage,
):
    return (
        stage_number(current_stage)
        >= stage_number(required_stage)
    )


# ================================================================
# BUILD PROGRESSION
# ================================================================

def build_progression():

    state = load_json(
        PREDICTION_DIR
        / "weekend_state.json"
    )

    if state is None:
        print("weekend_state.json not found.")
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

    current_stage = get_current_stage(
        state
    )

    print()
    print("F1 RACE PREDICTOR")
    print("BUILD RACE WEEKEND PROGRESSION")
    print()

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
        f"Weekend: "
        f"{str(weekend_format).upper()}"
    )

    print(
        f"Current stage: "
        f"{current_stage.upper()}"
    )

    print()

    # ------------------------------------------------------------
    # DRIVERS
    # ------------------------------------------------------------

    print(
        "Loading current driver list..."
    )

    drivers = load_current_drivers(
        season,
        round_number,
    )

    if len(drivers) != 22:
        raise ValueError(
            f"Expected 22 current drivers, "
            f"found {len(drivers)}."
        )

    # ------------------------------------------------------------
    # PRE-PRACTICE
    # ------------------------------------------------------------

    print(
        "Loading no-practice prediction..."
    )

    no_practice = (
        load_no_practice_prediction(
            season,
            round_number,
        )
    )

    # ------------------------------------------------------------
    # FP1
    # ------------------------------------------------------------

    after_fp1 = {}

    if stage_has_happened(
        current_stage,
        "fp1",
    ):

        print(
            "Loading FP1 prediction..."
        )

        after_fp1 = load_prediction_file(
            "current_2026_fp1_prediction.csv",
            season,
            round_number,
        )

    # ------------------------------------------------------------
    # FP2
    # ------------------------------------------------------------

    after_fp2 = {}

    if stage_has_happened(
        current_stage,
        "fp2",
    ):

        print(
            "Loading FP2 prediction..."
        )

        after_fp2 = load_prediction_file(
            "current_2026_fp2_prediction.csv",
            season,
            round_number,
        )

    # ------------------------------------------------------------
    # FP3
    # ------------------------------------------------------------

    after_fp3 = {}

    if stage_has_happened(
        current_stage,
        "fp3",
    ):

        print(
            "Loading FP3 prediction..."
        )

        after_fp3 = load_prediction_file(
            "current_2026_fp3_prediction.csv",
            season,
            round_number,
        )

    # ------------------------------------------------------------
    # QUALIFYING
    # ------------------------------------------------------------

    after_qualifying = {}

    if stage_has_happened(
        current_stage,
        "qualifying",
    ):

        print(
            "Loading qualifying prediction..."
        )

        after_qualifying = (
            load_prediction_file(
                "current_2026_qualifying_prediction.csv",
                season,
                round_number,
            )
        )

    # ------------------------------------------------------------
    # PRACTICE RESULTS
    # ------------------------------------------------------------

    print(
        "Loading live practice results..."
    )

    live_practice = (
        load_live_practice_results(
            season
        )
    )

    practice = {
        "fp1": {},
        "fp2": {},
        "fp3": {},
    }

    if stage_has_happened(
        current_stage,
        "fp1",
    ):
        practice["fp1"] = (
            live_practice["fp1"]
        )

    if stage_has_happened(
        current_stage,
        "fp2",
    ):
        practice["fp2"] = (
            live_practice["fp2"]
        )

    if stage_has_happened(
        current_stage,
        "fp3",
    ):
        practice["fp3"] = (
            live_practice["fp3"]
        )

    # ------------------------------------------------------------
    # QUALIFYING RESULT
    # ------------------------------------------------------------

    qualifying = {}

    if stage_has_happened(
        current_stage,
        "qualifying",
    ):

        print(
            "Loading qualifying result..."
        )

        qualifying = load_qualifying_result(
            season,
            round_number,
        )

    else:

        print(
            "Qualifying has not occurred yet."
        )

    # ------------------------------------------------------------
    # RACE RESULT
    # ------------------------------------------------------------

    race = {}

    if stage_has_happened(
        current_stage,
        "race",
    ):

        print(
            "Loading race result..."
        )

        race = load_race_result(
            season,
            round_number,
        )

    else:

        print(
            "Race has not occurred yet."
        )

    # ------------------------------------------------------------
    # BUILD ROWS
    # ------------------------------------------------------------

    rows = []

    for driver_key, driver_info in drivers.items():

        driver_name = (
            driver_info["driver_name"]
        )

        # --------------------------------------------------------
        # PRE-PRACTICE
        # --------------------------------------------------------

        no_practice_position = pd.NA

        pre_info = find_pre_practice_for_driver(
            driver_name,
            no_practice,
        )

        if pre_info is not None:
            no_practice_position = (
                pre_info["position"]
            )

        # --------------------------------------------------------
        # LIVE PREDICTIONS
        # --------------------------------------------------------

        fp1_prediction = pd.NA
        fp2_prediction = pd.NA
        fp3_prediction = pd.NA
        qualifying_prediction = pd.NA

        normalized_driver = normalise_text(
            driver_name
        )

        if normalized_driver in after_fp1:
            fp1_prediction = (
                after_fp1[
                    normalized_driver
                ]["position"]
            )

        if normalized_driver in after_fp2:
            fp2_prediction = (
                after_fp2[
                    normalized_driver
                ]["position"]
            )

        if normalized_driver in after_fp3:
            fp3_prediction = (
                after_fp3[
                    normalized_driver
                ]["position"]
            )

        if normalized_driver in after_qualifying:
            qualifying_prediction = (
                after_qualifying[
                    normalized_driver
                ]["position"]
            )

        # --------------------------------------------------------
        # ACTUAL RESULTS
        # --------------------------------------------------------

        fp1_result = pd.NA
        fp2_result = pd.NA
        fp3_result = pd.NA
        qualifying_result = pd.NA
        race_result = pd.NA

        info = find_practice_result_for_driver(
            driver_name,
            practice["fp1"],
        )

        if info is not None:
            fp1_result = info["position"]

        info = find_practice_result_for_driver(
            driver_name,
            practice["fp2"],
        )

        if info is not None:
            fp2_result = info["position"]

        info = find_practice_result_for_driver(
            driver_name,
            practice["fp3"],
        )

        if info is not None:
            fp3_result = info["position"]

        if normalized_driver in qualifying:
            qualifying_result = (
                qualifying[
                    normalized_driver
                ]
            )

        if normalized_driver in race:
            race_result = (
                race[
                    normalized_driver
                ]
            )

        # --------------------------------------------------------
        # LATEST AVAILABLE PREDICTION
        # --------------------------------------------------------

        latest_prediction = pd.NA

        if pd.notna(
            qualifying_prediction
        ):

            latest_prediction = (
                qualifying_prediction
            )

        elif pd.notna(
            fp3_prediction
        ):

            latest_prediction = (
                fp3_prediction
            )

        elif pd.notna(
            fp2_prediction
        ):

            latest_prediction = (
                fp2_prediction
            )

        elif pd.notna(
            fp1_prediction
        ):

            latest_prediction = (
                fp1_prediction
            )

        elif pd.notna(
            no_practice_position
        ):

            latest_prediction = (
                no_practice_position
            )

        # --------------------------------------------------------
        # FINAL COMPARISON
        # --------------------------------------------------------

        comparison = pd.NA

        if (
            pd.notna(
                qualifying_prediction
            )
            and pd.notna(race_result)
        ):

            comparison = (
                int(race_result)
                - int(
                    qualifying_prediction
                )
            )

        rows.append(
            {
                "driver_name":
                    driver_name,

                "team_name":
                    driver_info["team_name"],

                "no_practice_prediction":
                    no_practice_position,

                "fp1_result":
                    fp1_result,

                "prediction_after_fp1":
                    fp1_prediction,

                "fp2_result":
                    fp2_result,

                "prediction_after_fp2":
                    fp2_prediction,

                "fp3_result":
                    fp3_result,

                "prediction_after_fp3":
                    fp3_prediction,

                "qualifying_result":
                    qualifying_result,

                "final_prediction":
                    latest_prediction,

                "race_result":
                    race_result,

                "prediction_difference":
                    comparison,
            }
        )

    # ------------------------------------------------------------
    # DATAFRAME
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
    # SORT BY LATEST AVAILABLE PREDICTION
    # ------------------------------------------------------------

    result["_sort"] = pd.to_numeric(
        result["final_prediction"],
        errors="coerce",
    )

    result = (
        result
        .sort_values(
            [
                "_sort",
                "driver_name",
            ],
            na_position="last",
        )
        .drop(
            columns="_sort"
        )
        .reset_index(
            drop=True
        )
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

    print()
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

    print()
    print(
        f"Saved -> {output_path}"
    )

    print()
    print(
        "RACE PROGRESSION COMPLETE"
    )


if __name__ == "__main__":
    build_progression()