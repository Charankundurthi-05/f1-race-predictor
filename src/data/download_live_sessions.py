from pathlib import Path
import json

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PREDICTION_DIR = ROOT / "data" / "predictions"
WEEKEND_STATE_FILE = PREDICTION_DIR / "weekend_state.json"

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BASE_URL = "https://api.jolpi.ca/ergast/f1"


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


def get_season_round():
    state = load_weekend_state()

    season = int(
        state.get(
            "season",
            2026,
        )
    )

    round_number = int(
        state.get(
            "round",
            13,
        )
    )

    return season, round_number


def fetch(endpoint, season, round_number):
    url = (
        f"{BASE_URL}/"
        f"{season}/"
        f"{round_number}/"
        f"{endpoint}.json"
    )

    try:
        response = requests.get(
            url,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    except Exception as exc:
        print(
            f"Request failed: {endpoint}"
        )
        print(exc)
        return None


def get_race(data):
    if not data:
        return None

    try:
        races = (
            data
            .get("MRData", {})
            .get("RaceTable", {})
            .get("Races", [])
        )

        if not races:
            return None

        return races[0]

    except Exception:
        return None


def driver_name(item):
    driver = item.get(
        "Driver",
        {},
    )

    return (
        str(driver.get("givenName", "")).strip()
        + " "
        + str(driver.get("familyName", "")).strip()
    ).strip()


def driver_code(item):
    return (
        item.get(
            "Driver",
            {},
        ).get("code")
    )


def team_name(item):
    return (
        item.get(
            "Constructor",
            {},
        ).get("name")
    )


def extract_qualifying(
    data,
    season,
    round_number,
):
    race = get_race(data)

    if race is None:
        return pd.DataFrame()

    rows = []

    for item in race.get(
        "QualifyingResults",
        [],
    ):
        rows.append(
            {
                "season": season,
                "round": round_number,
                "race_name": race.get("raceName"),
                "driver_name": driver_name(item),
                "driver_code": driver_code(item),
                "team_name": team_name(item),
                "qualifying_position": item.get("position"),
                "q1": item.get("Q1"),
                "q2": item.get("Q2"),
                "q3": item.get("Q3"),
            }
        )

    return pd.DataFrame(rows)


def extract_sprint(
    data,
    season,
    round_number,
):
    race = get_race(data)

    if race is None:
        return pd.DataFrame()

    rows = []

    for item in race.get(
        "SprintResults",
        [],
    ):
        rows.append(
            {
                "season": season,
                "round": round_number,
                "race_name": race.get("raceName"),
                "driver_name": driver_name(item),
                "driver_code": driver_code(item),
                "team_name": team_name(item),
                "position": item.get("position"),
                "points": item.get("points"),
                "laps": item.get("laps"),
            }
        )

    return pd.DataFrame(rows)


def extract_race(
    data,
    season,
    round_number,
):
    race = get_race(data)

    if race is None:
        return pd.DataFrame()

    rows = []

    for item in race.get(
        "Results",
        [],
    ):
        rows.append(
            {
                "season": season,
                "round": round_number,
                "race_name": race.get("raceName"),
                "driver_name": driver_name(item),
                "driver_code": driver_code(item),
                "team_name": team_name(item),
                "finish_position": item.get("position"),
                "points": item.get("points"),
                "laps": item.get("laps"),
                "status": item.get("status"),
            }
        )

    return pd.DataFrame(rows)


def save_if_available(
    df,
    filename,
    label,
):
    if df.empty:
        print(
            f"{label}: NOT AVAILABLE"
        )
        return False

    path = PREDICTION_DIR / filename

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"{label}: "
        f"{len(df)} drivers"
    )

    print(
        f"Saved -> {path}"
    )

    return True


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("LIVE 2026 SESSION DATA")
    print("=" * 80)
    print()

    season, round_number = get_season_round()

    print(
        f"Season: {season}"
    )
    print(
        f"Round: {round_number}"
    )
    print()

    # ------------------------------------------------------------
    # QUALIFYING
    # ------------------------------------------------------------

    qualifying_data = fetch(
        "qualifying",
        season,
        round_number,
    )

    qualifying = extract_qualifying(
        qualifying_data,
        season,
        round_number,
    )

    save_if_available(
        qualifying,
        "live_qualifying_results.csv",
        "Qualifying results",
    )

    # ------------------------------------------------------------
    # SPRINT
    # ------------------------------------------------------------

    sprint_data = fetch(
        "sprint",
        season,
        round_number,
    )

    sprint = extract_sprint(
        sprint_data,
        season,
        round_number,
    )

    save_if_available(
        sprint,
        "live_sprint_results.csv",
        "Sprint results",
    )

    # ------------------------------------------------------------
    # RACE
    # ------------------------------------------------------------

    race_data = fetch(
        "results",
        season,
        round_number,
    )

    race = extract_race(
        race_data,
        season,
        round_number,
    )

    save_if_available(
        race,
        "live_race_results.csv",
        "Race results",
    )

    print()
    print("=" * 80)
    print("LIVE SESSION CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
