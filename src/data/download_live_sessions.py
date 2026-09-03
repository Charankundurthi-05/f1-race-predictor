from pathlib import Path
import requests
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = ROOT / "data" / "predictions"

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BASE_URL = "https://api.jolpi.ca/ergast/f1"

SEASON = 2026
ROUND = 13


def fetch(endpoint):
    url = f"{BASE_URL}/{SEASON}/{ROUND}/{endpoint}.json"

    try:
        response = requests.get(
            url,
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    except Exception as exc:
        print(f"Request failed: {endpoint}")
        print(exc)
        return None


def extract_results(data, endpoint):
    if not data:
        return pd.DataFrame()

    try:
        races = (
            data
            ["MRData"]
            ["RaceTable"]
            .get("Races", [])
        )

        if not races:
            return pd.DataFrame()

        race = races[0]

        if endpoint == "qualifying":

            rows = []

            for item in race.get(
                "QualifyingResults",
                []
            ):

                rows.append({
                    "season": SEASON,
                    "round": ROUND,
                    "race_name": race.get(
                        "raceName"
                    ),
                    "driver_name": (
                        item.get("Driver", {})
                        .get("givenName", "")
                        + " "
                        + item.get("Driver", {})
                        .get("familyName", "")
                    ).strip(),
                    "driver_code": (
                        item.get("Driver", {})
                        .get("code")
                    ),
                    "team_name": (
                        item.get("Constructor", {})
                        .get("name")
                    ),
                    "qualifying_position": item.get(
                        "position"
                    ),
                    "q1": item.get("Q1"),
                    "q2": item.get("Q2"),
                    "q3": item.get("Q3")
                })

            return pd.DataFrame(rows)

        return pd.DataFrame()

    except Exception as exc:
        print(f"Could not parse {endpoint}: {exc}")
        return pd.DataFrame()


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("LIVE 2026 SESSION DATA")
    print("=" * 80)
    print()

    print(f"Season: {SEASON}")
    print(f"Round: {ROUND}")
    print()

    # ---------------------------------------------------------------
    # QUALIFYING
    # ---------------------------------------------------------------

    qualifying_data = fetch("qualifying")

    qualifying = extract_results(
        qualifying_data,
        "qualifying"
    )

    if qualifying.empty:

        print("Qualifying results: NOT AVAILABLE")

    else:

        qualifying_file = (
            PREDICTION_DIR
            / "live_qualifying_results.csv"
        )

        qualifying.to_csv(
            qualifying_file,
            index=False
        )

        print(
            f"Qualifying results: "
            f"{len(qualifying)} drivers"
        )

        print(
            f"Saved -> {qualifying_file}"
        )

    # ---------------------------------------------------------------
    # SPRINT
    # ---------------------------------------------------------------

    sprint_data = fetch("sprint")

    if sprint_data:

        try:

            races = (
                sprint_data
                ["MRData"]
                ["RaceTable"]
                .get("Races", [])
            )

            if races:

                sprint_rows = []

                for item in races[0].get(
                    "SprintResults",
                    []
                ):

                    sprint_rows.append({
                        "season": SEASON,
                        "round": ROUND,
                        "driver_name": (
                            item.get("Driver", {})
                            .get("givenName", "")
                            + " "
                            + item.get("Driver", {})
                            .get("familyName", "")
                        ).strip(),
                        "driver_code": (
                            item.get("Driver", {})
                            .get("code")
                        ),
                        "team_name": (
                            item.get("Constructor", {})
                            .get("name")
                        ),
                        "position": item.get(
                            "position"
                        ),
                        "points": item.get(
                            "points"
                        ),
                        "laps": item.get(
                            "laps"
                        )
                    })

                if sprint_rows:

                    sprint_file = (
                        PREDICTION_DIR
                        / "live_sprint_results.csv"
                    )

                    pd.DataFrame(
                        sprint_rows
                    ).to_csv(
                        sprint_file,
                        index=False
                    )

                    print(
                        f"Sprint results: "
                        f"{len(sprint_rows)} drivers"
                    )

                else:
                    print(
                        "Sprint results: "
                        "NOT AVAILABLE"
                    )

            else:
                print(
                    "Sprint results: "
                    "NOT AVAILABLE"
                )

        except Exception as exc:

            print(
                f"Sprint parsing failed: {exc}"
            )

    else:

        print(
            "Sprint results: "
            "NOT AVAILABLE"
        )

    # ---------------------------------------------------------------
    # RACE
    # ---------------------------------------------------------------

    race_data = fetch("results")

    if race_data:

        try:

            races = (
                race_data
                ["MRData"]
                ["RaceTable"]
                .get("Races", [])
            )

            if races:

                race_rows = []

                for item in races[0].get(
                    "Results",
                    []
                ):

                    race_rows.append({
                        "season": SEASON,
                        "round": ROUND,
                        "driver_name": (
                            item.get("Driver", {})
                            .get("givenName", "")
                            + " "
                            + item.get("Driver", {})
                            .get("familyName", "")
                        ).strip(),
                        "driver_code": (
                            item.get("Driver", {})
                            .get("code")
                        ),
                        "team_name": (
                            item.get("Constructor", {})
                            .get("name")
                        ),
                        "finish_position": item.get(
                            "position"
                        ),
                        "points": item.get(
                            "points"
                        ),
                        "laps": item.get(
                            "laps"
                        ),
                        "status": item.get(
                            "status"
                        )
                    })

                if race_rows:

                    race_file = (
                        PREDICTION_DIR
                        / "live_race_results.csv"
                    )

                    pd.DataFrame(
                        race_rows
                    ).to_csv(
                        race_file,
                        index=False
                    )

                    print(
                        f"Race results: "
                        f"{len(race_rows)} drivers"
                    )

                else:
                    print(
                        "Race results: "
                        "NOT AVAILABLE"
                    )

            else:
                print(
                    "Race results: "
                    "NOT AVAILABLE"
                )

        except Exception as exc:

            print(
                f"Race parsing failed: {exc}"
            )

    else:

        print(
            "Race results: "
            "NOT AVAILABLE"
        )

    print()
    print("=" * 80)
    print("LIVE SESSION CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()