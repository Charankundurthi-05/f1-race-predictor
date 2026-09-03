from pathlib import Path
import requests
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "data" / "raw"

DRIVER_OUTPUT = OUTPUT_DIR / "driver_standings_2026.csv"
CONSTRUCTOR_OUTPUT = OUTPUT_DIR / "constructor_standings_2026.csv"


BASE_URL = "https://api.jolpi.ca/ergast/f1/2026"


def download(endpoint):

    url = f"{BASE_URL}/{endpoint}.json"

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("DOWNLOAD 2026 STANDINGS")
    print("=" * 80)
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------------
    # Driver standings
    # ---------------------------------------------------------------

    print("Downloading driver standings...")

    data = download(
        "driverstandings"
    )

    standings_lists = (
        data
        .get("MRData", {})
        .get("StandingsTable", {})
        .get("StandingsLists", [])
    )

    if not standings_lists:

        print(
            "No driver standings available."
        )

    else:

        standings = (
            standings_lists[0]
            .get("DriverStandings", [])
        )

        driver_rows = []

        for row in standings:

            driver = row.get(
                "Driver",
                {}
            )

            constructors = row.get(
                "Constructors",
                []
            )

            constructor_name = ""

            if constructors:
                constructor_name = constructors[0].get(
                    "name",
                    ""
                )

            driver_rows.append(
                {
                    "position": row.get(
                        "position"
                    ),
                    "points": row.get(
                        "points"
                    ),
                    "wins": row.get(
                        "wins"
                    ),
                    "driver_id": driver.get(
                        "driverId"
                    ),
                    "driver_code": driver.get(
                        "code"
                    ),
                    "driver_number": driver.get(
                        "permanentNumber"
                    ),
                    "driver_name": (
                        f"{driver.get('givenName', '')} "
                        f"{driver.get('familyName', '')}"
                    ).strip(),
                    "constructor": constructor_name
                }
            )

        driver_df = pd.DataFrame(
            driver_rows
        )

        driver_df.to_csv(
            DRIVER_OUTPUT,
            index=False
        )

        print(
            f"Driver standings: "
            f"{len(driver_df)} drivers"
        )

        print(
            f"Saved -> {DRIVER_OUTPUT}"
        )

    print()

    # ---------------------------------------------------------------
    # Constructor standings
    # ---------------------------------------------------------------

    print(
        "Downloading constructor standings..."
    )

    data = download(
        "constructorstandings"
    )

    standings_lists = (
        data
        .get("MRData", {})
        .get("StandingsTable", {})
        .get("StandingsLists", [])
    )

    if not standings_lists:

        print(
            "No constructor standings available."
        )

    else:

        standings = (
            standings_lists[0]
            .get("ConstructorStandings", [])
        )

        constructor_rows = []

        for row in standings:

            constructor = row.get(
                "Constructor",
                {}
            )

            constructor_rows.append(
                {
                    "position": row.get(
                        "position"
                    ),
                    "points": row.get(
                        "points"
                    ),
                    "wins": row.get(
                        "wins"
                    ),
                    "constructor_id": constructor.get(
                        "constructorId"
                    ),
                    "constructor_name": constructor.get(
                        "name"
                    ),
                    "nationality": constructor.get(
                        "nationality"
                    )
                }
            )

        constructor_df = pd.DataFrame(
            constructor_rows
        )

        constructor_df.to_csv(
            CONSTRUCTOR_OUTPUT,
            index=False
        )

        print(
            f"Constructor standings: "
            f"{len(constructor_df)} teams"
        )

        print(
            f"Saved -> {CONSTRUCTOR_OUTPUT}"
        )

    print()

    print("=" * 80)
    print("2026 STANDINGS DOWNLOAD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()