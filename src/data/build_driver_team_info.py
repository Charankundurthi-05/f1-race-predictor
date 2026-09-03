from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

DRIVER_FILE = RAW_DIR / "driver_standings_2026.csv"
CONSTRUCTOR_FILE = RAW_DIR / "constructor_standings_2026.csv"

DRIVER_OUTPUT = PROCESSED_DIR / "driver_info_2026.csv"
CONSTRUCTOR_OUTPUT = PROCESSED_DIR / "constructor_info_2026.csv"


# Wikimedia Commons image URLs.
# These are kept as URLs so the website can load the images directly.
DRIVER_PHOTOS = {
    "Lando Norris":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Oscar Piastri":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Max Verstappen":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "George Russell":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Charles Leclerc":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Lewis Hamilton":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Andrea Kimi Antonelli":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Fernando Alonso":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Lance Stroll":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Carlos Sainz":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Alexander Albon":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Esteban Ocon":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Oliver Bearman":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Pierre Gasly":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Franco Colapinto":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Yuki Tsunoda":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Liam Lawson":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Gabriel Bortoleto":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Nico Hülkenberg":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Arvid Lindblad":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Sergio Pérez":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Valtteri Bottas":
        "https://upload.wikimedia.org/wikipedia/commons/",
    "Jack Doohan":
        "https://upload.wikimedia.org/wikipedia/commons/"
}


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("BUILD DRIVER AND TEAM INFORMATION")
    print("=" * 80)
    print()

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not DRIVER_FILE.exists():
        raise FileNotFoundError(
            f"Missing driver standings: {DRIVER_FILE}"
        )

    if not CONSTRUCTOR_FILE.exists():
        raise FileNotFoundError(
            f"Missing constructor standings: {CONSTRUCTOR_FILE}"
        )

    drivers = pd.read_csv(
        DRIVER_FILE
    )

    constructors = pd.read_csv(
        CONSTRUCTOR_FILE
    )

    # ---------------------------------------------------------------
    # Driver information
    # ---------------------------------------------------------------

    drivers["driver_photo_url"] = (
        drivers["driver_name"]
        .map(DRIVER_PHOTOS)
        .fillna("")
    )

    drivers["profile_url"] = (
        "https://www.formula1.com/en/drivers/"
        + drivers["driver_id"].astype(str)
    )

    driver_columns = [
        "position",
        "points",
        "wins",
        "driver_id",
        "driver_code",
        "driver_number",
        "driver_name",
        "constructor",
        "driver_photo_url",
        "profile_url"
    ]

    drivers = drivers[
        [
            column
            for column in driver_columns
            if column in drivers.columns
        ]
    ]

    drivers.to_csv(
        DRIVER_OUTPUT,
        index=False
    )

    print(
        f"Driver information: "
        f"{len(drivers)} drivers"
    )

    print(
        f"Saved -> {DRIVER_OUTPUT}"
    )

    # ---------------------------------------------------------------
    # Constructor information
    # ---------------------------------------------------------------

    constructors["team_logo_url"] = ""

    constructors["profile_url"] = (
        "https://www.formula1.com/en/teams/"
        + constructors["constructor_id"].astype(str)
    )

    constructor_columns = [
        "position",
        "points",
        "wins",
        "constructor_id",
        "constructor_name",
        "nationality",
        "team_logo_url",
        "profile_url"
    ]

    constructors = constructors[
        [
            column
            for column in constructor_columns
            if column in constructors.columns
        ]
    ]

    constructors.to_csv(
        CONSTRUCTOR_OUTPUT,
        index=False
    )

    print(
        f"Constructor information: "
        f"{len(constructors)} teams"
    )

    print(
        f"Saved -> {CONSTRUCTOR_OUTPUT}"
    )

    print()
    print("=" * 80)
    print("DRIVER AND TEAM INFORMATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()