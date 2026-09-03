from pathlib import Path
import requests
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "driver_info_2026.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "driver_info_2026_with_photos.csv"
)

# Official Formula 1 driver profile pages.
# The page URLs are stored rather than guessing image filenames.
F1_DRIVER_PAGES = {
    "Lando Norris":
        "https://www.formula1.com/en/drivers/lando-norris",

    "Oscar Piastri":
        "https://www.formula1.com/en/drivers/oscar-piastri",

    "Max Verstappen":
        "https://www.formula1.com/en/drivers/max-verstappen",

    "George Russell":
        "https://www.formula1.com/en/drivers/george-russell",

    "Charles Leclerc":
        "https://www.formula1.com/en/drivers/charles-leclerc",

    "Lewis Hamilton":
        "https://www.formula1.com/en/drivers/lewis-hamilton",

    "Andrea Kimi Antonelli":
        "https://www.formula1.com/en/drivers/andrea-kimi-antonelli",

    "Fernando Alonso":
        "https://www.formula1.com/en/drivers/fernando-alonso",

    "Lance Stroll":
        "https://www.formula1.com/en/drivers/lance-stroll",

    "Carlos Sainz":
        "https://www.formula1.com/en/drivers/carlos-sainz",

    "Alexander Albon":
        "https://www.formula1.com/en/drivers/alexander-albon",

    "Esteban Ocon":
        "https://www.formula1.com/en/drivers/esteban-ocon",

    "Oliver Bearman":
        "https://www.formula1.com/en/drivers/oliver-bearman",

    "Pierre Gasly":
        "https://www.formula1.com/en/drivers/pierre-gasly",

    "Franco Colapinto":
        "https://www.formula1.com/en/drivers/franco-colapinto",

    "Yuki Tsunoda":
        "https://www.formula1.com/en/drivers/yuki-tsunoda",

    "Liam Lawson":
        "https://www.formula1.com/en/drivers/liam-lawson",

    "Gabriel Bortoleto":
        "https://www.formula1.com/en/drivers/gabriel-bortoleto",

    "Nico Hülkenberg":
        "https://www.formula1.com/en/drivers/nico-hulkenberg",

    "Arvid Lindblad":
        "https://www.formula1.com/en/drivers/arvid-lindblad",

    "Sergio Pérez":
        "https://www.formula1.com/en/drivers/sergio-perez",

    "Valtteri Bottas":
        "https://www.formula1.com/en/drivers/valtteri-bottas",

    "Jack Doohan":
        "https://www.formula1.com/en/drivers/jack-doohan",
}


def verify_url(url):

    try:

        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent":
                    "Mozilla/5.0"
            }
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("VERIFY 2026 DRIVER PROFILE PHOTOS")
    print("=" * 80)
    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing driver information file: "
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    df["official_f1_profile_url"] = (
        df["driver_name"]
        .map(F1_DRIVER_PAGES)
        .fillna("")
    )

    df["official_f1_profile_verified"] = (
        False
    )

    print(
        f"Drivers found: {len(df)}"
    )

    print()

    verified = 0

    for index, row in df.iterrows():

        name = row["driver_name"]
        url = row["official_f1_profile_url"]

        if not url:

            print(
                f"[MISSING] {name}"
            )

            continue

        valid = verify_url(
            url
        )

        df.loc[
            index,
            "official_f1_profile_verified"
        ] = valid

        if valid:

            verified += 1

            print(
                f"[PASS] {name}"
            )

        else:

            print(
                f"[CHECK] {name} "
                f"-> {url}"
            )

    # Keep the existing photo column, but remove
    # the placeholder values from the previous step.
    df["driver_photo_url"] = ""

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 80)
    print("DRIVER PHOTO PROFILE VERIFICATION")
    print("=" * 80)

    print(
        f"Verified official F1 profiles: "
        f"{verified}/{len(df)}"
    )

    print(
        f"Saved -> {OUTPUT_FILE}"
    )

    print()
    print(
        "Note: official F1 profile pages are verified."
    )

    print(
        "Direct image URLs will be populated "
        "from the verified pages when we build the website."
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()