from pathlib import Path
import zipfile
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ZIP_FILE = ROOT / "data" / "raw" / "f1db-csv.zip"


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("2026 SPRINT WEEKEND CHECK")
    print("=" * 80)
    print()

    if not ZIP_FILE.exists():
        raise FileNotFoundError(
            f"F1DB file not found: {ZIP_FILE}"
        )

    with zipfile.ZipFile(ZIP_FILE) as z:

        race_file = next(
            x for x in z.namelist()
            if x.endswith("f1db-races.csv")
        )

        with z.open(race_file) as f:
            races = pd.read_csv(f)

    races = races[
        races["year"] == 2026
    ].copy()

    sprint_qualifying_date = pd.to_datetime(
        races["sprintQualifyingDate"],
        errors="coerce"
    )

    sprint_race_date = pd.to_datetime(
        races["sprintRaceDate"],
        errors="coerce"
    )

    races["is_sprint"] = (
        sprint_qualifying_date.notna()
        | sprint_race_date.notna()
    )

    sprint_races = races[
        races["is_sprint"]
    ].copy()

    print("2026 SPRINT WEEKENDS")
    print("-" * 80)

    for _, row in sprint_races.iterrows():

        print(
            f"Round {int(row['round']):2d} | "
            f"Sprint Qualifying: "
            f"{row['sprintQualifyingDate']} | "
            f"Sprint: "
            f"{row['sprintRaceDate']}"
        )

    print()
    print(
        f"Total sprint weekends: "
        f"{len(sprint_races)}"
    )

    print()
    print("=" * 80)
    print("SPRINT WEEKEND CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()