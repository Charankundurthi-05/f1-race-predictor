from pathlib import Path
import zipfile
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ZIP_FILE = ROOT / "data" / "raw" / "f1db-csv.zip"


def find_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("WEEKEND FORMAT DETECTOR")
    print("=" * 80)
    print()

    if not ZIP_FILE.exists():
        raise FileNotFoundError(
            f"F1DB file not found: {ZIP_FILE}"
        )

    with zipfile.ZipFile(ZIP_FILE) as z:

        race_file = next(
            name
            for name in z.namelist()
            if name.endswith("f1db-races.csv")
        )

        with z.open(race_file) as f:
            races = pd.read_csv(f)

    # ---------------------------------------------------------------
    # Identify important columns dynamically
    # ---------------------------------------------------------------

    year_column = find_column(
        races,
        ["year", "season"]
    )

    round_column = find_column(
        races,
        ["round"]
    )

    name_column = find_column(
        races,
        [
            "name",
            "raceName",
            "officialName",
            "fullName"
        ]
    )

    date_column = find_column(
        races,
        [
            "date",
            "raceDate"
        ]
    )

    if year_column is None:
        raise ValueError("Could not find year column.")

    if round_column is None:
        raise ValueError("Could not find round column.")

    if name_column is None:
        raise ValueError(
            "Could not find race-name column.\n"
            f"Available columns:\n{list(races.columns)}"
        )

    if date_column is None:
        raise ValueError("Could not find race-date column.")

    races = races[
        races[year_column] == 2026
    ].copy()

    races[date_column] = pd.to_datetime(
        races[date_column],
        errors="coerce"
    )

    races = races.sort_values(
        [round_column, date_column]
    )

    # ---------------------------------------------------------------
    # Sprint schedule columns
    # ---------------------------------------------------------------

    sprint_columns = [
        column
        for column in races.columns
        if "sprint" in column.lower()
    ]

    print("Sprint-related columns found:")

    for column in sprint_columns:
        print(f"  {column}")

    print()

    # ---------------------------------------------------------------
    # Detect sprint weekend using sprint qualifying date.
    #
    # A populated sprintQualifyingDate means the weekend has a
    # sprint format. This is much safer than checking whether a
    # generic sprint column happens to contain a value.
    # ---------------------------------------------------------------

    sprint_date_column = find_column(
        races,
        ["sprintQualifyingDate"]
    )

    sprint_race_date_column = find_column(
        races,
        ["sprintRaceDate"]
    )

    print("2026 WEEKEND FORMAT")
    print("-" * 80)

    for _, row in races.iterrows():

        sprint = False

        if sprint_date_column is not None:
            sprint = pd.notna(
                row[sprint_date_column]
            )

        if (
            not sprint
            and sprint_race_date_column is not None
        ):
            sprint = pd.notna(
                row[sprint_race_date_column]
            )

        format_name = (
            "SPRINT"
            if sprint
            else "NORMAL"
        )

        print(
            f"Round {int(row[round_column]):2d} | "
            f"{str(row[name_column]):<45} | "
            f"{format_name}"
        )

    print()

    print("=" * 80)
    print("WEEKEND FORMAT DETECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()