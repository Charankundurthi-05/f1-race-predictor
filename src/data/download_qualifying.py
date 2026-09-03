import requests
import pandas as pd
from pathlib import Path
import time

BASE_URL = "https://api.jolpi.ca/ergast/f1"

CALENDAR_FILE = Path("data/raw/calendars.csv")
OUTPUT_FILE = Path("data/raw/qualifying_results.csv")

calendar = pd.read_csv(CALENDAR_FILE)

rows = []

session = requests.Session()

for _, race in calendar.iterrows():
    season = int(race["season"])
    round_number = int(race["round"])

    print(f"Downloading qualifying {season} Round {round_number}...")

    url = f"{BASE_URL}/{season}/{round_number}/qualifying/"

    success = False

    for attempt in range(1, 4):
        try:
            response = session.get(
                url,
                timeout=(10, 60)
            )

            if response.status_code == 200:
                success = True
                break

            print(
                f"  Attempt {attempt}: HTTP {response.status_code}"
            )

        except requests.exceptions.RequestException as e:
            print(
                f"  Attempt {attempt} failed: {type(e).__name__}"
            )

        if attempt < 3:
            time.sleep(3)

    if not success:
        print(
            f"  Skipping {season} Round {round_number} after 3 attempts."
        )
        continue

    data = response.json()

    races = data["MRData"]["RaceTable"]["Races"]

    if not races:
        print(
            f"  No qualifying data for {season} Round {round_number}"
        )
        continue

    race_data = races[0]

    for result in race_data.get("QualifyingResults", []):
        rows.append({
            "season": season,
            "round": round_number,
            "race_name": race_data["raceName"],
            "race_date": race_data.get("date"),
            "circuit_id": race_data["Circuit"]["circuitId"],
            "circuit_name": race_data["Circuit"]["circuitName"],
            "driver_id": result["Driver"]["driverId"],
            "driver_code": result["Driver"].get("code"),
            "driver_name": (
                f'{result["Driver"].get("givenName", "")} '
                f'{result["Driver"].get("familyName", "")}'
            ).strip(),
            "team_id": result["Constructor"]["constructorId"],
            "team_name": result["Constructor"]["name"],
            "qualifying_position": result.get("position"),
            "q1": result.get("Q1"),
            "q2": result.get("Q2"),
            "q3": result.get("Q3")
        })

    time.sleep(0.5)

df = pd.DataFrame(rows)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print()
print(f"Downloaded {len(df)} qualifying records.")
print(f"Saved to: {OUTPUT_FILE}")
print()
print("Records by season:")
print(df.groupby("season").size())