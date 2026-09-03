import requests
import pandas as pd
from pathlib import Path
import time

BASE_URL = "https://api.jolpi.ca/ergast/f1"

CALENDAR_FILE = Path("data/raw/calendars.csv")
OUTPUT_FILE = Path("data/raw/race_results.csv")

calendar = pd.read_csv(CALENDAR_FILE)

rows = []

for _, race in calendar.iterrows():
    season = int(race["season"])
    round_number = int(race["round"])

    print(f"Downloading {season} Round {round_number}...")

    url = f"{BASE_URL}/{season}/{round_number}/results/"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()

    races = data["MRData"]["RaceTable"]["Races"]

    if not races:
        print(f"  No results found for {season} Round {round_number}")
        continue

    race_data = races[0]

    for result in race_data.get("Results", []):
        fastest_lap = result.get("FastestLap", {})

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
            "number": result.get("number"),
            "grid_position": result.get("grid"),
            "finish_position": result.get("position"),
            "position_text": result.get("positionText"),
            "points": result.get("points"),
            "laps": result.get("laps"),
            "status": result.get("status"),
            "fastest_lap_rank": fastest_lap.get("rank"),
            "fastest_lap_time": fastest_lap.get("Time", {}).get("time"),
            "fastest_lap_speed": fastest_lap.get("AverageSpeed", {}).get("speed"),
            "fastest_lap_speed_unit": fastest_lap.get("AverageSpeed", {}).get("units")
        })

    time.sleep(0.3)

df = pd.DataFrame(rows)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print()
print(f"Downloaded {len(df)} race-result records.")
print(f"Saved to: {OUTPUT_FILE}")
print()
print("Records by season:")
print(df.groupby("season").size())