import requests
import pandas as pd
from pathlib import Path

BASE_URL = "https://api.jolpi.ca/ergast/f1"

OUTPUT_FILE = Path("data/raw/calendars.csv")

rows = []

for season in range(2014, 2027):
    url = f"{BASE_URL}/{season}/races/"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()
    races = data["MRData"]["RaceTable"]["Races"]

    for race in races:
        rows.append({
            "season": season,
            "round": int(race["round"]),
            "race_name": race["raceName"],
            "circuit_id": race["Circuit"]["circuitId"],
            "circuit_name": race["Circuit"]["circuitName"],
            "country": race["Circuit"]["Location"]["country"],
            "city": race["Circuit"]["Location"]["locality"],
            "race_date": race.get("date"),
            "race_time": race.get("time"),
            "fp1_date": race.get("FirstPractice", {}).get("date"),
            "fp1_time": race.get("FirstPractice", {}).get("time"),
            "fp2_date": race.get("SecondPractice", {}).get("date"),
            "fp2_time": race.get("SecondPractice", {}).get("time"),
            "fp3_date": race.get("ThirdPractice", {}).get("date"),
            "fp3_time": race.get("ThirdPractice", {}).get("time"),
            "qualifying_date": race.get("Qualifying", {}).get("date"),
            "qualifying_time": race.get("Qualifying", {}).get("time"),
            "sprint_date": race.get("Sprint", {}).get("date"),
            "sprint_time": race.get("Sprint", {}).get("time"),
            "sprint_qualifying_date": (
                race.get("SprintQualifying", {}).get("date")
                or race.get("SprintShootout", {}).get("date")
            ),
            "sprint_qualifying_time": (
                race.get("SprintQualifying", {}).get("time")
                or race.get("SprintShootout", {}).get("time")
            ),
            "is_sprint_weekend": "Sprint" in race
        })

df = pd.DataFrame(rows)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Downloaded {len(df)} races.")
print(f"Saved to: {OUTPUT_FILE}")
print()
print(df.groupby("season").size())