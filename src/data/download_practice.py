import requests
import pandas as pd
from pathlib import Path
import time

SESSIONS_URL = "https://api.jolpi.ca/f1/alpha/core/sessions/"
ENTRIES_URL = "https://api.jolpi.ca/f1/alpha/core/session-entries/"

OUTPUT_FILE = Path("data/raw/practice_results.csv")

START_YEAR = 2014
END_YEAR = 2026

PRACTICE_TYPES = {"FP1", "FP2", "FP3"}

session = requests.Session()


def get_json(url, params=None):

    for attempt in range(1, 4):

        try:

            response = session.get(
                url,
                params=params,
                timeout=(10, 60)
            )

            if response.status_code == 200:
                return response.json()

            print(
                f"  Attempt {attempt}: "
                f"HTTP {response.status_code}"
            )

        except requests.exceptions.RequestException as e:

            print(
                f"  Attempt {attempt} failed: "
                f"{type(e).__name__}"
            )

        if attempt < 3:
            time.sleep(5)

    return None


def get_all_sessions():

    print("Getting F1 sessions...")

    sessions = []

    page = 1

    while True:

        data = get_json(
            SESSIONS_URL,
            {
                "page": page
            }
        )

        if data is None:
            print(
                f"Failed to download sessions page {page}"
            )
            break

        metadata = data.get(
            "metadata",
            {}
        )

        records = data.get(
            "data",
            []
        )

        sessions.extend(records)

        total_pages = metadata.get(
            "total_pages",
            page
        )

        print(
            f"  Sessions page "
            f"{page}/{total_pages} "
            f"({len(records)} records)"
        )

        if page >= total_pages:
            break

        page += 1

        time.sleep(0.5)

    return sessions


def extract_year(session_data):

    timestamp = session_data.get(
        "timestamp"
    )

    if timestamp:

        try:
            return pd.to_datetime(
                timestamp
            ).year

        except Exception:
            pass

    return None


def get_practice_sessions():

    all_sessions = get_all_sessions()

    practice_sessions = []

    for session_data in all_sessions:

        session_type = session_data.get(
            "type"
        )

        if session_type not in PRACTICE_TYPES:
            continue

        year = extract_year(
            session_data
        )

        if year is None:
            continue

        if START_YEAR <= year <= END_YEAR:

            practice_sessions.append(
                session_data
            )

    return practice_sessions


def download_entries(session_data):

    session_id = session_data.get(
        "id"
    )

    session_type = session_data.get(
        "type"
    )

    timestamp = session_data.get(
        "timestamp"
    )

    round_data = session_data.get(
        "round",
        {}
    )

    round_number = round_data.get(
        "number"
    )

    race_name = round_data.get(
        "name"
    )

    year = extract_year(
        session_data
    )

    print(
        f"Downloading {year} "
        f"Round {round_number} "
        f"{race_name} "
        f"{session_type}..."
    )

    page = 1

    rows = []

    while True:

        data = get_json(
            ENTRIES_URL,
            {
                "session_id": session_id,
                "page": page
            }
        )

        if data is None:

            print(
                f"  Failed session "
                f"{session_id} page {page}"
            )

            break

        metadata = data.get(
            "metadata",
            {}
        )

        records = data.get(
            "data",
            []
        )

        for entry in records:

            driver_data = entry.get(
                "driver",
                {}
            )

            constructor_data = entry.get(
                "constructor",
                {}
            )

            rows.append({

                "season": year,

                "round": round_number,

                "race_name": race_name,

                "session_type": session_type,

                "session_id": session_id,

                "driver_id": driver_data.get(
                    "id"
                ),

                "driver_code": driver_data.get(
                    "code"
                ),

                "driver_name": (
                    f'{driver_data.get("given_name", "")} '
                    f'{driver_data.get("family_name", "")}'
                ).strip(),

                "constructor_id": constructor_data.get(
                    "id"
                ),

                "constructor_name": constructor_data.get(
                    "name"
                ),

                "position": entry.get(
                    "position"
                ),

                "laps_completed": entry.get(
                    "laps_completed"
                ),

                "fastest_lap_rank": entry.get(
                    "fastest_lap_rank"
                ),

                "fastest_lap_time": entry.get(
                    "fastest_lap_time"
                ),

                "average_lap_time": entry.get(
                    "average_lap_time"
                ),

                "top_speed": entry.get(
                    "top_speed"
                )
            })

        total_pages = metadata.get(
            "total_pages",
            page
        )

        if page >= total_pages:
            break

        page += 1

        time.sleep(0.5)

    return rows


def main():

    practice_sessions = get_practice_sessions()

    print()
    print(
        f"Found {len(practice_sessions)} "
        f"practice sessions."
    )

    if not practice_sessions:

        print(
            "No practice sessions found."
        )

        return

    print()

    all_rows = []

    for session_data in practice_sessions:

        rows = download_entries(
            session_data
        )

        all_rows.extend(
            rows
        )

        time.sleep(0.5)

    df = pd.DataFrame(
        all_rows
    )

    if df.empty:

        print(
            "No practice-session records downloaded."
        )

        return

    df = df.drop_duplicates(
        subset=[
            "session_id",
            "driver_id"
        ]
    )

    df = df.sort_values(
        [
            "season",
            "round",
            "session_type",
            "position"
        ],
        na_position="last"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"Downloaded {len(df)} "
        f"practice-session records."
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print()
    print(
        "Records by season:"
    )

    print(
        df.groupby("season").size()
    )

    print()
    print(
        "Records by session type:"
    )

    print(
        df.groupby("session_type").size()
    )


if __name__ == "__main__":
    main()