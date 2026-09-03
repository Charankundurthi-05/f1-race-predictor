from pathlib import Path
import requests
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"

BASE_URL = "https://api.openf1.org/v1"


def get_json(endpoint, params=None):
    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        params=params,
        timeout=30
    )

    if response.status_code == 404:
        return []

    response.raise_for_status()

    return response.json()


def get_current_meeting():
    meetings = get_json(
        "meetings",
        {
            "year": 2026
        }
    )

    if not meetings:
        return None

    now = pd.Timestamp.now(tz="UTC")

    meetings = sorted(
        meetings,
        key=lambda x: x["date_start"]
    )

    current_or_next = None

    for meeting in meetings:
        start = pd.to_datetime(
            meeting["date_start"],
            utc=True
        )

        end = pd.to_datetime(
            meeting["date_end"],
            utc=True
        )

        if start <= now <= end:
            return meeting

        if start > now:
            current_or_next = meeting
            break

    return current_or_next


def get_sessions(meeting_key):
    return get_json(
        "sessions",
        {
            "meeting_key": meeting_key
        }
    )


def get_session_results(session_key):
    return get_json(
        "session_result",
        {
            "session_key": session_key
        }
    )


def get_drivers(session_key):
    return get_json(
        "drivers",
        {
            "session_key": session_key
        }
    )


def process_session(session):
    session_key = session["session_key"]
    session_name = session["session_name"]

    print()
    print(
        f"{session_name}: "
        f"session_key={session_key}"
    )

    results = get_session_results(
        session_key
    )

    if not results:
        print(
            "  Results not available yet"
        )
        return []

    drivers = get_drivers(
        session_key
    )

    driver_map = {
        driver["driver_number"]: driver
        for driver in drivers
    }

    rows = []

    for result in results:
        number = result.get(
            "driver_number"
        )

        driver = driver_map.get(
            number,
            {}
        )

        rows.append({
            "season": 2026,
            "meeting_key": session["meeting_key"],
            "session_key": session_key,
            "session_name": session_name,
            "driver_number": number,
            "driver_name": driver.get(
                "full_name"
            ),
            "driver_code": driver.get(
                "name_acronym"
            ),
            "team_name": driver.get(
                "team_name"
            ),
            "position": result.get(
                "position"
            ),
            "duration": result.get(
                "duration"
            ),
            "gap_to_leader": result.get(
                "gap_to_leader"
            ),
            "number_of_laps": result.get(
                "number_of_laps"
            ),
            "dnf": result.get(
                "dnf"
            ),
            "dns": result.get(
                "dns"
            ),
            "dsq": result.get(
                "dsq"
            )
        })

    return rows


def main():
    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("LIVE 2026 PRACTICE DATA")
    print("=" * 80)

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    meeting = get_current_meeting()

    if meeting is None:
        print()
        print(
            "No 2026 meeting found."
        )
        return

    print()
    print(
        f"Meeting: "
        f"{meeting['meeting_name']}"
    )

    print(
        f"Location: "
        f"{meeting['location']}"
    )

    print(
        f"Meeting key: "
        f"{meeting['meeting_key']}"
    )

    sessions = get_sessions(
        meeting["meeting_key"]
    )

    practice_sessions = [
        session
        for session in sessions
        if session.get("session_name")
        in [
            "Practice 1",
            "Practice 2",
            "Practice 3"
        ]
    ]

    print()
    print(
        f"Practice sessions found: "
        f"{len(practice_sessions)}"
    )

    all_rows = []

    for session in practice_sessions:
        rows = process_session(
            session
        )

        all_rows.extend(rows)

    if not all_rows:
        print()
        print(
            "No practice results are "
            "available yet."
        )

        print(
            "The downloader is ready for "
            "the session once results are published."
        )

        print()
        print("=" * 80)
        print("LIVE PRACTICE CHECK COMPLETE")
        print("=" * 80)

        return

    df = pd.DataFrame(
        all_rows
    )

    df = df.sort_values(
        [
            "session_key",
            "position"
        ]
    )

    output_path = (
        RAW_DIR
        / "live_2026_practice.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"Saved {len(df)} rows -> "
        f"{output_path}"
    )

    print()
    print(
        df[
            [
                "session_name",
                "position",
                "driver_name",
                "team_name",
                "duration"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("LIVE PRACTICE DATA CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()