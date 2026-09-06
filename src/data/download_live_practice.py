from pathlib import Path
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PREDICTION_DIR = BASE_DIR / "data" / "predictions"

BASE_URL = "https://api.openf1.org/v1"

LIVE_PRACTICE_PATH = RAW_DIR / "live_2026_practice.csv"


def get_json(endpoint, params=None):
    url = f"{BASE_URL}/{endpoint}"

    try:
        response = requests.get(
            url,
            params=params,
            timeout=30,
        )
    except requests.RequestException as exc:
        print(
            f"OpenF1 request failed for {endpoint}: {exc}"
        )
        return None

    if response.status_code == 404:
        return []

    if response.status_code in (401, 403):
        print(
            f"OpenF1 live access unavailable for {endpoint} "
            f"(HTTP {response.status_code})."
        )
        print("Continuing without fresh live practice data.")
        return None

    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        print(
            f"OpenF1 request failed for {endpoint}: {exc}"
        )
        return None

    try:
        return response.json()
    except ValueError:
        print(
            f"OpenF1 returned invalid JSON for {endpoint}."
        )
        return None


def get_current_meeting():
    meetings = get_json(
        "meetings",
        {"year": 2026},
    )

    if meetings is None or not meetings:
        return None

    now = pd.Timestamp.now(tz="UTC")

    meetings = sorted(
        meetings,
        key=lambda x: x.get("date_start", ""),
    )

    current_or_next = None

    for meeting in meetings:
        try:
            start = pd.to_datetime(
                meeting["date_start"],
                utc=True,
            )
            end = pd.to_datetime(
                meeting["date_end"],
                utc=True,
            )
        except Exception:
            continue

        if start <= now <= end:
            return meeting

        if start > now:
            current_or_next = meeting
            break

    if current_or_next is not None:
        return current_or_next

    return meetings[-1]


def get_sessions(meeting_key):
    return get_json(
        "sessions",
        {"meeting_key": meeting_key},
    )


def get_session_results(session_key):
    return get_json(
        "session_result",
        {"session_key": session_key},
    )


def get_drivers(session_key):
    return get_json(
        "drivers",
        {"session_key": session_key},
    )


def process_session(session):
    session_key = session["session_key"]
    session_name = session["session_name"]

    print()
    print(
        f"{session_name}: session_key={session_key}"
    )

    results = get_session_results(session_key)

    if results is None:
        print("  Live results could not be accessed.")
        return []

    if not results:
        print("  Results not available yet")
        return []

    drivers = get_drivers(session_key)

    if drivers is None:
        print(
            "  Driver information could not be accessed."
        )
        return []

    driver_map = {
        driver["driver_number"]: driver
        for driver in drivers
        if "driver_number" in driver
    }

    rows = []

    for result in results:
        number = result.get("driver_number")
        driver = driver_map.get(number, {})

        rows.append(
            {
                "season": 2026,
                "meeting_key": session["meeting_key"],
                "session_key": session_key,
                "session_name": session_name,
                "driver_number": number,
                "driver_name": driver.get("full_name"),
                "driver_code": driver.get("name_acronym"),
                "team_name": driver.get("team_name"),
                "position": result.get("position"),
                "duration": result.get("duration"),
                "gap_to_leader": result.get("gap_to_leader"),
                "number_of_laps": result.get("number_of_laps"),
                "dnf": result.get("dnf"),
                "dns": result.get("dns"),
                "dsq": result.get("dsq"),
            }
        )

    return rows


def print_footer():
    print()
    print("=" * 80)
    print("LIVE PRACTICE CHECK COMPLETE")
    print("=" * 80)


def main():
    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("LIVE 2026 PRACTICE DATA")
    print("=" * 80)

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    meeting = get_current_meeting()

    if meeting is None:
        print()
        print("Live practice data is unavailable.")
        print(
            "The public OpenF1 endpoint did not provide "
            "accessible 2026 meeting data."
        )
        print(
            "The pipeline will continue using "
            "the latest valid prediction."
        )
        print_footer()
        return

    print()
    print(
        f"Meeting: {meeting.get('meeting_name', 'Unknown')}"
    )
    print(
        f"Location: {meeting.get('location', 'Unknown')}"
    )
    print(
        f"Meeting key: {meeting.get('meeting_key', 'Unknown')}"
    )

    sessions = get_sessions(meeting["meeting_key"])

    if sessions is None:
        print()
        print(
            "Practice session information is "
            "currently unavailable."
        )
        print(
            "The pipeline will continue without "
            "new live practice data."
        )
        print_footer()
        return

    practice_sessions = [
        session
        for session in sessions
        if session.get("session_name")
        in ["Practice 1", "Practice 2", "Practice 3"]
    ]

    print()
    print(
        f"Practice sessions found: {len(practice_sessions)}"
    )

    all_rows = []

    for session in practice_sessions:
        all_rows.extend(
            process_session(session)
        )

    if not all_rows:
        print()
        print(
            "No new live practice results are available."
        )
        print("Keeping the latest valid prediction.")
        print_footer()
        return

    df = pd.DataFrame(all_rows)

    if "position" in df.columns:
        df["position"] = pd.to_numeric(
            df["position"],
            errors="coerce",
        )

    df = df.sort_values(
        ["session_key", "position"],
        na_position="last",
    )

    output_path = RAW_DIR / "live_2026_practice.csv"
    prediction_copy = (
        PREDICTION_DIR / "live_practice_results.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    df.to_csv(
        prediction_copy,
        index=False,
    )

    print()
    print(
        f"Saved {len(df)} rows -> {output_path}"
    )
    print(
        f"Saved {len(df)} rows -> {prediction_copy}"
    )

    print()
    print(
        df[
            [
                "session_name",
                "position",
                "driver_name",
                "team_name",
                "duration",
            ]
        ].to_string(index=False)
    )

    print_footer()


if __name__ == "__main__":
    main()
