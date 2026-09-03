from pathlib import Path
import requests
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    ROOT
    / "data"
    / "predictions"
    / "live_sprint_qualifying_results.csv"
)


def get_current_2026_meeting():

    url = (
        "https://api.openf1.org/v1/meetings"
        "?year=2026"
    )

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    meetings = response.json()

    if not meetings:
        return None

    meetings_df = pd.DataFrame(meetings)

    meetings_df["date_start"] = pd.to_datetime(
        meetings_df["date_start"],
        errors="coerce"
    )

    now = pd.Timestamp.now(
        tz="UTC"
    )

    meetings_df["date_start"] = (
        meetings_df["date_start"]
        .dt.tz_convert("UTC")
    )

    upcoming = meetings_df[
        meetings_df["date_start"] >= now
    ].sort_values(
        "date_start"
    )

    if not upcoming.empty:
        return upcoming.iloc[0].to_dict()

    return meetings_df.sort_values(
        "date_start"
    ).iloc[-1].to_dict()


def get_sprint_qualifying_session(meeting_key):

    url = (
        "https://api.openf1.org/v1/sessions"
        f"?meeting_key={meeting_key}"
    )

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    sessions = response.json()

    if not sessions:
        return None

    for session in sessions:

        name = str(
            session.get(
                "session_name",
                ""
            )
        ).upper()

        if (
            "SPRINT QUALIFYING" in name
            or name == "SPRINT QUALI"
        ):
            return session

    return None


def download_results(session_key):

    url = (
        "https://api.openf1.org/v1/session_result"
        f"?session_key={session_key}"
    )

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    results = response.json()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    return df


def add_driver_names(df):

    if df.empty:
        return df

    if "driver_number" not in df.columns:
        return df

    url = (
        "https://api.openf1.org/v1/drivers"
        "?session_key=latest"
    )

    try:

        response = requests.get(
            url,
            timeout=30
        )

        if response.status_code != 200:
            return df

        drivers = response.json()

        if not drivers:
            return df

        drivers_df = pd.DataFrame(
            drivers
        )

        if "driver_number" not in drivers_df.columns:
            return df

        name_columns = [
            column
            for column in [
                "full_name",
                "broadcast_name",
                "name_acronym"
            ]
            if column in drivers_df.columns
        ]

        if not name_columns:
            return df

        name_column = name_columns[0]

        drivers_df = drivers_df[
            [
                "driver_number",
                name_column
            ]
        ].drop_duplicates(
            "driver_number"
        )

        df = df.merge(
            drivers_df,
            on="driver_number",
            how="left"
        )

        df = df.rename(
            columns={
                name_column: "driver_name"
            }
        )

    except Exception:
        pass

    return df


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("DOWNLOAD LIVE SPRINT QUALIFYING")
    print("=" * 80)
    print()

    try:

        meeting = get_current_2026_meeting()

        if meeting is None:

            print(
                "No 2026 meeting found."
            )

            return

        meeting_key = meeting.get(
            "meeting_key"
        )

        meeting_name = meeting.get(
            "meeting_name",
            "Unknown"
        )

        location = meeting.get(
            "location",
            "Unknown"
        )

        print(
            f"Meeting: {meeting_name}"
        )

        print(
            f"Location: {location}"
        )

        print(
            f"Meeting key: {meeting_key}"
        )

        print()

        session = get_sprint_qualifying_session(
            meeting_key
        )

        if session is None:

            print(
                "Sprint Qualifying session not found "
                "for the current meeting."
            )

            print(
                "This may be a normal weekend."
            )

            return

        session_key = session.get(
            "session_key"
        )

        session_name = session.get(
            "session_name",
            "Sprint Qualifying"
        )

        print(
            f"Session: {session_name}"
        )

        print(
            f"Session key: {session_key}"
        )

        print()

        results = download_results(
            session_key
        )

        if results.empty:

            print(
                "Sprint Qualifying results "
                "are not available yet."
            )

            print(
                "The downloader is ready "
                "once the official results are published."
            )

            return

        results = add_driver_names(
            results
        )

        # -----------------------------------------------------------
        # Standardize columns for the prediction pipeline
        # -----------------------------------------------------------

        if "position" in results.columns:

            results["sprint_qualifying_position"] = pd.to_numeric(
                results["position"],
                errors="coerce"
            )

        elif "position_number" in results.columns:

            results["sprint_qualifying_position"] = pd.to_numeric(
                results["position_number"],
                errors="coerce"
            )

        if "driver_name" not in results.columns:

            print(
                "Driver names could not be obtained."
            )

            return

        results["driver_name"] = (
            results["driver_name"]
            .astype(str)
            .str.strip()
        )

        results = results[
            results[
                "sprint_qualifying_position"
            ].notna()
        ].copy()

        results = results.sort_values(
            "sprint_qualifying_position"
        )

        if len(results) < 22:

            print(
                f"Only {len(results)} drivers "
                "have Sprint Qualifying results."
            )

            print(
                "Waiting for the complete official classification."
            )

            return

        # Keep the first 22 classified drivers.
        results = results.head(
            22
        ).copy()

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        results.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            f"Sprint Qualifying results: "
            f"{len(results)} drivers"
        )

        print(
            f"Saved -> {OUTPUT_FILE}"
        )

        print()
        print("=" * 80)
        print("LIVE SPRINT QUALIFYING DOWNLOAD COMPLETE")
        print("=" * 80)

    except requests.RequestException as exc:

        print(
            f"API error: {exc}"
        )

    except Exception as exc:

        print(
            f"Error: {exc}"
        )


if __name__ == "__main__":
    main()