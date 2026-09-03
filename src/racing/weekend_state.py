from pathlib import Path
from datetime import datetime
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

F1DB_FILE = (
    ROOT
    / "data"
    / "raw"
    / "f1db-races.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "predictions"
    / "weekend_state.json"
)


def load_schedule():

    if not F1DB_FILE.exists():
        raise FileNotFoundError(
            f"F1DB race schedule not found: {F1DB_FILE}"
        )

    df = pd.read_csv(
        F1DB_FILE
    )

    df = df[
        df["year"] == 2026
    ].copy()

    if df.empty:
        raise ValueError(
            "No 2026 races found in F1DB schedule."
        )

    return df


def parse_datetime(date_value, time_value=None):

    if pd.isna(date_value):
        return None

    date_text = str(date_value).strip()

    if not date_text:
        return None

    if (
        time_value is not None
        and not pd.isna(time_value)
    ):
        time_text = str(time_value).strip()

        if time_text:
            date_text = (
                f"{date_text} {time_text}"
            )

    try:
        return pd.Timestamp(
            date_text
        )

    except Exception:
        return None


def get_session_datetime(row, date_column, time_column):

    if date_column not in row.index:
        return None

    date_value = row[date_column]

    time_value = (
        row[time_column]
        if time_column in row.index
        else None
    )

    return parse_datetime(
        date_value,
        time_value
    )


def detect_weekend_format(row):

    sprint_qualifying = get_session_datetime(
        row,
        "sprintQualifyingDate",
        "sprintQualifyingTime"
    )

    sprint_race = get_session_datetime(
        row,
        "sprintRaceDate",
        "sprintRaceTime"
    )

    if (
        sprint_qualifying is not None
        or sprint_race is not None
    ):
        return "sprint"

    return "normal"


def determine_stage(
    now,
    race_date,
    weekend_format,
    row
):

    if weekend_format == "sprint":

        sprint_qualifying = get_session_datetime(
            row,
            "sprintQualifyingDate",
            "sprintQualifyingTime"
        )

        sprint_race = get_session_datetime(
            row,
            "sprintRaceDate",
            "sprintRaceTime"
        )

        qualifying = get_session_datetime(
            row,
            "qualifyingDate",
            "qualifyingTime"
        )

        if (
            sprint_qualifying is not None
            and now < sprint_qualifying
        ):
            return "pre_practice"

        if (
            sprint_qualifying is not None
            and sprint_race is not None
            and sprint_qualifying <= now < sprint_race
        ):
            return "sprint_qualifying"

        if (
            sprint_race is not None
            and qualifying is not None
            and sprint_race <= now < qualifying
        ):
            return "sprint"

        if (
            qualifying is not None
            and now < race_date
            and now >= qualifying
        ):
            return "race_ready"

        if now >= race_date:
            return "race_or_completed"

        return "pre_practice"

    # ---------------------------------------------------------------
    # Normal weekend
    # ---------------------------------------------------------------

    fp1 = get_session_datetime(
        row,
        "freePractice1Date",
        "freePractice1Time"
    )

    fp2 = get_session_datetime(
        row,
        "freePractice2Date",
        "freePractice2Time"
    )

    fp3 = get_session_datetime(
        row,
        "freePractice3Date",
        "freePractice3Time"
    )

    qualifying = get_session_datetime(
        row,
        "qualifyingDate",
        "qualifyingTime"
    )

    if now < fp1:
        return "pre_practice"

    if (
        fp1 is not None
        and fp2 is not None
        and now < fp2
    ):
        return "fp1"

    if (
        fp2 is not None
        and fp3 is not None
        and now < fp3
    ):
        return "fp2"

    if (
        fp3 is not None
        and qualifying is not None
        and now < qualifying
    ):
        return "fp3"

    if (
        qualifying is not None
        and now < race_date
    ):
        return "qualifying"

    if now >= race_date:
        return "race_or_completed"

    return "pre_practice"


def main():

    print("=" * 80)
    print("F1 RACE PREDICTOR")
    print("CURRENT WEEKEND STATE")
    print("=" * 80)
    print()

    schedule = load_schedule()

    # ---------------------------------------------------------------
    # Race dates
    # ---------------------------------------------------------------

    schedule["raceDateParsed"] = schedule.apply(
        lambda row: get_session_datetime(
            row,
            "date",
            "time"
        ),
        axis=1
    )

    now = pd.Timestamp.now()

    # ---------------------------------------------------------------
    # Find next upcoming race
    # ---------------------------------------------------------------

    future = schedule[
        schedule["raceDateParsed"] >= now
    ].sort_values(
        "raceDateParsed"
    )

    if future.empty:

        row = (
            schedule
            .sort_values("raceDateParsed")
            .iloc[-1]
        )

    else:

        row = future.iloc[0]

    race_date = row["raceDateParsed"]

    if race_date is None:
        raise ValueError(
            "Could not determine race date."
        )

    weekend_format = detect_weekend_format(
        row
    )

    current_stage = determine_stage(
        now,
        race_date,
        weekend_format,
        row
    )

    countdown_seconds = max(
        0,
        int(
            (
                race_date - now
            ).total_seconds()
        )
    )

    # ---------------------------------------------------------------
    # Build state
    # ---------------------------------------------------------------

    state = {
        "season": 2026,

        "round": int(
            row["round"]
        ),

        "race_name": str(
            row["officialName"]
        ),

        "circuit_id": str(
            row["circuitId"]
        ),

        "circuit_name": str(
            row["circuitId"]
        ),

        "race_date": race_date.isoformat(),

        "weekend_format": weekend_format,

        "current_stage": current_stage,

        "countdown_seconds": countdown_seconds,

        "generated_at": datetime.now().isoformat()
    }

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            state,
            f,
            indent=2
        )

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

    print(
        f"Season: {state['season']}"
    )

    print(
        f"Round: {state['round']}"
    )

    print(
        f"Race: {state['race_name']}"
    )

    print(
        f"Circuit ID: {state['circuit_id']}"
    )

    print(
        f"Race date: {state['race_date']}"
    )

    print(
        f"Weekend format: "
        f"{state['weekend_format']}"
    )

    print(
        f"Current stage: "
        f"{state['current_stage']}"
    )

    print(
        f"Countdown to race: "
        f"{state['countdown_seconds']} seconds"
    )

    print(
        f"Generated at: "
        f"{state['generated_at']}"
    )

    print()

    print(
        f"Saved -> {OUTPUT_FILE}"
    )

    print()
    print("=" * 80)
    print("WEEKEND STATE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()