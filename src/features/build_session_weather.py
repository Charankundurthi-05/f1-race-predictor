from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

MASTER_PATH = BASE_DIR / "data" / "processed" / "master_dataset_with_sprint.csv"
WEATHER_PATH = BASE_DIR / "data" / "raw" / "weather_hourly.csv"
F1DB_ZIP_PATH = BASE_DIR / "data" / "raw" / "f1db-csv.zip"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "session_weather_features.csv"


SESSION_COLUMNS = {
    "fp1": ("freePractice1Date", "freePractice1Time"),
    "fp2": ("freePractice2Date", "freePractice2Time"),
    "fp3": ("freePractice3Date", "freePractice3Time"),
    "qualifying": ("qualifyingDate", "qualifyingTime"),
    "sprint_qualifying": ("sprintQualifyingDate", "sprintQualifyingTime"),
    "sprint": ("sprintRaceDate", "sprintRaceTime"),
}


WEATHER_NUMERIC_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
]


def build_session_schedule(races):
    records = []

    for _, row in races.iterrows():
        for session_name, (date_col, time_col) in SESSION_COLUMNS.items():
            date_value = row.get(date_col)
            time_value = row.get(time_col)

            if pd.isna(date_value) or pd.isna(time_value):
                continue

            date_text = str(date_value).strip()
            time_text = str(time_value).strip()

            if not date_text or not time_text:
                continue

            timestamp = pd.to_datetime(
                f"{date_text} {time_text}",
                errors="coerce"
            )

            if pd.isna(timestamp):
                continue

            records.append(
                {
                    "race_id": row["id"],
                    "season": int(row["year"]),
                    "round": int(row["round"]),
                    "session": session_name,
                    "session_time": timestamp,
                    "circuit_id": row["circuitId"],
                }
            )

    return pd.DataFrame(records)


def aggregate_session_weather(session_row, weather_race):
    session_time = session_row["session_time"]

    start_time = session_time - pd.Timedelta(hours=2)
    end_time = session_time + pd.Timedelta(hours=2)

    subset = weather_race[
        (weather_race["weather_time"] >= start_time)
        & (weather_race["weather_time"] <= end_time)
    ].copy()

    if subset.empty:
        return None

    result = {
        "season": session_row["season"],
        "round": session_row["round"],
        "race_id": session_row["race_id"],
        "session": session_row["session"],
        "session_time": session_time,
        "weather_observations": len(subset),
    }

    result["temperature_avg"] = subset["temperature_2m"].mean()
    result["temperature_min"] = subset["temperature_2m"].min()
    result["temperature_max"] = subset["temperature_2m"].max()

    result["humidity_avg"] = subset["relative_humidity_2m"].mean()

    result["precipitation_total"] = subset["precipitation"].sum()

    result["rain_hours"] = (subset["rain"] > 0).sum()
    result["rain_flag"] = int((subset["rain"] > 0).any())

    result["wind_speed_avg"] = subset["wind_speed_10m"].mean()
    result["wind_speed_max"] = subset["wind_speed_10m"].max()

    # Circular mean is more appropriate for wind direction.
    wind_values = subset["wind_direction_10m"].dropna().to_numpy()

    if len(wind_values) > 0:
        radians = np.deg2rad(wind_values)

        sin_mean = np.sin(radians).mean()
        cos_mean = np.cos(radians).mean()

        direction = np.rad2deg(
            np.arctan2(sin_mean, cos_mean)
        )

        if direction < 0:
            direction += 360

        result["wind_direction_avg"] = direction
    else:
        result["wind_direction_avg"] = np.nan

    result["cloud_cover_avg"] = subset["cloud_cover"].mean()

    return result


def main():
    print("=" * 70)
    print("BUILDING SESSION-SPECIFIC WEATHER FEATURES")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load master dataset
    # ------------------------------------------------------------

    master = pd.read_csv(MASTER_PATH)

    print()
    print(f"Master dataset shape: {master.shape}")

    # ------------------------------------------------------------
    # Load weather
    # ------------------------------------------------------------

    weather = pd.read_csv(WEATHER_PATH)

    print(f"Hourly weather rows: {len(weather)}")

    print()
    print("Hourly weather columns:")
    print(list(weather.columns))

    if "weather_time" not in weather.columns:
        raise ValueError(
            "Expected weather timestamp column 'weather_time' "
            "was not found."
        )

    weather["weather_time"] = pd.to_datetime(
        weather["weather_time"],
        errors="coerce"
    )

    invalid_weather = weather["weather_time"].isna().sum()

    print()
    print(f"Invalid weather timestamps: {invalid_weather}")

    if invalid_weather > 0:
        weather = weather.dropna(subset=["weather_time"]).copy()

    # ------------------------------------------------------------
    # Ensure weather numeric columns are numeric
    # ------------------------------------------------------------

    for column in WEATHER_NUMERIC_COLUMNS:
        if column in weather.columns:
            weather[column] = pd.to_numeric(
                weather[column],
                errors="coerce"
            )

    # ------------------------------------------------------------
    # Load F1DB race schedule
    # ------------------------------------------------------------

    with zipfile.ZipFile(F1DB_ZIP_PATH) as z:
        races = pd.read_csv(z.open("f1db-races.csv"))

    races = races[
        (races["year"] >= 2014)
        & (races["year"] <= 2026)
    ].copy()

    print()
    print(f"F1DB schedule rows 2014-2026: {len(races)}")

    # ------------------------------------------------------------
    # Build local-time session schedule
    #
    # IMPORTANT:
    # F1DB's session date/time represents the circuit's local
    # scheduled time. The Open-Meteo weather timestamps were
    # downloaded using the circuit-local timezone, but saved
    # without timezone information.
    #
    # Therefore BOTH sides are intentionally treated as naive
    # local timestamps here.
    # ------------------------------------------------------------

    session_schedule = build_session_schedule(races)

    if session_schedule.empty:
        raise ValueError(
            "No session schedule timestamps could be created."
        )

    print()
    print("Session schedule coverage:")
    print(
        session_schedule["session"]
        .value_counts()
        .sort_index()
        .rename("scheduled_sessions")
        .to_string()
    )

    # ------------------------------------------------------------
    # Prepare race keys
    # ------------------------------------------------------------

    weather["season"] = pd.to_numeric(
        weather["season"],
        errors="coerce"
    )

    weather["round"] = pd.to_numeric(
        weather["round"],
        errors="coerce"
    )

    weather["season"] = weather["season"].astype("Int64")
    weather["round"] = weather["round"].astype("Int64")

    # ------------------------------------------------------------
    # Generate session weather
    # ------------------------------------------------------------

    generated = []

    weather_groups = {
        key: group.copy()
        for key, group in weather.groupby(
            ["season", "round"],
            dropna=False
        )
    }

    for _, session_row in session_schedule.iterrows():

        key = (
            int(session_row["season"]),
            int(session_row["round"])
        )

        weather_race = weather_groups.get(key)

        if weather_race is None:
            continue

        result = aggregate_session_weather(
            session_row,
            weather_race
        )

        if result is not None:
            generated.append(result)

    session_weather = pd.DataFrame(generated)

    print()
    print("Generated session weather rows:")
    print(len(session_weather))

    if session_weather.empty:
        raise ValueError(
            "No session weather rows were generated. "
            "Check the relationship between F1DB session times "
            "and weather timestamps."
        )

    # ------------------------------------------------------------
    # Sort output
    # ------------------------------------------------------------

    session_order = {
        "fp1": 1,
        "sprint_qualifying": 2,
        "sprint": 3,
        "fp2": 4,
        "fp3": 5,
        "qualifying": 6,
    }

    session_weather["session_order"] = (
        session_weather["session"]
        .map(session_order)
    )

    session_weather = session_weather.sort_values(
        ["season", "round", "session_order"]
    ).drop(columns=["session_order"])

    session_weather = session_weather.reset_index(drop=True)

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    duplicate_count = session_weather.duplicated(
        subset=["season", "round", "session"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "(season, round, session) weather rows."
        )

    print()
    print("Generated sessions by type:")
    print(
        session_weather["session"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Generated sessions by season:")
    season_summary = pd.crosstab(
        session_weather["season"],
        session_weather["session"]
    )

    print(season_summary.to_string())

    # ------------------------------------------------------------
    # Coverage comparison
    # ------------------------------------------------------------

    scheduled_count = len(session_schedule)
    generated_count = len(session_weather)

    print()
    print("Session coverage:")
    print(
        f"{generated_count} / {scheduled_count} "
        f"scheduled sessions have usable weather"
    )

    # ------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------

    print()
    print("Session weather preview:")
    print(
        session_weather.head(20).to_string(index=False)
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    session_weather.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("Saved successfully:")
    print(OUTPUT_PATH)

    print()
    print("=" * 70)
    print("SESSION WEATHER BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()