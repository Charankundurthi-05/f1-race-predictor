import os
import time
import requests
import pandas as pd


INPUT_FILE = "data/processed/master_dataset_with_regulation.csv"
OUTPUT_FILE = "data/raw/weather_hourly.csv"

API_URL = "https://archive-api.open-meteo.com/v1/archive"

START_YEAR = 2014
END_YEAR = 2026

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
]

MAX_RETRIES = 5
REQUEST_DELAY = 1.0


def load_races():
    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "season",
        "round",
        "race_name",
        "race_date",
        "circuit_id",
        "circuit_name",
        "circuit_latitude",
        "circuit_longitude",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {INPUT_FILE}: {missing}"
        )

    races = (
        df[
            [
                "season",
                "round",
                "race_name",
                "race_date",
                "circuit_id",
                "circuit_name",
                "circuit_latitude",
                "circuit_longitude",
            ]
        ]
        .drop_duplicates(
            subset=["season", "round", "circuit_id"]
        )
        .copy()
    )

    races["season"] = pd.to_numeric(
        races["season"],
        errors="coerce"
    )

    races = races[
        races["season"].between(
            START_YEAR,
            END_YEAR
        )
    ].copy()

    races["race_date"] = pd.to_datetime(
        races["race_date"],
        errors="coerce"
    )

    races = races.dropna(
        subset=[
            "race_date",
            "circuit_latitude",
            "circuit_longitude",
        ]
    )

    # ------------------------------------------------------------
    # Full F1 weekend weather window
    #
    # We need coverage for:
    #
    # Normal weekend:
    # Thursday -> Friday -> Saturday -> Sunday
    #
    # Sprint weekend:
    # Friday -> Saturday -> Sunday
    #
    # Two days before the race gives sufficient coverage for
    # Thursday FP1 while one day after the race provides a small
    # safety margin.
    # ------------------------------------------------------------

    races["start_date"] = (
        races["race_date"]
        - pd.Timedelta(days=2)
    ).dt.strftime("%Y-%m-%d")

    races["end_date"] = (
        races["race_date"]
        + pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")

    races = races.sort_values(
        ["season", "round"]
    ).reset_index(drop=True)

    return races


def fetch_weather(
    latitude,
    longitude,
    start_date,
    end_date,
):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "auto",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=60,
            )

            if response.status_code == 200:
                return response.json()

            print(
                f"HTTP {response.status_code} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

        except requests.RequestException as exc:
            print(
                f"Request failed: {exc} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

        if attempt < MAX_RETRIES:
            wait_time = attempt * 3
            time.sleep(wait_time)

    return None


def weather_to_dataframe(
    weather_json,
    race,
):
    if weather_json is None:
        return pd.DataFrame()

    hourly = weather_json.get("hourly")

    if not hourly:
        return pd.DataFrame()

    times = hourly.get("time", [])

    if not times:
        return pd.DataFrame()

    row_count = len(times)

    data = {
        "season": [race["season"]] * row_count,
        "round": [race["round"]] * row_count,
        "race_name": [race["race_name"]] * row_count,
        "race_date": [race["race_date"]] * row_count,
        "circuit_id": [race["circuit_id"]] * row_count,
        "circuit_name": [race["circuit_name"]] * row_count,
        "latitude": [weather_json.get("latitude")] * row_count,
        "longitude": [weather_json.get("longitude")] * row_count,
        "weather_time": times,
    }

    for variable in HOURLY_VARIABLES:

        values = hourly.get(variable, [])

        if len(values) != row_count:

            values = values[:row_count]

            if len(values) < row_count:
                values += [None] * (
                    row_count - len(values)
                )

        data[variable] = values

    result = pd.DataFrame(data)

    result["weather_time"] = pd.to_datetime(
        result["weather_time"],
        errors="coerce"
    )

    return result


def main():

    print("=" * 80)
    print("F1 WEATHER DATA DOWNLOADER")
    print("=" * 80)

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    races = load_races()

    print()
    print(
        f"Races requiring weather data: {len(races)}"
    )

    print()
    print(
        "Weather window: race date - 2 days "
        "through race date + 1 day"
    )

    print()
    print(
        "IMPORTANT: existing weather data will be "
        "rebuilt because the previous dataset used "
        "an incomplete weekend window."
    )

    # ------------------------------------------------------------
    # Start fresh.
    #
    # The previous file cannot be reused because the downloader
    # originally requested only race_date - 1 through race_date + 1.
    # ------------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

        print()
        print(
            f"Removed previous weather file: {OUTPUT_FILE}"
        )

    all_results = []

    successful = 0
    failed = 0

    for index, race in races.iterrows():

        print()
        print(
            f"[{index + 1}/{len(races)}] "
            f"{int(race['season'])} "
            f"Round {int(race['round'])}: "
            f"{race['race_name']}"
        )

        print(
            f"Location: "
            f"{race['circuit_latitude']:.4f}, "
            f"{race['circuit_longitude']:.4f}"
        )

        print(
            f"Date range: "
            f"{race['start_date']} -> "
            f"{race['end_date']}"
        )

        weather_json = fetch_weather(
            latitude=race["circuit_latitude"],
            longitude=race["circuit_longitude"],
            start_date=race["start_date"],
            end_date=race["end_date"],
        )

        result = weather_to_dataframe(
            weather_json,
            race,
        )

        if result.empty:

            print(
                "FAILED: no weather data returned."
            )

            failed += 1
            continue

        all_results.append(result)

        successful += 1

        print(
            f"Downloaded {len(result)} hourly rows."
        )

        # --------------------------------------------------------
        # Checkpoint every 10 successful races
        # --------------------------------------------------------

        if successful % 10 == 0:

            combined = pd.concat(
                all_results,
                ignore_index=True
            )

            combined.to_csv(
                OUTPUT_FILE,
                index=False
            )

            print()
            print(
                f"Checkpoint saved: "
                f"{len(combined):,} rows"
            )

        time.sleep(REQUEST_DELAY)

    # ------------------------------------------------------------
    # No data protection
    # ------------------------------------------------------------

    if not all_results:

        print()
        print(
            "No weather data was downloaded."
        )

        return

    # ------------------------------------------------------------
    # Combine everything
    # ------------------------------------------------------------

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    combined = combined.drop_duplicates(
        subset=[
            "season",
            "round",
            "circuit_id",
            "weather_time",
        ]
    )

    combined = combined.sort_values(
        [
            "season",
            "round",
            "weather_time",
        ]
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Save final dataset
    # ------------------------------------------------------------

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)

    print(
        f"Successful races: {successful}"
    )

    print(
        f"Failed races: {failed}"
    )

    print(
        f"Total hourly rows: {len(combined):,}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    # ------------------------------------------------------------
    # Coverage summary
    # ------------------------------------------------------------

    coverage = (
        combined
        .groupby(
            ["season", "round"]
        )
        .size()
        .reset_index(name="hourly_rows")
    )

    print()
    print(
        "Weather coverage summary:"
    )

    print(
        f"Unique races with weather: "
        f"{len(coverage)}"
    )

    print(
        f"Expected races: "
        f"{len(races)}"
    )

    print()
    print(
        "Hourly rows per race:"
    )

    print(
        coverage["hourly_rows"]
        .describe()
        .to_string()
    )

    print()
    print("Weather variables:")

    for variable in HOURLY_VARIABLES:
        print(f"  - {variable}")

    print()
    print("=" * 80)
    print("WEATHER DOWNLOAD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()