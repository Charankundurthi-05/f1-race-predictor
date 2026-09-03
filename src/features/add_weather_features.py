import os
import pandas as pd


INPUT_FILE = "data/raw/weather_hourly.csv"
OUTPUT_FILE = "data/processed/weather_features.csv"


def load_weather():
    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "season",
        "round",
        "race_name",
        "race_date",
        "circuit_id",
        "circuit_name",
        "weather_time",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required weather columns: {missing}"
        )

    df["weather_time"] = pd.to_datetime(
        df["weather_time"],
        errors="coerce"
    )

    df["race_date"] = pd.to_datetime(
        df["race_date"],
        errors="coerce"
    )

    numeric_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "season",
            "round",
            "circuit_id",
            "weather_time",
            "race_date",
        ]
    )

    return df


def calculate_statistics(data, prefix):
    result = {}

    if data.empty:
        result[f"{prefix}_temperature_avg"] = None
        result[f"{prefix}_temperature_min"] = None
        result[f"{prefix}_temperature_max"] = None
        result[f"{prefix}_humidity_avg"] = None
        result[f"{prefix}_precipitation_total"] = None
        result[f"{prefix}_rain_hours"] = None
        result[f"{prefix}_rain_flag"] = None
        result[f"{prefix}_wind_speed_avg"] = None
        result[f"{prefix}_wind_speed_max"] = None
        result[f"{prefix}_cloud_cover_avg"] = None

        return result

    result[f"{prefix}_temperature_avg"] = (
        data["temperature_2m"].mean()
    )

    result[f"{prefix}_temperature_min"] = (
        data["temperature_2m"].min()
    )

    result[f"{prefix}_temperature_max"] = (
        data["temperature_2m"].max()
    )

    result[f"{prefix}_humidity_avg"] = (
        data["relative_humidity_2m"].mean()
    )

    result[f"{prefix}_precipitation_total"] = (
        data["precipitation"].sum()
    )

    result[f"{prefix}_rain_hours"] = (
        (data["rain"] > 0).sum()
    )

    result[f"{prefix}_rain_flag"] = int(
        (data["rain"] > 0).any()
    )

    result[f"{prefix}_wind_speed_avg"] = (
        data["wind_speed_10m"].mean()
    )

    result[f"{prefix}_wind_speed_max"] = (
        data["wind_speed_10m"].max()
    )

    result[f"{prefix}_cloud_cover_avg"] = (
        data["cloud_cover"].mean()
    )

    return result


def calculate_features(group, season, round_number, circuit_id):
    group = group.sort_values(
        "weather_time"
    ).copy()

    race_date = pd.to_datetime(
        group["race_date"].iloc[0]
    )

    race_day_start = race_date.normalize()

    weekend_start = (
        race_day_start - pd.Timedelta(days=1)
    )

    weekend_end = (
        race_day_start
        + pd.Timedelta(days=1)
        - pd.Timedelta(hours=1)
    )

    fp1_start = weekend_start

    fp1_end = (
        race_day_start
        - pd.Timedelta(hours=1)
    )

    race_day_start_time = race_day_start

    race_day_end_time = weekend_end

    weekend_mask = (
        (group["weather_time"] >= weekend_start)
        & (group["weather_time"] <= weekend_end)
    )

    fp1_mask = (
        (group["weather_time"] >= fp1_start)
        & (group["weather_time"] <= fp1_end)
    )

    race_day_mask = (
        (group["weather_time"] >= race_day_start_time)
        & (group["weather_time"] <= race_day_end_time)
    )

    weekend = group.loc[
        weekend_mask
    ]

    fp1_period = group.loc[
        fp1_mask
    ]

    race_day = group.loc[
        race_day_mask
    ]

    result = {
        "season": season,
        "round": round_number,
        "race_name": group["race_name"].iloc[0],
        "race_date": group["race_date"].iloc[0],
        "circuit_id": circuit_id,
        "circuit_name": group["circuit_name"].iloc[0],
    }

    result.update(
        calculate_statistics(
            weekend,
            "weekend"
        )
    )

    result.update(
        calculate_statistics(
            fp1_period,
            "fp1_period"
        )
    )

    result.update(
        calculate_statistics(
            race_day,
            "race_day"
        )
    )

    result["weather_observations"] = len(
        group
    )

    return pd.Series(result)


def main():
    print("=" * 80)
    print("F1 WEATHER FEATURE ENGINEERING")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    weather = load_weather()

    print()
    print(
        f"Hourly weather rows: {len(weather):,}"
    )

    unique_races = (
        weather[
            [
                "season",
                "round",
                "circuit_id",
            ]
        ]
        .drop_duplicates()
    )

    print(
        f"Unique races: {len(unique_races)}"
    )

    results = []

    grouped = weather.groupby(
        [
            "season",
            "round",
            "circuit_id",
        ],
        dropna=False,
    )

    for (
        season,
        round_number,
        circuit_id,
    ), group in grouped:

        result = calculate_features(
            group=group,
            season=season,
            round_number=round_number,
            circuit_id=circuit_id,
        )

        results.append(result)

    features = pd.DataFrame(
        results
    )

    features = features.sort_values(
        [
            "season",
            "round",
        ]
    ).reset_index(drop=True)

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 80)
    print("WEATHER FEATURE ENGINEERING COMPLETE")
    print("=" * 80)

    print(
        f"Feature rows: {len(features):,}"
    )

    print(
        f"Feature columns: {len(features.columns)}"
    )

    print()
    print("Columns:")

    for column in features.columns:
        print(f"  - {column}")

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()
    print("Weather feature preview:")

    preview_columns = [
        "season",
        "round",
        "race_name",
        "weekend_temperature_avg",
        "weekend_precipitation_total",
        "weekend_rain_flag",
        "fp1_period_temperature_avg",
        "fp1_period_rain_flag",
        "race_day_temperature_avg",
        "race_day_rain_flag",
    ]

    print(
        features[
            preview_columns
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()