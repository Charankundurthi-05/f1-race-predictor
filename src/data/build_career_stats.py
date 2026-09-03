from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
F1DB_DIR = PROJECT_ROOT / "data" / "raw" / "f1db"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_f1db_file(filename):
    paths = [
        F1DB_DIR / filename,
        PROJECT_ROOT / "data" / "raw" / filename,
    ]

    for path in paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find {filename}\n"
        f"Checked:\n" +
        "\n".join(str(path) for path in paths)
    )


def load_f1db(filename):
    path = find_f1db_file(filename)
    print(f"Using F1DB file: {path.name}")
    return pd.read_csv(path, low_memory=False)


def num(series):
    return pd.to_numeric(series, errors="coerce")


print("=" * 80)
print("F1 RACE PREDICTOR")
print("BUILD ALL-TIME CAREER STATISTICS")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

drivers = load_f1db("f1db-drivers.csv")
constructors = load_f1db("f1db-constructors.csv")
races = load_f1db("f1db-races.csv")
race_results = load_f1db("f1db-races-race-results.csv")
qualifying = load_f1db("f1db-races-qualifying-results.csv")
driver_standings = load_f1db("f1db-races-driver-standings.csv")
constructor_standings = load_f1db(
    "f1db-races-constructor-standings.csv"
)

# ============================================================================
# NORMALIZE IDS
# ============================================================================

drivers["id"] = drivers["id"].astype(str)
constructors["id"] = constructors["id"].astype(str)

race_results["driverId"] = race_results["driverId"].astype(str)
race_results["constructorId"] = race_results["constructorId"].astype(str)

qualifying["driverId"] = qualifying["driverId"].astype(str)
driver_standings["driverId"] = driver_standings["driverId"].astype(str)

constructor_standings["constructorId"] = (
    constructor_standings["constructorId"].astype(str)
)

# ============================================================================
# ALL-TIME DRIVER CAREER STATISTICS
# ============================================================================

print("Building all-time driver statistics...")

driver_columns = {
    "id": "driver_id",
    "name": "driver_name",
    "abbreviation": "driver_code",
    "totalRaceEntries": "race_entries",
    "totalRaceStarts": "starts",
    "totalRaceWins": "wins",
    "totalPodiums": "podiums",
    "totalPoints": "points",
    "totalPolePositions": "poles",
    "totalFastestLaps": "fastest_laps",
    "totalChampionshipWins": "championships",
    "totalRaceLaps": "race_laps",
    "totalSprintRaceStarts": "sprint_starts",
    "totalSprintRaceWins": "sprint_wins",
    "totalDriverOfTheDay": "driver_of_the_day",
    "totalGrandSlams": "grand_slams",
    "bestChampionshipPosition": "best_championship_position",
    "bestStartingGridPosition": "best_starting_grid",
    "bestRaceResult": "best_finish",
    "bestSprintRaceResult": "best_sprint_finish",
}

available = {
    source: target
    for source, target in driver_columns.items()
    if source in drivers.columns
}

driver_career = drivers[
    list(available.keys())
].rename(columns=available)

# Make numerical career columns numeric.
for column in driver_career.columns:
    if column not in [
        "driver_id",
        "driver_name",
        "driver_code",
    ]:
        driver_career[column] = num(driver_career[column])

# ============================================================================
# LATEST CONSTRUCTOR FOR EACH DRIVER
# ============================================================================

print("Determining latest constructor for each driver...")

race_info = races[
    ["id", "year", "round"]
].copy()

race_info["id"] = num(race_info["id"])
race_info["year"] = num(race_info["year"])
race_info["round"] = num(race_info["round"])

team_history = race_results[
    ["raceId", "driverId", "constructorId"]
].copy()

team_history["raceId"] = num(team_history["raceId"])

team_history = team_history.merge(
    race_info,
    left_on="raceId",
    right_on="id",
    how="left",
)

team_history = team_history.sort_values(
    ["year", "round", "raceId"]
)

latest_team = (
    team_history
    .drop_duplicates("driverId", keep="last")
    [["driverId", "constructorId"]]
)

latest_team = latest_team.merge(
    constructors[["id", "name"]],
    left_on="constructorId",
    right_on="id",
    how="left",
)

latest_team = latest_team[
    ["driverId", "name"]
].rename(
    columns={
        "driverId": "driver_id",
        "name": "last_team",
    }
)

driver_career = driver_career.merge(
    latest_team,
    on="driver_id",
    how="left",
)

# ============================================================================
# DRIVER SEASON STATISTICS
# ============================================================================

print("Building driver season statistics...")

race_results["year"] = num(race_results["year"])
race_results["round"] = num(race_results["round"])
race_results["positionNumber"] = num(
    race_results["positionNumber"]
)
race_results["points"] = num(race_results["points"])

driver_season = (
    race_results
    .groupby(
        ["year", "driverId"],
        as_index=False
    )
    .agg(
        race_entries=("raceId", "count"),
        starts=("raceId", "count"),
        wins=(
            "positionNumber",
            lambda x: int((x == 1).sum())
        ),
        podiums=(
            "positionNumber",
            lambda x: int((x <= 3).sum())
        ),
        points=("points", "sum"),
        best_finish=("positionNumber", "min"),
    )
)

# ============================================================================
# SEASON POLES
# ============================================================================

qualifying["year"] = num(qualifying["year"])
qualifying["positionNumber"] = num(
    qualifying["positionNumber"]
)

season_poles = (
    qualifying[
        qualifying["positionNumber"] == 1
    ]
    .groupby(
        ["year", "driverId"]
    )
    .size()
    .reset_index(
        name="poles"
    )
)

driver_season = driver_season.merge(
    season_poles,
    on=["year", "driverId"],
    how="left",
)

driver_season["poles"] = (
    driver_season["poles"]
    .fillna(0)
    .astype(int)
)

# ============================================================================
# SEASON FASTEST LAPS
# ============================================================================

if "fastestLap" in race_results.columns:

    fastest = race_results[
        race_results["fastestLap"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    ]

    season_fastest = (
        fastest
        .groupby(
            ["year", "driverId"]
        )
        .size()
        .reset_index(
            name="fastest_laps"
        )
    )

    driver_season = driver_season.merge(
        season_fastest,
        on=["year", "driverId"],
        how="left",
    )

else:
    driver_season["fastest_laps"] = 0

driver_season["fastest_laps"] = (
    driver_season["fastest_laps"]
    .fillna(0)
    .astype(int)
)

# ============================================================================
# DRIVER CHAMPIONSHIP POSITIONS
# ============================================================================

print("Building driver championship history...")

driver_standings["year"] = num(
    driver_standings["year"]
)

driver_standings["round"] = num(
    driver_standings["round"]
)

driver_standings["positionNumber"] = num(
    driver_standings["positionNumber"]
)

final_driver_standings = (
    driver_standings
    .sort_values(
        ["year", "driverId", "round"]
    )
    .groupby(
        ["year", "driverId"],
        as_index=False
    )
    .tail(1)
)

final_driver_standings = final_driver_standings[
    [
        "year",
        "driverId",
        "positionNumber",
    ]
].rename(
    columns={
        "positionNumber": "championship_position"
    }
)

driver_season = driver_season.merge(
    final_driver_standings,
    on=["year", "driverId"],
    how="left",
)

# ============================================================================
# DRIVER NAMES
# ============================================================================

driver_names = drivers[
    ["id", "name", "abbreviation"]
].copy()

driver_names["id"] = driver_names["id"].astype(str)

driver_season = driver_season.merge(
    driver_names,
    left_on="driverId",
    right_on="id",
    how="left",
)

driver_season = driver_season.rename(
    columns={
        "name": "driver_name",
        "abbreviation": "driver_code",
        "driverId": "driver_id",
    }
)

driver_season = driver_season.drop(
    columns=["id"]
)

driver_season = driver_season[
    [
        "year",
        "driver_id",
        "driver_name",
        "driver_code",
        "race_entries",
        "starts",
        "wins",
        "podiums",
        "poles",
        "fastest_laps",
        "points",
        "best_finish",
        "championship_position",
    ]
]

driver_season = driver_season.sort_values(
    [
        "year",
        "championship_position",
        "driver_name",
    ],
    na_position="last",
)

# ============================================================================
# ALL-TIME CONSTRUCTOR CAREER STATISTICS
# ============================================================================

print("Building all-time constructor statistics...")

constructor_columns = {
    "id": "constructor_id",
    "name": "constructor_name",
    "totalRaceEntries": "race_entries",
    "totalRaceStarts": "starts",
    "totalRaceWins": "wins",
    "totalPodiums": "podiums",
    "totalPoints": "points",
    "totalChampionshipWins": "championships",
    "totalPolePositions": "poles",
    "totalFastestLaps": "fastest_laps",
    "bestChampionshipPosition": "best_championship_position",
    "bestStartingGridPosition": "best_starting_grid",
    "bestRaceResult": "best_finish",
}

available = {
    source: target
    for source, target in constructor_columns.items()
    if source in constructors.columns
}

constructor_career = constructors[
    list(available.keys())
].rename(columns=available)

for column in constructor_career.columns:
    if column not in [
        "constructor_id",
        "constructor_name",
    ]:
        constructor_career[column] = num(
            constructor_career[column]
        )

# ============================================================================
# CONSTRUCTOR SEASON STATISTICS
# ============================================================================

print("Building constructor season statistics...")

race_results["constructorId"] = (
    race_results["constructorId"].astype(str)
)

constructor_season = (
    race_results
    .groupby(
        ["year", "constructorId"],
        as_index=False
    )
    .agg(
        race_entries=("raceId", "count"),
        wins=(
            "positionNumber",
            lambda x: int((x == 1).sum())
        ),
        podiums=(
            "positionNumber",
            lambda x: int((x <= 3).sum())
        ),
        points=("points", "sum"),
        best_finish=("positionNumber", "min"),
    )
)

# IMPORTANT:
# Keep constructorId until after all merges.
# This prevents the KeyError from the previous version.

constructor_names = constructors[
    ["id", "name"]
].copy()

constructor_names["id"] = (
    constructor_names["id"].astype(str)
)

constructor_season = constructor_season.merge(
    constructor_names,
    left_on="constructorId",
    right_on="id",
    how="left",
)

constructor_season = constructor_season.rename(
    columns={
        "name": "constructor_name",
        "constructorId": "constructor_id",
    }
)

constructor_season = constructor_season.drop(
    columns=["id"]
)

# ============================================================================
# CONSTRUCTOR CHAMPIONSHIP POSITIONS
# ============================================================================

constructor_standings["year"] = num(
    constructor_standings["year"]
)

constructor_standings["round"] = num(
    constructor_standings["round"]
)

constructor_standings["positionNumber"] = num(
    constructor_standings["positionNumber"]
)

final_constructor_standings = (
    constructor_standings
    .sort_values(
        [
            "year",
            "constructorId",
            "round",
        ]
    )
    .groupby(
        ["year", "constructorId"],
        as_index=False
    )
    .tail(1)
)

final_constructor_standings = (
    final_constructor_standings[
        [
            "year",
            "constructorId",
            "positionNumber",
        ]
    ]
    .rename(
        columns={
            "constructorId": "constructor_id",
            "positionNumber": "championship_position",
        }
    )
)

# Correct merge: both sides now use constructor_id.

constructor_season = constructor_season.merge(
    final_constructor_standings,
    on=[
        "year",
        "constructor_id",
    ],
    how="left",
)

constructor_season = constructor_season[
    [
        "year",
        "constructor_id",
        "constructor_name",
        "race_entries",
        "wins",
        "podiums",
        "points",
        "best_finish",
        "championship_position",
    ]
]

constructor_season = constructor_season.sort_values(
    [
        "year",
        "championship_position",
        "constructor_name",
    ],
    na_position="last",
)

# ============================================================================
# CLEAN TYPES
# ============================================================================

driver_integer_columns = [
    "race_entries",
    "starts",
    "wins",
    "podiums",
    "poles",
    "fastest_laps",
    "championships",
    "race_laps",
    "sprint_starts",
    "sprint_wins",
    "driver_of_the_day",
    "grand_slams",
    "best_championship_position",
    "best_starting_grid",
    "best_finish",
    "best_sprint_finish",
]

for column in driver_integer_columns:
    if column in driver_career.columns:
        driver_career[column] = (
            num(driver_career[column])
            .fillna(0)
            .astype(int)
        )

constructor_integer_columns = [
    "race_entries",
    "starts",
    "wins",
    "podiums",
    "championships",
    "poles",
    "fastest_laps",
    "best_championship_position",
    "best_starting_grid",
    "best_finish",
]

for column in constructor_integer_columns:
    if column in constructor_career.columns:
        constructor_career[column] = (
            num(constructor_career[column])
            .fillna(0)
            .astype(int)
        )

# ============================================================================
# SORT CAREER TABLES
# ============================================================================

driver_career = driver_career.sort_values(
    [
        "wins",
        "podiums",
        "points",
    ],
    ascending=[
        False,
        False,
        False,
    ],
).reset_index(drop=True)

constructor_career = constructor_career.sort_values(
    [
        "wins",
        "podiums",
        "points",
    ],
    ascending=[
        False,
        False,
        False,
    ],
).reset_index(drop=True)

# ============================================================================
# SAVE
# ============================================================================

driver_career_path = (
    OUTPUT_DIR / "driver_career_stats.csv"
)

constructor_career_path = (
    OUTPUT_DIR / "constructor_career_stats.csv"
)

driver_season_path = (
    OUTPUT_DIR / "driver_season_stats.csv"
)

constructor_season_path = (
    OUTPUT_DIR / "constructor_season_stats.csv"
)

driver_career.to_csv(
    driver_career_path,
    index=False,
)

constructor_career.to_csv(
    constructor_career_path,
    index=False,
)

driver_season.to_csv(
    driver_season_path,
    index=False,
)

constructor_season.to_csv(
    constructor_season_path,
    index=False,
)

# ============================================================================
# OUTPUT
# ============================================================================

print()
print(f"Driver career records: {len(driver_career)}")
print(
    f"Constructor career records: "
    f"{len(constructor_career)}"
)
print(
    f"Driver season records: "
    f"{len(driver_season)}"
)
print(
    f"Constructor season records: "
    f"{len(constructor_season)}"
)

print()
print(f"Saved -> {driver_career_path}")
print(f"Saved -> {constructor_career_path}")
print(f"Saved -> {driver_season_path}")
print(f"Saved -> {constructor_season_path}")

print()
print("Top drivers by wins:")

print(
    driver_career[
        [
            "driver_name",
            "wins",
            "podiums",
            "poles",
            "championships",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print()
print("Top constructors by wins:")

print(
    constructor_career[
        [
            "constructor_name",
            "wins",
            "podiums",
            "championships",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

# ============================================================================
# SANITY CHECKS
# ============================================================================

print()
print("SANITY CHECKS")

alonso = driver_career[
    driver_career["driver_name"]
    .astype(str)
    .str.contains(
        "Fernando Alonso",
        case=False,
        na=False,
    )
]

if not alonso.empty:
    row = alonso.iloc[0]

    print(
        "Fernando Alonso: "
        f"{row['wins']} wins, "
        f"{row['podiums']} podiums, "
        f"{row['poles']} poles, "
        f"{row['championships']} championships, "
        f"{row['points']} points"
    )

    if (
        row["wins"] == 32
        and row["championships"] == 2
    ):
        print(
            "Fernando Alonso career totals: PASS"
        )
    else:
        print(
            "Fernando Alonso career totals: CHECK"
        )

hamilton = driver_career[
    driver_career["driver_name"]
    .astype(str)
    .str.contains(
        "Lewis Hamilton",
        case=False,
        na=False,
    )
]

if not hamilton.empty:
    row = hamilton.iloc[0]

    print(
        "Lewis Hamilton: "
        f"{row['wins']} wins, "
        f"{row['podiums']} podiums, "
        f"{row['poles']} poles, "
        f"{row['championships']} championships"
    )

print()
print("ALL-TIME CAREER STATISTICS COMPLETE")
print("=" * 80)