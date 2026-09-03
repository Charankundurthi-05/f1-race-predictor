PRE_PRACTICE_FEATURES = [
    "season",
    "round",
    "number",
    "driver_previous_finish",
    "driver_avg_finish_last_3",
    "driver_avg_finish_last_5",
    "driver_previous_points",
    "driver_avg_points_last_5",
    "team_avg_finish_last_5",
    "team_avg_points_last_5",
    "driver_circuit_avg_finish",
    "driver_circuit_races",
]

FP1_FEATURES = PRE_PRACTICE_FEATURES + [
    "fp1_position",
    "practice_sessions_available",
]

FP2_FEATURES = PRE_PRACTICE_FEATURES + [
    "fp1_position",
    "fp2_position",
    "practice_avg_position",
    "practice_best_position",
    "practice_sessions_available",
]

FP3_FEATURES = PRE_PRACTICE_FEATURES + [
    "fp1_position",
    "fp2_position",
    "fp3_position",
    "practice_avg_position",
    "practice_best_position",
    "practice_sessions_available",
]

QUALIFYING_FEATURES = PRE_PRACTICE_FEATURES + [
    "fp1_position",
    "fp2_position",
    "fp3_position",
    "practice_avg_position",
    "practice_best_position",
    "practice_sessions_available",
    "qualifying_position",
    "q1",
    "q2",
    "q3",
    "grid_position",
]

STAGE_FEATURES = {
    "pre_practice": PRE_PRACTICE_FEATURES,
    "fp1": FP1_FEATURES,
    "fp2": FP2_FEATURES,
    "fp3": FP3_FEATURES,
    "qualifying": QUALIFYING_FEATURES,
}

TARGET = "finish_position"

ALWAYS_FORBIDDEN_FEATURES = [
    "finish_position",
    "points",
    "status",
    "laps",
    "fastest_lap_rank",
    "fastest_lap_time",
    "fastest_lap_speed",
    "fastest_lap_speed_unit",
]

STAGE_FORBIDDEN_FEATURES = {
    "pre_practice": [
        "fp1_position",
        "fp2_position",
        "fp3_position",
        "practice_avg_position",
        "practice_best_position",
        "qualifying_position",
        "q1",
        "q2",
        "q3",
        "grid_position",
    ],
    "fp1": [
        "fp2_position",
        "fp3_position",
        "practice_avg_position",
        "practice_best_position",
        "qualifying_position",
        "q1",
        "q2",
        "q3",
        "grid_position",
    ],
    "fp2": [
        "fp3_position",
        "qualifying_position",
        "q1",
        "q2",
        "q3",
        "grid_position",
    ],
    "fp3": [
        "qualifying_position",
        "q1",
        "q2",
        "q3",
        "grid_position",
    ],
    "qualifying": [],
}


def get_features(stage):
    if stage not in STAGE_FEATURES:
        raise ValueError(f"Unknown stage: {stage}")

    return STAGE_FEATURES[stage].copy()


def get_forbidden_features(stage):
    if stage not in STAGE_FORBIDDEN_FEATURES:
        raise ValueError(f"Unknown stage: {stage}")

    return ALWAYS_FORBIDDEN_FEATURES + STAGE_FORBIDDEN_FEATURES[stage]


def validate_features(stage, dataframe):
    features = get_features(stage)
    forbidden = get_forbidden_features(stage)

    missing = [
        column
        for column in features
        if column not in dataframe.columns
    ]

    leakage = [
        column
        for column in features
        if column in forbidden
    ]

    if missing:
        raise ValueError(
            f"{stage}: missing feature columns: {missing}"
        )

    if leakage:
        raise ValueError(
            f"{stage}: forbidden leakage columns used: {leakage}"
        )

    return True