import zipfile
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "f1db-csv.zip"
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset_with_circuit.csv"
)


def main():
    print("=" * 80)
    print("ADDING CIRCUIT FEATURES")
    print("=" * 80)

    with zipfile.ZipFile(ZIP_PATH) as z:
        circuit_file = "f1db-circuits.csv"
        circuits = pd.read_csv(z.open(circuit_file))

    print(f"\nCircuit records loaded: {len(circuits)}")

    master = pd.read_csv(MASTER_PATH)

    print(f"Master dataset shape: {master.shape}")

    circuit_features = circuits[
        [
            "id",
            "type",
            "direction",
            "latitude",
            "longitude",
            "length",
            "turns",
            "totalRacesHeld",
        ]
    ].copy()

    circuit_features = circuit_features.rename(
        columns={
            "id": "f1db_circuit_id",
            "type": "circuit_type",
            "direction": "circuit_direction",
            "latitude": "circuit_latitude",
            "longitude": "circuit_longitude",
            "length": "circuit_length_km",
            "turns": "circuit_turns",
            "totalRacesHeld": "circuit_total_races_held",
        }
    )

    circuit_features["f1db_circuit_id"] = (
        circuit_features["f1db_circuit_id"]
        .astype(str)
        .str.strip()
    )

    circuit_features["circuit_type"] = (
        circuit_features["circuit_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    circuit_features["circuit_direction"] = (
        circuit_features["circuit_direction"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    numeric_columns = [
        "circuit_latitude",
        "circuit_longitude",
        "circuit_length_km",
        "circuit_turns",
        "circuit_total_races_held",
    ]

    for column in numeric_columns:
        circuit_features[column] = pd.to_numeric(
            circuit_features[column],
            errors="coerce",
        )

    if "circuit_id" not in master.columns:
        raise ValueError(
            "master_dataset.csv does not contain circuit_id"
        )

    master["circuit_id"] = (
        master["circuit_id"]
        .astype(str)
        .str.strip()
    )

    circuit_id_mapping = {
        "albert_park": "melbourne",
        "americas": "austin",
        "losail": "lusail",
        "marina_bay": "marina-bay",
        "red_bull_ring": "spielberg",
        "ricard": "paul-ricard",
        "rodriguez": "mexico-city",
        "spa": "spa-francorchamps",
        "vegas": "las-vegas",
        "villeneuve": "montreal",
        "yas_marina": "yas-marina",
    }

    master["f1db_circuit_id"] = master["circuit_id"].replace(
        circuit_id_mapping
    )

    duplicate_circuits = (
        circuit_features["f1db_circuit_id"].duplicated().sum()
    )

    if duplicate_circuits > 0:
        raise ValueError(
            f"Found {duplicate_circuits} duplicate circuit IDs in F1DB"
        )

    merged = master.merge(
        circuit_features,
        on="f1db_circuit_id",
        how="left",
        validate="many_to_one",
    )

    merged = merged.drop(columns=["f1db_circuit_id"])

    new_columns = [
        "circuit_type",
        "circuit_direction",
        "circuit_latitude",
        "circuit_longitude",
        "circuit_length_km",
        "circuit_turns",
        "circuit_total_races_held",
    ]

    print("\nCircuit feature availability:")

    for column in new_columns:
        available = merged[column].notna().sum()
        total = len(merged)
        percentage = available / total * 100

        print(
            f"{column:30s} "
            f"{available:5d} / {total:5d} "
            f"({percentage:6.2f}%)"
        )

    print("\nCircuit types:")
    print(
        merged["circuit_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nCircuit directions:")
    print(
        merged["circuit_direction"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nNumeric circuit feature summary:")

    print(
        merged[
            [
                "circuit_length_km",
                "circuit_turns",
                "circuit_total_races_held",
            ]
        ].describe().to_string()
    )

    missing_circuit = merged["circuit_type"].isna().sum()

    if missing_circuit > 0:
        print(
            f"\nWARNING: {missing_circuit} rows could not be "
            "matched to circuit data."
        )

        unmatched = (
            merged.loc[
                merged["circuit_type"].isna(),
                ["circuit_id", "circuit_name"],
            ]
            .drop_duplicates()
            .sort_values("circuit_id")
        )

        print("\nUnmatched circuits:")
        print(unmatched.to_string(index=False))

    else:
        print("\nAll master rows matched to circuit data.")

    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Final shape: {merged.shape}")

    print("\n" + "=" * 80)
    print("CIRCUIT FEATURES ADDED")
    print("=" * 80)


if __name__ == "__main__":
    main()