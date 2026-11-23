import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROPERTIES = ["qed", "logp", "sas", "tpsa"]
CHUNK_SIZE = 100_000


def process_split(input_path, output_path, split_name):
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    required = (
        ["smiles"]
        + PROPERTIES
        + [f"mean_{p}" for p in PROPERTIES]
        + [f"std_{p}" for p in PROPERTIES]
    )

    print(f"\nProcessing {split_name}: {input_path}")
    print(f"Output: {output_path}")

    metadata_means = None
    metadata_stds = None
    rows_read = 0
    rows_written = 0
    first_chunk = True

    minimums = {p: float("inf") for p in PROPERTIES}
    maximums = {p: float("-inf") for p in PROPERTIES}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove an older output so this run starts cleanly.
    if output_path.exists():
        output_path.unlink()

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_path,
            usecols=required,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        rows_read += len(chunk)

        # Read the dataset's normalization metadata.
        if metadata_means is None:
            metadata_means = {
                p: float(chunk[f"mean_{p}"].dropna().iloc[0])
                for p in PROPERTIES
            }
            metadata_stds = {
                p: float(chunk[f"std_{p}"].dropna().iloc[0])
                for p in PROPERTIES
            }

            for p in PROPERTIES:
                if not np.isfinite(metadata_means[p]):
                    raise ValueError(f"Invalid mean for {p}")
                if not np.isfinite(metadata_stds[p]) or metadata_stds[p] <= 0:
                    raise ValueError(f"Invalid standard deviation for {p}")

        # Confirm normalization metadata remains consistent.
        for p in PROPERTIES:
            if not np.allclose(
                chunk[f"mean_{p}"].dropna().to_numpy(),
                metadata_means[p],
                atol=1e-5,
            ):
                raise ValueError(f"Inconsistent mean metadata for {p}")

            if not np.allclose(
                chunk[f"std_{p}"].dropna().to_numpy(),
                metadata_stds[p],
                atol=1e-5,
            ):
                raise ValueError(f"Inconsistent std metadata for {p}")

        result = pd.DataFrame()
        result["smiles"] = chunk["smiles"].fillna("").astype(str).str.strip()

        keep = result["smiles"].ne("") & result["smiles"].ne("nan")

        # Convert z-scores back to original descriptor values:
        # original_value = normalized_value * std + mean
        for p in PROPERTIES:
            normalized = pd.to_numeric(chunk[p], errors="coerce")
            original = normalized * metadata_stds[p] + metadata_means[p]

            result[p] = original
            keep &= np.isfinite(original)

        result = result.loc[keep].reset_index(drop=True)

        if len(result):
            for p in PROPERTIES:
                minimums[p] = min(minimums[p], float(result[p].min()))
                maximums[p] = max(maximums[p], float(result[p].max()))

            result.to_csv(
                output_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
                float_format="%.8f",
            )
            first_chunk = False
            rows_written += len(result)

        print(
            f"  Chunk {chunk_number}: "
            f"read={rows_read:,}, retained={rows_written:,}"
        )

    if rows_written == 0:
        raise ValueError(f"No usable rows found in {input_path}")

    summary = {
        "split": split_name,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "rows_read": rows_read,
        "rows_written": rows_written,
        "rows_dropped": rows_read - rows_written,
        "normalization_means": metadata_means,
        "normalization_stds": metadata_stds,
        "original_property_ranges": {
            p: {"min": minimums[p], "max": maximums[p]}
            for p in PROPERTIES
        },
    }

    print(f"\n{split_name} summary:")
    print(json.dumps(summary, indent=2))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "CFG_test/cfg/data/guacamol_full",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs/processed/guacamol_full"

    train_summary = process_split(
        args.data_dir / "train.csv",
        output_dir / "train.csv",
        "train",
    )

    val_summary = process_split(
        args.data_dir / "val.csv",
        output_dir / "val.csv",
        "validation",
    )

    metadata = {
        "dataset": "Full GuacaMol",
        "properties": PROPERTIES,
        "property_units": {
            "qed": "unitless",
            "logp": "MolLogP",
            "sas": "SA score",
            "tpsa": "Å²",
        },
        "train": train_summary,
        "validation": val_summary,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n======================================")
    print("DATASET PREPROCESSING COMPLETE")
    print("======================================")
    print("Train:", output_dir / "train.csv")
    print("Validation:", output_dir / "val.csv")
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
