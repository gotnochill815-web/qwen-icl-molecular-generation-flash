import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, RDConfig, Crippen

sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
import sascorer

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]


def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def calculate_property(mol, prop):
    if prop == "logp":
        return float(Crippen.MolLogP(mol))
    if prop == "tpsa":
        return float(rdMolDescriptors.CalcTPSA(mol))
    if prop == "sas":
        return float(sascorer.calculateScore(mol))
    raise ValueError(f"Unsupported property: {prop}")


def load_training_reference(train_path, cache_path):
    if cache_path.exists():
        print("Loading canonical training reference:", cache_path)
        with open(cache_path) as f:
            return {line.strip() for line in f if line.strip()}

    print("Building canonical training reference...")
    train_set = set()

    with open(train_path, newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, 1):
            smiles = canonicalize(row["smiles"])
            if smiles:
                train_set.add(smiles)

            if i % 50000 == 0:
                print(
                    f"Processed {i:,} training molecules | "
                    f"{len(train_set):,} unique valid molecules"
                )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for smiles in sorted(train_set):
            f.write(smiles + "\n")

    print("Training reference size:", len(train_set))
    return train_set


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise RuntimeError(f"No records found: {path}")

    job_ids = [r.get("job_id") for r in records]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Duplicate job IDs found.")

    return records


def prepare_molecules(records, train_set, properties):
    rows = []

    for record in records:
        smiles = canonicalize(record.get("smiles", ""))
        valid = bool(smiles)

        row = {
            "job_id": record.get("job_id"),
            "target_id": record.get("target_id"),
            "raw_smiles": record.get("smiles", ""),
            "canonical_smiles": smiles,
            "valid": valid,
            "in_train_data": (
                smiles in train_set if valid else None
            ),
        }

        for prop in properties:
            target = record.get("targets", {}).get(prop)
            row[f"target_{prop}"] = (
                float(target) if target is not None else None
            )

        mol = Chem.MolFromSmiles(smiles) if valid else None

        for prop in properties:
            target = record.get("targets", {}).get(prop)
            measured = calculate_property(mol, prop) if mol else None

            error = (
                abs(measured - float(target))
                if measured is not None and target is not None
                else None
            )

            row[f"measured_{prop}"] = measured
            row[f"{prop}_absolute_error"] = error

        rows.append(row)

    return rows


def compute_metrics(rows, train_set, properties):
    total = len(rows)
    valid_rows = [r for r in rows if r["valid"]]

    unique_smiles = {
        r["canonical_smiles"] for r in valid_rows
    }

    novel_smiles = {
        s for s in unique_smiles if s not in train_set
    }

    result = {
        "total_generations": total,
        "valid_count": len(valid_rows),
        "invalid_count": total - len(valid_rows),
        "validity": len(valid_rows) / total if total else None,
        "unique_valid_count": len(unique_smiles),
        "uniqueness": (
            len(unique_smiles) / len(valid_rows)
            if valid_rows else None
        ),
        "novel_unique_count": len(novel_smiles),
        "novelty": (
            len(novel_smiles) / len(unique_smiles)
            if unique_smiles else None
        ),
    }

    for prop in properties:
        errors = [
            r[f"{prop}_absolute_error"]
            for r in valid_rows
            if r[f"{prop}_absolute_error"] is not None
        ]

        result[f"{prop}_diff_count"] = len(errors)
        result[f"{prop}_diff_mean"] = (
            statistics.mean(errors) if errors else None
        )
        result[f"{prop}_diff_std"] = (
            statistics.stdev(errors) if len(errors) > 1 else None
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/tpsa_sas_zero_shot.json"
    )
    parser.add_argument(
        "--input",
        default="outputs/raw/tpsa_sas_zero_shot.jsonl"
    )
    parser.add_argument(
        "--tag",
        default="tpsa_sas_zero_shot_chemfm"
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    input_path = ROOT / args.input

    with open(config_path) as f:
        config = json.load(f)

    experiments = config["experiments"]
    if len(experiments) != 1:
        raise ValueError("Expected one experiment.")

    experiment, properties = next(iter(experiments.items()))

    if (
        len(properties) not in (2, 3)
        or not set(properties).issubset({"logp", "tpsa", "sas"})
    ):
        raise ValueError(
            f"Expected a supported two- or three-property experiment, "
            f"received {properties}"
        )

    print("Experiment:", experiment)
    print("Properties:", properties)

    records = load_jsonl(input_path)
    print("Input records:", len(records))

    train_set = load_training_reference(
        ROOT / config["dataset"]["train"],
        ROOT / "outputs/processed/guacamol_full/"
        / "train_canonical_chemfm.txt"
    )

    rows = prepare_molecules(records, train_set, properties)
    overall = compute_metrics(rows, train_set, properties)

    target_groups = {}
    for row in rows:
        target_groups.setdefault(row["target_id"], []).append(row)

    per_target = {}
    for target_id, target_rows in target_groups.items():
        per_target[target_id] = {
            **{
                f"target_{prop}": target_rows[0][f"target_{prop}"]
                for prop in properties
            },
            **compute_metrics(target_rows, train_set, properties)
        }

    summary = {
        "experiment": experiment,
        "evaluation_method": "ChemFM conditional_generation score.py",
        "novelty_definition": (
            "Novel unique valid canonical SMILES / "
            "unique valid canonical SMILES"
        ),
        "overall": overall,
        "per_target": per_target,
    }

    output_dir = ROOT / "outputs/evaluation" / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "molecules.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    summary_rows = [
        {"scope": "overall", "target_id": "all", **overall}
    ]

    for target_id, metrics in per_target.items():
        summary_rows.append({
            "scope": "target",
            "target_id": target_id,
            **metrics
        })

    fields = list(dict.fromkeys(
        key for row in summary_rows for key in row
    ))

    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n========== OVERALL TPSA + SAS METRICS ==========")
    print(json.dumps(overall, indent=2))

    print("\n========== PER-TARGET METRICS ==========")
    for target_id, metrics in per_target.items():
        print(f"\n{target_id}")
        print(json.dumps(metrics, indent=2))

    print("\nEvaluation complete.")
    print("Results:", output_dir)


if __name__ == "__main__":
    main()
