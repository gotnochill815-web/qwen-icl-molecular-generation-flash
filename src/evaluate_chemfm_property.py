import argparse
import csv
import json
import statistics
from pathlib import Path

import sys
from rdkit import Chem, RDLogger
from rdkit.Chem import RDConfig
from rdkit.Chem import QED, Crippen, rdMolDescriptors

sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
import sascorer

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]


def canonicalize(smiles):
    try:
        if not smiles:
            return ""
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def calculate_property(mol, prop):
    if prop == "qed":
        return float(QED.qed(mol))
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

        if not reader.fieldnames or "smiles" not in reader.fieldnames:
            raise ValueError("Training CSV must contain a smiles column.")

        for i, row in enumerate(reader, start=1):
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
        raise ValueError("Duplicate job IDs found in generation output.")

    return records


def prepare_molecules(records, train_set, prop):
    rows = []

    for record in records:
        smiles = canonicalize(record.get("smiles", ""))
        valid = bool(smiles)

        target = record.get("targets", {}).get(prop)
        measured = None
        error = None

        if valid:
            mol = Chem.MolFromSmiles(smiles)
            measured = calculate_property(mol, prop)

            if target is not None:
                error = abs(measured - float(target))

        rows.append({
            "job_id": record.get("job_id"),
            "target_id": record.get("target_id"),
            "target_index": record.get("target_index"),
            f"target_{prop}": (
                float(target) if target is not None else None
            ),
            "raw_smiles": record.get("smiles", ""),
            "canonical_smiles": smiles,
            "valid": valid,
            "in_train_data": (
                smiles in train_set if valid else None
            ),
            f"measured_{prop}": measured,
            f"{prop}_absolute_error": error,
        })

    return rows


def compute_metrics(rows, train_set, prop):
    total = len(rows)
    valid_rows = [r for r in rows if r["valid"]]

    unique_smiles = {
        r["canonical_smiles"] for r in valid_rows
    }

    novel_smiles = {
        smiles for smiles in unique_smiles
        if smiles not in train_set
    }

    errors = [
        r[f"{prop}_absolute_error"]
        for r in valid_rows
        if r[f"{prop}_absolute_error"] is not None
    ]

    return {
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
        f"{prop}_diff_count": len(errors),
        f"{prop}_diff_mean": (
            statistics.mean(errors) if errors else None
        ),
        f"{prop}_diff_std": (
            statistics.stdev(errors) if len(errors) > 1 else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    with open(config_path) as f:
        config = json.load(f)

    experiments = config["experiments"]
    if len(experiments) != 1:
        raise ValueError("Expected exactly one experiment.")

    experiment, properties = next(iter(experiments.items()))
    if len(properties) != 1:
        raise ValueError("Expected exactly one target property.")

    prop = properties[0]
    if prop not in {"qed", "logp", "tpsa", "sas"}:
        raise ValueError(f"Unsupported property: {prop}")

    input_path = Path(
        args.input or config["paths"]["generation_output"]
    )
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    train_path = ROOT / config["dataset"]["train"]
    cache_path = (
        ROOT / "outputs/processed/guacamol_full/"
        / "train_canonical_chemfm.txt"
    )

    tag = args.tag or f"{experiment}_zero_shot_chemfm"
    output_dir = ROOT / "outputs/evaluation" / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
    print("Input records:", len(records))

    train_set = load_training_reference(train_path, cache_path)
    rows = prepare_molecules(records, train_set, prop)

    overall = compute_metrics(rows, train_set, prop)

    target_groups = {}
    for row in rows:
        target_groups.setdefault(row["target_id"], []).append(row)

    per_target = {}
    for target_id, target_rows in target_groups.items():
        per_target[target_id] = {
            f"target_{prop}": target_rows[0][f"target_{prop}"],
            **compute_metrics(target_rows, train_set, prop)
        }

    summary = {
        "experiment": experiment,
        "property": prop,
        "evaluation_method": "ChemFM conditional_generation score.py",
        "novelty_definition": (
            "Novel unique valid canonical SMILES / "
            "unique valid canonical SMILES"
        ),
        "overall": overall,
        "per_target": per_target,
    }

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

    print(f"\n========== OVERALL {prop.upper()} METRICS ==========")
    print(json.dumps(overall, indent=2))

    print("\n========== PER-TARGET METRICS ==========")
    for target_id, metrics in per_target.items():
        print(f"\n{target_id}")
        print(json.dumps(metrics, indent=2))

    print("\nEvaluation complete.")
    print("Results:", output_dir)


if __name__ == "__main__":
    main()
