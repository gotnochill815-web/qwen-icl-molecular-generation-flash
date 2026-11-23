import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import evaluate_chemfm_property as single_evaluator
import evaluate_chemfm_multi as multi_evaluator


def root_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path):
    records = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

    return records


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")

    fields = list(dict.fromkeys(
        key for row in rows for key in row
    ))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_experiment(
    experiment,
    properties,
    records,
    train_set,
    output_root,
):
    selected = [
        record for record in records
        if record.get("experiment") == experiment
    ]

    if not selected:
        print(f"SKIPPED: {experiment} (no generated records)")
        return False

    if len(properties) == 1:
        evaluator = single_evaluator
        evaluator_properties = properties[0]
    else:
        if not set(properties).issubset({"logp", "tpsa", "sas"}):
            raise ValueError(
                f"Unsupported multi-property experiment: {experiment}"
            )
        evaluator = multi_evaluator
        evaluator_properties = properties

    rows = evaluator.prepare_molecules(
        selected,
        train_set,
        evaluator_properties,
    )

    overall = evaluator.compute_metrics(
        rows,
        train_set,
        evaluator_properties,
    )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["target_id"]].append(row)

    per_target = {}

    for target_id, target_rows in grouped.items():
        target_values = {
            f"target_{prop}": target_rows[0].get(f"target_{prop}")
            for prop in properties
        }

        per_target[target_id] = {
            **target_values,
            **evaluator.compute_metrics(
                target_rows,
                train_set,
                evaluator_properties,
            ),
        }

    summary = {
        "experiment": experiment,
        "properties": properties,
        "evaluation_method": (
            "ChemFM-compatible RDKit descriptor evaluation"
        ),
        "novelty_definition": (
            "Novel unique valid canonical SMILES / "
            "unique valid canonical SMILES"
        ),
        "overall": overall,
        "per_target": per_target,
    }

    output_dir = output_root / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "molecules.csv", rows)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    summary_rows = [
        {"scope": "overall", "target_id": "all", **overall}
    ]

    for target_id, metrics in per_target.items():
        summary_rows.append({
            "scope": "target",
            "target_id": target_id,
            **metrics,
        })

    write_csv(output_dir / "summary.csv", summary_rows)

    print(
        f"Evaluated {experiment}: "
        f"{len(selected)} records, "
        f"{overall['valid_count']} valid"
    )
    print(f"  Results: {output_dir}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot Flash molecular generation."
    )
    parser.add_argument(
        "--config",
        default="configs/flash_zero_shot.json",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Generation JSONL. Defaults to the config path.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Evaluate one experiment instead of all experiments.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/evaluation/flash_zero_shot",
        help="Separate output directory for Flash evaluation.",
    )
    args = parser.parse_args()

    config_path = root_path(args.config)

    with open(config_path) as f:
        config = json.load(f)

    input_path = root_path(
        args.input or config["paths"]["generation_output"]
    )
    output_root = root_path(args.output_root)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Generation output does not exist: {input_path}"
        )

    experiments = config["experiments"]

    if args.experiment:
        if args.experiment not in experiments:
            raise ValueError(
                f"Unknown experiment '{args.experiment}'. "
                f"Available: {list(experiments)}"
            )
        experiment_names = [args.experiment]
    else:
        experiment_names = list(experiments)

    manifest_path = root_path(config["paths"]["job_manifest"])
    manifest_jobs = read_jsonl(manifest_path)

    expected_jobs = {
        job["job_id"]: job for job in manifest_jobs
    }

    records = read_jsonl(input_path)

    if not records:
        raise RuntimeError(f"No generation records found: {input_path}")

    job_ids = [record.get("job_id") for record in records]

    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Duplicate job IDs in generation output.")

    for record in records:
        job_id = record.get("job_id")
        experiment = record.get("experiment")

        if job_id not in expected_jobs:
            raise ValueError(
                f"Job ID {job_id!r} is not in the Flash manifest."
            )

        expected_experiment = expected_jobs[job_id]["experiment"]

        if experiment != expected_experiment:
            raise ValueError(
                f"Experiment mismatch for {job_id}: "
                f"record={experiment}, manifest={expected_experiment}"
            )

    print("Config:", config_path)
    print("Generation records:", len(records))
    print("Input:", input_path)
    print("Output root:", output_root)

    train_path = root_path(config["dataset"]["train"])
    cache_path = (
        ROOT
        / "outputs/processed/guacamol_full/"
        / "train_canonical_chemfm.txt"
    )

    train_set = single_evaluator.load_training_reference(
        train_path,
        cache_path,
    )

    completed = []

    for experiment in experiment_names:
        properties = experiments[experiment]

        print(f"\n===== {experiment.upper()} =====")

        expected_count = sum(
            job["experiment"] == experiment
            for job in manifest_jobs
        )

        actual_count = sum(
            record.get("experiment") == experiment
            for record in records
        )

        print(
            f"Generated records: {actual_count}/{expected_count}"
        )

        success = evaluate_experiment(
            experiment=experiment,
            properties=properties,
            records=records,
            train_set=train_set,
            output_root=output_root,
        )

        if success:
            completed.append(experiment)

    print("\n========== FLASH EVALUATION SUMMARY ==========")
    print(f"Completed: {len(completed)}/{len(experiment_names)}")
    print("Completed experiments:", completed)
    print("Results root:", output_root)


if __name__ == "__main__":
    main()
