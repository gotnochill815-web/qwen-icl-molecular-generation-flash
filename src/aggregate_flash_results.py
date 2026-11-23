import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path):
    with open(path) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def write_csv(path, rows):
    if not rows:
        print(f"No rows to write: {path}")
        return

    fields = list(dict.fromkeys(
        key for row in rows for key in row
    ))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Flash zero-shot molecular generation results."
    )
    parser.add_argument(
        "--config",
        default="configs/flash_zero_shot.json",
    )
    parser.add_argument(
        "--input-root",
        default="outputs/evaluation/flash_zero_shot",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation/flash_zero_shot/all_experiments",
    )
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    input_root = resolve_path(args.input_root)
    output_dir = resolve_path(args.output_dir)

    with open(config_path) as f:
        config = json.load(f)

    experiments = config["experiments"]

    manifest_path = resolve_path(config["paths"]["job_manifest"])
    jobs = read_jsonl(manifest_path)

    expected_counts = {
        experiment: 0 for experiment in experiments
    }

    for job in jobs:
        experiment = job.get("experiment")

        if experiment not in expected_counts:
            raise ValueError(
                f"Unexpected experiment in manifest: {experiment}"
            )

        expected_counts[experiment] += 1

    if any(count == 0 for count in expected_counts.values()):
        raise ValueError("At least one experiment has no manifest jobs.")

    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    target_rows = []
    results = {}
    missing = []
    partial = []

    for experiment, properties in experiments.items():
        summary_path = (
            input_root / experiment / "summary.json"
        )

        expected = expected_counts[experiment]

        if not summary_path.exists():
            print(f"MISSING: {experiment}")
            missing.append(experiment)
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        if summary.get("experiment") != experiment:
            raise ValueError(
                f"Experiment mismatch in {summary_path}"
            )

        if summary.get("properties") != properties:
            raise ValueError(
                f"Property mismatch in {summary_path}: "
                f"expected {properties}, "
                f"received {summary.get('properties')}"
            )

        overall = summary["overall"]
        per_target = summary.get("per_target", {})

        evaluated = overall.get("total_generations")

        if evaluated is None:
            raise ValueError(
                f"Missing total_generations in {summary_path}"
            )

        evaluated = int(evaluated)

        if evaluated > expected:
            raise ValueError(
                f"{experiment} has {evaluated} generations, "
                f"but the manifest contains only {expected} jobs."
            )

        status = "complete" if evaluated == expected else "partial"

        if status == "partial":
            partial.append(
                f"{experiment}: {evaluated}/{expected}"
            )

        row = {
            "experiment": experiment,
            "properties": "+".join(properties),
            "expected_jobs": expected,
            "evaluated_generations": evaluated,
            "status": status,
            **overall,
        }
        overall_rows.append(row)

        for target_id, metrics in per_target.items():
            target_rows.append({
                "experiment": experiment,
                "target_id": target_id,
                **{
                    f"target_{prop}": metrics.get(f"target_{prop}")
                    for prop in properties
                },
                **metrics,
            })

        results[experiment] = {
            "properties": properties,
            "expected_jobs": expected,
            "evaluated_generations": evaluated,
            "status": status,
            "overall": overall,
            "per_target": per_target,
        }

        print(
            f"Loaded: {experiment} | "
            f"{evaluated}/{expected} | {status}"
        )

    write_csv(
        output_dir / "all_results.csv",
        overall_rows,
    )
    write_csv(
        output_dir / "per_target_results.csv",
        target_rows,
    )

    with open(output_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "missing_results.txt", "w") as f:
        if not missing and not partial:
            f.write("All experiments completed.\n")
        else:
            if missing:
                f.write("Missing experiments:\n")
                f.write("\n".join(missing) + "\n")
            if partial:
                f.write("\nPartial experiments:\n")
                f.write("\n".join(partial) + "\n")

    complete_count = sum(
        row["status"] == "complete"
        for row in overall_rows
    )

    print("\n========== FLASH AGGREGATION SUMMARY ==========")
    print(f"Configured experiments: {len(experiments)}")
    print(f"Results found: {len(overall_rows)}")
    print(f"Complete experiments: {complete_count}")
    print(f"Missing experiments: {len(missing)}")
    print(f"Partial experiments: {len(partial)}")
    print(f"Per-target rows: {len(target_rows)}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
