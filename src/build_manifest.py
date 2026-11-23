import argparse
import json
from pathlib import Path

import pandas as pd

from common import ROOT, load_config


def select_targets(
    df, properties, count, seed, quantiles, fixed_targets=None
):
    if fixed_targets is not None:
        if len(fixed_targets) != count:
            raise ValueError(
                f"Expected {count} fixed targets, "
                f"received {len(fixed_targets)}"
            )

        if len(properties) == 1:
            prop = properties[0]
            return [
                {prop: float(value)}
                for value in fixed_targets
            ]

        selected = []
        for target in fixed_targets:
            if not isinstance(target, dict):
                raise ValueError(
                    "Multi-property fixed targets must be dictionaries."
                )

            if set(target) != set(properties):
                raise ValueError(
                    f"Expected target properties {properties}, "
                    f"received {list(target)}"
                )

            selected.append({
                prop: float(target[prop])
                for prop in properties
            })

        return selected

    if len(properties) == 1:
        prop = properties[0]
        values = df[prop].dropna().sort_values().reset_index(drop=True)

        selected = []
        for q in quantiles:
            idx = round(q * (len(values) - 1))
            selected.append({prop: float(values.iloc[idx])})

        return selected

    available = df[properties].dropna().drop_duplicates()

    if len(available) < count:
        raise ValueError(
            f"Only {len(available)} unique combinations for {properties}"
        )

    sampled = available.sample(n=count, random_state=seed)

    return [
        {p: float(row[p]) for p in properties}
        for _, row in sampled.iterrows()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    experiments = config["experiments"]
    if len(experiments) != 1:
        raise ValueError(
            "Each config must define exactly one experiment."
        )

    target_count = config["targets"]["targets_per_experiment"]

    validation_path = ROOT / config["dataset"]["validation"]
    print("Config:", args.config)
    print("Loading:", validation_path)

    all_properties = config["dataset"]["properties"]
    df = pd.read_csv(
        validation_path,
        usecols=["smiles"] + all_properties
    )

    target_manifest = {
        "project": config["project"],
        "seed": config["seed"],
        "source_split": (
            "fixed_reference_targets"
            if all(
                config.get("targets", {}).get("fixed_targets", {}).get(exp)
                is not None
                for exp in experiments
            )
            else "validation"
        ),
        "config": args.config,
        "experiments": {}
    }

    jobs = []
    generation = config["generation"]

    job_index = 0

    for experiment_index, (experiment, properties) in enumerate(
        experiments.items()
    ):
        seed = (
            config["seed"]
            + config.get("target_seed_offset", experiment_index)
        )

        fixed_targets = (
            config.get("targets", {})
            .get("fixed_targets", {})
            .get(experiment)
        )

        targets = select_targets(
            df,
            properties,
            target_count,
            seed,
            config["targets"]["single_property_quantiles"],
            fixed_targets=fixed_targets
        )

        target_manifest["experiments"][experiment] = []

        for target_index, target_values in enumerate(targets):
            target_id = f"{experiment}_target_{target_index + 1}"

            target_manifest["experiments"][experiment].append({
                "target_id": target_id,
                "target_index": target_index,
                "properties": properties,
                "targets": target_values
            })

            per_target_counts = generation.get(
                "generations_per_replicate_by_target"
            )

            if per_target_counts is not None:
                if len(per_target_counts) != len(targets):
                    raise ValueError(
                        "generations_per_replicate_by_target must have "
                        "one entry per target."
                    )
                samples_this_target = int(
                    per_target_counts[target_index]
                )
            else:
                samples_this_target = int(
                    generation["generations_per_replicate"]
                )

            for replicate in range(
                generation["replicates_per_target"]
            ):
                for sample in range(samples_this_target):
                    job_seed = (
                        config["seed"]
                        + experiment_index * 1_000_000
                        + target_index * 100_000
                        + replicate * 1_000
                        + sample
                    )

                    jobs.append({
                        "job_index": job_index,
                        "job_id": (
                            f"{experiment}"
                            f"__t{target_index:02d}"
                            f"__r{replicate:02d}"
                            f"__s{sample:03d}"
                        ),
                        "experiment": experiment,
                        "properties": properties,
                        "target_id": target_id,
                        "target_index": target_index,
                        "targets": target_values,
                        "replicate_index": replicate,
                        "sample_index": sample,
                        "seed": job_seed
                    })

                    job_index += 1

    target_path = ROOT / config["paths"]["target_manifest"]
    jobs_path = ROOT / config["paths"]["job_manifest"]

    target_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "w") as f:
        json.dump(target_manifest, f, indent=2)

    with open(jobs_path, "w") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")

    print("\nManifest created.")
    print("Experiments:", len(experiments))
    print("Targets:", sum(len(x) for x in target_manifest["experiments"].values()))
    print("Total jobs:", len(jobs))
    print("Target manifest:", target_path)
    print("Job manifest:", jobs_path)


if __name__ == "__main__":
    main()
