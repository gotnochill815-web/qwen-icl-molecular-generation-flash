import argparse

from common import (
    ROOT,
    load_config,
    read_jsonl,
    append_jsonl,
    make_record
)

SMOKE_SMILES = [
    "CCO", "CCN", "CCCO", "c1ccccc1",
    "CC(=O)O", "CCOC", "CC(C)O", "C1CCCCC1",
    "COc1ccccc1", "NCCO", "CC(=O)N", "CNC",
    "O=C=O", "CCS"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if len(config["experiments"]) != 1:
        raise ValueError(
            "Each config must define exactly one experiment."
        )

    jobs_path = ROOT / config["paths"]["job_manifest"]
    output_path = ROOT / config["paths"]["smoke_output"]

    jobs = read_jsonl(jobs_path)

    if not jobs:
        raise RuntimeError(f"No jobs found: {jobs_path}")

    # Two samples from the first target of each configured experiment.
    selected = [
        job for job in jobs
        if job["target_index"] == 0
        and job["replicate_index"] == 0
        and job["sample_index"] < 2
    ]

    expected = 2 * len(config["experiments"])

    if len(selected) != expected:
        raise RuntimeError(
            f"Expected {expected} smoke jobs, found {len(selected)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    for i, job in enumerate(selected):
        record = make_record(
            job=job,
            raw_output=SMOKE_SMILES[i % len(SMOKE_SMILES)],
            backend="smoke_test_stub"
        )

        append_jsonl(output_path, record)

        print(
            f"[{i+1}/{expected}] "
            f"{job['experiment']} | "
            f"valid={record['valid']} | "
            f"SMILES={record['smiles']}"
        )

    records = read_jsonl(output_path)

    experiments = {r["experiment"] for r in records}
    valid_count = sum(r["valid"] for r in records)

    assert len(records) == expected
    assert len(experiments) == len(config["experiments"])
    assert valid_count == expected

    print("\n========== SMOKE TEST ==========")
    print("Config:", args.config)
    print("Records:", len(records))
    print("Experiments:", len(experiments))
    print("Valid molecules:", valid_count)
    print("Output:", output_path)
    print("STATUS: PASSED")


if __name__ == "__main__":
    main()
