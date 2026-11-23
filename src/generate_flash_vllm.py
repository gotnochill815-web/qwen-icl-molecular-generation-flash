import argparse
import math
import time
from pathlib import Path


from common import (
    ROOT,
    load_config,
    read_jsonl,
    append_jsonl,
    build_prompt,
    make_record,
)
from generate_flash import format_chat_prompt


def preflight(engine_config, torch):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Request a GPU allocation before running."
        )

    gpu_count = torch.cuda.device_count()
    tensor_parallel_size = int(
        engine_config.get("tensor_parallel_size", 1)
    )

    if gpu_count < tensor_parallel_size:
        raise RuntimeError(
            f"Requested tensor_parallel_size={tensor_parallel_size}, "
            f"but only {gpu_count} GPUs are visible."
        )

    gpu_memory = []
    for index in range(gpu_count):
        props = torch.cuda.get_device_properties(index)
        memory_gb = props.total_memory / (1024 ** 3)
        gpu_memory.append(memory_gb)
        print(
            f"GPU {index}: {props.name} | "
            f"{memory_gb:.1f} GiB"
        )

    visible_memory = sum(gpu_memory)
    required_memory = float(
        engine_config.get("required_total_gpu_memory_gb", 0)
    )

    print("Total visible GPU memory:", round(visible_memory, 1), "GiB")
    print("Required GPU memory:", required_memory, "GiB")

    # Fail before model initialization rather than attempting a known
    # impossible full-precision load.
    if required_memory and visible_memory < required_memory:
        raise RuntimeError(
            "\nInsufficient GPU memory for the configured checkpoint. "
            f"Visible: {visible_memory:.1f} GiB; "
            f"required: {required_memory:.1f} GiB. "
            "No model loading was attempted."
        )


def load_engine(config, model_id):
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "The vLLM backend requires PyTorch and vLLM. "
            "Install the optional vLLM dependencies before "
            "running generation."
        ) from exc

    model_config = config["model"]
    engine_config = config["vllm"]

    print("\nPyTorch:", torch.__version__)
    print("vLLM:", vllm.__version__)
    print("Model:", model_id)

    preflight(engine_config, torch)

    engine_kwargs = {
        "model": model_id,
        "trust_remote_code": model_config.get(
            "trust_remote_code", True
        ),
        "dtype": model_config.get("dtype", "bfloat16"),
        "tensor_parallel_size": int(
            engine_config.get("tensor_parallel_size", 1)
        ),
        "max_model_len": int(
            engine_config.get("max_model_len", 4096)
        ),
        "max_num_seqs": int(
            engine_config.get("max_num_seqs", 4)
        ),
        "max_num_batched_tokens": int(
            engine_config.get("max_num_batched_tokens", 4096)
        ),
        "gpu_memory_utilization": float(
            engine_config.get("gpu_memory_utilization", 0.90)
        ),
        "enable_prefix_caching": bool(
            engine_config.get("enable_prefix_caching", True)
        ),
        "language_model_only": bool(
            engine_config.get("language_model_only", True)
        ),
        "seed": int(config.get("seed", 42)),
        "runner": "generate",
    }

    cpu_offload = float(engine_config.get("cpu_offload_gb", 0))
    if cpu_offload > 0:
        engine_kwargs["cpu_offload_gb"] = cpu_offload

    print("\nInitializing vLLM engine...")
    llm = LLM(**engine_kwargs)

    tokenizer = llm.get_tokenizer()

    print("vLLM engine initialized.")
    print("Tokenizer:", type(tokenizer).__name__)

    return llm, tokenizer, SamplingParams, vllm.__version__


def prepare_prompt(tokenizer, job):
    prompt = build_prompt(job["properties"], job["targets"])

    formatted_prompt, add_special_tokens = format_chat_prompt(
        tokenizer,
        prompt,
        properties=job.get("properties"),
        targets=job.get("targets"),
    )

    # Convert to token IDs explicitly. This prevents vLLM from adding
    # an additional BOS token to a chat-template-formatted prompt.
    encoded = tokenizer(
        formatted_prompt,
        add_special_tokens=add_special_tokens,
    )

    input_ids = encoded["input_ids"]

    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    return {"prompt_token_ids": input_ids}


def make_sampling_params(job, generation_config, SamplingParams):
    do_sample = bool(generation_config.get("do_sample", True))

    temperature = (
        float(generation_config.get("temperature", 0.7))
        if do_sample else 0.0
    )

    return SamplingParams(
        n=1,
        temperature=temperature,
        top_p=float(generation_config.get("top_p", 0.95)),
        max_tokens=int(
            generation_config.get("max_new_tokens", 128)
        ),
        repetition_penalty=float(
            generation_config.get("repetition_penalty", 1.0)
        ),
        seed=int(job["seed"]),
        skip_special_tokens=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    jobs_path = ROOT / config["paths"]["job_manifest"]

    output_path = (
        Path(args.output)
        if args.output
        else ROOT / config["paths"]["generation_output"]
    )

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    jobs = read_jsonl(jobs_path)

    if not jobs:
        raise ValueError(
            f"No jobs found in {jobs_path}. "
            "Run build_manifest.py first."
        )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        jobs = jobs[:args.limit]

    existing_records = read_jsonl(output_path)
    completed_ids = {
        record["job_id"] for record in existing_records
    }

    pending_jobs = [
        job for job in jobs
        if job["job_id"] not in completed_ids
    ]

    print("Total selected jobs:", len(jobs))
    print("Already completed:", len(completed_ids))
    print("Pending jobs:", len(pending_jobs))

    if not pending_jobs:
        print("All selected jobs are already complete.")
        return

    model_id = args.model_id or config["model"]["model_id"]

    llm, tokenizer, SamplingParams, vllm_version = load_engine(
        config, model_id
    )

    generation_config = config["generation"]
    engine_config = config["vllm"]

    batch_size = int(engine_config.get("batch_size", 4))

    if batch_size <= 0:
        raise ValueError("vllm.batch_size must be positive.")

    if batch_size > int(engine_config.get("max_num_seqs", 4)):
        raise ValueError(
            "batch_size cannot exceed max_num_seqs."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    completed_this_run = 0

    for batch_start in range(0, len(pending_jobs), batch_size):
        batch_jobs = pending_jobs[
            batch_start:batch_start + batch_size
        ]

        batch_number = batch_start // batch_size + 1
        total_batches = math.ceil(len(pending_jobs) / batch_size)

        print(
            f"\nBatch {batch_number}/{total_batches} "
            f"({len(batch_jobs)} jobs)"
        )

        prompts = [
            prepare_prompt(tokenizer, job)
            for job in batch_jobs
        ]

        sampling_params = [
            make_sampling_params(job, generation_config, SamplingParams)
            for job in batch_jobs
        ]

        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        if len(outputs) != len(batch_jobs):
            raise RuntimeError(
                "vLLM returned a different number of outputs "
                "than submitted prompts."
            )

        for position, (job, result) in enumerate(
            zip(batch_jobs, outputs),
            start=1,
        ):
            raw_output = (
                result.outputs[0].text
                if result.outputs else ""
            )

            record = make_record(
                job=job,
                raw_output=raw_output,
                backend=f"{model_id} | vLLM {vllm_version}",
            )

            append_jsonl(output_path, record)
            completed_this_run += 1

            print(
                f"  [{position}/{len(batch_jobs)}] "
                f"{job['job_id']} | "
                f"valid={record['valid']} | "
                f"SMILES={record['smiles'] or '<invalid>'}"
            )

            if args.debug:
                token_count = (
                    len(result.outputs[0].token_ids)
                    if result.outputs else 0
                )
                print("    Generated tokens:", token_count)
                print("    Raw output:", repr(raw_output))

    elapsed = time.perf_counter() - start_time

    print("\nGeneration finished.")
    print("New records:", completed_this_run)
    print("Total records in output:", len(read_jsonl(output_path)))
    print("Elapsed seconds:", round(elapsed, 2))
    print("Output:", output_path)


if __name__ == "__main__":
    main()
