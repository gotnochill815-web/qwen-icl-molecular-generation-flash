
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


def minimum_gpu_memory(model_id, engine_config):
    """
    Current vLLM Qwen3.8 recipe memory estimates, in GiB:
      NVFP4: 130
      FP8:   250
      BF16:  423
    """
    model_name = model_id.upper()

    if "NVFP4" in model_name or "MXFP4" in model_name:
        return 130.0

    if "FP8" in model_name:
        return 250.0

    if "QWEN3.8-FLASH-NEXT" in model_name:
        return 423.0

    return float(
        engine_config.get("default_minimum_gpu_memory_gb", 130)
    )


def preflight(engine_config, torch, vllm_version, model_id):
    from packaging.version import Version

    minimum_version = engine_config.get(
        "min_vllm_version", "0.30.0"
    )

    if Version(vllm_version) < Version(minimum_version):
        raise RuntimeError(
            f"Qwen3.8-Flash-Next requires vLLM >= "
            f"{minimum_version}; found {vllm_version}."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Request a supported GPU allocation."
        )

    gpu_count = torch.cuda.device_count()

    tensor_parallel_size = int(
        engine_config.get("tensor_parallel_size", 1)
    )

    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be at least 1.")

    if gpu_count < tensor_parallel_size:
        raise RuntimeError(
            f"tensor_parallel_size={tensor_parallel_size}, "
            f"but only {gpu_count} GPUs are visible."
        )

    gpu_memory = []
    gpu_names = []

    for index in range(gpu_count):
        props = torch.cuda.get_device_properties(index)

        memory_gb = props.total_memory / (1024 ** 3)

        gpu_memory.append(memory_gb)
        gpu_names.append(props.name)

        print(
            f"GPU {index}: {props.name} | "
            f"{memory_gb:.1f} GiB"
        )

    total_memory = sum(gpu_memory)
    required_memory = minimum_gpu_memory(
        model_id, engine_config
    )

    print("Visible total GPU memory:", round(total_memory, 1), "GiB")
    print("Checkpoint memory requirement:", required_memory, "GiB")

    if any("A100" in name.upper() for name in gpu_names):
        print(
            "WARNING: A100 is not listed in the current official "
            "vLLM Qwen3.8 hardware validation matrix."
        )

    if total_memory < required_memory:
        raise RuntimeError(
            "\nInsufficient GPU memory. Model initialization was "
            "not attempted.\n"
            f"Checkpoint: {model_id}\n"
            f"Visible: {total_memory:.1f} GiB\n"
            f"Required: {required_memory:.1f} GiB\n"
            "Request a compatible multi-GPU allocation or select "
            "a hardware-supported checkpoint configuration."
        )


def load_engine(config, model_id):
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "The vLLM backend needs PyTorch and vLLM. "
            "Install vLLM >= 0.30.0 in a CUDA-compatible environment."
        ) from exc

    engine_config = config["vllm"]
    model_config = config["model"]

    version = getattr(vllm, "__version__", "0.0.0")

    print("PyTorch:", torch.__version__)
    print("vLLM:", version)
    print("Model:", model_id)

    preflight(
        engine_config,
        torch,
        version,
        model_id
    )

    engine_kwargs = {
        "model": model_id,
        "runner": "generate",
        "trust_remote_code": model_config.get(
            "trust_remote_code", True
        ),
        "dtype": engine_config.get("dtype", "auto"),
        "tensor_parallel_size": int(
            engine_config.get("tensor_parallel_size", 1)
        ),
        "max_model_len": int(
            engine_config.get("max_model_len", 4096)
        ),
        "max_num_seqs": int(
            engine_config.get("max_num_seqs", 2)
        ),
        "max_num_batched_tokens": int(
            engine_config.get("max_num_batched_tokens", 2048)
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
        "engram_config": engine_config.get(
            "engram_config", {"cpu_offload": True}
        ),
    }

    # Optional general model-weight CPU offload.
    cpu_offload = float(
        engine_config.get("cpu_offload_gb", 0)
    )

    if cpu_offload > 0:
        engine_kwargs["cpu_offload_gb"] = cpu_offload

    if engine_config.get("distributed_executor_backend"):
        engine_kwargs["distributed_executor_backend"] = (
            engine_config["distributed_executor_backend"]
        )

    if engine_config.get("quantization"):
        engine_kwargs["quantization"] = (
            engine_config["quantization"]
        )

    print("\nInitializing vLLM engine...")
    llm = LLM(**engine_kwargs)

    tokenizer = llm.get_tokenizer()

    print("vLLM engine initialized.")
    print("Tokenizer:", type(tokenizer).__name__)

    return llm, tokenizer, SamplingParams, version


def prepare_prompt(tokenizer, job):
    prompt = build_prompt(
        job["properties"],
        job["targets"]
    )

    formatted, add_special_tokens = format_chat_prompt(
        tokenizer,
        prompt,
        properties=job.get("properties"),
        targets=job.get("targets"),
    )

    encoded = tokenizer(
        formatted,
        add_special_tokens=add_special_tokens
    )

    input_ids = encoded["input_ids"]

    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()

    if input_ids and isinstance(input_ids[0], (list, tuple)):
        input_ids = input_ids[0]

    if not input_ids:
        raise RuntimeError(
            f"Tokenizer returned empty prompt for {job['job_id']}"
        )

    return {
        "prompt_token_ids": [int(token) for token in input_ids]
    }


def make_sampling_params(job, generation_config, SamplingParams):
    do_sample = bool(
        generation_config.get("do_sample", True)
    )

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
    parser = argparse.ArgumentParser(
        description="Zero-shot Flash-Next generation using vLLM."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    config = load_config(args.config)

    if len(config["experiments"]) != 1:
        raise ValueError(
            "Each config must define exactly one experiment."
        )

    if int(config["generation"].get("shots", 0)) != 0:
        raise ValueError("This repository supports zero-shot only.")

    experiment = next(iter(config["experiments"]))

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
            f"No jobs found: {jobs_path}. "
            "Run build_manifest.py first."
        )

    unexpected = {
        job.get("experiment")
        for job in jobs
        if job.get("experiment") != experiment
    }

    if unexpected:
        raise ValueError(
            f"Manifest contains unexpected experiments: {unexpected}"
        )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        jobs = jobs[:args.limit]

    existing = read_jsonl(output_path)

    completed_ids = {
        record["job_id"] for record in existing
    }

    pending = [
        job for job in jobs
        if job["job_id"] not in completed_ids
    ]

    print("Config:", args.config)
    print("Experiment:", experiment)
    print("Selected jobs:", len(jobs))
    print("Already completed:", len(completed_ids))
    print("Pending jobs:", len(pending))

    if not pending:
        print("All selected jobs are already complete.")
        return

    engine_config = config["vllm"]

    model_id = (
        args.model_id
        or engine_config.get("model_id")
        or config["model"]["model_id"]
    )

    llm, tokenizer, SamplingParams, version = load_engine(
        config,
        model_id
    )

    generation_config = config["generation"]

    batch_size = int(engine_config.get("batch_size", 2))
    max_num_seqs = int(engine_config.get("max_num_seqs", 2))

    if batch_size <= 0 or batch_size > max_num_seqs:
        raise ValueError(
            "batch_size must be positive and no greater "
            "than vllm.max_num_seqs."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    written = 0

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[
            batch_start:batch_start + batch_size
        ]

        batch_number = batch_start // batch_size + 1
        total_batches = math.ceil(len(pending) / batch_size)

        print(
            f"\nBatch {batch_number}/{total_batches} "
            f"({len(batch)} jobs)"
        )

        prompts = [
            prepare_prompt(tokenizer, job)
            for job in batch
        ]

        sampling = [
            make_sampling_params(
                job,
                generation_config,
                SamplingParams
            )
            for job in batch
        ]

        outputs = llm.generate(
            prompts,
            sampling_params=sampling,
            use_tqdm=False,
        )

        if len(outputs) != len(batch):
            raise RuntimeError(
                "vLLM returned a different number of outputs "
                "than submitted prompts."
            )

        for position, (job, result) in enumerate(
            zip(batch, outputs), start=1
        ):
            raw_output = (
                result.outputs[0].text
                if result.outputs else ""
            )

            record = make_record(
                job=job,
                raw_output=raw_output,
                backend=f"{model_id} | vLLM {version}",
            )

            append_jsonl(output_path, record)
            written += 1

            print(
                f"  [{position}/{len(batch)}] "
                f"{job['job_id']} | "
                f"valid={record['valid']} | "
                f"SMILES={record['smiles'] or '<invalid>'}"
            )

            if args.debug:
                print("    Raw output:", repr(raw_output))

    elapsed = time.perf_counter() - started

    print("\nGeneration finished.")
    print("New records:", written)
    print("Total records:", len(read_jsonl(output_path)))
    print("Elapsed seconds:", round(elapsed, 2))
    print("Output:", output_path)


if __name__ == "__main__":
    main()
