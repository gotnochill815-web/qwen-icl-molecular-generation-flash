
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



def _read_int_file(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def available_host_memory_gb():
    """
    Estimate available host RAM, respecting cgroup limits when discoverable.
    Returns None if the system does not expose usable memory information.
    """
    candidates = []

    # System-wide available memory.
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available_kb = int(line.split()[1])
                candidates.append(available_kb * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass

    # Cgroup v2 and v1 memory limits, where available.
    try:
        cgroup_lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        cgroup_lines = []

    for line in cgroup_lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue

        hierarchy, controllers, relative_path = parts
        relative_path = relative_path.lstrip("/")

        if hierarchy == "0" and not controllers:
            base = Path("/sys/fs/cgroup") / relative_path
            limit = _read_int_file(base / "memory.max")
            usage = _read_int_file(base / "memory.current")

        elif "memory" in controllers.split(","):
            bases = [
                Path("/sys/fs/cgroup/memory") / relative_path,
                Path("/sys/fs/cgroup") / relative_path,
            ]

            limit = usage = None
            for base in bases:
                limit = _read_int_file(base / "memory.limit_in_bytes")
                usage = _read_int_file(base / "memory.usage_in_bytes")
                if limit is not None and usage is not None:
                    break
        else:
            continue

        # Very large cgroup limits represent "unlimited".
        if (
            limit is not None
            and usage is not None
            and 0 < limit < (1 << 60)
        ):
            candidates.append(max(0, limit - usage))

    if not candidates:
        return None

    return min(candidates) / (1024 ** 3)


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

    # vLLM tensor parallelism uses the selected TP ranks.
    # Do not count unrelated visible GPUs toward model memory.
    selected_memory = []
    selected_names = []

    print("Visible GPUs:", gpu_count)
    print("Tensor-parallel GPUs selected:", tensor_parallel_size)

    for index in range(tensor_parallel_size):
        props = torch.cuda.get_device_properties(index)
        memory_gb = props.total_memory / (1024 ** 3)

        selected_memory.append(memory_gb)
        selected_names.append(props.name)

        print(
            f"TP GPU {index}: {props.name} | "
            f"{memory_gb:.1f} GiB"
        )

    total_selected_memory = sum(selected_memory)
    required_memory = minimum_gpu_memory(
        model_id, engine_config
    )

    print(
        "Selected TP GPU memory:",
        round(total_selected_memory, 1),
        "GiB",
    )
    print(
        "Checkpoint memory requirement:",
        required_memory,
        "GiB",
    )

    if any("A100" in name.upper() for name in selected_names):
        print(
            "WARNING: A100 is not listed in the current official "
            "vLLM Qwen3.8 hardware validation matrix."
        )

    if total_selected_memory < required_memory:
        raise RuntimeError(
            "\nInsufficient GPU memory. Model initialization was "
            "not attempted.\n"
            f"Checkpoint: {model_id}\n"
            f"Selected TP memory: {total_selected_memory:.1f} GiB\n"
            f"Required: {required_memory:.1f} GiB\n"
        )

    engram_config = engine_config.get(
        "engram_config", {"cpu_offload": True}
    )

    if engram_config.get("cpu_offload", True):
        required_host_memory = float(
            engine_config.get(
                "minimum_available_host_memory_gb", 51.0
            )
        )

        available_host_memory = available_host_memory_gb()

        if available_host_memory is None:
            print(
                "WARNING: Could not determine available host memory. "
                "Check the scheduler memory allocation manually."
            )
        else:
            print(
                "Available host memory:",
                round(available_host_memory, 1),
                "GiB",
            )
            print(
                "Minimum host memory for PLE offload:",
                required_host_memory,
                "GiB",
            )

            if available_host_memory < required_host_memory:
                raise RuntimeError(
                    "Insufficient available host memory for PLE CPU "
                    "offload. Need at least "
                    f"{required_host_memory:.1f} GiB available; "
                    f"found {available_host_memory:.1f} GiB. "
                    "Additional runtime headroom is also required."
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



def validate_engine_config(engine_config):
    tensor_parallel_size = int(
        engine_config.get("tensor_parallel_size", 1)
    )
    batch_size = int(engine_config.get("batch_size", 2))
    max_num_seqs = int(engine_config.get("max_num_seqs", 2))
    max_model_len = int(engine_config.get("max_model_len", 4096))
    max_batched_tokens = int(
        engine_config.get("max_num_batched_tokens", 2048)
    )
    gpu_memory_utilization = float(
        engine_config.get("gpu_memory_utilization", 0.90)
    )

    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive.")

    if batch_size < 1 or max_num_seqs < 1:
        raise ValueError("batch_size and max_num_seqs must be positive.")

    if batch_size > max_num_seqs:
        raise ValueError(
            "batch_size must not exceed vllm.max_num_seqs."
        )

    if max_model_len < 1 or max_batched_tokens < 1:
        raise ValueError(
            "max_model_len and max_num_batched_tokens must be positive."
        )

    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError(
            "gpu_memory_utilization must be in the interval (0, 1]."
        )

    print("vLLM engine configuration validation: PASS")


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

    generation_config = config["generation"]

    # Validate settings before importing/initializing the vLLM engine.
    validate_engine_config(engine_config)

    batch_size = int(engine_config.get("batch_size", 2))

    llm, tokenizer, SamplingParams, version = load_engine(
        config,
        model_id
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
