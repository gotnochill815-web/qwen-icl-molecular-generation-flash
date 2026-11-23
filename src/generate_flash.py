import argparse
import sys
from pathlib import Path

import torch
import transformers

from common import (
    ROOT,
    load_config,
    read_jsonl,
    append_jsonl,
    build_prompt,
    make_record
)


def load_model(model_id, model_config):
    """
    Load Qwen3.8-Flash-Next from either a Hugging Face Hub ID
    or a local checkpoint directory.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Request a GPU allocation first."
        )

    model_path = Path(model_id).expanduser()

    # Resolve a local checkpoint when it exists. Otherwise, use the
    # supplied string as a Hugging Face repository ID.
    if model_path.is_absolute() and model_path.is_dir():
        model_source = str(model_path.resolve())
    elif not model_path.is_absolute() and (ROOT / model_path).is_dir():
        model_source = str((ROOT / model_path).resolve())
    else:
        model_source = model_id

    local_files_only = bool(
        model_config.get("local_files_only", False)
    )
    trust_remote_code = bool(
        model_config.get("trust_remote_code", True)
    )

    if local_files_only and not Path(model_source).is_dir():
        raise FileNotFoundError(
            "local_files_only=True, but the supplied model is not "
            f"a local directory: {model_source}"
        )

    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model source:", model_source)
    print("Local-files-only:", local_files_only)

    dtype_name = model_config.get("dtype", "bfloat16")

    if dtype_name == "auto":
        dtype = "auto"
    else:
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_name}")

    model_class_name = model_config.get(
        "auto_model_class", "AutoModelForImageTextToText"
    )
    model_class = getattr(transformers, model_class_name, None)

    if model_class is None:
        raise RuntimeError(
            f"{model_class_name} is unavailable in Transformers "
            f"{transformers.__version__}."
        )

    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }

    cache_dir = model_config.get("cache_dir")
    if cache_dir:
        load_kwargs["cache_dir"] = str(
            Path(cache_dir).expanduser()
        )

    print("Loading processor/tokenizer...")

    try:
        processor = transformers.AutoProcessor.from_pretrained(
            model_source, **load_kwargs
        )
        tokenizer = getattr(processor, "tokenizer", None)

        if tokenizer is None:
            raise RuntimeError(
                "AutoProcessor did not expose a tokenizer."
            )

    except Exception as exc:
        print("AutoProcessor unavailable:", repr(exc))
        print("Trying AutoTokenizer...")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_source, **load_kwargs
        )

    if (
        tokenizer.pad_token_id is None
        and tokenizer.eos_token_id is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "dtype": dtype,
        "device_map": model_config.get("device_map", "auto"),
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }

    if cache_dir:
        model_kwargs["cache_dir"] = str(
            Path(cache_dir).expanduser()
        )

    if model_config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = (
            model_config["attn_implementation"]
        )

    print("Loading model weights...")

    try:
        model = model_class.from_pretrained(
            model_source, **model_kwargs
        )
    except TypeError as exc:
        # Compatibility with Transformers versions that use
        # torch_dtype instead of dtype.
        if "dtype" not in str(exc):
            raise

        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = model_class.from_pretrained(
            model_source, **model_kwargs
        )

    model.eval()

    print("Model loaded successfully.")
    print("Model class:", type(model).__name__)
    print(
        "EOS IDs:",
        getattr(model.generation_config, "eos_token_id", None)
    )
    print(
        "PAD ID:",
        getattr(model.generation_config, "pad_token_id", None)
    )

    for index in range(torch.cuda.device_count()):
        print(
            f"GPU {index} memory allocated:",
            round(
                torch.cuda.memory_allocated(index) / 1024**3, 2
            ),
            "GB"
        )

    return model, tokenizer


def format_chat_prompt(tokenizer, prompt, properties=None, targets=None):
    def get_property_names(value):
        if isinstance(value, dict):
            items = value.keys()
        elif isinstance(value, str):
            items = [value]
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = []

        return {
            str(item).strip().lower()
            for item in items
            if str(item).strip()
        }

    names = (
        get_property_names(properties)
        | get_property_names(targets)
    )

    if names == {"logp"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "calculated octanol/water partition coefficient (LogP), "
            "calculated using RDKit Crippen MolLogP, is as close as "
            "possible to the requested target. A valid SMILES is not "
            "sufficient by itself: try to match the requested LogP while "
            "producing a chemically plausible molecule. Treat the examples "
            "as demonstrations of the requested input-output format. "
            "For each example, the supplied LogP is the reference property "
            "value for that example. Do not copy an example unnecessarily. "
            "Return only the generated SMILES. Do not include explanations, "
            "Markdown fences, labels, or multiple candidates."
        )
    elif names == {"qed"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "quantitative estimate of drug-likeness (QED), calculated "
            "using RDKit QED.qed(), is as close as possible to the "
            "requested target. A valid SMILES is not sufficient by itself: "
            "try to match the requested QED while producing a chemically "
            "plausible molecule. Treat the examples as demonstrations of "
            "the requested input-output format. For each example, the "
            "supplied QED is the reference property value for that example. "
            "Do not copy an example unnecessarily. Return only the "
            "generated SMILES. Do not include explanations, Markdown "
            "fences, labels, or multiple candidates."
        )
    elif names == {"tpsa"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "topological polar surface area (TPSA), calculated using "
            "RDKit rdMolDescriptors.CalcTPSA, is as close as possible "
            "to the requested target. A valid SMILES is not sufficient "
            "by itself: try to match the requested TPSA while producing "
            "a chemically plausible molecule. Treat the examples as "
            "demonstrations of the requested input-output format. "
            "For each example, the supplied TPSA is the reference property "
            "value for that example. Do not copy an example unnecessarily. "
            "Return only the generated SMILES. Do not include explanations, "
            "Markdown fences, labels, or multiple candidates."
        )
    elif names == {"sas"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "synthetic accessibility score (SAS), calculated using "
            "RDKit sascorer.calculateScore, is as close as possible "
            "to the requested target. A valid SMILES is not sufficient "
            "by itself: try to match the requested SAS while producing "
            "a chemically plausible molecule. Treat the examples as "
            "demonstrations of the requested input-output format. "
            "For each example, the supplied SAS is the reference property "
            "value for that example. Do not copy an example unnecessarily. "
            "Return only the generated SMILES. Do not include explanations, "
            "Markdown fences, labels, or multiple candidates."
        )
    elif names == {"tpsa", "sas"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "topological polar surface area (TPSA), calculated using "
            "RDKit rdMolDescriptors.CalcTPSA, and synthetic accessibility "
            "score (SAS), calculated using RDKit sascorer.calculateScore, "
            "are as close as possible to their requested targets. "
            "A valid SMILES is not sufficient by itself: try to match "
            "both requested properties simultaneously while producing "
            "a chemically plausible molecule. Treat the examples as "
            "demonstrations of the requested input-output format. "
            "For each example, the supplied TPSA and SAS values are the "
            "reference property values for that example. Do not copy an "
            "example unnecessarily. Return only the generated SMILES. "
            "Do not include explanations, Markdown fences, labels, "
            "or multiple candidates."
        )
    elif names == {"logp", "tpsa"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "calculated octanol/water partition coefficient (LogP), "
            "calculated using RDKit Crippen MolLogP, and topological "
            "polar surface area (TPSA), calculated using RDKit "
            "rdMolDescriptors.CalcTPSA, are as close as possible to "
            "their requested targets. A valid SMILES is not sufficient "
            "by itself: try to match both requested properties "
            "simultaneously while producing a chemically plausible "
            "molecule. Treat the examples as demonstrations of the "
            "requested input-output format. For each example, the "
            "supplied LogP and TPSA values are the reference property "
            "values for that example. Do not copy an example "
            "unnecessarily. Return only the generated SMILES. "
            "Do not include explanations, Markdown fences, labels, "
            "or multiple candidates."
        )
    elif names == {"sas", "logp"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "synthetic accessibility score (SAS), calculated using "
            "RDKit sascorer.calculateScore, and calculated octanol/water "
            "partition coefficient (LogP), calculated using RDKit "
            "Crippen MolLogP, are as close as possible to their "
            "requested targets. A valid SMILES is not sufficient by "
            "itself: try to match both requested properties "
            "simultaneously while producing a chemically plausible "
            "molecule. Treat the examples as demonstrations of the "
            "requested input-output format. For each example, the "
            "supplied SAS and LogP values are the reference property "
            "values for that example. Do not copy an example "
            "unnecessarily. Return only the generated SMILES. "
            "Do not include explanations, Markdown fences, labels, "
            "or multiple candidates."
        )
    elif names == {"logp", "sas", "tpsa"}:
        system_prompt = (
            "You are a molecular generation model. "
            "Generate exactly ONE valid molecular SMILES string whose "
            "calculated octanol/water partition coefficient (LogP), "
            "calculated using RDKit Crippen MolLogP, topological polar "
            "surface area (TPSA), calculated using RDKit "
            "rdMolDescriptors.CalcTPSA, and synthetic accessibility "
            "score (SAS), calculated using RDKit sascorer.calculateScore, "
            "are as close as possible to their requested targets. "
            "A valid SMILES is not sufficient by itself: try to match "
            "all three requested properties simultaneously while "
            "producing a chemically plausible molecule. Treat the "
            "examples as demonstrations of the requested input-output "
            "format. For each example, the supplied LogP, SAS, and TPSA "
            "values are the reference property values for that example. "
            "Do not copy an example unnecessarily. Return only the "
            "generated SMILES. Do not include explanations, Markdown "
            "fences, labels, or multiple candidates."
        )
    else:
        system_prompt = (
            "You are a computational chemistry assistant. "
            "Follow the requested output format exactly."
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return text, False

    return prompt, True


def get_input_device(model):
    try:
        device = model.get_input_embeddings().weight.device
        if device.type != "meta":
            return device
    except Exception:
        pass

    return torch.device("cuda:0")


def generate_one(model, tokenizer, job, config, debug_tokens=False):
    prompt = build_prompt(job["properties"], job["targets"])
    formatted_prompt, add_special_tokens = format_chat_prompt(
        tokenizer,
        prompt,
        properties=job.get("properties"),
        targets=job.get("targets")
    )

    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=add_special_tokens
    )

    input_device = get_input_device(model)
    encoded = {
        key: value.to(input_device)
        for key, value in encoded.items()
    }

    seed = int(job["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    generation_config = config["generation"]

    generation_kwargs = {
        "max_new_tokens": generation_config["max_new_tokens"],
        "do_sample": generation_config["do_sample"],
        "temperature": generation_config["temperature"],
        "top_p": generation_config["top_p"],
        "repetition_penalty": generation_config["repetition_penalty"],
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            **generation_kwargs
        )

    input_length = encoded["input_ids"].shape[-1]
    generated_ids = output_ids[0, input_length:]

    if debug_tokens:
        token_ids = generated_ids.detach().cpu().tolist()

        print("\\n========== RAW GENERATION DEBUG ==========")
        print("Generated token count:", len(token_ids))
        print("Generated token IDs:", token_ids)
        print(
            "Generated token strings:",
            tokenizer.convert_ids_to_tokens(token_ids)
        )
        print(
            "Decoded with special tokens:",
            repr(tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            ))
        )

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    ).strip()

    if debug_tokens:
        print("Decoded without special tokens:", repr(raw_output))
        print("========== END GENERATION DEBUG ==========")

    return raw_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Experiment configuration JSON file."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of jobs to run. Useful for a small GPU test."
    )
    parser.add_argument(
        "--debug-tokens",
        action="store_true",
        help="Print raw generated token IDs and special-token decoding."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSONL path. Defaults to the configured path."
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional Hugging Face model ID override."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    jobs_path = ROOT / config["paths"]["job_manifest"]
    output_path = (
        Path(args.output) if args.output
        else ROOT / config["paths"]["generation_output"]
    )

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    jobs = read_jsonl(jobs_path)

    if not jobs:
        raise ValueError(
            "No jobs found. Run python src/build_manifest.py first."
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
    model, tokenizer = load_model(
        model_id,
        config["model"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for index, job in enumerate(pending_jobs, start=1):
        print(
            f"\n[{index}/{len(pending_jobs)}] "
            f"{job['job_id']} | {job['experiment']} | "
            f"targets={job['targets']}"
        )

        raw_output = generate_one(
            model,
            tokenizer,
            job,
            config,
            debug_tokens=args.debug_tokens and index == 1
        )

        record = make_record(
            job=job,
            raw_output=raw_output,
            backend=model_id
        )

        append_jsonl(output_path, record)

        print(
            "Valid:", record["valid"],
            "| SMILES:", record["smiles"] or "<invalid output>"
        )

    print("\nGeneration batch finished.")
    print("Output:", output_path)
    print("Records written:", len(read_jsonl(output_path)))


if __name__ == "__main__":
    main()
