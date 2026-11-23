import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, rdMolDescriptors, RDConfig

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/logp_zero_shot.json"

# RDKit's Ertl synthetic accessibility implementation.
import sys
sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
import sascorer


PROPERTY_INFO = {
    "logp": {
        "name": "LogP",
        "description": "RDKit Crippen MolLogP",
        "precision": 3
    },
    "tpsa": {
        "name": "TPSA",
        "description": "RDKit topological polar surface area in square angstroms",
        "precision": 2
    },
    "sas": {
        "name": "SAS",
        "description": "RDKit Ertl synthetic accessibility score",
        "precision": 3
    },
    "qed": {
        "name": "QED",
        "description": "RDKit quantitative estimate of drug-likeness",
        "precision": 3
    }
}


def load_config(path=None):
    config_path = Path(path) if path else CONFIG_PATH

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    with open(config_path) as f:
        return json.load(f)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def calculate_properties(smiles):
    """
    Parse SMILES, canonicalize it, and calculate all four descriptors.
    Returns (canonical_smiles, descriptor_dict).
    """
    if not smiles:
        return "", {}

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "", {}

        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True
        )

        values = {
            "logp": float(Crippen.MolLogP(mol)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "sas": float(sascorer.calculateScore(mol)),
            "qed": float(QED.qed(mol))
        }

        return canonical, values

    except Exception:
        return "", {}


def extract_smiles(raw_output):
    """
    Parse one complete SMILES candidate.

    Accepts a plain SMILES, an optional SMILES: label, or a Markdown
    code fence. Never extracts individual tokens from explanatory prose.
    """
    if raw_output is None:
        return "", ""

    text = str(raw_output).strip()

    if not text:
        return "", ""

    # Allow a standalone SMILES label before the actual candidate.
    lines = text.splitlines()

    if lines and re.fullmatch(
        r"\s*(?:smiles|molecule|output)\s*[:=]?\s*",
        lines[0],
        flags=re.IGNORECASE
    ):
        text = "\n".join(lines[1:]).strip()

    if not text:
        return "", ""

    # Allow one surrounding Markdown code fence.
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()

        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
        else:
            return "", ""

    # Only one non-empty candidate is allowed.
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) != 1:
        return "", ""

    candidate = re.sub(
        r"^(?:smiles|molecule|output)\s*[:=]\s*",
        "",
        lines[0],
        flags=re.IGNORECASE
    ).strip()

    # Remove matching surrounding quotes or backticks.
    if (
        len(candidate) >= 2
        and candidate[0] == candidate[-1]
        and candidate[0] in {"'", '"', "`"}
    ):
        candidate = candidate[1:-1].strip()

    # A SMILES candidate should not contain explanatory whitespace.
    if not candidate or any(char.isspace() for char in candidate):
        return "", ""

    canonical, _ = calculate_properties(candidate)

    if not canonical:
        return "", ""

    return candidate, canonical

def build_prompt(properties, targets):
    """
    Build a zero-shot prompt. There are intentionally no demonstrations.
    """
    condition_lines = []

    for prop in properties:
        info = PROPERTY_INFO[prop]
        value = float(targets[prop])
        condition_lines.append(
            f"- {info['name']}: {value!r} "
            f"({info['description']})"
        )

    conditions = "\n".join(condition_lines)

    return f"""You are an expert computational chemist specializing in molecular design.

Generate exactly one chemically valid molecule in SMILES format.

The molecule should satisfy these target molecular properties as closely as possible:

{conditions}

Requirements:
1. Generate a chemically valid molecule.
2. Aim to match every requested property simultaneously.
3. Use the property definitions specified above.
4. Return exactly one SMILES string.
5. Do not return explanations, labels, Markdown, or multiple candidates.

SMILES:"""


def make_record(job, raw_output, backend):
    raw_smiles, canonical = extract_smiles(raw_output)
    _, measured = calculate_properties(canonical)

    return {
        "job_id": job["job_id"],
        "experiment": job["experiment"],
        "properties": job["properties"],
        "target_id": job["target_id"],
        "target_index": job["target_index"],
        "targets": job["targets"],
        "replicate_index": job["replicate_index"],
        "sample_index": job["sample_index"],
        "seed": job["seed"],
        "shots": 0,
        "backend": backend,
        "prompt": build_prompt(job["properties"], job["targets"]),
        "raw_output": str(raw_output),
        "raw_smiles": raw_smiles,
        "smiles": canonical,
        "valid": bool(canonical),
        "measured_properties": measured,
        "timestamp": timestamp()
    }


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records
