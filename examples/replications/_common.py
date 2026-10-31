"""Shared pieces of the paper replications: the pinned model, the data files, the report.

Every replication script in this folder states one published claim, measures it with
existing TorchLens spellings, and prints a verdict computed from a rule fixed before the
first run. The rules live in each script's ``RULE`` string and in the folder README.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

#: Public GPT-2 small checkpoint and the exact Hub revision every replication uses.
MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"  # pragma: allowlist secret

DATA_DIR = Path(__file__).resolve().parent / "data"

#: Pinned public sources of the vendored data (``make_data.py`` rebuilds it from them).
IOI_REPO = "mib-bench/ioi"
IOI_REVISION = "e5f3468f3af4c0883be35cd3bced8c711c95d286"  # pragma: allowlist secret
IOI_FILE = "train.json"
IOI_SHA256 = (
    "2ee41861d725eee2b71da3846872d5e5ea29d9d8c9aa3a8954f8430584b2f243"  # pragma: allowlist secret
)
FACTS_URL = "https://rome.baulab.info/data/dsets/known_1000.json"
FACTS_SHA256 = (
    "61daca55318bb5260c1e62e133debef9102c7b278bbc7b19dd2ac655543f333a"  # pragma: allowlist secret
)

REPLICATED = "replicated"
NOT_REPLICATED = "did not replicate"


def load_gpt2() -> tuple[torch.nn.Module, Any]:
    """Load GPT-2 small (float32, eager attention) and its tokenizer at the pinned revision.

    Returns
    -------
    tuple
        ``(model, tokenizer)``; the model is in eval mode on the CPU.
    """

    from transformers import AutoTokenizer, GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, attn_implementation="eager"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    return model, tokenizer


def load_data(name: str) -> dict[str, Any]:
    """Read one vendored data file from ``data/``.

    Parameters
    ----------
    name:
        File name inside ``data/``.

    Returns
    -------
    dict
        The parsed JSON document.
    """

    return json.loads((DATA_DIR / name).read_text())


def rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    """Hash a list of data rows canonically (sorted keys, compact separators).

    Parameters
    ----------
    rows:
        JSON-serialisable rows.

    Returns
    -------
    str
        Hex sha256 of the canonical JSON encoding.
    """

    blob = json.dumps(list(rows), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash an integer tensor's values (int64, row-major bytes).

    Parameters
    ----------
    tensor:
        Integer tensor.

    Returns
    -------
    str
        Hex sha256 of its int64 bytes.
    """

    return hashlib.sha256(tensor.to(torch.int64).contiguous().numpy().tobytes()).hexdigest()


def single_token(tokenizer: Any, word: str) -> int | None:
    """Return the GPT-2 token id of ``" " + word`` when it is exactly one token.

    Parameters
    ----------
    tokenizer:
        GPT-2 tokenizer.
    word:
        Word without its leading space.

    Returns
    -------
    int or None
        The single token id, or ``None`` when the word splits into several tokens.
    """

    ids = tokenizer(" " + word, add_special_tokens=False).input_ids
    return int(ids[0]) if len(ids) == 1 else None


def save_trace(trace: Any, out_dir: Path | None, stem: str) -> Path | None:
    """Write a portable ``.tlspec`` of one representative trace beside the summary.

    Parameters
    ----------
    trace:
        A finished trace.
    out_dir:
        Output directory, or ``None`` to skip writing.
    stem:
        File stem for the bundle.

    Returns
    -------
    Path or None
        The bundle path, or ``None`` when nothing was written.
    """

    if out_dir is None:
        return None
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{stem}.tlspec"
    trace.save(path, level="portable", overwrite=True)
    return path


def report(summary: dict[str, Any], out_dir: Path | None, stem: str) -> dict[str, Any]:
    """Print the claim, the measured numbers and the verdict; write the JSON summary.

    Parameters
    ----------
    summary:
        Must hold ``claim``, ``rule``, ``measured`` and ``verdict``.
    out_dir:
        Output directory, or ``None`` to print only.
    stem:
        File stem for the summary JSON.

    Returns
    -------
    dict
        The same summary.
    """

    print(f"CLAIM: {summary['claim']}")
    print(f"RULE: {summary['rule']}")
    for key, value in summary["measured"].items():
        print(f"MEASURED {key}: {value}")
    print(f"VERDICT: {summary['verdict']}")
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        target = Path(out_dir) / f"{stem}.json"
        target.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary
