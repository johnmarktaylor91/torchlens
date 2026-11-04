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


def save_trace(
    trace: Any, out_dir: Path | None, stem: str, *, model: torch.nn.Module, inputs: Any
) -> dict[str, Any] | None:
    """Write a portable ``.tlspec`` of one representative trace, then verify it replays.

    The bundle is loaded back with ``tl.load`` and rerun on ``model`` with ``inputs``
    (``loaded.run``). Every op's saved value in the loaded trace and in the replay must
    equal the original bitwise, op by op and label by label. A mismatch raises: a replay
    that differs is a finding to explain, never a tolerance to widen.

    Parameters
    ----------
    trace:
        A finished trace captured with ``save="all"``.
    out_dir:
        Output directory, or ``None`` to skip writing.
    stem:
        File stem for the bundle.
    model:
        The model the trace was captured from.
    inputs:
        The exact input the trace was captured with.

    Returns
    -------
    dict or None
        ``{"bundle", "ops", "tensors_compared", "loaded_equal", "replay_equal"}``, or
        ``None`` when nothing was written.

    Raises
    ------
    RuntimeError
        When the loaded bundle or its replay differs from the original trace.
    """

    import torchlens as tl

    if out_dir is None:
        return None
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{stem}.tlspec"
    trace.save(path, level="portable", overwrite=True)
    loaded = tl.load(path)
    with torch.no_grad():
        replayed = loaded.run(model, inputs)
    try:
        compared = _require_same_ops(trace, loaded, f"{path.name} loaded back")
        _require_same_ops(trace, replayed, f"{path.name} replayed on the model")
    finally:
        replayed.cleanup()
    print(f"REPLAY: {path.name} loads back and replays bitwise equal ({compared} tensors)")
    return {
        "bundle": path.name,
        "ops": len(trace.ops),
        "tensors_compared": compared,
        "loaded_equal": True,
        "replay_equal": True,
    }


def _require_same_ops(original: Any, other: Any, what: str) -> int:
    """Raise unless ``other`` has the original's op labels and bitwise-equal saved values.

    Parameters
    ----------
    original:
        The trace as captured.
    other:
        The loaded or replayed trace.
    what:
        Phrase naming ``other`` in the error.

    Returns
    -------
    int
        Number of tensor values compared.
    """

    mine = [op.label for op in original.ops]
    theirs = [op.label for op in other.ops]
    if mine != theirs:
        first = next(
            (i for i, pair in enumerate(zip(mine, theirs, strict=False)) if pair[0] != pair[1]),
            min(len(mine), len(theirs)),
        )
        raise RuntimeError(
            f"{what}: op labels differ from the original at index {first} "
            f"({len(mine)} ops originally, {len(theirs)} now)"
        )
    compared = 0
    for before, after in zip(original.ops, other.ops, strict=True):
        if not isinstance(before.out, torch.Tensor):
            continue
        if not isinstance(after.out, torch.Tensor) or not torch.equal(before.out, after.out):
            raise RuntimeError(f"{what}: op {before.label} differs from the original")
        compared += 1
    return compared


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
