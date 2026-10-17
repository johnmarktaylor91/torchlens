"""Rebuild the vendored replication data from its pinned public sources.

Run: ``python examples/replications/make_data.py``. It downloads MIB's IOI ``train.json``
(Hugging Face dataset ``mib-bench/ioi`` at a pinned revision, MIT) and ROME's
``known_1000.json`` (MIT), checks both sha256 digests, applies the selection rules below
and rewrites ``data/ioi_rows.json`` and ``data/known_facts.json``. Each file keeps the whole
scanned prefix of its source, in file order, with a ``kept`` flag, so the selection can be
re-checked without the source.

Selection rules (fixed before any replication ran):

- IOI: the first 50 rows whose ``" " + indirect_object`` and ``" " + subject`` are each
  exactly one GPT-2 token.
- Known facts: the first 10 records whose subject occurs in the prompt and for which GPT-2
  small's greedy next token after the prompt is the first token of ``" " + attribute``.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import _common as common  # the sibling module; scripts here run from this folder
import torch

IOI_COUNT = 50
FACTS_COUNT = 10


def _checked(blob: bytes, expected: str, what: str) -> bytes:
    """Refuse a download whose sha256 is not the pinned one."""

    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected:
        raise RuntimeError(f"{what}: sha256 {digest} is not the pinned {expected}")
    return blob


def ioi_rows(tokenizer: Any) -> dict[str, Any]:
    """Select the IOI prompt set from MIB's pinned ``train.json``.

    Parameters
    ----------
    tokenizer:
        GPT-2 tokenizer.

    Returns
    -------
    dict
        Provenance plus the scanned rows with their ``kept`` flags.
    """

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        common.IOI_REPO, common.IOI_FILE, repo_type="dataset", revision=common.IOI_REVISION
    )
    source = json.loads(_checked(Path(path).read_bytes(), common.IOI_SHA256, common.IOI_FILE))
    rows: list[dict[str, Any]] = []
    kept = 0
    for index, row in enumerate(source):
        if kept == IOI_COUNT:
            break
        meta = row["metadata"]
        ok = all(
            common.single_token(tokenizer, meta[key]) is not None
            for key in ("indirect_object", "subject")
        )
        kept += ok
        rows.append(
            {
                "index": index,
                "prompt": row["prompt"],
                "indirect_object": meta["indirect_object"],
                "subject": meta["subject"],
                "kept": ok,
            }
        )
    return {
        "source": f"https://huggingface.co/datasets/{common.IOI_REPO} {common.IOI_FILE}",
        "pins": "revision and sha256 in _common.IOI_REVISION and _common.IOI_SHA256",
        "license": "MIT",
        "rule": "first 50 rows in file order whose ' '+indirect_object and ' '+subject are "
        "each exactly one GPT-2 token",
        "rows": rows,
    }


def known_facts(model: torch.nn.Module, tokenizer: Any) -> dict[str, Any]:
    """Select the known-facts set from ROME's pinned ``known_1000.json``.

    Parameters
    ----------
    model:
        GPT-2 small at the pinned revision.
    tokenizer:
        Its tokenizer.

    Returns
    -------
    dict
        Provenance plus the scanned records with their ``kept`` flags.
    """

    with urllib.request.urlopen(common.FACTS_URL, timeout=60) as response:  # noqa: S310 -- pinned https URL
        source = json.loads(_checked(response.read(), common.FACTS_SHA256, "known_1000.json"))
    records: list[dict[str, Any]] = []
    kept = 0
    for record in source:
        if kept == FACTS_COUNT:
            break
        ok = record["subject"] in record["prompt"] and is_known(model, tokenizer, record)
        kept += ok
        records.append(
            {
                key: record[key]
                for key in ("known_id", "subject", "attribute", "prompt", "relation_id")
            }
            | {"kept": ok}
        )
    return {
        "source": common.FACTS_URL,
        "pins": "sha256 in _common.FACTS_SHA256",
        "license": "MIT (ROME, Meng et al. 2022)",
        "rule": "first 10 records in file order whose subject occurs in the prompt and whose "
        "GPT-2 small greedy next token is the first token of ' '+attribute",
        "records": records,
    }


def is_known(model: torch.nn.Module, tokenizer: Any, record: dict[str, Any]) -> bool:
    """Return whether GPT-2 small's greedy next token starts the record's attribute.

    Parameters
    ----------
    model:
        GPT-2 small.
    tokenizer:
        Its tokenizer.
    record:
        One known-facts record (``prompt``, ``attribute``).

    Returns
    -------
    bool
        ``True`` when the top-1 next token is the first token of ``" " + attribute``.
    """

    ids = tokenizer(record["prompt"], return_tensors="pt").input_ids
    target = tokenizer(" " + record["attribute"], add_special_tokens=False).input_ids[0]
    with torch.no_grad():
        logits = model(ids).logits
    return int(logits[0, -1].argmax()) == int(target)


def main() -> int:
    """Rewrite both data files from their pinned sources.

    Returns
    -------
    int
        Process exit code.
    """

    model, tokenizer = common.load_gpt2()
    for name, payload in (
        ("ioi_rows.json", ioi_rows(tokenizer)),
        ("known_facts.json", known_facts(model, tokenizer)),
    ):
        (common.DATA_DIR / name).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote data/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
