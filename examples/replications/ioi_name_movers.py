"""Replication: the IOI name-mover heads in GPT-2 small (Wang et al. 2022).

Run: ``python examples/replications/ioi_name_movers.py [--out DIR]``.
Downloads GPT-2 small (public, about 550 MB) at a pinned revision on first use; the
prompts are vendored in ``data/ioi_rows.json`` (MIB IOI, MIT).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import _common as common  # the sibling module; scripts here run from this folder
import torch

import torchlens as tl
import torchlens.mechinterp

CLAIM = (
    "Wang et al. 2022 ('Interpretability in the Wild: a Circuit for Indirect Object "
    "Identification in GPT-2 small', arXiv 2211.00593, Section 3.1 and Figure 3b): the Name "
    "Mover Heads 9.9, 9.6 and 10.0 have the largest positive direct effect on the IO-minus-S "
    "logit difference at the final position. Section 2 reports a mean logit difference of "
    "3.56 (IO over S 99.3% of the time) on the paper's own 100,000-prompt sample."
)
RULE = (
    "replicated when heads 9.9, 9.6 and 10.0 are all among the top 5 of the 144 heads ranked "
    "by mean direct logit contribution to logit(IO) - logit(S) over the 50 prompts"
)
NAME_MOVERS = ("9.9", "9.6", "10.0")
TOP_K = 5
PAPER_LOGIT_DIFF = 3.56
PAPER_IO_OVER_S = 0.993
_HEAD_LABEL = re.compile(r"h\.(\d+)\..*?(\d+)$")


def head_name(label: str) -> str | None:
    """Turn a head row's label into ``"layer.head"``; ``None`` for non-head rows.

    Parameters
    ----------
    label:
        A ``full_decomposition`` row label.

    Returns
    -------
    str or None
        ``"layer.head"`` for a head row.
    """

    match = _HEAD_LABEL.search(label)
    return f"{match.group(1)}.{match.group(2)}" if match else None


def prompt_contributions(
    model: torch.nn.Module, tokenizer: Any, row: dict[str, Any]
) -> tuple[dict[str, float], float]:
    """Direct logit contribution of every head for one prompt, plus its clean logit diff.

    Parameters
    ----------
    model:
        GPT-2 small.
    tokenizer:
        Its tokenizer.
    row:
        One IOI row (``prompt``, ``indirect_object``, ``subject``).

    Returns
    -------
    tuple
        ``({"layer.head": contribution}, native logit difference)`` at the last position.
    """

    ids = tokenizer(row["prompt"], return_tensors="pt").input_ids
    answer = common.single_token(tokenizer, row["indirect_object"])
    versus = common.single_token(tokenizer, row["subject"])
    trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save="all"))
    try:
        scores = torchlens.mechinterp.direct_logit_contributions(
            trace, torchlens.mechinterp.full_decomposition(trace), answer=answer, vs=versus
        )
    finally:
        trace.cleanup()
    heads = {}
    for coordinate, value in zip(scores.coordinates, scores.values, strict=True):
        name = head_name(coordinate.label) if coordinate.kind == "attention" else None
        if name is not None:
            heads[name] = float(value.detach().reshape(-1)[0])
    return heads, float(scores.native.detach().reshape(-1)[0])


def run(out_dir: Path | None = None) -> dict[str, Any]:
    """Rank the heads by mean direct logit contribution and report the verdict.

    Parameters
    ----------
    out_dir:
        Where to write the JSON summary and the ``.tlspec``; ``None`` prints only.

    Returns
    -------
    dict
        The printed summary (claim, rule, measured numbers, verdict).
    """

    data = common.load_data("ioi_rows.json")
    rows = [row for row in data["rows"] if row["kept"]]
    model, tokenizer = common.load_gpt2()
    totals: dict[str, float] = {}
    diffs = []
    for row in rows:
        heads, diff = prompt_contributions(model, tokenizer, row)
        for name, value in heads.items():
            totals[name] = totals.get(name, 0.0) + value
        diffs.append(diff)
    means = {name: total / len(rows) for name, total in totals.items()}
    ranked = sorted(means, key=lambda name: means[name], reverse=True)
    top = ranked[:TOP_K]
    passed = len(means) == 144 and all(name in top for name in NAME_MOVERS)
    _save_first_prompt(model, tokenizer, rows[0], out_dir)
    summary = {
        "replication": "ioi_name_movers",
        "claim": CLAIM,
        "rule": RULE,
        "model": f"{common.MODEL_ID}@{common.MODEL_REVISION} float32 eager",
        "data": {
            "source": data["source"],
            "source_sha256": common.IOI_SHA256,
            "n_prompts": len(rows),
            "rows_sha256": common.rows_sha256(rows),
        },
        "measured": {
            "n_heads_ranked": len(means),
            "top5_heads": [[name, round(means[name], 4)] for name in top],
            "name_mover_ranks": {name: ranked.index(name) + 1 for name in NAME_MOVERS},
            "mean_logit_diff": round(sum(diffs) / len(diffs), 4),
            "paper_mean_logit_diff": PAPER_LOGIT_DIFF,
            "fraction_io_over_s": round(sum(diff > 0 for diff in diffs) / len(diffs), 4),
            "paper_fraction_io_over_s": PAPER_IO_OVER_S,
        },
        "verdict": common.REPLICATED if passed else common.NOT_REPLICATED,
    }
    return common.report(summary, out_dir, "ioi_name_movers")


def _save_first_prompt(
    model: torch.nn.Module, tokenizer: Any, row: dict[str, Any], out_dir: Path | None
) -> None:
    """Write a portable ``.tlspec`` of the first prompt's run."""

    if out_dir is None:
        return
    ids = tokenizer(row["prompt"], return_tensors="pt").input_ids
    trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save="all"))
    try:
        common.save_trace(trace, out_dir, "ioi_name_movers")
    finally:
        trace.cleanup()


def main(argv: list[str] | None = None) -> int:
    """Command-line entry: run the replication and exit 0 whatever the verdict.

    Parameters
    ----------
    argv:
        Arguments (``--out DIR``).

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args(argv)
    run(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
