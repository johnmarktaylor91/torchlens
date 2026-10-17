"""Replication: induction heads in GPT-2 small (Olsson et al. 2022).

Run: ``python examples/replications/induction_heads.py [--out DIR]``.
Downloads GPT-2 small (public, about 550 MB) at a pinned revision on first use.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import _common as common  # the sibling module; scripts here run from this folder
import torch

import torchlens as tl

CLAIM = (
    "Olsson et al. 2022 ('In-context Learning and Induction Heads', Transformer Circuits "
    "Thread): induction heads attend from a token to the token right after its earlier "
    "occurrence. On GPT-2 small, the heads with the top prefix-matching scores on repeated "
    "random tokens are 5.5, 6.9 and 7.10 (independent Hugging Face eager-attention oracle)."
)
RULE = (
    "replicated when TorchLens's top 3 heads, in order, equal the oracle's top 3 and every "
    "one of the 144 float32 scores is within 1e-4 of the oracle's float64 score"
)
TOP_K = 3
SCORE_TOLERANCE = 1e-4
PERIOD = 50


def induction_tokens(protocol: dict[str, Any]) -> torch.Tensor:
    """Build the oracle's input: ``[bos] + R + R`` with ``R`` seeded random tokens.

    Parameters
    ----------
    protocol:
        The oracle file's ``protocol`` block (seed, batch, repeat length, token range).

    Returns
    -------
    torch.Tensor
        Token ids, shape ``[batch, 1 + 2 * repeat_len]``.
    """

    low, high = protocol["token_range"]
    generator = torch.Generator().manual_seed(protocol["seed"])
    repeat = torch.randint(
        low, high, (protocol["batch"], protocol["repeat_len"]), generator=generator
    )
    bos = torch.full((protocol["batch"], 1), 50256)
    return torch.cat([bos, repeat, repeat], dim=1)


def ranked_heads(scores: torch.Tensor) -> list[str]:
    """Return ``"layer.head"`` names ordered by score, highest first.

    Parameters
    ----------
    scores:
        ``[layers, heads]`` score grid.

    Returns
    -------
    list of str
        Every head, ranked.
    """

    flat = [
        (float(scores[layer, head]), layer, head)
        for layer in range(scores.shape[0])
        for head in range(scores.shape[1])
    ]
    return [f"{layer}.{head}" for _, layer, head in sorted(flat, reverse=True)]


def run(out_dir: Path | None = None) -> dict[str, Any]:
    """Measure the induction scores, compare them with the oracle and report.

    Parameters
    ----------
    out_dir:
        Where to write the JSON summary and the ``.tlspec``; ``None`` prints only.

    Returns
    -------
    dict
        The printed summary (claim, rule, measured numbers, verdict).
    """

    oracle = common.load_data("induction_oracle.json")
    model, _tokenizer = common.load_gpt2()
    tokens = induction_tokens(oracle["protocol"])
    trace = tl.trace(model, tokens, capture=tl.options.CaptureOptions(layers_to_save="all"))
    try:
        result = tl.mechinterp.head_scores(
            trace, kind=tl.mechinterp.FixedPeriodInduction(period=PERIOD), measure="mul"
        )
    finally:
        trace.cleanup()
    got = torch.tensor(result.scores, dtype=torch.float64)
    want = torch.tensor(oracle["scores_float64"], dtype=torch.float64)
    gap = float((got - want).abs().max())
    mine = ranked_heads(got)[:TOP_K]
    theirs = ranked_heads(want)[:TOP_K]
    passed = mine == theirs and gap <= SCORE_TOLERANCE
    _save_small_trace(model, tokens, out_dir)
    summary = {
        "replication": "R-IND",
        "claim": CLAIM,
        "rule": RULE,
        "model": f"{common.MODEL_ID}@{common.MODEL_REVISION} float32 eager",
        "data": {
            "protocol": oracle["protocol"],
            "tokens_sha256": common.tensor_sha256(tokens),
            "eligible_rows_per_layer": list(result.eligible_rows),
        },
        "measured": {
            "top3_torchlens": mine,
            "top3_oracle": theirs,
            "max_abs_gap_vs_oracle": gap,
            "top5_torchlens_scores": [
                [name, round(float(got[int(name.split(".")[0]), int(name.split(".")[1])]), 6)]
                for name in ranked_heads(got)[:5]
            ],
        },
        "verdict": common.REPLICATED if passed else common.NOT_REPLICATED,
    }
    return common.report(summary, out_dir, "induction_heads")


def _save_small_trace(model: torch.nn.Module, tokens: torch.Tensor, out_dir: Path | None) -> None:
    """Write a portable ``.tlspec`` of the first sequence's run (one row keeps it small).

    Parameters
    ----------
    model:
        The loaded GPT-2 small.
    tokens:
        The full protocol batch; only row 0 is traced again.
    out_dir:
        Output directory, or ``None``.
    """

    if out_dir is None:
        return
    trace = tl.trace(model, tokens[:1], capture=tl.options.CaptureOptions(layers_to_save="all"))
    try:
        common.save_trace(trace, out_dir, "induction_heads")
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
