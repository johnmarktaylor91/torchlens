"""Induction heads on GPT-2 small against an independent oracle (pre-registered bar CIRC-3a).

The oracle (``data/circ3_induction_oracle.json``, read here as data: the
oracle file (sha256 in ``_SOURCE_SHA256``) minus its ``model_revision``
field, which is the r1-gpt2 artifact's revision; computed
with Hugging Face eager attention and no TorchLens): GPT-2 small at revision
607a30d7, ``[bos] + R + R`` with ``R`` 50 random tokens in ``[1000, 50000)``
drawn by a ``torch.Generator`` seeded 0, batch 8; a head's score is its mean
attention from each second-copy query to the token after that token's first
occurrence. With the model in float64, ``tl.mechinterp.head_scores(...,
kind="induction")`` equals the oracle's float64 scores within 1e-9 absolute
for all 144 heads and its top 5 heads are the oracle's top 5; float32 is held
to a 1e-4 ceiling against the same scores.

The scores come from ``head_scores(..., kind="induction", period=50)``, the
fixed-period mask (destination ``i`` on the one source ``i - 49``, counted
when its token equals the token at ``i - 50``), which is the oracle's
definition. The default token-equality mask differs on this input: one row
of ``R`` repeats a token, so it adds a scored row and extra sources (401
eligible rows instead of 400); the protocol test pins that repeat.
Runs only in the preflighted offline venue (GATE-ID R1_OFFLINE_VENUE).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.real_model, pytest.mark.real_checkpoint, pytest.mark.slow]

_ORACLE_PATH = Path(__file__).parent / "data" / "circ3_induction_oracle.json"
_SOURCE_SHA256 = (
    "7b80017f020b0edc5599bc94a77cfd75f6d4e21e907a6a14e43490dfd962ec93"  # pragma: allowlist secret
)
_ORACLE_SHA256 = (
    "90fcfa62802a26d8b78d6046bebc0f50191f6398dfab48c85618a5d855aecfbf"  # pragma: allowlist secret
)
_ORACLE = json.loads(_ORACLE_PATH.read_text())


def _tokens() -> torch.Tensor:
    protocol = _ORACLE["protocol"]
    low, high = protocol["token_range"]
    generator = torch.Generator().manual_seed(protocol["seed"])
    repeat = torch.randint(
        low, high, (protocol["batch"], protocol["repeat_len"]), generator=generator
    )
    bos = torch.full((protocol["batch"], 1), 50256)
    return torch.cat([bos, repeat, repeat], dim=1)


def _scores(model, dtype: torch.dtype):
    import torchlens as tl

    model = copy.deepcopy(model).to(dtype).eval()
    model.set_attn_implementation("eager")
    trace = tl.trace(model, _tokens(), capture=tl.options.CaptureOptions(layers_to_save="all"))
    try:
        return tl.mechinterp.head_scores(trace, kind="induction", measure="mul", period=50)
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def gpt2(r1_loader):
    return r1_loader("r1-gpt2")["model"]


def test_oracle_file_and_protocol_are_the_pinned_ones() -> None:
    assert hashlib.sha256(_ORACLE_PATH.read_bytes()).hexdigest() == _ORACLE_SHA256
    tokens = _tokens()
    assert (
        tokens[:, : len(_ORACLE["tokens_sha_preview"])].flatten().tolist()[:8]
        == _ORACLE["tokens_sha_preview"]
    )
    repeat = tokens[:, 1:51]
    assert sum(len(set(row.tolist())) < 50 for row in repeat) == 1  # one row repeats a token in R


def test_float64_induction_scores_equal_the_oracle(gpt2) -> None:
    result = _scores(gpt2, torch.float64)
    want = torch.tensor(_ORACLE["scores_float64"], dtype=torch.float64)
    got = torch.tensor(result.scores, dtype=torch.float64)
    assert got.shape == want.shape == (12, 12)
    assert result.eligible_rows == (8 * 50,) * 12
    gap = float((got - want).abs().max())
    print(f"CIRC-3a float64 max |TorchLens - oracle| = {gap:.3e}")
    assert gap <= 1e-9
    flat = sorted(
        ((float(got[layer, h]), layer, h) for layer in range(12) for h in range(12)), reverse=True
    )
    mine = [f"{layer}.{h}" for _, layer, h in flat[:5]]
    print(f"CIRC-3a top 5: {mine}")
    assert mine == [row["head"] for row in _ORACLE["top10_float64"][:5]]


def test_float32_induction_scores_stay_within_the_ceiling(gpt2) -> None:
    result = _scores(gpt2, torch.float32)
    want = torch.tensor(_ORACLE["scores_float64"], dtype=torch.float64)
    gap = float((torch.tensor(result.scores, dtype=torch.float64) - want).abs().max())
    print(f"CIRC-3a float32 max |TorchLens - oracle float64| = {gap:.3e}")
    assert gap <= 1e-4
