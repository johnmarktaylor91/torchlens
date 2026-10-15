"""``head_scores(..., kind="induction", period=n)``: the fixed-period induction mask.

On ``[bos] + R + R`` the score of every head equals its mean attention from
each second-copy query to the token after that token's first occurrence,
read straight off the captured pattern, even when a token repeats inside
``R`` (where the token-equality mask adds rows and sources). Float64
patterns are scored in float64. A bad ``period=`` refuses typed and the
remedy runs.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.mechinterp._errors import MechInterpError

pytest.importorskip("transformers")
pytestmark = pytest.mark.heavy


@pytest.fixture(scope="module")
def repeated():
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        n_layer=2, n_head=2, n_embd=8, vocab_size=64, n_positions=32, use_cache=False
    )
    config._attn_implementation = "eager"
    torch.manual_seed(1)
    model = GPT2LMHeadModel(config).eval().double()
    rep = torch.tensor([[5, 9, 12, 9, 30, 41], [7, 8, 20, 33, 2, 50]])  # row 0 repeats 9 inside R
    ids = torch.cat([torch.full((2, 1), 63), rep, rep], dim=1)
    trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save="all"))
    yield trace
    trace.cleanup()


def _oracle(trace, period: int) -> torch.Tensor:
    rows = []
    for module in mi._heads._attention_modules(trace):
        pattern = module.facets["pattern"]
        pattern = pattern.value if hasattr(pattern, "value") else pattern
        q = torch.arange(1 + period, 1 + 2 * period)
        rows.append(pattern[:, :, q, q - period + 1].mean(dim=(0, 2)))
    return torch.stack(rows)


def test_period_mask_is_the_positional_induction_score(repeated) -> None:
    result = mi.head_scores(repeated, kind="induction", period=6)
    got = torch.tensor(result.scores, dtype=torch.float64)
    torch.testing.assert_close(got, _oracle(repeated, 6), rtol=0, atol=1e-12)
    assert result.eligible_rows == (12, 12)
    assert result.provenance["mask_definition"].startswith("fixed period 6")
    token_equality = mi.head_scores(repeated, kind="induction")
    assert token_equality.eligible_rows != result.eligible_rows  # the repeat inside R adds rows
    appliance = repeated.using("mi").head_scores(kind="induction", period=6)
    assert appliance.scores == result.scores


@pytest.mark.parametrize(
    ("kind", "period"), [("duplicate_token", 6), ("induction", 0), ("induction", 2.0)]
)
def test_a_bad_period_refuses_and_the_remedy_runs(repeated, kind, period) -> None:
    with pytest.raises(MechInterpError) as excinfo:
        mi.head_scores(repeated, kind=kind, period=period)
    assert excinfo.value.fields["code"] == "mi_head_score_period_invalid"
    assert mi.head_scores(repeated, kind="induction", period=6).eligible_rows == (12, 12)
    assert mi.head_scores(repeated, kind=kind).kind == kind
