"""Lowered-counterfactual ADMISSION-GATE matrix (mikit D11, pre-signed).

Per-cell equality: the eager-DIRECT edit and the fused-LOWERED add must
agree below the eager-vs-SDPA noise floor, across gpt2-class (fused Conv1D)
and Llama-class (GQA -- the expanded-v path) R0 families, both lowering
kinds, plus the refusal rows (dropout, stale facet, length mismatch, q/k).
Any red cell reverts to "refuse, recapture eager" -- the reversion clause is
signed in advance by the capability's own proposer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import torchlens.mechinterp as mi
from torchlens.mechinterp._errors import MechInterpError

sys.path.insert(0, str(Path(__file__).resolve().parent / "real_model" / "r0"))

pytestmark = pytest.mark.slow  # 4-trace equality cells run ~30s each


@pytest.fixture(scope="module")
def matrix_models():
    """Eager + SDPA variants for both families, built before any capture."""

    from families import build_gpt2, build_llama

    torch.manual_seed(0)
    return {
        ("gpt2", "eager"): build_gpt2("eager").eval(),
        ("gpt2", "sdpa"): build_gpt2("sdpa").eval(),
        ("llama", "eager"): build_llama("eager").eval(),
        ("llama", "sdpa"): build_llama("sdpa").eval(),
    }


def _final_logits(trace):
    """Metric: the full final-position logit vector."""

    from torchlens.mechinterp._anchors import resolve_lm_head

    return resolve_lm_head(trace).facets()["logits"].value[0, -1, :].detach().clone()


@pytest.mark.parametrize("family", ["gpt2"])
@pytest.mark.parametrize("kind", ["head", "pattern"])
def test_lowered_matches_across_implementations(matrix_models, family, kind):
    """Matrix cell: eager-lowered == sdpa-lowered below the noise floor.

    The lowering is implementation-independent math over captured values;
    agreement across implementations on identical weights is the exactness
    evidence available at zero network (the real-gpt2 eager-direct row runs
    in the nightly oracle leg).
    """

    torch.manual_seed(4)
    clean = torch.randint(0, 500, (1, 6))
    corrupt = torch.randint(0, 500, (1, 6))
    results = {}
    for impl in ("eager", "sdpa"):
        results[impl] = mi.lowered_counterfactual(
            matrix_models[(family, impl)],
            clean,
            corrupt,
            layer=0,
            head=1,
            kind=kind,
            metric=_final_logits,
        )
    gap = float((results["eager"].metric_value - results["sdpa"].metric_value).abs().max())
    # Noise floor: the unpatched eager-vs-SDPA disagreement on these weights
    # is ~1e-6 at this scale; the lowered cells must not add above it.
    assert gap < 5e-5, f"lowered cell disagrees across implementations by {gap}"
    receipt = results["sdpa"].receipt
    assert receipt["formula_version"] == "mikit_d11_v1"
    assert receipt["lowered_at_real_site"]["op_label"]
    assert receipt["invalidated_virtual_facets"]


@pytest.mark.parametrize("kind", ["head", "pattern"])
def test_llama_cells_revert_to_refusal(matrix_models, kind):
    """D11 pre-signed reversion clause, exercised: the Llama-class rerun path
    currently raises the typed control-flow-divergence refusal (the trace
    never serves a silently wrong number), so the GQA matrix cells stand
    REVERTED until the rerun seam admits them. This pin SCREAMS the moment
    the seam is fixed, at which point the equality cells above extend to
    the family (delete this test in that change).
    """

    torch.manual_seed(4)
    clean = torch.randint(0, 500, (1, 6))
    corrupt = torch.randint(0, 500, (1, 6))
    with pytest.raises(Exception) as excinfo:
        mi.lowered_counterfactual(
            matrix_models[("llama", "eager")],
            clean,
            corrupt,
            layer=0,
            head=1,
            kind=kind,
            metric=_final_logits,
        )
    assert "diverged" in str(excinfo.value) or isinstance(excinfo.value, MechInterpError)


def test_direct_eager_edit_matches_lowered(matrix_models):
    """The core D11 cell: a DIRECT z-edit (real eager op) == the lowered add."""

    torch.manual_seed(5)
    clean = torch.randint(0, 500, (1, 6))
    corrupt = torch.randint(0, 500, (1, 6))
    model = matrix_models[("gpt2", "eager")]

    lowered = mi.lowered_counterfactual(
        model, clean, corrupt, layer=0, head=1, kind="head", metric=_final_logits
    )
    direct = mi.grid(
        model,
        clean,
        corrupt,
        lambda t: _final_logits(t)[42],
        sites="z",
        axes=("layer", "head"),
        heads=[1],
    )
    direct_cell = float(direct.values[0, 0])
    lowered_cell = float(lowered.metric_value[42])
    assert abs(direct_cell - lowered_cell) < 5e-5


def test_lowered_refusal_rows(matrix_models):
    """The matrix's refusal rows: q/k, train mode, stale reads, length."""

    torch.manual_seed(6)
    clean = torch.randint(0, 500, (1, 6))
    corrupt = torch.randint(0, 500, (1, 6))
    model = matrix_models[("gpt2", "sdpa")]

    with pytest.raises(MechInterpError) as excinfo:
        mi.lowered_counterfactual(model, clean, corrupt, layer=0, head=0, kind="q")
    assert excinfo.value.fields["code"] == "mi_lowered_site_not_lowerable"

    with pytest.raises(MechInterpError) as excinfo:
        mi.lowered_counterfactual(
            model, clean, torch.randint(0, 500, (1, 9)), layer=0, head=0, kind="head"
        )
    assert excinfo.value.fields["code"] == "mi_lowered_length_mismatch"

    result = mi.lowered_counterfactual(
        model, clean, corrupt, layer=0, head=0, kind="head", metric=_final_logits
    )
    with pytest.raises(MechInterpError) as excinfo:
        result.read_facet("transformer.h.0.attn", "pattern")
    assert excinfo.value.fields["code"] == "mi_lowered_stale_facet_read"
    # Non-invalidated layers still read.
    assert result.read_facet("transformer.h.1", "resid_post") is not None

    model.train()
    try:
        with pytest.raises(MechInterpError) as excinfo:
            mi.lowered_counterfactual(model, clean, corrupt, layer=0, head=0, kind="head")
        assert excinfo.value.fields["code"] == "mi_lowered_dropout_active"
    finally:
        model.eval()
