"""Lowered-counterfactual ADMISSION-GATE matrix (mikit D11, pre-signed).

Per-cell equality: the eager-DIRECT edit and the fused-LOWERED add must
agree below the eager-vs-SDPA noise floor, across gpt2-class (fused Conv1D)
and Llama-class (GQA -- the expanded-v path) R0 families, both lowering
kinds, plus the refusal rows (dropout, stale facet, length mismatch, q/k).
Any red cell reverts to "refuse, recapture eager" -- the reversion clause is
signed in advance by the capability's own proposer. The Llama cells pin the
rotary-embedding body per instance: the direct-buffer form (transformers>=5.19)
is admitted, the buffer-view form (transformers<=5.18) stays reverted.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.intervention.errors import ControlFlowDivergenceWarning
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


def _buffer_view_rotary_forward(self, x, position_ids):
    """transformers<=5.18 ``LlamaRotaryEmbedding.forward`` body (buffer VIEW form).

    ``inv_freq[None, :, None].expand(...)`` hands a storage-sharing view of the
    ``inv_freq`` buffer to the next op. The patched rerun of a lowered
    counterfactual logs two extra (1, 8, 1) source events for those views, so
    the raw-event shape hash diverges (the open rerun-seam gap).
    """

    inv_freq = (
        self.inv_freq[None, :, None]
        .expand(position_ids.shape[0], -1, 1)
        .to(dtype=torch.float, device=x.device)
    )
    freqs = (inv_freq @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _direct_buffer_rotary_forward(self, x, position_ids):
    """transformers>=5.19 ``LlamaRotaryEmbedding.forward`` body (direct buffer form).

    Mathematically identical rotary embedding; the buffer is read whole
    (``inv_freq.to(...)``) with no storage-sharing view, so the rerun seam
    admits it.
    """

    freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


_ROTARY_FORMS = {
    "buffer_view": _buffer_view_rotary_forward,
    "direct_buffer": _direct_buffer_rotary_forward,
}


@pytest.fixture(scope="module")
def llama_rotary_models():
    """Llama (GQA) models with the rotary body PINNED, independent of transformers.

    The installed transformers decides which rotary body a stock Llama runs
    (5.18 ships the buffer-view form, 5.19 the direct-buffer form), and only
    the buffer-view form trips the rerun seam. Pinning the body per instance
    (a subclass swapped onto ``model.model.rotary_emb`` before any capture)
    makes both the admitted cells and the reversion pin hold on every
    transformers release instead of flipping with the installed version.
    """

    from families import build_llama
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    models = {}
    for form, body in _ROTARY_FORMS.items():
        rotary_cls = type(
            f"_{form.title().replace('_', '')}Rotary",
            (LlamaRotaryEmbedding,),
            {"forward": torch.no_grad()(body)},
        )
        for impl in ("eager", "sdpa"):
            model = build_llama(impl).eval()
            model.model.rotary_emb.__class__ = rotary_cls
            models[(form, impl)] = model
    return models


def _strict_lowered(model, clean, corrupt, **kwargs):
    """Run a lowered cell with control-flow divergence escalated to an error."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", ControlFlowDivergenceWarning)
        return mi.lowered_counterfactual(model, clean, corrupt, **kwargs)


def test_llama_rotary_forms_agree(llama_rotary_models):
    """Both pinned rotary bodies compute the same model (the pin changes no math)."""

    torch.manual_seed(4)
    tokens = torch.randint(0, 500, (1, 6))
    with torch.no_grad():
        view = llama_rotary_models[("buffer_view", "eager")](tokens).logits
        direct = llama_rotary_models[("direct_buffer", "eager")](tokens).logits
    assert float((view - direct).abs().max()) < 1e-5


@pytest.mark.parametrize("kind", ["head", "pattern"])
def test_llama_lowered_matches_across_implementations(llama_rotary_models, kind):
    """GQA matrix cell (the expanded-v path): eager-lowered == sdpa-lowered.

    The direct-buffer rotary body (transformers>=5.19) is admitted by the
    rerun seam: no control-flow divergence, and the lowered cells agree across
    implementations below the same noise floor as the gpt2 cells.
    """

    torch.manual_seed(4)
    clean = torch.randint(0, 500, (1, 6))
    corrupt = torch.randint(0, 500, (1, 6))
    results = {
        impl: _strict_lowered(
            llama_rotary_models[("direct_buffer", impl)],
            clean,
            corrupt,
            layer=0,
            head=1,
            kind=kind,
            metric=_final_logits,
        )
        for impl in ("eager", "sdpa")
    }
    gap = float((results["eager"].metric_value - results["sdpa"].metric_value).abs().max())
    assert gap < 5e-5, f"llama lowered cell disagrees across implementations by {gap}"
    receipt = results["sdpa"].receipt
    assert receipt["formula_version"] == "mikit_d11_v1"
    assert receipt["lowered_at_real_site"]["op_label"]
    assert receipt["invalidated_virtual_facets"]


def test_llama_direct_eager_edit_matches_lowered(llama_rotary_models):
    """The core D11 cell on the GQA family: a DIRECT z-edit == the lowered add.

    The capture itself also passes forward validation on both
    implementations, so the admitted cells rest on a validated trace.
    """

    torch.manual_seed(5)
    clean = torch.randint(0, 500, (1, 6))
    corrupt = torch.randint(0, 500, (1, 6))
    model = llama_rotary_models[("direct_buffer", "eager")]

    for impl in ("eager", "sdpa"):
        assert tl.validate(llama_rotary_models[("direct_buffer", impl)], clean, scope="forward")
    lowered = _strict_lowered(
        model, clean, corrupt, layer=0, head=1, kind="head", metric=_final_logits
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ControlFlowDivergenceWarning)
        direct = mi.grid(
            model,
            clean,
            corrupt,
            lambda t: _final_logits(t)[42],
            sites="z",
            axes=("layer", "head"),
            heads=[1],
        )
    assert abs(float(direct.values[0, 0]) - float(lowered.metric_value[42])) < 5e-5


@pytest.mark.parametrize("kind", ["head", "pattern"])
def test_llama_buffer_view_rotary_reverts_to_refusal(llama_rotary_models, kind):
    """D11 pre-signed reversion clause, exercised: the buffer-view rotary body
    (transformers<=5.18) still trips the rerun seam, so its cells raise the
    typed control-flow-divergence refusal (the trace never serves a silently
    wrong number). This pin SCREAMS the moment the seam is fixed, at which
    point the buffer-view form joins the admitted cells above (delete this
    test in that change).
    """

    torch.manual_seed(4)
    clean = torch.randint(0, 500, (1, 6))
    corrupt = torch.randint(0, 500, (1, 6))
    with pytest.raises(ControlFlowDivergenceWarning, match="diverged"):
        _strict_lowered(
            llama_rotary_models[("buffer_view", "eager")],
            clean,
            corrupt,
            layer=0,
            head=1,
            kind=kind,
            metric=_final_logits,
        )


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
