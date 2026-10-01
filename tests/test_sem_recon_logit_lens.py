"""Logit-lens 5.x slice handling + streaming prediction extractor (A03).

Two fixes pinned here:

* transformers 5.x causal LMs project only the last ``logits_to_keep``
  positions through the head, so the captured ``logits`` facet is
  sequence-sliced; lens validation now compares the reconstruction's exact
  suffix instead of refusing on shape (and still refuses WRONG
  reconstructions on sliced captures -- the tripwire survives the fix).
* ``logit_lens_predictions`` streams per-layer reductions (top-k, full-vocab
  logsumexp, requested-token values/ranks) without retaining any
  ``[batch, positions, vocab]`` projection, and only the row served from the
  captured output logits may say "native output".
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.semantic import FacetSpec
from torchlens.semantic.logit_lens import (
    PROVENANCE_NATIVE,
    PROVENANCE_PROJECTED,
    LogitLensError,
    logit_lens,
    logit_lens_predictions,
)

pytestmark = pytest.mark.smoke

VOCAB, D_MODEL, SEQ, KEEP = 11, 6, 5, 2


class LensBlock(nn.Module):
    """Residual toy block exposing resid_post."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.lin(x))


class SlicedLM(nn.Module):
    """Toy causal LM projecting only the LAST ``KEEP`` positions (logits_to_keep)."""

    def __init__(self) -> None:
        super().__init__()
        self.b0 = LensBlock()
        self.b1 = LensBlock()
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.b1(self.b0(x))
        return self.lm_head(self.ln_f(hidden)[:, -KEEP:, :])


class FullLM(SlicedLM):
    """Same weights, full-sequence projection (the pre-5.x shape)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.b1(self.b0(x))
        return self.lm_head(self.ln_f(hidden))


class WrongNormLM(SlicedLM):
    """Sliced LM whose head math the reconstruction does NOT capture."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.b1(self.b0(x))
        normed = self.ln_f(hidden) * 1.5  # nonstandard scaling
        return self.lm_head(normed[:, -KEEP:, :])


def _block_recipe(module: Any) -> dict[str, Any]:
    op = module.trace.ops[module.calls[0].output_ops[0]]
    return {"resid_post": FacetSpec.from_home(op, recipe_id="lens_toy_block")}


def _head_recipe(module: Any) -> dict[str, Any]:
    from torchlens.semantic.recipes._helpers import config_value, parameter_spec

    trace = module.trace
    head = trace.modules["lm_head"]
    norm = trace.modules["ln_f"]
    head_out = trace.ops[head.calls[0].output_ops[0]]
    result: dict[str, Any] = {
        "logits": FacetSpec.from_home(head_out, recipe_id="lens_toy_head"),
        "unembed_weight": parameter_spec(head, "weight", "lens_toy_head"),
        "final_norm_kind": "layer_norm",
        "final_norm_gamma": parameter_spec(norm, "weight", "lens_toy_head"),
        "final_norm_beta": parameter_spec(norm, "bias", "lens_toy_head"),
    }
    eps = config_value(norm, "eps", "variance_epsilon")
    if isinstance(eps, (int, float)):
        result["final_norm_eps"] = eps
    return result


_HEAD_FACETS = (
    "logits",
    "unembed_weight",
    "final_norm_kind",
    "final_norm_eps",
    "final_norm_gamma",
    "final_norm_beta",
)


@pytest.fixture(autouse=True, scope="module")
def _register_recipes() -> Iterator[None]:
    """Register the toy recipes at RUN time, restoring the registry after."""

    from torchlens.semantic import facets as _facets

    saved = list(_facets._REGISTRY)
    tl.facets.register(class_name="LensBlock", target_scope="module", facets=("resid_post",))(
        _block_recipe
    )
    for class_name in ("SlicedLM", "FullLM", "WrongNormLM"):
        tl.facets.register(class_name=class_name, target_scope="module", facets=_HEAD_FACETS)(
            _head_recipe
        )
    try:
        yield
    finally:
        _facets._REGISTRY[:] = saved
        _facets._REGISTRY_VERSION += 1


@pytest.fixture(scope="module")
def _models() -> dict[str, nn.Module]:
    """Build every toy variant once (same weights across variants)."""

    torch.manual_seed(11)
    sliced = SlicedLM()
    full = FullLM()
    full.load_state_dict(sliced.state_dict())
    wrong = WrongNormLM()
    wrong.load_state_dict(sliced.state_dict())
    return {"sliced": sliced, "full": full, "wrong": wrong}


def _capture(model: nn.Module) -> Any:
    torch.manual_seed(5)
    x = torch.randn(1, SEQ, D_MODEL)
    return tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )


def _manual_projection(log: Any, model: SlicedLM, address: str) -> torch.Tensor:
    hidden = log.modules[address].facets["resid_post"].value
    normed = torch.nn.functional.layer_norm(
        hidden, (D_MODEL,), model.ln_f.weight, model.ln_f.bias, model.ln_f.eps
    )
    return normed @ model.lm_head.weight.T


def test_sliced_capture_validates_and_projects(_models: dict[str, nn.Module]) -> None:
    """The flagship 5.x shape: sliced captured logits validate via suffix alignment."""

    log = _capture(_models["sliced"])
    try:
        result = logit_lens(log)
        assert result.validated
        assert result.lens_source == "model_head"
        assert tuple(result.final_logits.shape) == (1, KEEP, VOCAB)
        # Projections are full-sequence and numerically right.
        assert tuple(result.entries[-1].logits.shape) == (1, SEQ, VOCAB)
        manual = _manual_projection(log, _models["sliced"], "b1")
        assert torch.allclose(result.entries[-1].logits, manual, atol=1e-5)
        # The projected suffix reproduces the model's real sliced output.
        assert torch.allclose(
            result.entries[-1].logits[:, -KEEP:, :], result.final_logits, atol=1e-5
        )
    finally:
        log.cleanup()


def test_full_capture_still_validates(_models: dict[str, nn.Module]) -> None:
    """Regression: the exact-shape path is unchanged."""

    log = _capture(_models["full"])
    try:
        result = logit_lens(log)
        assert result.validated
        assert tuple(result.final_logits.shape) == (1, SEQ, VOCAB)
    finally:
        log.cleanup()


def test_wrong_reconstruction_on_sliced_capture_still_refuses(
    _models: dict[str, nn.Module],
) -> None:
    """The slice fix must not weaken the validation tripwire (LOCKED principle)."""

    log = _capture(_models["wrong"])
    try:
        with pytest.raises(LogitLensError, match="failed validation"):
            logit_lens(log)
    finally:
        log.cleanup()


def test_predictions_stream_math_and_provenance(_models: dict[str, nn.Module]) -> None:
    """Top-k, full-vocab denominator, one-based ranks, native-row honesty."""

    log = _capture(_models["sliced"])
    try:
        result = logit_lens(log)
        preds = logit_lens_predictions(log, k=3, tokens=[0, 5])
        assert preds.validated and preds.lens_source == "model_head"
        assert len(preds.rows) == 3  # b0, b1, native
        projected_rows = preds.rows[:2]
        native = preds.rows[-1]
        for row in projected_rows:
            assert row.provenance == PROVENANCE_PROJECTED
            assert row.positions == tuple(range(SEQ))
            assert tuple(row.top_ids.shape) == (1, SEQ, 3)
        # Native row: served from CAPTURED logits, covering only the kept suffix.
        assert native.provenance == PROVENANCE_NATIVE
        assert native.layer_index is None and native.facet is None
        assert native.positions == tuple(range(SEQ - KEEP, SEQ))
        captured = result.final_logits.float()
        assert torch.allclose(native.logsumexp, torch.logsumexp(captured, dim=-1), atol=1e-6)
        # Full-vocabulary denominator on a projected row.
        full_logits = result.entries[-1].logits.float()
        row = projected_rows[-1]
        assert torch.allclose(row.logsumexp, torch.logsumexp(full_logits, dim=-1), atol=1e-5)
        expected_top_probs = torch.exp(row.top_logits - row.logsumexp.unsqueeze(-1))
        assert torch.allclose(row.top_probs, expected_top_probs, atol=1e-6)
        # One-based ranks with the strictly-greater convention.
        token5 = row.token_ranks[..., 1]
        manual_rank = (full_logits > full_logits[..., 5:6]).sum(-1) + 1
        assert torch.equal(token5, manual_rank)
        assert "1 + count" in preds.tie_convention
    finally:
        log.cleanup()


def test_predictions_retain_no_vocab_width_tensors(_models: dict[str, nn.Module]) -> None:
    """FIX-L memory contract: no retained tensor keeps the vocabulary axis."""

    log = _capture(_models["sliced"])
    try:
        preds = logit_lens_predictions(log, k=2, tokens=[3])
        for row in preds.rows:
            for name in ("top_ids", "top_logits", "top_probs"):
                assert getattr(row, name).shape[-1] == 2, name
            for name in ("token_logits", "token_probs", "token_ranks"):
                assert getattr(row, name).shape[-1] == 1, name
            assert row.logsumexp.shape[-1] == len(row.positions)
            for name in (
                "top_ids",
                "top_logits",
                "top_probs",
                "logsumexp",
                "token_logits",
                "token_probs",
                "token_ranks",
            ):
                assert VOCAB not in tuple(getattr(row, name).shape), name
    finally:
        log.cleanup()


def test_predictions_tie_ranks_share_smallest_rank(_models: dict[str, nn.Module]) -> None:
    """The recorded tie convention is what the math actually does."""

    log = _capture(_models["sliced"])
    try:
        tied = torch.zeros(1, 1, VOCAB)
        tied[0, 0, 0] = 1.0  # unique top; all other tokens tied at 0
        preds = logit_lens_predictions(
            log, k=1, tokens=[1, 2], layers=["b0"], lens=lambda hidden: tied
        )
        row = preds.rows[0]
        # Both tied tokens rank 2 (one strictly-greater logit each).
        assert row.token_ranks[0, 0, 0] == 2
        assert row.token_ranks[0, 0, 1] == 2
        assert preds.lens_source == "user" and not preds.validated
    finally:
        log.cleanup()


def test_predictions_positions_normalize_and_refuse(_models: dict[str, nn.Module]) -> None:
    log = _capture(_models["sliced"])
    try:
        preds = logit_lens_predictions(log, k=1, positions=[-1, 0])
        projected = preds.rows[0]
        assert projected.positions == (SEQ - 1, 0)
        native = preds.rows[-1]
        # The native row covers only the kept suffix; position 0 drops from it.
        assert native.positions == (SEQ - 1,)
        with pytest.raises(LogitLensError, match="outside the sequence") as position_exc:
            logit_lens_predictions(log, k=1, positions=[SEQ + 3])
        assert position_exc.value.fields["code"] == "logit_lens_position_invalid"
    finally:
        log.cleanup()


def test_predictions_token_and_k_refusals(_models: dict[str, nn.Module]) -> None:
    log = _capture(_models["sliced"])
    try:
        with pytest.raises(LogitLensError, match="outside the vocabulary") as token_exc:
            logit_lens_predictions(log, k=1, tokens=[VOCAB + 4])
        assert token_exc.value.fields["code"] == "logit_lens_token_id_invalid"
        with pytest.raises(LogitLensError, match="positive integer") as k_exc:
            logit_lens_predictions(log, k=0)
        assert k_exc.value.fields["code"] == "logit_lens_k_invalid"
    finally:
        log.cleanup()


def test_predictions_projection_rank_refusal(_models: dict[str, nn.Module]) -> None:
    """A rank-1 lens projection refuses typed instead of mis-extracting."""

    log = _capture(_models["sliced"])
    try:
        with pytest.raises(LogitLensError, match="at least") as rank_exc:
            logit_lens_predictions(log, k=1, lens=lambda hidden: hidden.reshape(-1)[:1])
        assert rank_exc.value.fields["code"] == "logit_lens_projection_rank_invalid"
    finally:
        log.cleanup()
