"""Phase 4 facet patching helper tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.semantic import FacetSpec


class HeadResult(nn.Module):
    """Return two explicit head-result tensors."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stack one important head and one inert head."""

        return torch.stack((x, torch.zeros_like(x)), dim=2)


class PatchableAttention(nn.Module):
    """Tiny attention-like module with a writable per-head result facet."""

    def __init__(self) -> None:
        """Initialize the result-producing child."""

        super().__init__()
        self.n_heads = 2
        self.result_source = HeadResult()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum per-head residual contributions."""

        return self.result_source(x).sum(dim=2)


class PatchableMLP(nn.Module):
    """Tiny MLP-like module with a simple output facet."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a small linear contribution."""

        return 0.25 * x


class ToyTransformerBlock(nn.Module):
    """Transformer-like block with attention, MLP, and residual adds."""

    def __init__(self) -> None:
        """Initialize child modules."""

        super().__init__()
        self.attn = PatchableAttention()
        self.mlp = PatchableMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention and MLP residual updates."""

        resid_mid = x + self.attn(x)
        return resid_mid + self.mlp(resid_mid)


class PatchingToyModel(nn.Module):
    """Wrapper exposing a named transformer block."""

    def __init__(self) -> None:
        """Initialize the block."""

        super().__init__()
        self.block = ToyTransformerBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the toy block."""

        return self.block(x)


@tl.facets.register(
    class_name="PatchableAttention",
    target_scope="module",
    facets=("result", "attn_out", "n_heads", "head"),
)
def patchable_attention(module: Any) -> dict[str, Any]:
    """Expose writable attention-output facets for the toy module."""

    trace = module.trace
    result_module = trace.modules[f"{module.address}.result_source"]
    result_op = trace.ops[result_module.calls[0].output_ops[0]]
    output_op = trace.ops[module.calls[0].output_ops[0]]
    return {
        "result": FacetSpec.from_home(result_op, recipe_id="patchable_attention"),
        "attn_out": FacetSpec.from_home(output_op, recipe_id="patchable_attention"),
        "n_heads": 2,
        "head": module.facets.head,
    }


@tl.facets.register(
    class_name="PatchableMLP",
    target_scope="module",
    facets=("output", "up_out"),
)
def patchable_mlp(module: Any) -> dict[str, Any]:
    """Expose writable MLP output facets for the toy module."""

    output_op = module.trace.ops[module.calls[0].output_ops[0]]
    spec = FacetSpec.from_home(output_op, recipe_id="patchable_mlp")
    return {"output": spec, "up_out": spec}


def _metric(log: Any) -> torch.Tensor:
    """Return the scalar metric used by patching tests."""

    return log[log.output_layers[0]].out[0, 0, 0]


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return clean and corrupted toy inputs."""

    clean = torch.zeros(1, 3, 4)
    corrupted = torch.zeros(1, 3, 4)
    clean[0, 0, 0] = 2.0
    return clean.requires_grad_(), corrupted.requires_grad_()


def test_activation_patch_attention_heads_identifies_important_head() -> None:
    """Activation patching returns ``[layer, head]`` and finds the important head."""

    model = PatchingToyModel()
    clean, corrupted = _inputs()

    scores = tl.facets.patching.activation_patch_attention_heads(model, clean, corrupted, _metric)

    assert scores.shape == (1, 2)
    assert scores[0, 0] > 2.0
    assert torch.isclose(scores[0, 1], torch.tensor(0.0), atol=1e-6)


def test_activation_patch_head_cell_matches_manual_patch() -> None:
    """One helper cell matches an explicit clean-into-corrupted facet patch."""

    model = PatchingToyModel()
    clean, corrupted = _inputs()
    scores = tl.facets.patching.activation_patch_attention_heads(model, clean, corrupted, _metric)
    clean_log = tl.trace(model, clean, layers_to_save="all", save_arg_values=True)
    corrupted_log = tl.trace(model, corrupted, layers_to_save="all", save_arg_values=True)
    clean_head = clean_log.modules["block.attn"].facets.head(0).result.detach().clone()

    def _manual_patch(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        """Return the clean head activation."""

        del out, hook
        return clean_head

    manual = corrupted_log.fork("manual_head_patch")
    manual.attach_hooks(tl.facet("result").head(0).in_module("block.attn"), _manual_patch)
    manual.rerun(model, corrupted)

    assert torch.equal(scores[0, 0], _metric(manual).detach())


def test_residual_attention_and_mlp_patch_shapes() -> None:
    """Residual, attention-output, and MLP helpers return expected layer shapes."""

    model = PatchingToyModel()
    clean, corrupted = _inputs()

    residual = tl.facets.patching.activation_patch_residual_stream(model, clean, corrupted, _metric)
    residual_layer = tl.facets.patching.activation_patch_residual_stream(
        model, clean, corrupted, _metric, patch_positions=False
    )
    attn_out = tl.facets.patching.activation_patch_attention_output(
        model, clean, corrupted, _metric
    )
    mlp_out = tl.facets.patching.activation_patch_mlp_output(model, clean, corrupted, _metric)

    assert residual.shape == (1, 3)
    assert residual_layer.shape == (1,)
    assert attn_out.shape == (1,)
    assert mlp_out.shape == (1,)


def test_attribution_patch_attention_heads_matches_activation_ordering() -> None:
    """Attribution patching has the same sign and top component as activation patching."""

    model = PatchingToyModel()
    clean, corrupted = _inputs()
    activation_scores = tl.facets.patching.activation_patch_attention_heads(
        model, clean, corrupted, _metric
    )
    attribution_scores = tl.facets.patching.attribution_patch_attention_heads(
        model, clean, corrupted, _metric
    )

    assert attribution_scores.shape == (1, 2)
    assert torch.sign(attribution_scores[0, 0]) == torch.sign(activation_scores[0, 0])
    assert attribution_scores.argmax() == activation_scores.argmax()
    assert (
        torch.corrcoef(torch.stack((activation_scores.flatten(), attribution_scores.flatten())))[
            0, 1
        ]
        > 0.99
    )
    assert torch.allclose(attribution_scores, activation_scores, atol=1e-5, rtol=1e-5)


def test_attribution_patch_requires_grad_capture() -> None:
    """Attribution patching raises a clear error when facet gradients are missing."""

    model = PatchingToyModel()
    clean, corrupted = _inputs()

    with pytest.raises(RuntimeError, match="requires grad capture.*Facet gradient unavailable"):
        tl.facets.patching.attribution_patch_attention_heads(
            model,
            clean,
            corrupted,
            _metric,
            trace_kwargs={"backward_ready": True, "gradients_to_save": None},
        )
