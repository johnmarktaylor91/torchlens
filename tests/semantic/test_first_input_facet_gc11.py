"""Regression tests for the ``first_input``/``"input"`` facet on the torch backend.

cert9-f5 fix (round-8-prep peripheral MAJOR): ``first_input``/``first_input_spec``
(``torchlens/semantic/recipes/_helpers.py``) read ``ModuleCall.forward_args``, which
finalization ("GC-11", ``torchlens/postprocess/finalization.py``) unconditionally
nulls for the default ``torch_module`` module-identity mode -- the mode ordinary
``nn.Module`` tracing uses. That made the built-in ``"input"``/``"indices"`` facet
falsely report "not captured -- re-run with save_arg_values=True" even when the
value WAS captured and was trivially readable via the op graph, exactly the
fallback its sibling ``module_input_op_spec`` already implements.

The fix gives ``first_input_spec`` the same op-graph fallback. These tests assert
the FACET VALUE (not just presence), mirroring the ``up_out``/``intermediate``/
``down_out``/``output`` assertions in ``test_mlp_recipes.py`` -- the exact
assertion gap the cert8 peripheral report flagged as the reason this regression
had no tripwire.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


def test_layer_norm_input_facet_resolves_on_default_backend() -> None:
    """LayerNorm's "input" facet resolves via the op-graph fallback."""

    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    x = torch.randn(2, 4)
    log = tl.trace(model, x, layers_to_save="all", save_arg_values=True)
    ln_module = log.modules["1"]

    assert ln_module.calls[0].forward_args is None, "GC-11 should still null forward_args"
    assert ln_module.facets.has("input")
    facet_value = ln_module.facets.input.value
    linear_out = log["linear_1_1"].out
    assert torch.equal(facet_value, linear_out)


def test_rms_norm_input_facet_resolves_on_default_backend() -> None:
    """RMSNorm-family's "input" facet resolves via the op-graph fallback."""

    class TinyRMSNorm(nn.Module):
        """Minimal RMSNorm surrogate matched by the ``rms_norm`` recipe."""

        def __init__(self, dim: int) -> None:
            """Initialize weight-only RMSNorm."""

            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply RMS normalization."""

            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + 1e-6) * self.weight

    TinyRMSNorm.__name__ = "RMSNorm"
    model = nn.Sequential(nn.Linear(4, 4), TinyRMSNorm(4))
    x = torch.randn(2, 4)
    log = tl.trace(model, x, layers_to_save="all", save_arg_values=True)
    norm_module = log.modules["1"]

    assert norm_module.facets.recipe_source == "rms_norm"
    assert norm_module.facets.has("input")
    facet_value = norm_module.facets.input.value
    linear_out = log["linear_1_1"].out
    assert torch.equal(facet_value, linear_out)


def test_embedding_indices_facet_resolves_on_default_backend() -> None:
    """Embedding's "indices" facet resolves via the op-graph fallback."""

    model = nn.Embedding(10, 4)
    x = torch.tensor([1, 2, 3])
    log = tl.trace(model, x, layers_to_save="all", save_arg_values=True)
    embedding_module = log.modules["self"]

    assert embedding_module.calls[0].forward_args is None
    assert embedding_module.facets.has("indices")
    facet_value = embedding_module.facets.indices.value
    assert torch.equal(facet_value, x)


def test_first_input_spec_falls_back_when_forward_args_absent() -> None:
    """Direct unit check: first_input_spec resolves via module_input_op_spec.

    Exercises the helper directly (not just through a recipe) so the fallback
    logic itself is pinned, independent of any one recipe's wiring.
    """

    from torchlens.semantic.recipes._helpers import first_input_spec
    from torchlens.semantic.facets import AbsenceReason

    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    x = torch.randn(2, 4)
    log = tl.trace(model, x, layers_to_save="all", save_arg_values=True)
    ln_module = log.modules["1"]

    spec = first_input_spec(ln_module, "layer_norm")
    assert not isinstance(spec, AbsenceReason)
    assert spec.home_kind == "op"
