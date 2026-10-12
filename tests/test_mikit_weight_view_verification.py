"""``head_weight_views`` payload verification: true views pass, wrong views refuse.

The verification reconstructs each q/k/v facet as ``input @ W (+ b)`` and
compares it with the captured payload under the cancellation-aware
reconstruction budget (``semantic/tolerances.py``). These tests pin both
sides of that tripwire on real-class models: the true views verify, and a
wrong view refuses typed whether the error is gross (permuted heads,
permuted d_head columns, a transposed per-head view) or small (one head
scaled by 1.001).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp as mi
import torchlens.mechinterp._params as params_module
from torchlens.mechinterp._errors import MechInterpError

sys.path.insert(0, str(Path(__file__).resolve().parent / "real_model" / "r0"))


def _scaled_first_head(view: torch.Tensor) -> torch.Tensor:
    scaled = view.clone()
    scaled[0] = scaled[0] * 1.001
    return scaled


PERTURBATIONS = {
    "permuted_heads": lambda view: view.flip(0),
    "permuted_d_head_columns": lambda view: view.flip(2),
    "transposed_view": lambda view: view.transpose(1, 2).reshape(view.shape),
    "one_head_scaled_1.001": _scaled_first_head,
}


def _trace(builder_name: str):
    import families

    model = getattr(families, builder_name)("eager")
    torch.manual_seed(8)
    x = torch.randint(0, 500, (2, 7))
    return tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))


@pytest.fixture(scope="module")
def traces():
    built = {
        "gpt2": (_trace("build_gpt2"), "transformer.h.1.attn"),
        "llama": (_trace("build_llama"), "model.layers.1.self_attn"),
    }
    try:
        yield built
    finally:
        for trace, _ in built.values():
            trace.cleanup()


def test_true_views_verify_on_every_role(traces):
    for trace, address in traces.values():
        views = mi.head_weight_views(trace, address)
        for role in ("q", "k", "v"):
            assert views.sources[role].verification == "payload_verified", (address, role)


@pytest.mark.parametrize("family", ["gpt2", "llama"])
@pytest.mark.parametrize("perturbation", sorted(PERTURBATIONS))
@pytest.mark.parametrize("role_index", [0, 2])  # the q view and the v view
def test_wrong_view_refuses(traces, monkeypatch, family, perturbation, role_index):
    trace, address = traces[family]
    original = params_module._head_view
    calls = {"n": 0}

    def perturbed(weight, n_heads, d_head, *, out_proj):
        view = original(weight, n_heads, d_head, out_proj=out_proj)
        if out_proj:
            return view
        index = calls["n"]
        calls["n"] += 1
        return PERTURBATIONS[perturbation](view) if index == role_index else view

    monkeypatch.setattr(params_module, "_head_view", perturbed)
    with pytest.raises(MechInterpError) as excinfo:
        mi.head_weight_views(trace, address)
    assert excinfo.value.fields["code"] == "mi_weight_view_unverified"
