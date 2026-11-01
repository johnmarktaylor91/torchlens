"""Replay validation of a loaded torch bundle resolves op callables from ``func_id``.

A bundle never stores callables. A capture that saves arguments records each
op's portable ``FunctionRegistryKey``; validation (an execution boundary)
resolves it through the purity-gated resolver, so a bundle saved with its
re-execution payloads (``level="executable_with_callables"``) replays after
load. Foreign (``custom``) keys and impure torch callables never resolve, and
an op with no key keeps the by-name refusal.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import ArtifactError
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.options import CaptureOptions
from torchlens.validation._loaded_callables import resolve_loaded_callables


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(v)) + v * 2.0


def _capture(save_args: bool) -> tuple[tl.Trace, nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    model = _Block().eval()
    x = torch.randn(3, 4)
    reference = model(x)
    capture = CaptureOptions(save_args=True, save_rng=True) if save_args else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x, capture=capture) if capture else tl.trace(model, x)
    return trace, model, reference


def test_loaded_replay_ready_bundle_validates(tmp_path: Path) -> None:
    trace, _model, reference = _capture(save_args=True)
    path = tmp_path / "trace.tlz"
    trace.save(path, level="executable_with_callables")
    loaded = tl.load(path)
    computational = [op for op in loaded.layer_list if op.func_name not in {"none"}]
    assert computational and all(op.func is None for op in computational)
    assert loaded.validate_forward_pass(reference) is True
    status = loaded.validation_replay_status
    assert status.state == "passed", status.message
    assert all(op.func is not None for op in computational)


def test_loaded_bundle_without_function_identities_still_refuses(tmp_path: Path) -> None:
    trace, _model, reference = _capture(save_args=False)
    path = tmp_path / "trace.tlz"
    trace.save(path)
    loaded = tl.load(path)
    with pytest.raises(ArtifactError, match="requires resolved func callables"):
        loaded.validate_forward_pass(reference)


@pytest.mark.parametrize(
    "key",
    [
        FunctionRegistryKey("custom", "system", "function", import_path="os:system"),
        FunctionRegistryKey("torch", "load", "function"),
        FunctionRegistryKey("torch", "no_such_op_anywhere", "function"),
    ],
    ids=["custom_foreign", "torch_impure", "torch_missing"],
)
def test_foreign_impure_or_missing_keys_never_resolve(key: FunctionRegistryKey) -> None:
    layer = SimpleNamespace(func=None, func_id=key)
    assert resolve_loaded_callables(SimpleNamespace(layer_list=[layer])) == 0
    assert layer.func is None


def test_pure_torch_keys_resolve() -> None:
    layers = [
        SimpleNamespace(func=None, func_id=FunctionRegistryKey("torch", "relu", "function")),
        SimpleNamespace(
            func=None, func_id=FunctionRegistryKey("torch.Tensor", "__add__", "dunder")
        ),
    ]
    assert resolve_loaded_callables(SimpleNamespace(layer_list=layers)) == 2
    assert all(callable(layer.func) for layer in layers)
