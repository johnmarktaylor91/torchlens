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
from torchlens.validation._loaded_callables import (
    release_loaded_callables,
    resolve_loaded_callables,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # A literal producer (``ones``), a C-module builtin (``F.gelu``) and a
        # reduction (``mean``, a receptive-field rule that reads input shapes).
        h = nn.functional.gelu(self.lin(v)) + torch.ones(4)
        return torch.relu(h) + v * 2.0 + h.mean(dim=0, keepdim=True)


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
    # Callables live for the replay phase only; the artifact keeps its load shape.
    assert all(op.func is None for op in computational)


def test_loaded_portable_bundle_refuses_by_name(tmp_path: Path) -> None:
    """A ``portable`` save drops saved arguments; validation refuses, never a verdict."""

    trace, _model, reference = _capture(save_args=True)
    path = tmp_path / "trace.tlz"
    trace.save(path)
    loaded = tl.load(path)
    with pytest.raises(ArtifactError, match="level='executable_with_callables'"):
        loaded.validate_forward_pass(reference)


def test_loaded_bundle_without_function_identities_still_refuses(tmp_path: Path) -> None:
    trace, _model, reference = _capture(save_args=False)
    path = tmp_path / "trace.tlz"
    trace.save(path)
    loaded = tl.load(path)
    with pytest.raises(ArtifactError, match="requires resolved func callables"):
        loaded.validate_forward_pass(reference)


def _layer(key: FunctionRegistryKey, func_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        func=None,
        func_id=key,
        func_name=func_name,
        is_input=False,
        is_output=False,
        is_buffer=False,
        is_internal_source=False,
    )


@pytest.mark.parametrize(
    ("key", "func_name"),
    [
        (FunctionRegistryKey("custom", "system", "function", import_path="os:system"), "system"),
        (FunctionRegistryKey("torch", "load", "function"), "load"),
        (FunctionRegistryKey("torch", "no_such_op_anywhere", "function"), "no_such_op_anywhere"),
        (FunctionRegistryKey("torch", "relu", "function"), "sigmoid"),
    ],
    ids=["custom_foreign", "torch_impure", "torch_missing", "key_names_another_function"],
)
def test_foreign_impure_missing_or_mismatched_keys_never_resolve(
    key: FunctionRegistryKey, func_name: str
) -> None:
    layer = _layer(key, func_name)
    assert resolve_loaded_callables(SimpleNamespace(layer_list=[layer])) == []
    assert layer.func is None


def test_structural_nodes_never_resolve() -> None:
    """An output node inherits its producer's ``func_id``; it must not replay that function."""

    output = _layer(FunctionRegistryKey("torch", "relu", "function"), "none")
    output.is_output = True
    assert resolve_loaded_callables(SimpleNamespace(layer_list=[output])) == []
    assert output.func is None


def test_pure_torch_keys_resolve_and_release() -> None:
    layers = [
        _layer(FunctionRegistryKey("torch", "relu", "function"), "relu"),
        _layer(FunctionRegistryKey("torch.Tensor", "__add__", "dunder"), "__add__"),
    ]
    resolved = resolve_loaded_callables(SimpleNamespace(layer_list=layers))
    assert resolved == layers and all(callable(layer.func) for layer in layers)
    release_loaded_callables(resolved)
    assert all(layer.func is None for layer in layers)


def test_literal_producer_and_torch_c_builtin_keys_resolve() -> None:
    """A parentless ``ones`` replays (SPEC 2 rule 3); ``F.gelu``'s C-module key maps exactly."""

    ones = _layer(FunctionRegistryKey("torch", "ones", "function"), "ones")
    ones.is_internal_source = True
    gelu = _layer(
        FunctionRegistryKey("custom", "gelu", "function", import_path="torch._C._nn:gelu"), "gelu"
    )
    forged = _layer(
        FunctionRegistryKey("custom", "gelu", "function", import_path="torch._C._nn:os.system"),
        "gelu",
    )
    resolved = resolve_loaded_callables(SimpleNamespace(layer_list=[ones, gelu, forged]))
    assert resolved == [ones, gelu]
    assert forged.func is None
