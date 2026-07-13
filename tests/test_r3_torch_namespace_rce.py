"""Round-3 security regression: torch-namespace RCE / arbitrary-file-write.

A crafted ``.tlspec`` bundle op keyed ``("torch", "load")`` or ``("torch",
"save")`` historically resolved ``torch.load`` / ``torch.save`` through the
untrusted callable resolvers (filtered only by ``callable(...)``), giving RCE
(unpickle) or arbitrary-file-write during ``Trace.run`` / intervention replay.
Both resolvers now admit only pure, side-effect-free forward/tensor ops.

These tests pin the door shut on BOTH the runnable-load resolver and the
intervention resolver, while proving genuine forward-op models still load, run,
and VERIFY, and that operator / functional / tensor-method ops still resolve.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_function_registry_key
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness
from torchlens.utils._callable_safety import is_pure_forward_callable

_CAP = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


class _TinyLinear(nn.Module):
    """Minimal parameterized graph whose first op is a torch tensor op."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(value))


class _TinyCNN(nn.Module):
    """Small convolutional network (conv2d / relu / pool / linear)."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3)
        self.fc = nn.Linear(16, 10)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.relu(self.c1(value))
        value = torch.relu(self.c2(value))
        value = torch.nn.functional.adaptive_avg_pool2d(value, 1).flatten(1)
        return self.fc(value)


class _TinyAttention(nn.Module):
    """Attention block exercising MHA / layer_norm / linear forward ops."""

    def __init__(self) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(32, 4, batch_first=True)
        self.ln = nn.LayerNorm(32)
        self.fc = nn.Linear(32, 32)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        attended, _ = self.mha(value, value, value)
        value = self.ln(value + attended)
        return self.fc(value)


def _save_runnable(model: nn.Module, inputs: torch.Tensor, path: Path) -> Path:
    """Capture and persist a runnable bundle with embedded weights."""

    log = tl.trace(model.eval(), inputs, capture=_CAP)
    tl.save(log, path, level="runnable", include_weights=True)
    return path


def _relu_registry_index(run: dict) -> int:
    """Return the callable-registry index whose key is ``("torch", "relu")``."""

    for index, entry in enumerate(run["callable_registry"]):
        if entry["key"]["namespace"] == "torch" and entry["key"]["qualname"] == "relu":
            return index
    raise AssertionError("expected a torch.relu op in the tiny model")


def _relu_registry_id(run: dict, index: int) -> str:
    return str(run["callable_registry"][index]["registry_id"])


# --------------------------------------------------------------------------- #
# Intervention resolver: deny side-effecting torch callables, allow forward ops.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("qualname", ["load", "save"])
def test_intervention_resolver_denies_torch_serialization(qualname: str) -> None:
    """torch.load / torch.save must be refused with a typed security error."""

    key = FunctionRegistryKey("torch", qualname, "function")
    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key)


@pytest.mark.parametrize(
    ("namespace", "qualname", "dispatch"),
    [
        ("operator", "add", "function"),
        ("operator", "mul", "function"),
        ("torch.nn.functional", "relu", "function"),
        ("torch.Tensor", "view", "method"),
        ("torch.Tensor", "reshape", "method"),
        ("torch", "matmul", "function"),
        ("torch", "conv2d", "function"),
        ("torch", "layer_norm", "function"),
    ],
)
def test_intervention_resolver_allows_forward_ops(
    namespace: str, qualname: str, dispatch: str
) -> None:
    """Genuine forward/tensor ops still resolve through the intervention resolver."""

    resolved = resolve_function_registry_key(FunctionRegistryKey(namespace, qualname, dispatch))
    assert callable(resolved)


# --------------------------------------------------------------------------- #
# Shared safety helper: dangerous callables denied, forward ops admitted.
# --------------------------------------------------------------------------- #


def test_callable_safety_helper_denies_side_effecting() -> None:
    """The gate denies serialization, jit, and os/pickle-module callables."""

    for func in (
        torch.load,
        torch.save,
        torch.jit.load,
        torch.jit.save,
        os.system,
        pickle.loads,
        pickle.load,
    ):
        assert is_pure_forward_callable(func) is False


def test_callable_safety_helper_allows_forward_ops() -> None:
    """The gate admits torch / functional / tensor-method / operator ops."""

    import operator

    import torch.nn.functional as functional

    for func in (
        torch.matmul,
        torch.add,
        torch.einsum,
        torch.fft.fft,
        functional.relu,
        functional.linear,
        torch.Tensor.view,
        torch.Tensor.reshape,
        operator.add,
        operator.mul,
    ):
        assert is_pure_forward_callable(func) is True


# --------------------------------------------------------------------------- #
# Runnable-load resolver: end-to-end exploit reproductions must be blocked.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_runnable_load_blocks_torch_save_file_write(tmp_path: Path) -> None:
    """Repointing an op to torch.save must not write an attacker-chosen file."""

    src = _save_runnable(_TinyLinear(), torch.ones(1, 4), tmp_path / "clean.tlspec")
    evil = tmp_path / "evil.tlspec"
    shutil.copytree(src, evil)
    target = tmp_path / "PWNED_BY_TLSPEC_RUN"

    manifest_path = evil / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    run = manifest["run"]
    index = _relu_registry_index(run)
    registry_id = _relu_registry_id(run, index)
    run["callable_registry"][index]["key"]["qualname"] = "save"
    call = next(c for c in run["calls"] if c["registry_id"] == registry_id)
    call["argument_names"] = ["obj", "f"]
    call["num_positional_args"] = 2
    call["num_keyword_args"] = 0
    call["tensor_arguments"] = [{"argument_path": ["args", 0], "slot_id": "slot:input_1:1"}]
    call["literal_arguments"] = [
        {"argument_path": ["args", 1], "value": {"kind": "str", "value": str(target)}}
    ]
    manifest_path.write_text(json.dumps(manifest))

    loaded = tl.load(evil)
    with pytest.raises(Exception):
        loaded.run(inputs=torch.ones(1, 4))
    assert not target.exists()


@pytest.mark.smoke
def test_runnable_load_blocks_torch_load_rce(tmp_path: Path) -> None:
    """Repointing an op to torch.load(weights_only=False) must not execute code."""

    evil_pickle = tmp_path / "evil.pt"
    proof = tmp_path / "RCE_PROOF"

    class _Evil:
        def __reduce__(self):  # type: ignore[no-untyped-def]
            return (os.system, (f"touch {proof}",))

    evil_pickle.write_bytes(pickle.dumps(_Evil()))

    src = _save_runnable(_TinyLinear(), torch.ones(1, 4), tmp_path / "clean.tlspec")
    evil = tmp_path / "evil2.tlspec"
    shutil.copytree(src, evil)

    manifest_path = evil / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    run = manifest["run"]
    index = _relu_registry_index(run)
    registry_id = _relu_registry_id(run, index)
    run["callable_registry"][index]["key"]["qualname"] = "load"
    call = next(c for c in run["calls"] if c["registry_id"] == registry_id)
    call["argument_names"] = ["f"]
    call["num_positional_args"] = 1
    call["num_keyword_args"] = 1
    call["tensor_arguments"] = []
    call["literal_arguments"] = [
        {"argument_path": ["args", 0], "value": {"kind": "str", "value": str(evil_pickle)}},
        {
            "argument_path": ["kwargs", "weights_only"],
            "value": {"kind": "bool", "value": False},
        },
    ]
    manifest_path.write_text(json.dumps(manifest))

    loaded = tl.load(evil)
    with pytest.raises(Exception):
        loaded.run(inputs=torch.ones(1, 4))
    assert not proof.exists()


# --------------------------------------------------------------------------- #
# Genuine models must still load, run, and VERIFY through the runnable path.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_genuine_cnn_runnable_still_verified(tmp_path: Path) -> None:
    """A real CNN round-trips through the runnable path and verifies its path."""

    inputs = torch.randn(1, 3, 16, 16)
    path = _save_runnable(_TinyCNN(), inputs, tmp_path / "cnn.tlspec")
    result = tl.load(path).run(inputs=inputs)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_genuine_attention_runnable_still_verified(tmp_path: Path) -> None:
    """A real attention block round-trips through the runnable path and verifies."""

    inputs = torch.randn(1, 5, 32)
    path = _save_runnable(_TinyAttention(), inputs, tmp_path / "attn.tlspec")
    result = tl.load(path).run(inputs=inputs)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
