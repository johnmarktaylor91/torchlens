"""Resolver release-gate keys: private torch builtins and the module-boundary identity."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import runnable_coherence, runnable_load
from torchlens._io.runnable import (
    build_sparse_run_descriptor,
    preflight_sparse_run_descriptor,
)
from torchlens._runnable import (
    PathFaithfulness,
    ResolverStatus,
    RunnableErrorCode,
)
from torchlens._utils._callable_safety import is_pure_forward_callable
from torchlens.intervention._types import FunctionRegistryKey
from torchlens.options import CaptureOptions

_FFT_NAMES = (
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fft2",
    "ifft2",
    "rfft2",
    "irfft2",
    "fftn",
    "ifftn",
    "fftshift",
    "ifftshift",
)
_LEGACY_CTOR_KEY = FunctionRegistryKey("torch.Tensor", "__new__", "method")
_BOUNDARY_IDENTITY_KEY = FunctionRegistryKey(
    "custom", "identity", "function", import_path="torchlens._utils.display:identity"
)


class _FFTFamily(nn.Module):
    """Every ``torch.fft`` builtin the classics census recorded."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Chain the twelve FFT builtins and return a real tensor."""

        one_d = torch.fft.irfft(torch.fft.rfft(value), n=value.shape[-1])
        one_d = one_d + torch.fft.ifft(torch.fft.fft(value)).real
        two_d = torch.fft.irfft2(torch.fft.rfft2(value), s=value.shape[-2:])
        two_d = two_d + torch.fft.ifft2(torch.fft.fft2(value)).real
        n_d = torch.fft.ifftn(torch.fft.fftn(value)).real
        shifted = torch.fft.ifftshift(torch.fft.fftshift(value))
        return one_d + two_d + n_d + shifted


class _PassThrough(nn.Module):
    """Module whose output is its input object."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged."""

        return value


class _IdentityFamily(nn.Module):
    """``nn.Identity`` and a pass-through module between real ops."""

    def __init__(self) -> None:
        """Initialize compact affine state around the boundary modules."""

        super().__init__()
        self.first = nn.Linear(4, 4)
        self.skip = nn.Identity()
        self.through = _PassThrough()
        self.second = nn.Linear(4, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Route activations through both boundary modules."""

        hidden = self.through(self.skip(torch.relu(self.first(value))))
        return self.second(hidden + 1.0)


def _capture(model: nn.Module, inputs: torch.Tensor) -> Any:
    """Capture one runnable-ready trace."""

    return tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            replayable=True,
            capture_container_structure=True,
            cache=False,
        ),
    )


def _loaded_run_matches_live(model: nn.Module, inputs: torch.Tensor, tmp_path: Path) -> None:
    """Save, load and run a runnable artifact, and compare it with the live model."""

    path = tmp_path / "model.tlspec"
    _capture(model, inputs).save(path, level="runnable", include_weights=True)
    result = tl.load(path).run(inputs=inputs, seed=0)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    torch.testing.assert_close(result.output, model(inputs))


# Every fft cell runs the same resolver path; smoke keeps one per shape class (1-D,
# real inverse, 2-D, n-D, shift helper) and the rest run in the backstop tiers.


@pytest.mark.parametrize("name", _FFT_NAMES)
def test_public_fft_key_resolves_to_the_public_callable(name: str) -> None:
    """The ``torch.fft`` key form resolves exactly to the same object.

    Producers on torch <= 2.12 mint this public form (the private-to-public alias row
    applies at capture); torch >= 2.13 mints the private ``torch._C._fft`` form. Both
    must resolve on every running torch, so artifacts load across the whole range.
    """

    key = FunctionRegistryKey("torch.fft", name, "function")
    resolved = runnable_load._resolve_exact_key(key, runnable_load._stock_path_from_key(key))
    assert resolved is not None
    func, qualname = resolved
    # The public attribute may be the installed capture wrapper; compare originals.
    original = runnable_load._unwrap_decorated(getattr(torch.fft, name))
    assert runnable_load._unwrap_decorated(func) is original
    assert qualname == f"torch.fft.{name}"


class _SpecialFamily(nn.Module):
    """Public special functions, keyed as ``torch._C._special`` builtins."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Exercise two special functions."""

        return torch.special.erf(value) + torch.special.gammaln(value.abs() + 1.0)


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    ("module_name", "qualname", "public"),
    (("torch._C._special", "special_erf", "erf"),),
)
def test_registry_coherence_accepts_only_the_keys_own_public_name(
    module_name: str, qualname: str, public: str
) -> None:
    """The public spelling of a private builtin is coherent; any other name still refuses."""

    key = FunctionRegistryKey(
        "custom", qualname, "function", import_path=f"{module_name}:{qualname}"
    )
    check = runnable_coherence._callable_registry_contradiction
    assert check(key, ("op_1_1:1",), {"op_1_1": public}) is None
    assert check(key, ("op_1_1:1",), {"op_1_1": qualname}) is None
    assert check(key, ("op_1_1:1",), {"op_1_1": "relu"}) == ("relu", "op_1_1:1")
    foreign = FunctionRegistryKey("custom", qualname, "function", import_path=f"mymod:{qualname}")
    assert check(foreign, ("op_1_1:1",), {"op_1_1": public}) == (public, "op_1_1:1")


def test_identity_model_loaded_run_is_verified(tmp_path: Path) -> None:
    """A loaded artifact with module-boundary identity ops runs faithfully."""

    torch.manual_seed(0)
    _loaded_run_matches_live(_IdentityFamily().eval(), torch.randn(2, 4), tmp_path)


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "key",
    (
        FunctionRegistryKey(
            "custom",
            "identity",
            "function",
            import_path="torchlens._postprocess.graph_traversal:identity",
        ),
    ),
)
def test_synthetic_rung_admits_only_the_exact_identity_key(
    key: FunctionRegistryKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Near-miss keys keep the custom default-deny and are never imported."""

    descriptor = build_sparse_run_descriptor(_capture(_IdentityFamily().eval(), torch.randn(2, 4)))
    registry_id = next(
        entry.registry_id
        for entry in descriptor.callable_registry
        if entry.key == _BOUNDARY_IDENTITY_KEY
    )
    crafted = replace(
        descriptor,
        callable_registry=tuple(
            replace(entry, key=key) if entry.registry_id == registry_id else entry
            for entry in descriptor.callable_registry
        ),
    )
    original_import_module = importlib.import_module

    def guarded_import_module(name: str, package: str | None = None) -> Any:
        """Fail if readiness imports an artifact-selected module."""

        if name.startswith(("attacker_payload", "torchlens._utils")):
            raise AssertionError(f"artifact-selected import of {name!r}")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    report, attachments = preflight_sparse_run_descriptor(crafted)
    assert attachments is None
    record = next(item for item in report.resolver_records if item.registry_id == registry_id)
    assert record.status is ResolverStatus.UNAVAILABLE
    assert {diagnostic.code for diagnostic in record.diagnostics} == {
        RunnableErrorCode.UNTRUSTED_CUSTOM_IMPORT
    }


class _LegacyConstructorFamily(nn.Module):
    """Data, alias and size forms of the legacy ``torch.Tensor(...)`` constructor."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Build tensors with the legacy constructor and combine them with the input."""

        data = torch.Tensor([1.0, 2.0, 3.0, 4.0])
        alias = torch.Tensor(torch.zeros(2, 4))
        sized = torch.Tensor(2, 4).zero_()
        return value + data + alias + sized


def test_legacy_tensor_constructor_key_stays_unresolved() -> None:
    """Section-13 disposition: the resolver keeps refusing the raw legacy constructor.

    The raw callable has a hidden ``cdata=`` raw-pointer overload, so it must never
    resolve for an untrusted bundle; only a guarded adapter may replace this refusal.
    """

    assert not is_pure_forward_callable(torch.Tensor.__new__)
    descriptor = build_sparse_run_descriptor(
        _capture(_LegacyConstructorFamily().eval(), torch.randn(2, 4))
    )
    report, attachments = preflight_sparse_run_descriptor(descriptor)
    assert attachments is None
    records = [r for r in report.resolver_records if r.recorded_key == _LEGACY_CTOR_KEY]
    assert len(records) == 1
    record = records[0]
    assert record.status is ResolverStatus.UNAVAILABLE
    assert record.provenance == "nonforward_callable_denied"
    assert {diagnostic.code for diagnostic in record.diagnostics} == {
        RunnableErrorCode.UNTRUSTED_CUSTOM_IMPORT
    }
