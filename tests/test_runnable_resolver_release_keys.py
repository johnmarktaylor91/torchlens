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
from torchlens._io.runnable import build_sparse_run_descriptor, preflight_sparse_run_descriptor
from torchlens.errors import RunnablePreflightError
from torchlens.intervention.resolver import function_registry_key_from_callable
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    PathFaithfulness,
    ResolverStatus,
    RunnableErrorCode,
    SparseRunDescriptor,
)
from torchlens.utils._callable_safety import is_pure_forward_callable
from torchlens.utils.display import identity

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
    "custom", "identity", "function", import_path="torchlens.utils.display:identity"
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
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )


def _records_by_qualname(descriptor: SparseRunDescriptor) -> dict[str, Any]:
    """Return resolver records keyed by recorded qualname."""

    report, attachments = preflight_sparse_run_descriptor(descriptor)
    assert attachments is not None, report.diagnostics
    return {record.recorded_key.qualname: record for record in report.resolver_records}


def _loaded_run_matches_live(model: nn.Module, inputs: torch.Tensor, tmp_path: Path) -> None:
    """Save, load and run a runnable artifact, and compare it with the live model."""

    path = tmp_path / "model.tlspec"
    tl.save(_capture(model, inputs), path, level="runnable", include_weights=True)
    result = tl.load(path).run(inputs=inputs, seed=0)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    torch.testing.assert_close(result.output, model(inputs))


@pytest.mark.smoke
@pytest.mark.parametrize("name", _FFT_NAMES)
def test_private_fft_key_resolves_to_the_public_callable(name: str) -> None:
    """A ``torch._C._fft`` key resolves exactly, to the object ``torch.fft`` exports."""

    key = FunctionRegistryKey(
        "custom", f"fft_{name}", "function", import_path=f"torch._C._fft:fft_{name}"
    )
    resolved = runnable_load._resolve_exact_key(key, runnable_load._stock_path_from_key(key))
    assert resolved is not None
    func, qualname = resolved
    # Capture wrappers may be installed on ``torch.fft``; compare with the original.
    assert func is runnable_load._unwrap_decorated(getattr(torch.fft, name))
    assert qualname == f"torch._C._fft.fft_{name}"


@pytest.mark.smoke
def test_private_linalg_key_resolves_exactly() -> None:
    """A ``torch._C._linalg`` key resolves without the version-bounded alias rows."""

    key = FunctionRegistryKey(
        "custom", "linalg_inv", "function", import_path="torch._C._linalg:linalg_inv"
    )
    resolved = runnable_load._resolve_exact_key(key, runnable_load._stock_path_from_key(key))
    assert resolved is not None
    assert resolved[0] is runnable_load._unwrap_decorated(torch.linalg.inv)


@pytest.mark.smoke
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


def _minted_fft_key(name: str) -> FunctionRegistryKey:
    """Return the key the running torch mints for ``torch.fft.<name>`` at capture."""

    return function_registry_key_from_callable(
        runnable_load._unwrap_decorated(getattr(torch.fft, name))
    )


@pytest.mark.smoke
def test_fft_keys_resolve_exactly_whatever_the_producer_torch_version() -> None:
    """FFT readiness does not depend on the recorded producer torch minor.

    The key form is the one the running torch mints (public ``torch.fft`` on torch
    <= 2.12, private ``torch._C._fft`` on torch >= 2.13); either must resolve exactly
    whatever producer version the descriptor claims.
    """

    descriptor = build_sparse_run_descriptor(_capture(_FFTFamily().eval(), torch.randn(2, 4, 6)))
    for version in ("2.1.0", descriptor.compatibility.backend_version, "2.99.0"):
        stamped = replace(
            descriptor,
            compatibility=replace(descriptor.compatibility, backend_version=version),
        )
        records = _records_by_qualname(stamped)
        for name in _FFT_NAMES:
            key = _minted_fft_key(name)
            assert key.qualname in {name, f"fft_{name}"}, key
            record = records[key.qualname]
            assert record.status is ResolverStatus.RESOLVED_EXACT, (version, record)
            stock_path = runnable_load._stock_path_from_key(key)
            assert record.provenance == f"exact_getattr:{stock_path}", (version, record)


def test_fft_model_loaded_run_is_verified(tmp_path: Path) -> None:
    """A loaded FFT artifact runs (the op records ``rfft``, the key ``fft_rfft``)."""

    torch.manual_seed(0)
    _loaded_run_matches_live(_FFTFamily().eval(), torch.randn(2, 4, 6), tmp_path)


class _SpecialFamily(nn.Module):
    """Public special functions, keyed as ``torch._C._special`` builtins."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Exercise two special functions."""

        return torch.special.erf(value) + torch.special.gammaln(value.abs() + 1.0)


def test_special_model_loaded_run_is_verified(tmp_path: Path) -> None:
    """A loaded ``torch.special`` artifact runs; registry coherence accepts ``erf``."""

    torch.manual_seed(0)
    _loaded_run_matches_live(_SpecialFamily().eval(), torch.rand(2, 4), tmp_path)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("module_name", "qualname", "public"),
    (
        ("torch._C._fft", "fft_rfft", "rfft"),
        ("torch._C._linalg", "linalg_inv", "inv"),
        ("torch._C._special", "special_erf", "erf"),
    ),
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


@pytest.mark.smoke
def test_registry_coherence_still_refuses_a_swapped_fft_builtin() -> None:
    """``fft_irfft`` as the authority for an op recorded as ``rfft`` is a contradiction."""

    key = FunctionRegistryKey(
        "custom", "fft_irfft", "function", import_path="torch._C._fft:fft_irfft"
    )
    assert runnable_coherence._callable_registry_contradiction(
        key, ("rfft_1_1:1",), {"rfft_1_1": "rfft"}
    ) == ("rfft", "rfft_1_1:1")


@pytest.mark.smoke
def test_boundary_identity_table_key_is_the_minted_key() -> None:
    """The resolver table spells exactly the key capture mints for the boundary op."""

    assert function_registry_key_from_callable(identity) == _BOUNDARY_IDENTITY_KEY
    assert set(runnable_load._TORCHLENS_SYNTHETIC_CALLABLES) == {_BOUNDARY_IDENTITY_KEY}


@pytest.mark.smoke
def test_identity_boundary_op_resolves_exactly() -> None:
    """``nn.Identity`` and pass-through outputs resolve to the in-memory identity helper."""

    descriptor = build_sparse_run_descriptor(_capture(_IdentityFamily().eval(), torch.randn(2, 4)))
    keys = [entry.key for entry in descriptor.callable_registry]
    assert _BOUNDARY_IDENTITY_KEY in keys
    record = _records_by_qualname(descriptor)["identity"]
    assert record.status is ResolverStatus.RESOLVED_EXACT
    assert record.resolved_qualname == "torchlens.utils.display.identity"
    assert record.provenance == "torchlens_synthetic:torchlens.utils.display.identity"


def test_identity_model_loaded_run_is_verified(tmp_path: Path) -> None:
    """A loaded artifact with module-boundary identity ops runs faithfully."""

    torch.manual_seed(0)
    _loaded_run_matches_live(_IdentityFamily().eval(), torch.randn(2, 4), tmp_path)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "key",
    (
        FunctionRegistryKey(
            "custom", "identity", "function", import_path="torchlens.utils:identity"
        ),
        FunctionRegistryKey(
            "custom",
            "int_list_to_compact_str",
            "function",
            import_path="torchlens.utils.display:int_list_to_compact_str",
        ),
        FunctionRegistryKey(
            "custom", "identity", "method", import_path="torchlens.utils.display:identity"
        ),
        FunctionRegistryKey(
            "custom", "identity", "function", import_path="attacker_payload:identity"
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

        if name.startswith(("attacker_payload", "torchlens.utils")):
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


def _legacy_ctor_labels(trace: Any) -> list[str]:
    """Return the pass-qualified labels of the legacy-constructor ops."""

    ops = trace.ops.values() if hasattr(trace.ops, "values") else trace.ops
    return [str(op.label) for op in ops if getattr(op, "func_id", None) == _LEGACY_CTOR_KEY]


@pytest.mark.smoke
def test_legacy_tensor_constructor_save_refuses_typed(tmp_path: Path) -> None:
    """Section-13 disposition: runnable save refuses ``torch.Tensor.__new__`` typed.

    If this stops raising, the legacy constructor became runnable without the
    round-2 adapter (or the refusal went silent): re-open the disposition.
    """

    trace = _capture(_LegacyConstructorFamily().eval(), torch.randn(2, 4))
    labels = _legacy_ctor_labels(trace)
    assert len(labels) == 3
    with pytest.raises(RunnablePreflightError) as excinfo:
        tl.save(trace, tmp_path / "legacy.tlspec", level="runnable")
    refused = {
        label
        for diagnostic in excinfo.value.fields["diagnostics"]
        if diagnostic.code is RunnableErrorCode.UNSUPPORTED_LITERAL
        and diagnostic.detection_stage == "producer_literal"
        for label in diagnostic.affected_op_labels
    }
    assert set(labels) <= refused


@pytest.mark.smoke
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
