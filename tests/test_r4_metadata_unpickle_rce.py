"""Round-4 security regression: bundle ``metadata.pkl`` load-time unpickle RCE.

``tl.load`` historically read a bundle's ``metadata.pkl`` with an UNRESTRICTED
:class:`pickle.Unpickler`, so a crafted ``.tlspec`` whose ``metadata.pkl``
carried a ``__reduce__`` / ``os.system`` gadget executed arbitrary code at
``pickle.load`` time -- BEFORE any resolver / callable-safety defense (r2/r3)
could fire. This sat in front of the entire prior hardening effort.

``metadata.pkl`` is now read with :class:`SafeBundleUnpickler`, a default-deny
class allowlist (mirroring ``torch.load(..., weights_only=True)``). These tests
pin the door shut: a malicious ``metadata.pkl`` RAISES and executes NOTHING,
while every legitimately-saved bundle at every save level still loads and runs.
"""

from __future__ import annotations

import importlib
import io
import pickle
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens._io._safe_unpickle import (
    SafeBundleUnpickler,
    _safe_getattr,
    _safe_load_from_bytes,
)
from torchlens.options import CaptureOptions
from torchlens.utils.display import identity

_CAP = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


class _ControlFlow(nn.Module):
    """Control-flow graph; a portable save embeds a predicate tensor + method refs."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.sum() > 0:
            return torch.relu(value) * 2
        return torch.sigmoid(value) - 1


class _StatefulLinear(nn.Module):
    """Runnable-eligible parameterized graph with a persistent buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(value)) * self.scale


def _metadata_path(bundle_path: Path) -> Path:
    """Return the ``metadata.pkl`` path inside a saved bundle."""

    return bundle_path / "metadata.pkl"


def _overwrite_metadata(bundle_path: Path, obj: object) -> None:
    """Replace ONLY ``metadata.pkl`` with a pickled payload, leaving the rest intact."""

    with _metadata_path(bundle_path).open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


class _OsSystemGadget:
    """``__reduce__`` -> ``os.system`` payload that would run a shell command."""

    def __init__(self, marker: Path) -> None:
        self.marker = str(marker)

    def __reduce__(self):  # type: ignore[no-untyped-def]
        import os

        return (os.system, (f"touch {self.marker}",))


class _EvalGadget:
    """``__reduce__`` -> ``builtins.eval`` payload that would write a marker file."""

    def __init__(self, marker: Path) -> None:
        self.marker = str(marker)

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (eval, (f"open({self.marker!r}, 'w').close()",))


class _ImportlibGadget:
    """``__reduce__`` -> ``importlib.import_module`` gadget."""

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (importlib.import_module, ("os",))


class _GetattrPivotGadget:
    """``getattr(<trusted function>, "__globals__")`` classic unpickle pivot."""

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (getattr, (identity, "__globals__"))


def _save_all_levels(tmp_path: Path) -> dict[str, Path]:
    """Save one bundle at each required level and return their paths.

    Live source models are held for the duration of the save because the runnable
    ``include_weights`` path requires the capture ``Trace``'s live source model.
    """

    x = torch.randn(2, 4)
    paths: dict[str, Path] = {}
    # Keep models alive: the runnable weight/state payload reads the live model.
    keep_alive: list[nn.Module] = []

    audit = tmp_path / "audit"
    cf_a = _ControlFlow()
    keep_alive.append(cf_a)
    tl.trace(cf_a, x, layers_to_save="all").save(audit, level="audit")
    paths["analysis"] = audit

    portable = tmp_path / "portable"
    cf_p = _ControlFlow()
    keep_alive.append(cf_p)
    tl.trace(cf_p, x, layers_to_save="all").save(portable, level="portable")
    paths["portable"] = portable

    run = tmp_path / "runnable"
    lin = _StatefulLinear()
    keep_alive.append(lin)
    tl.trace(lin, x, save=tl.func("relu"), capture=_CAP).save(run, level="runnable")
    paths["runnable"] = run

    run_w = tmp_path / "runnable_weights"
    lin_w = _StatefulLinear()
    keep_alive.append(lin_w)
    tl.trace(lin_w, x, save=tl.func("relu"), capture=_CAP).save(
        run_w, level="runnable", include_weights=True
    )
    paths["runnable_weights"] = run_w

    run_a = tmp_path / "runnable_activations"
    lin_a = _StatefulLinear()
    keep_alive.append(lin_a)
    tl.trace(lin_a, x, save=tl.func("relu"), capture=_CAP).save(
        run_a, level="runnable", include_activations=True
    )
    paths["runnable_activations"] = run_a

    assert len(keep_alive) == 5
    return paths


# --------------------------------------------------------------------------- #
# Legitimate bundles at every save level still load (and run).
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_all_save_levels_still_load(tmp_path: Path) -> None:
    """Every legitimately-saved bundle at every level loads under the allowlist."""

    for name, bundle_path in _save_all_levels(tmp_path).items():
        loaded = tl.load(bundle_path)
        assert loaded is not None, name


@pytest.mark.smoke
def test_runnable_levels_still_run(tmp_path: Path) -> None:
    """Runnable bundles at all three variants still execute after restricted load."""

    x = torch.randn(2, 4)
    paths = _save_all_levels(tmp_path)
    for name in ("runnable", "runnable_weights", "runnable_activations"):
        loaded = tl.load(paths[name])
        result = loaded.run(inputs=x)
        assert tuple(result.output.shape) == (2, 4), name


@pytest.mark.smoke
def test_control_flow_bundle_round_trips(tmp_path: Path) -> None:
    """A control-flow portable bundle (embedded tensor + torch method refs) loads."""

    x = torch.randn(2, 4)
    bundle = tmp_path / "cf"
    tl.trace(_ControlFlow(), x, layers_to_save="all").save(bundle, level="portable")
    loaded = tl.load(bundle)
    assert loaded is not None


# --------------------------------------------------------------------------- #
# Malicious metadata.pkl is denied and executes nothing (through real tl.load).
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize(
    "gadget_factory",
    [
        pytest.param(lambda m: _OsSystemGadget(m), id="reduce_os_system"),
        pytest.param(lambda m: _EvalGadget(m), id="builtins_eval"),
        pytest.param(lambda _m: _ImportlibGadget(), id="importlib_import_module"),
        pytest.param(lambda _m: _GetattrPivotGadget(), id="getattr_globals_pivot"),
    ],
)
def test_malicious_metadata_is_denied_and_executes_nothing(tmp_path: Path, gadget_factory) -> None:
    """A crafted ``metadata.pkl`` raises and never runs the embedded gadget."""

    x = torch.randn(2, 4)
    bundle = tmp_path / "victim"
    tl.trace(_ControlFlow(), x, layers_to_save="all").save(bundle, level="portable")

    marker = tmp_path / "PWNED"
    _overwrite_metadata(bundle, gadget_factory(marker))

    with pytest.raises((TorchLensIOError, pickle.UnpicklingError)):
        tl.load(bundle)

    assert not marker.exists(), "load-time gadget executed"


@pytest.mark.smoke
def test_direct_unpickler_denies_os_system(tmp_path: Path) -> None:
    """The restricted unpickler raises ``UnpicklingError`` on an os.system gadget."""

    marker = tmp_path / "PWNED_DIRECT"
    blob = tmp_path / "meta.pkl"
    with blob.open("wb") as handle:
        pickle.dump(_OsSystemGadget(marker), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with blob.open("rb") as handle:
        with pytest.raises(pickle.UnpicklingError):
            SafeBundleUnpickler(handle).load()
    assert not marker.exists()


# --------------------------------------------------------------------------- #
# Unit-level guards on the two safe wrappers.
# --------------------------------------------------------------------------- #


def test_safe_getattr_blocks_non_allowlisted_target() -> None:
    """``_safe_getattr`` refuses any target outside the torch C holder classes."""

    with pytest.raises(pickle.UnpicklingError):
        _safe_getattr(identity, "__globals__")
    with pytest.raises(pickle.UnpicklingError):
        _safe_getattr(pickle, "loads")


def test_safe_getattr_allows_torch_tensor_method() -> None:
    """``_safe_getattr`` still reconstructs a torch C tensor-method reference."""

    resolved = _safe_getattr(torch._C.TensorBase, "sum")
    assert callable(resolved)


def test_safe_load_from_bytes_uses_weights_only(tmp_path: Path) -> None:
    """``_safe_load_from_bytes`` refuses an embedded os.system payload (weights_only)."""

    marker = tmp_path / "PWNED_STORAGE"
    buffer = io.BytesIO()
    torch.save(_OsSystemGadget(marker), buffer)

    with pytest.raises(pickle.UnpicklingError):
        _safe_load_from_bytes(buffer.getvalue())
    assert not marker.exists()


def test_safe_load_from_bytes_round_trips_a_real_tensor() -> None:
    """A benign embedded tensor still reconstructs through the weights-only wrapper."""

    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    restored = _safe_load_from_bytes(buffer.getvalue())
    assert torch.equal(restored, tensor)
