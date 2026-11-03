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

import io
import pickle
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io._safe_unpickle import (
    SafeBundleUnpickler,
    _safe_getattr,
)
from torchlens._utils.display import identity
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    replayable=True,
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


class _OsSystemGadget:
    """``__reduce__`` -> ``os.system`` payload that would run a shell command."""

    def __init__(self, marker: Path) -> None:
        self.marker = str(marker)

    def __reduce__(self):  # type: ignore[no-untyped-def]
        import os

        return (os.system, (f"touch {self.marker}",))


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
    tl.trace(cf_a, x, save="all").save(audit, level="audit")
    paths["analysis"] = audit

    portable = tmp_path / "portable"
    cf_p = _ControlFlow()
    keep_alive.append(cf_p)
    tl.trace(cf_p, x, save="all").save(portable, level="portable")
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


def test_control_flow_bundle_round_trips(tmp_path: Path) -> None:
    """A control-flow bundle keeps tensors out of metadata and restores arm bindings."""

    x = torch.randn(2, 4)
    bundle = tmp_path / "cf"
    tl.trace(_ControlFlow(), x, save="all").save(bundle, level="portable")
    assert b"_load_from_bytes" not in _metadata_path(bundle).read_bytes()
    loaded = tl.load(bundle)
    assert loaded is not None
    for conditional in loaded.graph.conditionals:
        for arm in conditional.arms:
            assert arm._trace is loaded


# --------------------------------------------------------------------------- #
# Malicious metadata.pkl is denied and executes nothing (through real tl.load).
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Unit-level guards on the two safe wrappers.
# --------------------------------------------------------------------------- #


def test_safe_getattr_blocks_non_allowlisted_target() -> None:
    """``_safe_getattr`` refuses any target outside the torch C holder classes."""

    with pytest.raises(pickle.UnpicklingError):
        _safe_getattr(identity, "__globals__")
    with pytest.raises(pickle.UnpicklingError):
        _safe_getattr(pickle, "loads")


def _benign_module_level_callable(value: str) -> str:
    """A benign module-level callable used to exercise foreign-callable gating."""

    return f"resolved:{value}"


class _ForeignCallableGadget:
    """``__reduce__`` referencing a benign, NON-registered foreign callable."""

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (_benign_module_level_callable, ("ok",))


def _dump(obj: object) -> bytes:
    """Pickle an object to bytes."""

    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def test_foreign_callable_admitted_with_module_allowlist() -> None:
    """A narrow ``allowed_custom_callable_modules`` admits only the listed module."""

    data = _dump(_ForeignCallableGadget())
    result = SafeBundleUnpickler(
        io.BytesIO(data), allowed_custom_callable_modules={__name__}
    ).load()
    assert result == "resolved:ok"

    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(
            io.BytesIO(data), allowed_custom_callable_modules={"some.other.module"}
        ).load()


def _global_ref_pickle(module: str, name: str) -> bytes:
    """Build a minimal pickle that is a bare GLOBAL reference to ``module.name``.

    A ``FacetRegistrySnapshot`` pickles its recipe funcs by identity, i.e. as a bare
    GLOBAL (not a called ``__reduce__``). This hand-writes that opcode so the load
    can be exercised for a module that is deliberately NOT importable, proving no
    import occurs at unpickle.
    """

    return b"c" + module.encode() + b"\n" + name.encode() + b"\n."
