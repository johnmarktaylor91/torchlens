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
    _DeferredForeignCallable,
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


# --------------------------------------------------------------------------- #
# Widened-allowlist regressions and the foreign custom-callable trust policy.
# --------------------------------------------------------------------------- #


def test_allowlist_admits_inert_stdlib_and_numpy_globals() -> None:
    """``builtins.object`` and NumPy numeric reconstruction remain resolvable."""

    import numpy as np

    for module, name in (("builtins", "object"), ("numpy", "dtype")):
        blob = io.BytesIO()
        pickle.dump(("marker",), blob)  # ensure a valid stream to attach unpickler
        unpickler = SafeBundleUnpickler(io.BytesIO(blob.getvalue()))
        assert unpickler.find_class(module, name) is (object if name == "object" else np.dtype)


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


def test_foreign_callable_denied_by_default() -> None:
    """A foreign callable that is not a live registered recipe is denied by default."""

    data = _dump(_ForeignCallableGadget())
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(data)).load()


def test_foreign_callable_admitted_with_trust_flag() -> None:
    """``trust_custom_callables=True`` admits a foreign already-imported callable."""

    data = _dump(_ForeignCallableGadget())
    result = SafeBundleUnpickler(io.BytesIO(data), trust_custom_callables=True).load()
    assert result == "resolved:ok"


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


def test_foreign_recipe_bare_reference_load_tolerated_without_import() -> None:
    """A foreign recipe bare reference load-tolerates as a placeholder, no import.

    This is the exact ordering-dependent failure surface: a saved snapshot embeds a
    foreign (user/test) recipe by identity. Under a default untrusted load the
    foreign module MUST NOT be imported (the RCE surface), yet the trace must still
    load structurally -- the reference resolves to an inert deferred placeholder.
    """

    import sys

    module_name = "torchlens_r4_absent_recipe_module"
    assert module_name not in sys.modules
    blob = _global_ref_pickle(module_name, "_probe_recipe")

    resolved = SafeBundleUnpickler(io.BytesIO(blob)).load()
    assert isinstance(resolved, _DeferredForeignCallable)
    assert resolved.module == module_name
    assert resolved.qualname == "_probe_recipe"
    # The foreign module was NEVER imported at unpickle time.
    assert module_name not in sys.modules
    # Execute-deny: calling the deferred recipe fails closed.
    with pytest.raises(pickle.UnpicklingError):
        resolved("record")


def test_torchlens_owned_recipe_admitted_without_trust() -> None:
    """A torchlens-owned recipe/transform resolves to the real callable, no trust."""

    from torchlens.utils.display import identity

    blob = _global_ref_pickle("torchlens.utils.display", "identity")
    resolved = SafeBundleUnpickler(io.BytesIO(blob)).load()
    assert resolved is identity


def test_foreign_recipe_admission_ignores_live_registry() -> None:
    """Admission is DETERMINISTIC: a live registration does not change the outcome.

    The removed live-registry snapshot made loads ordering-dependent (pass alone,
    fail in smoke). A foreign recipe reference now resolves to the same inert
    placeholder whether or not the recipe is currently registered.
    """

    from torchlens.semantic import facets

    module_name = "torchlens_r4_registry_probe_module"
    blob = _global_ref_pickle(module_name, "_reg_probe")

    def _reg_probe(value: object) -> dict:
        return {"seen": value}

    _reg_probe.__module__ = module_name  # simulate a foreign-module recipe

    entry = facets._RegisteredRecipe(
        public=None,  # type: ignore[arg-type]
        func=_reg_probe,
        predicate=None,
        declared_facets=(),
        order=-1,
    )
    facets._REGISTRY.append(entry)
    try:
        resolved = SafeBundleUnpickler(io.BytesIO(blob)).load()
    finally:
        facets._REGISTRY.remove(entry)
    assert isinstance(resolved, _DeferredForeignCallable)
