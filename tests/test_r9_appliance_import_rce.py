"""Round-9 security regression: foreign import at load via appliance modules.

Both untrusted-deserialization gates treated EVERY ``torchlens.*`` submodule as
inert first-party code safe to import + inspect without a trust opt-in:

* ``SafeBundleUnpickler.find_class`` (``torchlens/_io/_safe_unpickle.py``) --
  the ``torchlens`` branch called ``super().find_class(module, name)``, and
  ``pickle.Unpickler.find_class`` IMPORTS ``module`` before the global is
  (ultimately) rejected.
* ``resolve_function_registry_key`` (``torchlens/intervention/resolver.py``) --
  the ``path_claims_torchlens`` branch called
  ``importlib.import_module(module_name)`` unconditionally.

That premise is FALSE for the extras-gated appliance packages:
``torchlens/neuro/__init__.py`` imports ``rsatoolbox`` / ``brainscore_core`` at
import time, and ``torchlens/notebook/__init__.py`` imports ``IPython`` /
``jupyter_client`` at import time. A crafted bundle whose ``metadata.pkl``
carries a ``GLOBAL`` naming ``torchlens.neuro`` (or ``torchlens.notebook``)
therefore made a PLAIN ``tl.load()`` -- no run, no trust flags -- import those
foreign third-party packages and execute their top-level code.

MEDIUM (defense-in-depth): the fix excludes the two appliance packages from
the "torchlens is our own code" fast path at BOTH gate sites so a name
matching them falls through to the SAME deny-by-default / trust-opt-in /
load-tolerate treatment already applied to a genuinely foreign module --
denied by NAME, never imported to make that determination -- plus makes the
appliance packages' own ``__init__`` imports lazy (deferred to first
attribute access) so an incidental ``import torchlens.neuro`` is inert too.
"""

from __future__ import annotations

import io
import pickle
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io._safe_unpickle import (
    SafeBundleUnpickler,
    _DeferredForeignCallable,
    _is_torchlens_appliance_module,
)
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import (
    _is_torchlens_appliance_module as _resolver_is_appliance_module,
)
from torchlens.intervention.resolver import resolve_function_registry_key
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _global_only_pickle(module: str, name: str) -> bytes:
    """Hand-assemble a bare ``GLOBAL(module, name)`` pickle stream (no REDUCE)."""

    return (
        pickle.PROTO
        + bytes([2])
        + pickle.GLOBAL
        + (module + "\n" + name + "\n").encode()
        + pickle.STOP
    )


def _drop_foreign_modules() -> None:
    """Remove appliance third-party deps from ``sys.modules`` so the test is honest."""

    for foreign in ("rsatoolbox", "brainscore_core", "IPython", "jupyter_client"):
        for loaded in list(sys.modules):
            if loaded == foreign or loaded.startswith(f"{foreign}."):
                del sys.modules[loaded]
    for tl_appliance in ("torchlens.neuro", "torchlens.notebook"):
        for loaded in list(sys.modules):
            if loaded == tl_appliance or loaded.startswith(f"{tl_appliance}."):
                del sys.modules[loaded]


# --------------------------------------------------------------------------- #
# Gate site 1: SafeBundleUnpickler.find_class (metadata.pkl unpickle).
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("appliance_module", ["torchlens.neuro", "torchlens.notebook"])
def test_safe_unpickler_defers_appliance_module_without_importing(appliance_module: str) -> None:
    """A GLOBAL naming an appliance module is load-tolerated without ever importing it.

    Mirrors the treatment of a genuinely foreign module: the reference is
    resolved to an inert ``_DeferredForeignCallable`` placeholder by default
    (never denied outright, since a trust opt-in could still admit it), but
    the appliance module itself is NEVER imported to reach that decision, and
    actually calling the placeholder fails closed.
    """

    _drop_foreign_modules()
    payload = _global_only_pickle(appliance_module, "__name__")
    resolved = SafeBundleUnpickler(io.BytesIO(payload)).load()

    assert isinstance(resolved, _DeferredForeignCallable)
    assert resolved.module == appliance_module
    assert appliance_module not in sys.modules
    assert "rsatoolbox" not in sys.modules
    assert "brainscore_core" not in sys.modules
    assert "IPython" not in sys.modules
    assert "jupyter_client" not in sys.modules

    with pytest.raises(pickle.UnpicklingError):
        resolved()


@pytest.mark.smoke
def test_end_to_end_load_does_not_import_appliance_deps(tmp_path: Path) -> None:
    """A tampered bundle naming ``torchlens.neuro`` never imports rsatoolbox at ``tl.load()``.

    This mirrors the exact repro: a legitimate runnable bundle whose
    ``metadata.pkl`` is overwritten with a raw pickle GLOBAL referencing
    ``torchlens.neuro``. A plain ``tl.load()`` (no run, no trust flags) must
    either fail closed or otherwise never import the foreign package.
    """

    _drop_foreign_modules()
    model = _Linear()
    x = torch.randn(2, 4)
    bundle = tmp_path / "m.tlspec"
    tl.trace(model, x, capture=_CAP).save(bundle, level="runnable", include_weights=True)

    metadata_path = bundle / "metadata.pkl"
    metadata_path.write_bytes(_global_only_pickle("torchlens.neuro", "__name__"))

    assert "rsatoolbox" not in sys.modules
    try:
        tl.load(bundle)
    except Exception:
        pass
    assert "rsatoolbox" not in sys.modules
    assert "brainscore_core" not in sys.modules


@pytest.mark.parametrize("appliance_module", ["torchlens.neuro", "torchlens.notebook"])
def test_safe_unpickler_admits_appliance_module_only_under_explicit_trust(
    appliance_module: str,
) -> None:
    """Explicit trust opt-in still permits importing an appliance module (least-privilege escape hatch)."""

    _drop_foreign_modules()
    payload = _global_only_pickle(appliance_module, "__name__")
    resolved = SafeBundleUnpickler(
        io.BytesIO(payload), allowed_custom_callable_modules={appliance_module}
    ).load()
    assert resolved == appliance_module
    assert appliance_module in sys.modules


def test_safe_unpickler_still_denies_prior_dangerous_globals() -> None:
    """Prior gate denials are unaffected: ``os.system`` remains blocked."""

    payload = _global_only_pickle("os", "system")
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(payload)).load()


def test_safe_unpickler_still_admits_torchlens_types() -> None:
    """Non-appliance ``torchlens.*`` classes still resolve (unaffected)."""

    from torchlens.data_classes.trace import Trace

    payload = _global_only_pickle("torchlens.data_classes.trace", "Trace")
    resolved = SafeBundleUnpickler(io.BytesIO(payload)).load()
    assert resolved is Trace


def test_is_torchlens_appliance_module_classification() -> None:
    """Appliance-module detection matches by exact name and submodule prefix."""

    assert _is_torchlens_appliance_module("torchlens.neuro")
    assert _is_torchlens_appliance_module("torchlens.neuro.submodule")
    assert _is_torchlens_appliance_module("torchlens.notebook")
    assert _is_torchlens_appliance_module("torchlens.notebook.submodule")
    assert not _is_torchlens_appliance_module("torchlens")
    assert not _is_torchlens_appliance_module("torchlens.data_classes.trace")
    # A prefix-alike module must not false-positive.
    assert not _is_torchlens_appliance_module("torchlens.neurotransmitter")


# --------------------------------------------------------------------------- #
# Gate site 2: resolve_function_registry_key (intervention-spec resolver).
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("appliance_module", ["torchlens.neuro", "torchlens.notebook"])
def test_resolver_denies_appliance_module_by_name(appliance_module: str) -> None:
    """A custom key naming an appliance module is denied by default, no import."""

    _drop_foreign_modules()
    key = FunctionRegistryKey(
        "custom", "anything", "function", import_path=f"{appliance_module}:anything"
    )
    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key)
    assert appliance_module not in sys.modules
    assert "rsatoolbox" not in sys.modules
    assert "brainscore_core" not in sys.modules
    assert "IPython" not in sys.modules
    assert "jupyter_client" not in sys.modules


def test_resolver_admits_appliance_module_only_under_explicit_trust() -> None:
    """Explicit trust opt-in still permits importing an appliance module."""

    _drop_foreign_modules()
    key = FunctionRegistryKey(
        "custom", "anything", "function", import_path="torchlens.neuro:anything"
    )
    # No such public attribute exists yet, so resolution fails downstream --
    # but the important behavior is that the trust gate is what governs whether
    # the module is imported at all, not a name-based hard block.
    with pytest.raises(Exception):
        resolve_function_registry_key(key, allowed_custom_callable_modules={"torchlens.neuro"})
    assert "torchlens.neuro" in sys.modules


def test_resolver_still_denies_os_system_pivot() -> None:
    """Prior gate denial is unaffected: walking to ``os.system`` off torchlens is denied."""

    key = FunctionRegistryKey(
        "custom", "system", "function", import_path="torchlens._io.tlspec:os.system"
    )
    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key)


def test_resolver_still_admits_legit_torchlens_custom_callable() -> None:
    """A genuinely torchlens-owned custom callable still resolves without trust flags."""

    key = FunctionRegistryKey(
        "custom", "identity", "function", import_path="torchlens.utils.display:identity"
    )
    resolved = resolve_function_registry_key(key)
    assert callable(resolved)
    assert resolved.__module__.startswith("torchlens.")


def test_resolver_still_resolves_torch_and_operator_namespaces() -> None:
    """Prior fixed-namespace resolution is unaffected."""

    torch_key = FunctionRegistryKey("torch", "relu", "function")
    assert resolve_function_registry_key(torch_key) is torch.relu

    op_key = FunctionRegistryKey("operator", "add", "function")
    import operator

    assert resolve_function_registry_key(op_key) is operator.add


def test_resolver_appliance_classifier_matches_unpickler_classifier() -> None:
    """Both gate-site classifiers agree on the appliance module set (defense-in-depth parity)."""

    for name in (
        "torchlens.neuro",
        "torchlens.neuro.sub",
        "torchlens.notebook",
        "torchlens.notebook.sub",
        "torchlens",
        "torchlens.data_classes.trace",
    ):
        assert _is_torchlens_appliance_module(name) == _resolver_is_appliance_module(name)
