"""Round-23 security regression: load-time RCE via the unpickler's TYPE branch.

The r22 fix positive-allowlisted first-party CALLABLES, but the bundle
``metadata.pkl`` unpickler still admitted ANY ``torchlens.*`` TYPE unconditionally
("reconstruction path, never invoked with attacker args"). That premise is FALSE: a
pickle ``REDUCE`` INVOKES the admitted type (or its reconstructed instance) with
attacker arguments. ``torchlens.intervention.save.LazyImportRef`` is a frozen
dataclass whose ``__call__`` resolves an ARBITRARY import path under a trust flag
read from its OWN pickled field, so a crafted ``metadata.pkl`` reconstructs
``LazyImportRef(import_path="os:system", trust_custom_callables=True)`` and
REDUCE-invokes it -> ``os.system`` on plain ``tl.load`` (a confirmed load-time RCE;
marker file created).

The three-layer fix these tests pin:

* LAYER 1 (primary): the torchlens TYPE branch is a positive allowlist of
  vetted-inert DATA types (``_SAFE_TORCHLENS_TYPES``). ``LazyImportRef`` and every
  other non-allowlisted / import-sink type are DENIED; legit metadata data types
  (``Op`` / ``Trace`` / ``FacetRegistrySnapshot`` / rename targets) still admit.
* LAYER 2: ``LazyImportRef.__setstate__`` forces ``trust_custom_callables=False`` /
  ``allowed_custom_callable_modules=None`` on ANY unpickle reconstruction, so a
  forged instance can never carry attacker trust (the normal ``__init__``-threaded
  trusted-load path is unaffected).
* LAYER 3: the resolver ``_enforce_foreign_trust`` denies dangerous modules
  (``os`` / ``subprocess`` / ``builtins`` / ``importlib`` / ...) even on the
  trust-satisfied path -- trust never authorizes ``import os``.

Prior deny closures (r21 ``_module_is_installed``, ``os.system``) still hold.
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
import torchlens.utils as tlutils
from torchlens._io import TorchLensIOError
from torchlens._io._safe_unpickle import SafeBundleUnpickler
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_import_ref
from torchlens.intervention.save import LazyImportRef
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


class _SmallLinear(nn.Module):
    """Minimal model with a param + persistent buffer for a runnable bundle."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Return a relu'd, scaled linear projection."""

        return torch.relu(self.lin(v)) * self.scale


def _has_unpickling_error(exc: BaseException | None) -> bool:
    """Return whether an UnpicklingError appears anywhere in the cause/context chain."""

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, pickle.UnpicklingError):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class _LazyImportRefTypeReduceGadget:
    """``__reduce__`` -> a forged ``LazyImportRef`` INVOKED with an attacker command.

    On a vulnerable tree the type branch admits ``LazyImportRef``, pickle
    reconstructs the instance with attacker-forged ``trust_custom_callables=True``,
    and the REDUCE INVOKES it -> ``os.system(<cmd>)`` at ``tl.load`` time.
    """

    def __init__(self, command: str) -> None:
        self._command = command

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (
            LazyImportRef(import_path="os:system", trust_custom_callables=True),
            (self._command,),
        )


# --------------------------------------------------------------------------- #
# LAYER 1 -- the confirmed load-time RCE through real tl.load.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_lazyimportref_type_reduce_rce_denied_through_tl_load(tmp_path: Path) -> None:
    """A forged LazyImportRef REDUCE in metadata.pkl raises + never runs os.system."""

    marker = tmp_path / "R23_PWNED"
    victim = tmp_path / "victim.tlspec"
    tl.trace(_SmallLinear(), torch.randn(2, 4), save=tl.func("relu"), capture=_CAP).save(
        victim, level="runnable", include_weights=True
    )
    with (victim / "metadata.pkl").open("wb") as handle:
        pickle.dump(
            _LazyImportRefTypeReduceGadget(f"touch {marker}"),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    raised: BaseException | None = None
    try:
        tl.load(victim)
    except (TorchLensIOError, pickle.UnpicklingError) as exc:
        raised = exc

    assert raised is not None, "forged LazyImportRef type-reduce was not blocked"
    assert _has_unpickling_error(raised), "denial must surface an UnpicklingError"
    assert not marker.exists(), "os.system marker written -> RCE via type-reduce"


@pytest.mark.smoke
def test_lazyimportref_type_denied_at_unpickler() -> None:
    """The LazyImportRef TYPE is denied at find_class (the reduce callable is unresolvable)."""

    unpickler = SafeBundleUnpickler(io.BytesIO(b""))
    with pytest.raises(pickle.UnpicklingError):
        unpickler.find_class("torchlens.intervention.save", "LazyImportRef")


@pytest.mark.smoke
def test_legit_data_types_and_snapshot_still_admit() -> None:
    """Vetted-inert data types + FacetRegistrySnapshot + rename targets still resolve."""

    unpickler = SafeBundleUnpickler(io.BytesIO(b""))
    admitted = [
        ("torchlens.data_classes.op", "Op"),
        ("torchlens.data_classes.op", "TensorLog"),  # locked rename alias of Op
        ("torchlens.data_classes.trace", "Trace"),
        ("torchlens.data_classes.grad_fn_call", "GradFnCall"),  # rename target
        ("torchlens.intervention._super.super_op", "SuperOp"),  # rename target
        ("torchlens.intervention.types", "InterventionSpec"),
        ("torchlens.intervention.types", "HelperSpec"),
        ("torchlens.semantic.facets", "FacetRegistrySnapshot"),
        ("torchlens.ir.container_registry", "ContainerRecord"),
    ]
    for module, name in admitted:
        resolved = unpickler.find_class(module, name)
        assert isinstance(resolved, type), f"{module}:{name} should admit as a type"


@pytest.mark.smoke
def test_r21_and_import_sink_gadgets_still_denied() -> None:
    """The r21 private import gadget and other torchlens callables stay denied."""

    unpickler = SafeBundleUnpickler(io.BytesIO(b""))
    # r21 gadget: torchlens.utils._module_is_installed (importlib.import_module sink).
    assert hasattr(tlutils, "_module_is_installed")
    with pytest.raises(pickle.UnpicklingError):
        unpickler.find_class("torchlens.utils", "_module_is_installed")
    # os.system reached off a torchlens module path is still denied (real module posix).
    with pytest.raises(pickle.UnpicklingError):
        unpickler.find_class("torchlens._io._safe_unpickle", "io")


# --------------------------------------------------------------------------- #
# LAYER 2 -- LazyImportRef self-defense on unpickle.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_lazyimportref_setstate_sanitizes_forged_trust() -> None:
    """Any unpickle reconstruction forces trust off, ignoring pickled trust fields."""

    forged = LazyImportRef(
        import_path="os:system",
        trust_custom_callables=True,
        allowed_custom_callable_modules=("os",),
    )
    restored = pickle.loads(pickle.dumps(forged))
    assert restored.import_path == "os:system", "import_path preserved (inert until call)"
    assert restored.trust_custom_callables is False, "forged trust must be sanitized off"
    assert restored.allowed_custom_callable_modules is None, "forged module allowlist cleared"


def test_lazyimportref_normal_init_preserves_trust(tmp_path: Path) -> None:
    """The normal __init__-threaded trusted-load path is unaffected by LAYER 2.

    LAYER 2 only fires on ``__setstate__`` (unpickle), so a legitimately-trusted
    reference constructed in-process still carries its trust context and resolves a
    non-dangerous trusted module.
    """

    module_dir = tmp_path / "trusted_pkg"
    module_dir.mkdir()
    (module_dir / "r23_trusted_recipe.py").write_text("def double(x):\n    return x * 2\n")
    sys.path.insert(0, str(module_dir))
    sys.modules.pop("r23_trusted_recipe", None)
    try:
        ref = LazyImportRef(
            import_path="r23_trusted_recipe:double",
            trust_custom_callables=True,
        )
        assert ref.trust_custom_callables is True, "in-process trust preserved (not unpickled)"
        assert ref(21) == 42, "trusted non-dangerous module resolves + runs"
    finally:
        sys.path.remove(str(module_dir))
        sys.modules.pop("r23_trusted_recipe", None)


# --------------------------------------------------------------------------- #
# LAYER 3 -- resolver dangerous-module denylist even under trust.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize(
    "import_path",
    [
        "os:system",
        "subprocess:call",
        "builtins:eval",
        "importlib:import_module",
        "shutil:rmtree",
        "socket:socket",
    ],
)
def test_dangerous_modules_denied_even_with_forged_trust(import_path: str) -> None:
    """A dangerous module is denied even when trust is explicitly satisfied."""

    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path, trust_custom_callables=True)
    # Denied even when explicitly allowlisted -- trust never authorizes these modules.
    module_name = import_path.split(":", 1)[0]
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(import_path, allowed_custom_callable_modules={module_name})


def test_legit_trusted_custom_module_still_resolves(tmp_path: Path) -> None:
    """A non-dangerous custom module still resolves under trust (LAYER 3 not over-broad)."""

    module_dir = tmp_path / "safe_pkg"
    module_dir.mkdir()
    (module_dir / "r23_safe_recipe.py").write_text("def triple(x):\n    return x * 3\n")
    sys.path.insert(0, str(module_dir))
    sys.modules.pop("r23_safe_recipe", None)
    try:
        via_trust = resolve_import_ref("r23_safe_recipe:triple", trust_custom_callables=True)
        assert via_trust(4) == 12
        via_allow = resolve_import_ref(
            "r23_safe_recipe:triple", allowed_custom_callable_modules={"r23_safe_recipe"}
        )
        assert via_allow(5) == 15
    finally:
        sys.path.remove(str(module_dir))
        sys.modules.pop("r23_safe_recipe", None)


# --------------------------------------------------------------------------- #
# End-to-end: a legit runnable bundle + FacetRegistrySnapshot still loads + runs.
# --------------------------------------------------------------------------- #


def test_legit_runnable_bundle_round_trips(tmp_path: Path) -> None:
    """A real runnable bundle (data types + facet snapshot) still loads and runs."""

    path = tmp_path / "legit.tlspec"
    tl.trace(_SmallLinear(), torch.randn(2, 4), save=tl.func("relu"), capture=_CAP).save(
        path, level="runnable", include_weights=True, include_activations=True
    )

    loaded = tl.load(path)
    assert loaded.__class__.__name__ == "Trace"
    result = loaded.run(inputs=torch.randn(2, 4))
    assert result.__class__.__name__ == "RunResult"
