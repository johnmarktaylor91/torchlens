"""Round-21 security regression: first-party pickle-REDUCE RCE in the bundle unpickler.

Before r21 the bundle ``metadata.pkl`` unpickler admitted the ENTIRE ``torchlens.*``
callable namespace as a pickle REDUCE target (``callable(obj) and
_is_torchlens_owned(obj)``). Because ``torchlens.utils._module_is_installed`` does
``importlib.import_module(<arg>)``, a crafted ``metadata.pkl`` whose REDUCE named it
ran an arbitrary import -- attacker top-level code -- on a plain ``tl.load(path)``,
BEFORE ``.run()`` and before any downstream callable-safety gate could fire (a
confirmed load-time RCE).

The fix narrows first-party CALLABLE admission to a deterministic, registry-state-
INDEPENDENT set of vetted-INERT callables (public + side-effect-free), via
``torchlens.utils._callable_safety.is_inert_first_party_callable``. TYPES are still
admitted (reconstruction path, never invoked with attacker args). The private
import gadget is denied; genuine public facet recipes / transforms / helpers still
resolve. The same narrowing is applied defense-in-depth to the intervention
resolver's torchlens-owned auto-trust.

These tests pin: the ``_module_is_installed`` REDUCE gadget is DENIED (through real
``tl.load`` AND the direct unpickler, executing nothing), a vetted first-party recipe
/ transform still admits, the resolver denies the gadget while still resolving a
legit public helper, and every prior deny (``os.system``) still holds.
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
from torchlens.intervention.save import _resolve_import_ref
from torchlens.options import CaptureOptions
from torchlens.utils._callable_safety import is_inert_first_party_callable

_CAP = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


def _global_ref_pickle(module: str, name: str) -> bytes:
    """Build a minimal pickle that is a bare GLOBAL reference to ``module.name``."""

    return b"c" + module.encode() + b"\n" + name.encode() + b"\n."


def _load_ref(module: str, name: str) -> object:
    """Resolve a bare GLOBAL reference through the restricted bundle unpickler."""

    return SafeBundleUnpickler(io.BytesIO(_global_ref_pickle(module, name))).load()


class _ModuleInstalledGadget:
    """``__reduce__`` -> ``torchlens.utils._module_is_installed('<attacker>')``.

    On a vulnerable tree the unpickler admits the private helper and pickle INVOKES
    it, importing the attacker module (whose top-level code writes a sentinel).
    """

    def __init__(self, attacker_module: str) -> None:
        self._attacker = attacker_module

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (tlutils._module_is_installed, (self._attacker,))


def _has_unpickling_error(exc: BaseException | None) -> bool:
    """Return whether an UnpicklingError appears anywhere in the cause/context chain."""

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, pickle.UnpicklingError):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


# --------------------------------------------------------------------------- #
# The confirmed load-time RCE: crafted metadata.pkl -> private import gadget.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_module_is_installed_reduce_gadget_denied_through_tl_load(tmp_path: Path) -> None:
    """A crafted bundle whose metadata REDUCE is the import gadget raises + no import."""

    attacker_dir = tmp_path / "attacker_pkg"
    attacker_dir.mkdir()
    sentinel = tmp_path / "PWNED_R21"
    attacker_name = "evil_r21_first_party_rce_module"
    (attacker_dir / f"{attacker_name}.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('pwned')\n"
    )
    sys.path.insert(0, str(attacker_dir))
    sys.modules.pop(attacker_name, None)
    try:
        victim = tmp_path / "victim.tlspec"
        tl.trace(nn.Linear(4, 4), torch.randn(1, 4), capture=_CAP).save(
            victim, level="runnable", include_weights=True
        )
        with (victim / "metadata.pkl").open("wb") as handle:
            pickle.dump(
                _ModuleInstalledGadget(attacker_name), handle, protocol=pickle.HIGHEST_PROTOCOL
            )

        raised: BaseException | None = None
        try:
            tl.load(victim)
        except (TorchLensIOError, pickle.UnpicklingError) as exc:
            raised = exc

        assert raised is not None, "load-time import gadget was not blocked"
        assert _has_unpickling_error(raised), "denial must surface an UnpicklingError"
        assert not sentinel.exists(), "attacker sentinel written -> gadget executed (RCE)"
        assert attacker_name not in sys.modules, "attacker module imported -> gadget executed"
    finally:
        sys.path.remove(str(attacker_dir))
        sys.modules.pop(attacker_name, None)


@pytest.mark.smoke
def test_module_is_installed_denied_by_direct_unpickler() -> None:
    """The restricted unpickler refuses the private import gadget as a bare GLOBAL."""

    with pytest.raises(pickle.UnpicklingError):
        _load_ref("torchlens.utils", "_module_is_installed")


# --------------------------------------------------------------------------- #
# Vetted first-party callables (recipe + transform) still admit.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_vetted_facet_recipe_admitted() -> None:
    """A genuine public first-party facet recipe still resolves to the real callable."""

    from torchlens.semantic.recipes.norm import layer_norm

    assert _load_ref("torchlens.semantic.recipes.norm", "layer_norm") is layer_norm


@pytest.mark.smoke
def test_vetted_transform_helper_admitted() -> None:
    """A public first-party transform (identity) still resolves (prior r4 contract)."""

    from torchlens.utils.display import identity

    assert _load_ref("torchlens.utils.display", "identity") is identity


@pytest.mark.smoke
def test_full_facet_registry_snapshot_round_trips_through_unpickler() -> None:
    """A FacetRegistrySnapshot embedding a real registered recipe survives the door.

    A snapshot pickles its recipe funcs BY IDENTITY (bare GLOBALs). Pickling a
    snapshot built from the built-in registry and reloading it through the restricted
    unpickler must resolve every embedded first-party recipe to the real callable --
    proving the narrowed gate does not regress a legitimate recipe-bearing bundle.
    """

    from torchlens.semantic import facets as facets_mod

    # Snapshot the PURE built-in registry (deterministic, unaffected by any recipe a
    # sibling test may have left registered). The built-in set includes the residual
    # recipe registered with the PRIVATE predicate ``_is_transformer_block``, which
    # exercises the marker path (a private-named but genuinely registered callable).
    original = list(facets_mod._REGISTRY)
    try:
        facets_mod.reset()
        snapshot = facets_mod.snapshot()
        assert snapshot.recipes, "expected built-in recipes in the registry snapshot"
        recipe_names = {r.public.recipe_name for r in snapshot.recipes}
        assert "transformer_residuals" in recipe_names, "residual recipe (private predicate)"

        blob = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
        restored = SafeBundleUnpickler(io.BytesIO(blob)).load()

        assert isinstance(restored, facets_mod.FacetRegistrySnapshot)
        assert len(restored.recipes) == len(snapshot.recipes)
        for orig, roundtripped in zip(snapshot.recipes, restored.recipes):
            assert roundtripped.func is orig.func
            assert roundtripped.predicate is orig.predicate
    finally:
        facets_mod._REGISTRY[:] = original


# --------------------------------------------------------------------------- #
# The inert-first-party gate discriminates correctly.
# --------------------------------------------------------------------------- #


def test_inert_first_party_gate_discriminates() -> None:
    """The gate admits public inert helpers and denies private / side-effecting ones."""

    from torchlens.intervention.helpers import clamp, zero_ablate
    from torchlens.semantic.recipes.norm import layer_norm
    from torchlens.utils import _module_is_installed
    from torchlens.utils.display import identity

    assert is_inert_first_party_callable(identity)
    assert is_inert_first_party_callable(layer_norm)
    assert is_inert_first_party_callable(zero_ablate)
    assert is_inert_first_party_callable(clamp)
    # The confirmed RCE gadget (private, imports its argument).
    assert not is_inert_first_party_callable(_module_is_installed)
    # A public torchlens callable whose name marks file/serialization I/O is denied.
    assert not is_inert_first_party_callable(tl.load)
    assert not is_inert_first_party_callable(tl.save)


# --------------------------------------------------------------------------- #
# Prior denies still hold + resolver defense-in-depth.
# --------------------------------------------------------------------------- #


class _OsSystemGadget:
    """``__reduce__`` -> ``os.system`` payload (the classic r1-r20 gadget)."""

    def __reduce__(self):  # type: ignore[no-untyped-def]
        import os

        return (os.system, ("true",))


@pytest.mark.smoke
def test_os_system_still_denied() -> None:
    """The r1-r20 closure holds: an os.system REDUCE / bare ref is still refused."""

    # Bare reference to the dangerous module is module-denied.
    with pytest.raises(pickle.UnpicklingError):
        _load_ref("posix", "system")
    # A __reduce__ os.system gadget is refused before it can execute.
    blob = pickle.dumps(_OsSystemGadget(), protocol=pickle.HIGHEST_PROTOCOL)
    with pytest.raises(pickle.UnpicklingError):
        SafeBundleUnpickler(io.BytesIO(blob)).load()


def test_resolver_denies_first_party_import_gadget_but_resolves_helper() -> None:
    """Defense-in-depth: the resolver denies the gadget, still resolves a public helper."""

    from torchlens.intervention.helpers import zero_ablate

    assert _resolve_import_ref("torchlens.intervention.helpers:zero_ablate") is zero_ablate
    with pytest.raises((UntrustedCallableError, Exception)) as excinfo:
        _resolve_import_ref("torchlens.utils:_module_is_installed")
    # The refusal must originate from the untrusted-callable gate.
    chain: list[BaseException] = []
    exc: BaseException | None = excinfo.value
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        chain.append(exc)
        exc = exc.__cause__ or exc.__context__
    assert any(isinstance(item, UntrustedCallableError) for item in chain)
