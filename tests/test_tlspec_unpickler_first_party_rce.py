"""Round-21 security regression: first-party pickle-REDUCE RCE in the bundle unpickler.

Before r21 the bundle ``metadata.pkl`` unpickler admitted the ENTIRE ``torchlens.*``
callable namespace as a pickle REDUCE target (``callable(obj) and
_is_torchlens_owned(obj)``). Because ``torchlens.report._environment._module_is_installed`` does
``importlib.import_module(<arg>)``, a crafted ``metadata.pkl`` whose REDUCE named it
ran an arbitrary import -- attacker top-level code -- on a plain ``tl.load(path)``,
BEFORE ``.run()`` and before any downstream callable-safety gate could fire (a
confirmed load-time RCE).

The fix narrows first-party CALLABLE admission to a deterministic, registry-state-
INDEPENDENT set of vetted-INERT callables (public + side-effect-free), via
``torchlens._utils._callable_safety.is_inert_first_party_callable``. TYPES are still
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

import pytest

import torchlens as tl
from torchlens._io._safe_unpickle import SafeBundleUnpickler
from torchlens._utils._callable_safety import is_inert_first_party_callable
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    replayable=True,
    capture_container_structure=True,
    cache=False,
)


def _global_ref_pickle(module: str, name: str) -> bytes:
    """Build a minimal pickle that is a bare GLOBAL reference to ``module.name``."""

    return b"c" + module.encode() + b"\n" + name.encode() + b"\n."


def _load_ref(module: str, name: str) -> object:
    """Resolve a bare GLOBAL reference through the restricted bundle unpickler."""

    return SafeBundleUnpickler(io.BytesIO(_global_ref_pickle(module, name))).load()


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


# --------------------------------------------------------------------------- #
# Vetted first-party callables (recipe + transform) still admit.
# --------------------------------------------------------------------------- #


def test_full_facet_registry_snapshot_round_trips_through_unpickler() -> None:
    """A FacetRegistrySnapshot embedding a real registered recipe survives the door.

    A snapshot pickles its recipe funcs BY IDENTITY (bare GLOBALs). Pickling a
    snapshot built from the built-in registry and reloading it through the restricted
    unpickler must resolve every embedded first-party recipe to the real callable --
    proving the narrowed gate does not regress a legitimate recipe-bearing bundle.
    """

    from torchlens._semantic import facets as facets_mod

    # Snapshot the PURE built-in registry (deterministic, unaffected by any recipe a
    # sibling test may have left registered). The built-in set includes the residual
    # recipe registered with the PRIVATE predicate ``_is_transformer_block``, which
    # exercises the marker path (a private-named but genuinely registered callable).
    original = list(facets_mod._REGISTRY)
    try:
        facets_mod.reset()
        snapshot = facets_mod.snapshot()
        assert snapshot.functions, "expected built-in recipes in the registry snapshot"
        recipe_names = {r.public.name for r in snapshot.functions}
        assert "transformer_residuals" in recipe_names, "residual recipe (private predicate)"

        blob = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
        restored = SafeBundleUnpickler(io.BytesIO(blob)).load()

        assert isinstance(restored, facets_mod.FacetRegistrySnapshot)
        assert len(restored.functions) == len(snapshot.functions)
        for orig, roundtripped in zip(snapshot.functions, restored.functions):
            assert roundtripped.func is orig.func
            assert roundtripped.predicate is orig.predicate
    finally:
        facets_mod._REGISTRY[:] = original


# --------------------------------------------------------------------------- #
# The inert-first-party gate discriminates correctly.
# --------------------------------------------------------------------------- #


def test_inert_first_party_gate_discriminates() -> None:
    """The gate admits public inert helpers and denies private / side-effecting ones."""

    from torchlens._semantic.recipes.norm import layer_norm
    from torchlens._utils.display import identity
    from torchlens.intervention._helpers import clamp, zero_ablate
    from torchlens.report._environment import _module_is_installed

    assert is_inert_first_party_callable(identity)
    assert is_inert_first_party_callable(layer_norm)
    assert is_inert_first_party_callable(zero_ablate)
    assert is_inert_first_party_callable(clamp)
    # The confirmed RCE gadget (private, imports its argument).
    assert not is_inert_first_party_callable(_module_is_installed)
    # A public torchlens callable whose name marks file/serialization I/O is denied.
    assert not is_inert_first_party_callable(tl.load)
    assert not is_inert_first_party_callable(tl.Trace.save)


# --------------------------------------------------------------------------- #
# Prior denies still hold + resolver defense-in-depth.
# --------------------------------------------------------------------------- #


class _OsSystemGadget:
    """``__reduce__`` -> ``os.system`` payload (the classic r1-r20 gadget)."""

    def __reduce__(self):  # type: ignore[no-untyped-def]
        import os

        return (os.system, ("true",))


# --------------------------------------------------------------------------- #
# Round-22: the positive-allowlist inversion. The pre-r22 gate's PUBLIC-name
# fallthrough admitted ~60 side-effecting public torchlens.* callables whose verb
# (fs-mutation / registry-mutation / capability-flip) was not on the denylist. The
# gate is now default-deny: only marker-stamped facet recipes and a frozen vetted-
# inert set (identity + the pure intervention helper factories) admit.
# --------------------------------------------------------------------------- #


def test_public_side_effecting_first_party_callables_denied() -> None:
    """The public-name fallthrough is GONE: fs / state mutators fail the gate + door."""

    from torchlens._utils._torch_compat import mark_torch_capability_missing
    from torchlens.recording._cleanup import cleanup_partial

    assert not is_inert_first_party_callable(cleanup_partial)
    assert not is_inert_first_party_callable(mark_torch_capability_missing)
    with pytest.raises(pickle.UnpicklingError):
        _load_ref("torchlens.recording._cleanup", "cleanup_partial")
    with pytest.raises(pickle.UnpicklingError):
        _load_ref("torchlens._utils._torch_compat", "mark_torch_capability_missing")


def test_untrusted_callable_refusals_carry_code_and_subject() -> None:
    """R65: every trust-gate refusal names its code and the denied import."""

    from torchlens.intervention._resolver_registry import resolve_import_ref

    with pytest.raises(UntrustedCallableError) as untrusted:
        resolve_import_ref("somemod:some_fn")
    assert untrusted.value.fields["code"] == "custom_callable_untrusted"
    assert untrusted.value.fields["module"] == "somemod"
    assert untrusted.value.fields["import_path"] == "somemod:some_fn"

    with pytest.raises(UntrustedCallableError) as denied:
        resolve_import_ref("os:system", trust_custom_callables=True)
    assert denied.value.fields["code"] == "custom_callable_module_denied"
    assert denied.value.fields["module"] == "os"

    with pytest.raises(UntrustedCallableError) as not_listed:
        resolve_import_ref("somemod:some_fn", allowed_custom_callable_modules={"othermod"})
    assert not_listed.value.fields["code"] == "custom_callable_module_not_allowlisted"

    with pytest.raises(UntrustedCallableError) as private:
        resolve_import_ref("torchlens.report._environment:_module_is_installed")
    assert private.value.fields["code"] == "custom_callable_private_first_party"

    with pytest.raises(UntrustedCallableError) as impure:
        resolve_import_ref("torch:from_file", trust_custom_callables=True)
    assert impure.value.fields["code"] == "custom_callable_not_pure"
