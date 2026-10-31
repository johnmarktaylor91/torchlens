"""S4 predicate runtime extension point: contract + registry tests.

Covers the m1 (contract core: ``PredicateProtocol`` + ``coerce_predicate``
over raw callables) and m2 (named registry: ``register_predicate`` + name
acceptance) obligations from the S4 contract
(``docs/reference/predicate_runtime.md``).

The no-behavior-change guard for shipped selector semantics is the standing
``tests/golden/selector_semantics_matrix.json`` golden (this merge touches no
selector/capture file, so the matrix must not regenerate).
"""

from __future__ import annotations

import functools
import pickle

import pytest

import torchlens as tl
from torchlens._trace_selector_helpers import _predicate_cache_key
from torchlens.capture.predicates import (
    _is_supported_followed_by_predicate,
    _keep_op_needs_alias_retry,
)
from torchlens.fastlog.exceptions import PredicateError
from torchlens.fastlog.types import CaptureSpec
from torchlens.intervention.selectors import BaseSelector
from torchlens.ir import predicate_registry
from torchlens.ir.predicate import RetroactiveCaptureDecision
from torchlens.ir.predicate_registry import (
    coerce_predicate,
    register_predicate,
)

SLOTS = ("save", "halt", "until")


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate the process-global user registry per test."""

    predicate_registry._reset_for_tests()
    yield
    predicate_registry._reset_for_tests()


def _plain_predicate(ctx):
    """Module-level plain predicate (also the pickle-reference target)."""

    return True


class _SlotsCallable:
    """``__slots__`` callable instance (attribute assignment impossible)."""

    __slots__ = ()

    def __call__(self, ctx):
        return False


class _MethodHolder:
    """Bound-method predicate source."""

    def predicate(self, ctx):
        return None


def _selector_bearing_closure():
    """Closure carrying a ``.selector`` attribute (capture-introspected)."""

    inner = tl.func("relu")

    def predicate(ctx):
        return bool(inner(ctx))

    predicate.selector = inner
    return predicate


# ---------------------------------------------------------------------------
# m1: coerce_predicate — closed-domain acceptance matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slot", SLOTS)
def test_closed_domain_acceptance_matrix(slot):
    """Every shipped keep_op shape has an ENUMERATED coerce verdict per slot."""

    holder = _MethodHolder()
    identity_shapes = [
        _plain_predicate,  # plain function
        holder.predicate,  # bound method
        functools.partial(_plain_predicate),  # functools.partial
        _SlotsCallable(),  # __slots__ callable instance
        tl.func("relu"),  # BaseSelector instance (acceptance-by-identity)
        tl.func("conv2d") & tl.followed_by(tl.func("relu")),  # followed_by composite
        _selector_bearing_closure(),  # .selector-bearing closure
    ]
    for shape in identity_shapes:
        assert coerce_predicate(shape, slot=slot) is shape

    # Bare str -> registry-name path (miss refuses typed under m2).
    with pytest.raises(PredicateError) as excinfo:
        coerce_predicate("never_registered", slot=slot)
    assert excinfo.value.fields["code"] == "predicate_unregistered"

    # Non-callable non-str -> house-style typed ValueError.
    with pytest.raises(ValueError, match="raw predicate callable or a registered-name"):
        coerce_predicate(123, slot=slot)


def test_unknown_slot_refuses_typed():
    """Unknown slots refuse with the house-style ValueError naming the vocabulary."""

    with pytest.raises(ValueError, match="closed slot vocabulary"):
        coerce_predicate(_plain_predicate, slot="intervene")
    with pytest.raises(ValueError, match="closed slot vocabulary"):
        coerce_predicate(_plain_predicate, slot="live")


def test_three_point_introspection_parity():
    """Raw flow is bit-identical to shipped: the three introspection points
    see the user's original object after coercion."""

    composite = tl.func("conv2d") & tl.followed_by(tl.func("relu"))
    coerced = coerce_predicate(composite, slot="save")
    assert coerced is composite
    assert _is_supported_followed_by_predicate(coerced) == _is_supported_followed_by_predicate(
        composite
    )

    selector = tl.func("relu")
    coerced_selector = coerce_predicate(selector, slot="save")
    assert coerced_selector is selector
    assert _keep_op_needs_alias_retry(coerced_selector) == _keep_op_needs_alias_retry(selector)

    wrapped = _selector_bearing_closure()
    coerced_wrapped = coerce_predicate(wrapped, slot="save")
    assert coerced_wrapped is wrapped
    assert _predicate_cache_key(coerced_wrapped) == _predicate_cache_key(wrapped)

    # A .selector-bearing wrapper whose selector IS the retroactive composite
    # keeps its followed_by support through the recursion (bit-identical flow).
    def followed_wrapper(ctx):
        return bool(composite(ctx))

    followed_wrapper.selector = composite
    coerced_followed = coerce_predicate(followed_wrapper, slot="save")
    assert coerced_followed is followed_wrapper
    assert _is_supported_followed_by_predicate(coerced_followed)


# ---------------------------------------------------------------------------
# m2: register_predicate — registry state machine
# ---------------------------------------------------------------------------


def test_registration_returns_fn_unchanged_and_mutates_no_attribute():
    """The decorator returns the function truly unchanged (getattr sweep)."""

    def probe(ctx):
        return True

    before = dict(vars(probe))
    before_dir = set(dir(probe))
    returned = register_predicate("probe")(probe)
    assert returned is probe
    assert dict(vars(probe)) == before
    assert set(dir(probe)) == before_dir


def test_duplicate_registration_refuses_then_replace_bumps_version():
    """Dup without replace refuses typed; replace=True mints a new version."""

    register_predicate("dup")(_plain_predicate)
    with pytest.raises(PredicateError) as excinfo:
        register_predicate("dup")(_plain_predicate)
    assert excinfo.value.fields["code"] == "predicate_name_conflict"

    pre_replace_wrapper = coerce_predicate("dup", slot="save")
    old_key = pre_replace_wrapper.__torchlens_cache_key__

    def replacement(ctx):
        return False

    register_predicate("dup", replace=True)(replacement)
    new_wrapper = coerce_predicate("dup", slot="save")
    new_key = new_wrapper.__torchlens_cache_key__
    assert old_key != new_key
    assert old_key[:2] == ("registered", "dup")
    assert new_key[:2] == ("registered", "dup")
    # The pre-replace wrapper keeps its consistent (old callable, old key) pair.
    assert pre_replace_wrapper.__torchlens_cache_key__ == old_key
    assert pre_replace_wrapper(None) is True
    assert new_wrapper(None) is False


def test_builtin_names_never_replaceable(monkeypatch):
    """replace=True on a builtin refuses typed (state machine pin)."""

    monkeypatch.setitem(predicate_registry._BUILTIN_PREDICATES, "builtin_probe", _plain_predicate)
    for replace in (False, True):
        with pytest.raises(PredicateError) as excinfo:
            register_predicate("builtin_probe", replace=replace)(_plain_predicate)
        assert excinfo.value.fields["code"] == "predicate_name_conflict"


def test_registration_shape_matrix():
    """Every plain-callable shape registers; two names -> distinct keys."""

    holder = _MethodHolder()
    shapes = {
        "shape_fn": _plain_predicate,
        "shape_method": holder.predicate,
        "shape_partial": functools.partial(_plain_predicate),
        "shape_slots": _SlotsCallable(),
    }
    for name, shape in shapes.items():
        assert register_predicate(name)(shape) is shape
        wrapper = coerce_predicate(name, slot="save")
        assert wrapper is not shape
        assert wrapper.__torchlens_cache_key__[:2] == ("registered", name)

    # Same callable under two names -> two registrations, DISTINCT keys.
    register_predicate("alias_a")(_plain_predicate)
    register_predicate("alias_b")(_plain_predicate)
    key_a = coerce_predicate("alias_a", slot="save").__torchlens_cache_key__
    key_b = coerce_predicate("alias_b", slot="save").__torchlens_cache_key__
    assert key_a != key_b


def test_introspection_bearing_registration_refuses():
    """BaseSelector trees / followed_by composites / .selector closures refuse."""

    for shape in (
        tl.func("relu"),
        tl.func("conv2d") & tl.followed_by(tl.func("relu")),
        _selector_bearing_closure(),
    ):
        with pytest.raises(ValueError, match="introspects structurally"):
            register_predicate("nope")(shape)

    with pytest.raises(ValueError, match="requires a callable"):
        register_predicate("nope")(object())
    with pytest.raises(ValueError, match="non-empty str"):
        register_predicate("")
    with pytest.raises(ValueError, match="non-empty str"):
        register_predicate(42)


# ---------------------------------------------------------------------------
# m2: the enforcing wrapper
# ---------------------------------------------------------------------------


def test_name_vs_callable_return_parity_per_slot():
    """A registered name and the raw callable return the same values in-domain."""

    returns = {"save": [True, False, None, CaptureSpec(save_out=True, save_metadata=True)]}
    bool_returns = [True, False]

    for value in returns["save"]:

        def fn(ctx, _value=value):
            return _value

        register_predicate("parity", replace=True)(fn)
        wrapper = coerce_predicate("parity", slot="save")
        assert wrapper(None) is fn(None)
        assert coerce_predicate(fn, slot="save") is fn

    for slot in ("halt", "until"):
        for value in bool_returns:

            def fn(ctx, _value=value):
                return _value

            register_predicate("parity", replace=True)(fn)
            wrapper = coerce_predicate("parity", slot=slot)
            assert wrapper(None) is fn(None)


@pytest.mark.parametrize("slot", ("halt", "until"))
def test_halt_until_bool_only_enforcement(slot):
    """Registered halt/until predicates must return bool; others refuse typed."""

    def truthy(ctx):
        return 1  # truthy but NOT bool

    register_predicate("truthy", replace=True)(truthy)
    wrapper = coerce_predicate("truthy", slot=slot)
    with pytest.raises(PredicateError) as excinfo:
        wrapper(None)
    assert excinfo.value.fields["code"] == "predicate_return_invalid"
    assert excinfo.value.fields["reason"] == f"{slot}_slot_bool_only"


def test_registered_retroactive_refuses_typed():
    """RetroactiveCaptureDecision is RAW-CALLABLE-ONLY: registered names refuse."""

    decision = RetroactiveCaptureDecision(
        target_raw_labels=("relu_1_2_raw",),
        spec=CaptureSpec(save_out=True, save_metadata=True),
    )

    def retroactive(ctx):
        return decision

    # Raw-callable passthrough is untouched (shipped behavior).
    assert coerce_predicate(retroactive, slot="save") is retroactive

    register_predicate("retro")(retroactive)
    wrapper = coerce_predicate("retro", slot="save")
    with pytest.raises(PredicateError) as excinfo:
        wrapper(None)
    assert excinfo.value.fields["code"] == "predicate_return_invalid"
    assert excinfo.value.fields["reason"] == "registered_retroactive_unsupported"


def test_wrapper_transparency_contract():
    """The wrapper carries the cache key and otherwise presents plain."""

    register_predicate("transparent")(_plain_predicate)
    wrapper = coerce_predicate("transparent", slot="save")
    key = wrapper.__torchlens_cache_key__
    assert key[:2] == ("registered", "transparent")
    assert _predicate_cache_key(wrapper) == key
    assert getattr(wrapper, "selector", None) is None
    assert not isinstance(wrapper, BaseSelector)
    # Plain-callable conservative treatment: alias retry stays on.
    assert _keep_op_needs_alias_retry(wrapper)
    assert not _is_supported_followed_by_predicate(wrapper)


def test_coerce_time_binding_is_slot_aware_per_wrapper():
    """Wrappers are minted per (name, slot) coercion; keys match across slots."""

    def flexible(ctx):
        return True

    register_predicate("flex")(flexible)
    save_wrapper = coerce_predicate("flex", slot="save")
    halt_wrapper = coerce_predicate("flex", slot="halt")
    assert save_wrapper is not halt_wrapper
    assert save_wrapper.__torchlens_cache_key__ == halt_wrapper.__torchlens_cache_key__


# ---------------------------------------------------------------------------
# Security + purity pins
# ---------------------------------------------------------------------------


def test_no_loader_marker_and_pickled_reference_refuses():
    """Registration stamps NOTHING the restricted loader consults; a pickled
    reference to a registered predicate is load-tolerated only as the inert
    foreign placeholder and REFUSES typed at execution resolution (tamper
    pin: the UntrustedCallableError path is unchanged by registration)."""

    from torchlens._io._safe_unpickle import SafeBundleUnpickler, _DeferredForeignCallable
    from torchlens.intervention.errors import UntrustedCallableError
    from torchlens.intervention.resolver import resolve_import_ref
    from torchlens.utils._callable_safety import (
        FACET_RECIPE_MARKER_ATTR,
        is_inert_first_party_callable,
    )

    register_predicate("pickled")(_plain_predicate)
    # No attribute consulted by the restricted loader is set, and the
    # default-deny inertness gate still denies (not torchlens-owned).
    assert not hasattr(_plain_predicate, FACET_RECIPE_MARKER_ATTR)
    assert not is_inert_first_party_callable(_plain_predicate)

    payload = pickle.dumps(_plain_predicate)
    import io as _io

    loaded = SafeBundleUnpickler(_io.BytesIO(payload)).load()
    # Load-tolerated as the INERT placeholder, never the imported callable.
    assert isinstance(loaded, _DeferredForeignCallable)
    assert loaded is not _plain_predicate

    ref = f"{_plain_predicate.__module__}:{_plain_predicate.__qualname__}"
    with pytest.raises(UntrustedCallableError):
        resolve_import_ref(ref)


def test_alias_retry_purity_meta_contract():
    """Double invocation per event is the user's documented contract: a
    counting predicate observes N calls for N coerce-then-evaluate rounds
    (clause 1: predicates must be pure w.r.t. observable effects)."""

    calls = {"n": 0}

    def counting(ctx):
        calls["n"] += 1
        return False

    coerced = coerce_predicate(counting, slot="save")
    assert coerced is counting
    coerced(None)
    coerced(None)  # the alias retry MAY legally invoke a second time per event
    assert calls["n"] == 2


def test_reset_for_tests_refuses_outside_pytest(monkeypatch):
    """_reset_for_tests is pytest-only (prerelease-switch precedent)."""

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with pytest.raises(RuntimeError, match="TEST-ONLY"):
        predicate_registry._reset_for_tests()
