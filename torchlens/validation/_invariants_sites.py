"""Site-key invariants I-S1 / I-S2 / I-S3' (the L1 grouping-core tripwires).

APPLICABILITY DOMAIN (declared at the invariant's birth, never a later
weakening): capture-time traces always carry site keys (step 7 mints them),
so any trace where at least one op carries a key is IN domain and totality
is enforced; a trace where NO op carries a key is a legacy artifact written
before ``site_key_v1`` existed and is OUT of the declared domain (the
accessors refuse typed on it instead -- ``site_key_unavailable``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .._data_classes._layer_sites import layer_site_key
from .._data_classes._site_key import ROOT_CALL_INSTANCE, SITE_KEY_PREFIX, parse_site_key

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

#: Raw-label survival pattern (mirrors the roster sweep in invariants.py):
#: a raw label leaking into any site-key COMPONENT is an unresolvable
#: reference persisted into portable artifacts.
_RAW_LABEL_COMPONENT_PATTERN = re.compile(r"_raw$")


def _check_site_key_invariants(ml: Trace) -> None:
    """Check I-S1 (prefixed totality), I-S2 (per-instance uniqueness), and
    I-S3' (Layer accessor coherence) over one finished trace.

    Raises
    ------
    MetadataInvariantError
        On the first violated site invariant.
    """

    name = "site_key_invariants"
    ops = list(ml._layer_list)
    if not any(getattr(op, "site_key", None) is not None for op in ops):
        return  # legacy artifact: out of the invariant's declared domain
    _check_site_key_totality(ops, name)
    _check_site_key_uniqueness(ops, name, _root_call_instances(ml))
    _check_layer_site_coherence(ml, name)


def _check_site_key_totality(ops: list, name: str) -> None:
    """I-S1: every retained op carries a non-empty, prefixed, well-formed key."""

    from .invariants import MetadataInvariantError

    for op in ops:
        key = getattr(op, "site_key", None)
        if not isinstance(key, str) or not key.startswith(SITE_KEY_PREFIX + "|"):
            raise MetadataInvariantError(
                name,
                f"I-S1: op '{op.label}' carries site_key {key!r} (expected a "
                f"non-empty '{SITE_KEY_PREFIX}|'-prefixed string on every "
                "retained op once any op carries one)",
            )
        try:
            module_site, layer_type, _slot, _ordinal = parse_site_key(key)
        except ValueError as exc:
            raise MetadataInvariantError(
                name, f"I-S1: op '{op.label}' carries malformed site_key {key!r}: {exc}"
            ) from exc
        # Raw-label survival sweep over the key's COMPONENTS: the whole-string
        # scalar sweep cannot see a label embedded before the ordinal tail.
        for component in (*module_site, layer_type):
            if _RAW_LABEL_COMPONENT_PATTERN.search(component):
                raise MetadataInvariantError(
                    name,
                    f"Raw label {component!r} survived postprocessing inside "
                    f"{op.label}.site_key ({key!r})",
                )


def _root_call_instances(ml: Trace) -> dict[str, str]:
    """Map op labels to the root call that ran them, when the root ran more than once.

    A plain module root runs once per capture, so an op outside every submodule
    sits in the one root instance. A bound-method root's module runs once per
    call of the bound method (``self:1`` .. ``self:N``); those ops carry no root
    frame (their containment facts read as under a plain trace), so each root
    ModuleCall's op roster names their pass-qualified innermost call instance.
    """
    if "self" not in ml.modules:
        return {}
    root_calls = list(ml.modules["self"].ops.values())
    if len(root_calls) < 2:
        return {}
    return {str(op_label): str(call.label) for call in root_calls for op_label in call.ops}


def _call_instance(
    op: object, equivalence_counts: dict[object, int], root_call_of: dict[str, str]
) -> object:
    """The pass-qualified innermost call instance an op's site key must be unique within."""
    stack = tuple(getattr(op, "module_call_stack", ()) or ())
    if stack:
        return stack[-1]
    root_call = root_call_of.get(str(getattr(op, "label", "")))
    if root_call is not None:
        return root_call
    equivalence_class = getattr(op, "equivalence_class", None)
    if equivalence_counts.get(equivalence_class, 0) > 1:
        # A genuine multi-occurrence group (recurring passes of one
        # site): differentiate siblings by their unique raw label so
        # the designed sharing never falsely collides.
        return (ROOT_CALL_INSTANCE, getattr(op, "_label_raw", getattr(op, "label", None)))
    # A singleton occurrence: no legitimate reason to share a
    # site_key with anything else, so keep the strict coarse
    # identity that catches a forged collision.
    return ROOT_CALL_INSTANCE


def _check_site_key_uniqueness(
    ops: list, name: str, root_call_of: dict[str, str] | None = None
) -> None:
    """I-S2: (site_key, pass-qualified innermost call instance) unique.

    An op in no submodule takes the root call that ran it when the root ran
    more than once (a bound-method root; ``_root_call_instances``).
    Without a module stack (a bare ``function_root`` capture, e.g. JAX's
    source-path-based scan/while recurrence), there is no call-stack frame to
    tell passes apart. Site keys are INTENTIONALLY shared across recurring
    passes of one site (the whole point of ``site_key_v1``), and neither
    ``equivalence_class`` alone (shared by every structurally-interchangeable
    occurrence, so it alone collapses distinct passes back together) nor
    ``pass_index`` alone (restarts at 1 per Layer, so a site split across more
    than one Layer by a pre-existing grouping boundary -- e.g. a JAX while
    loop's pre-loop condition check, which shares its "lt" equivalence_class
    with every in-loop repeat but lands in its own singleton Layer because of
    the N+1-cond-vs-N-body arity mismatch -- collides on pass_index too)
    reliably reconstructs "which occurrence is this" across that boundary.

    The NARROW carve-out: only ops whose ``equivalence_class`` is genuinely
    shared by more than one retained op (a real multi-occurrence group --
    the one case site keys are designed to repeat across) get the
    per-occurrence raw-label differentiator, so siblings in that group never
    collide. Every op with a SINGLETON equivalence_class (the overwhelming
    majority, including pseudo input/output nodes) stays on the coarse,
    strict ``ROOT_CALL_INSTANCE`` identity, so two singleton ops forged to
    share one site_key still collide and still raise -- the check never goes
    fully blind for backends without a module stack.
    """

    from .invariants import MetadataInvariantError

    equivalence_counts: dict[object, int] = {}
    for op in ops:
        if not tuple(getattr(op, "module_call_stack", ()) or ()):
            equivalence_counts[getattr(op, "equivalence_class", None)] = (
                equivalence_counts.get(getattr(op, "equivalence_class", None), 0) + 1
            )

    seen: dict[tuple[str, object], str] = {}
    for op in ops:
        call_instance = _call_instance(op, equivalence_counts, root_call_of or {})
        identity = (str(op.site_key), call_instance)
        if identity in seen:
            raise MetadataInvariantError(
                name,
                f"I-S2: ops '{seen[identity]}' and '{op.label}' share site_key "
                f"{op.site_key!r} within call instance {call_instance!r}",
            )
        seen[identity] = op.label


def _check_layer_site_coherence(ml: Trace, name: str) -> None:
    """I-S3': the accessor returns a key iff the layer's ops share exactly one.

    A silent single-key read on a site-spanning layer is the failure this
    tripwire exists to catch.
    """

    from .._errors import InvalidArgumentError
    from .invariants import MetadataInvariantError

    for layer_label in getattr(ml, "layer_labels", ()) or ():
        layer = ml._layer_logs.get(layer_label)
        if layer is None:
            continue
        member_keys = {op.site_key for _, op in layer.ops.items()}
        member_keys.discard(None)
        if len(member_keys) == 1:
            accessor_value = layer_site_key(layer)
            expected = next(iter(member_keys))
            if accessor_value != expected:
                raise MetadataInvariantError(
                    name,
                    f"I-S3': layer_site_key('{layer_label}') returned "
                    f"{accessor_value!r}, expected {expected!r}",
                )
        elif len(member_keys) > 1:
            try:
                silently_read = layer_site_key(layer)
            except InvalidArgumentError:
                continue
            raise MetadataInvariantError(
                name,
                f"I-S3': Layer '{layer_label}' spans {len(member_keys)} sites "
                f"but layer_site_key silently returned {silently_read!r} instead of "
                "the typed ambiguity refusal",
            )
