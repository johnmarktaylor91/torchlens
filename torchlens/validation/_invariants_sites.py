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

from ..data_classes._site_key import ROOT_CALL_INSTANCE, SITE_KEY_PREFIX, parse_site_key

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

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
    ops = list(ml.layer_list)
    if not any(getattr(op, "site_key", None) is not None for op in ops):
        return  # legacy artifact: out of the invariant's declared domain
    _check_site_key_totality(ops, name)
    _check_site_key_uniqueness(ops, name)
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


def _check_site_key_uniqueness(ops: list, name: str) -> None:
    """I-S2: (site_key, pass-qualified innermost call instance) unique."""

    from .invariants import MetadataInvariantError

    seen: dict[tuple[str, object], str] = {}
    for op in ops:
        stack = tuple(getattr(op, "module_call_stack", ()) or ())
        if stack:
            call_instance: object = stack[-1]
        else:
            # No torch/object-module call stack to tell passes apart (a bare
            # function_root capture, e.g. JAX's source-path-based scan/while
            # recurrence, never populates one). Site keys are INTENTIONALLY
            # shared across recurring passes of one site (the whole point of
            # site_key_v1), so two retained ops of the SAME recurring site in
            # DIFFERENT passes must not collide here; ``pass_index`` is the
            # backend-neutral "which invocation of this site is this" signal
            # every op carries (1 for the overwhelming majority of ops, which
            # are single-pass and already rely on their site_key alone being
            # distinct -- this never masks a genuine same-pass collision).
            call_instance = (ROOT_CALL_INSTANCE, getattr(op, "pass_index", 1))
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
        layer = ml.layer_logs.get(layer_label)
        if layer is None:
            continue
        member_keys = {op.site_key for _, op in layer.ops.items()}
        member_keys.discard(None)
        if len(member_keys) == 1:
            accessor_value = layer.site_key
            expected = next(iter(member_keys))
            if accessor_value != expected:
                raise MetadataInvariantError(
                    name,
                    f"I-S3': Layer '{layer_label}'.site_key returned "
                    f"{accessor_value!r}, expected {expected!r}",
                )
        elif len(member_keys) > 1:
            try:
                silently_read = layer.site_key
            except InvalidArgumentError:
                continue
            raise MetadataInvariantError(
                name,
                f"I-S3': Layer '{layer_label}' spans {len(member_keys)} sites "
                f"but site_key silently returned {silently_read!r} instead of "
                "the typed ambiguity refusal",
            )
