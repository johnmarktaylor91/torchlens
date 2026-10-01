"""Structural fingerprints for collapse fold honesty, with the B1 memo cache.

This module owns the per-member structural-signature family used by the
"+N more" run-fold honesty checks (moved verbatim from ``auto_collapse.py``),
plus the collapse memo's B1 memoization (build item 6): one
:class:`MemberFingerprintCache` per collapse analysis computes each member's
``(structural signature, exterior bindings)`` pair ONCE, sharing a single
:func:`_module_wiring_walk` between the wiring digest and the
exterior-bindings read (historically the walk ran twice per member per
check, and the signature was recomputed for every candidate window the
member appeared in -- measured ~1,132 recomputations per module on
densenet201-class fan-outs).

The cache is keyed on the :class:`~.auto_collapse.CollapseAnalysis` instance,
which is itself revision-keyed (``_ANALYSIS_CACHE`` invalidates on any
collapse-relevant graph mutation), so a stale fingerprint can never be served
across a trace mutation and no extra O(ops) revision walk is paid in the hot
enumeration loops.
"""

from __future__ import annotations

import re
import weakref
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..data_classes.module import Module
    from ..data_classes.trace import Trace

#: Structural signature row type: ``(num_layers, num_params,
#: num_params_trainable, num_params_frozen, ops_signature, wiring_digest)``.
StructuralSignature = tuple[int, int, int, int, tuple[tuple[str, str, str], ...], object]

#: Exterior-bindings row type: sorted ``(exterior_label, slot)`` pairs, or
#: ``None`` when the member's wiring is unresolvable.
ExteriorBindings = "tuple[tuple[str, int], ...] | None"


def _module_structural_signature(module: Module) -> StructuralSignature:
    """Return a per-module structural fingerprint for fold-honesty checks.

    Two modules are only considered structurally interchangeable for the
    "+N more" repeat-fold ellipsis when this fingerprint matches exactly. It
    is used to require that EVERY member of a fold — the visible
    representative included — shares one structure, so a same-class,
    same-output-shape sibling with genuinely different internals (extra
    layers/params) can never be silently hidden inside a ``+N more`` box that
    claims uniformity.

    r-b6 R19-1: counts alone were not enough — a kwargs-different conv
    (dilation 2) and a tanh-for-relu block both matched the historical 4-int
    fingerprint, so two DIFFERENT models rendered byte-identical DOT under
    the homogeneity claim. The fingerprint therefore also carries the ordered
    per-layer op-type sequence and a canonical ``func_config`` digest.

    r3 b6-opus R19-1: op types and kwargs are still not enough — a residual
    ``x + y`` block and a self-add ``y + y`` block share the same ordered op
    list and params but are DIFFERENT DAGs. The fingerprint therefore also
    carries :func:`_module_wiring_digest`, a canonical intra-module dataflow
    component.

    r4 b6-opus R19-1: ``func_config`` is the MODULE configuration, so a
    functional/dunder op's scalar operand never entered the fingerprint —
    ``* 1.0`` and ``* 3.0`` blocks folded behind one ``+N more``. Each row
    therefore also carries a canonical digest of the op's captured
    non-tensor arguments (:func:`_non_tensor_args_digest`).

    Parameters
    ----------
    module:
        Module to fingerprint.

    Returns
    -------
    tuple
        ``(num_layers, num_params, num_params_trainable, num_params_frozen,
        ops_signature, wiring_digest)`` where ``ops_signature`` is a tuple of
        ``(op_type, func_config_digest, non_tensor_args_digest)`` rows in
        layer order and ``wiring_digest`` canonicalizes the member's
        interior edges plus boundary crossings.
    """

    ops_signature = tuple(
        (
            str(getattr(layer, "func_name", None) or getattr(layer, "layer_type", "")),
            _func_config_digest(getattr(layer, "func_config", None)),
            _non_tensor_args_digest(layer),
        )
        for layer in module.layers
    )
    return (
        int(module.num_layers),
        int(module.num_params),
        int(module.num_params_trainable),
        int(module.num_params_frozen),
        ops_signature,
        _module_wiring_digest(module),
    )


def _module_wiring_walk(module: Module) -> tuple[object, tuple[tuple[str, int], ...]]:
    """Walk one member's wiring; return ``(digest_rows, exterior_bindings)``.

    ``digest_rows`` encodes, per interior op in execution order, the ordered
    parent slots as either ``("i", position)`` — an edge from the interior op
    at that execution position — or ``("x", k)`` — a boundary crossing from
    the ``k``-th distinct exterior source first seen while walking this
    member. ``exterior_bindings`` is the sorted ``(exterior_label, k)``
    correspondence those crossings used, for the cross-member consistency
    check (:func:`_exterior_bindings_consistent`).

    Raises on unresolvable wiring; callers own the degrade policy.
    """

    trace = module.trace
    if trace is None:
        return "", ()
    canonical: list[str] = []
    seen: set[str] = set()
    for label in module._op_labels():
        resolved = trace.ops[label].label
        if resolved not in seen:
            seen.add(resolved)
            canonical.append(resolved)
    position = {label: index for index, label in enumerate(canonical)}
    exterior: dict[str, int] = {}
    rows: list[tuple[tuple[str, int], ...]] = []
    for label in canonical:
        slots: list[tuple[str, int]] = []
        for parent_label in trace.ops[label].parents:
            parent = trace.ops[parent_label].label
            if parent in position:
                slots.append(("i", position[parent]))
            else:
                slots.append(("x", exterior.setdefault(parent, len(exterior))))
        rows.append(tuple(slots))
    return tuple(rows), tuple(sorted(exterior.items()))


def _wiring_degrade_sentinel(module: Module, error: Exception) -> tuple[str, str, str]:
    """Return the unique per-member sentinel for unresolvable wiring.

    A member whose wiring cannot be resolved degrades to a UNIQUE
    per-member sentinel, so it can never fold (r4 b6-fable R19): degrading
    every failing member to a shared exception type name made two members
    with genuinely different-but-unresolvable wiring compare equal, silently
    falling back to the op-signature-only comparison the r3 HIGH proved
    insufficient.
    """

    return (
        "__torchlens_wiring_unresolved__",
        str(getattr(module, "address", "") or id(module)),
        type(error).__name__,
    )


def _walk_or_degrade(
    module: Module,
) -> tuple[object, tuple[tuple[str, int], ...] | None]:
    """Walk one member's wiring, degrading per the fold-honesty contract.

    Returns ``(digest, bindings)``: on success the walk's rows and exterior
    bindings; on ANY resolution failure the unique per-member sentinel (so
    the member can never fold) and ``None`` bindings. The blind catch is the
    contract -- unresolvable wiring of any shape must degrade, never crash
    a render -- and this is its ONE home.
    """

    try:
        rows, bindings = _module_wiring_walk(module)
    except Exception as error:  # noqa: BLE001 - degrade contract (one home)
        return _wiring_degrade_sentinel(module, error), None
    return rows, bindings


def _module_wiring_digest(module: Module) -> object:
    """Return a canonical intra-module dataflow digest for one fold member.

    Exterior sources are numbered per member (never by label), so two run
    members fed by different upstream blocks still compare equal when their
    interior wiring matches, while a residual skip (``x + y``) can never
    match a self-add (``y + y``): the former's add row reads
    ``(("x", 0), ("i", j))`` and the latter's ``(("i", j), ("i", j))``.

    Degrade policy: see :func:`_wiring_degrade_sentinel`.
    """

    digest, _bindings = _walk_or_degrade(module)
    return digest


def _module_exterior_bindings(module: Module) -> tuple[tuple[str, int], ...] | None:
    """Return one member's exterior-source binding, or ``None`` if unresolvable."""

    _digest, bindings = _walk_or_degrade(module)
    return bindings


def _exterior_bindings_consistent(
    merged: dict[str, int],
    bindings: tuple[tuple[str, int], ...] | None,
) -> bool:
    """Merge one member's exterior binding into the fold's shared frame.

    r4 b6-sol R19-1: per-member first-seen numbering alone erases the
    cross-member source correspondence — ``sub(a, b)`` and ``sub(b, a)``
    both canonicalize to ``(("x", 0), ("x", 1))``. When fold members SHARE
    an exterior source, that source must occupy the SAME operand slot in
    every member; members with disjoint exterior sets (consecutive chain
    blocks fed by different upstream blocks) impose no constraint and keep
    folding. Returns whether the member is consistent, updating ``merged``
    in place on success.
    """

    if bindings is None:
        return False
    return all(merged.setdefault(label, index) == index for label, index in bindings)


def _func_config_digest(func_config: Any) -> str:
    """Return a canonical, order-independent digest of one ``func_config``.

    ``func_config`` values are capture-recorded primitives (ints, tuples,
    strings), so ``repr`` over key-sorted items is deterministic. An exotic
    unsortable/unreprable config degrades to its type name — coarser matching,
    never a crash.
    """

    if not func_config:
        return ""
    try:
        return repr(sorted(func_config.items(), key=lambda item: str(item[0])))
    except Exception:
        return type(func_config).__name__


_MEMORY_ADDRESS_PATTERN = re.compile(r"0x[0-9a-fA-F]+")


def _non_tensor_args_digest(layer: Any) -> str:
    """Return a canonical digest of one op's captured non-tensor arguments.

    r4 b6-opus R19-1: ``func_config`` is empty for functional/dunder ops, so
    a scalar operand (``* 3.0`` vs ``* 1.0``) never entered the fold
    fingerprint and two models computing DIFFERENT functions folded behind
    one ``+N more`` ellipsis. The captured positional and keyword non-tensor
    argument values are already recorded per op; digest them canonically.

    Default-object reprs embed memory addresses, which are nondeterministic
    per process; they are masked so equal-valued members keep comparing
    equal (coarser matching for address-only-distinct objects, matching the
    ``_func_config_digest`` degrade discipline). An unreprable value
    degrades to its type name — coarser matching, never a crash.
    """

    try:
        # Multi-pass layers refuse per-pass reads at the aggregate (typed
        # layer_pass_ambiguous, not AttributeError), so read each pass's op
        # directly; the operand values of EVERY pass are fingerprint-relevant.
        ops = getattr(layer, "ops", None)
        sources = list(ops.values()) if ops is not None else [layer]
        parts = tuple(
            (
                getattr(source, "non_tensor_pos_args", None),
                getattr(source, "non_tensor_kwargs", None),
            )
            for source in (sources or [layer])
        )
        if not any(pos or kw for pos, kw in parts):
            return ""
        text = repr(parts)
    except Exception as error:
        return type(error).__name__
    return _MEMORY_ADDRESS_PATTERN.sub("0xADDR", text)


class MemberFingerprintCache:
    """Per-analysis memo of member ``(signature, bindings)`` pairs (B1).

    One instance is shared by every enumeration sweep over one trace
    revision. :meth:`signature` and :meth:`bindings` compute BOTH values on
    first access via a single :func:`_module_wiring_walk`, preserving the
    exact historical degrade semantics of :func:`_module_wiring_digest`
    (unique per-member sentinel) and :func:`_module_exterior_bindings`
    (``None``) on unresolvable wiring.
    """

    __slots__ = ("_pairs", "_trace_ref", "__weakref__")

    def __init__(self, trace: Trace) -> None:
        """Bind the cache to ``trace`` without pinning it."""

        self._trace_ref = weakref.ref(trace)
        self._pairs: dict[str, tuple[StructuralSignature, tuple[tuple[str, int], ...] | None]] = {}

    def _pair(self, address: str) -> tuple[StructuralSignature, tuple[tuple[str, int], ...] | None]:
        """Return the memoized ``(signature, bindings)`` pair for ``address``."""

        cached = self._pairs.get(address)
        if cached is not None:
            return cached
        trace = self._trace_ref()
        if trace is None:
            raise RuntimeError("MemberFingerprintCache outlived its trace")
        module = cast("Module", trace.modules[address])
        digest, bindings = _walk_or_degrade(module)
        ops_signature = tuple(
            (
                str(getattr(layer, "func_name", None) or getattr(layer, "layer_type", "")),
                _func_config_digest(getattr(layer, "func_config", None)),
                _non_tensor_args_digest(layer),
            )
            for layer in module.layers
        )
        signature: StructuralSignature = (
            int(module.num_layers),
            int(module.num_params),
            int(module.num_params_trainable),
            int(module.num_params_frozen),
            ops_signature,
            digest,
        )
        pair = (signature, bindings)
        self._pairs[address] = pair
        return pair

    def signature(self, address: str) -> StructuralSignature:
        """Return the structural signature for one member address."""

        return self._pair(address)[0]

    def bindings(self, address: str) -> tuple[tuple[str, int], ...] | None:
        """Return the exterior bindings for one member address."""

        return self._pair(address)[1]


#: Attribute slot for the per-analysis fingerprint cache. The analysis is a
#: frozen dataclass whose Mapping fields make it unhashable (no weak-dict
#: keying), so the cache rides the instance dict via ``object.__setattr__``.
#: The analysis object is itself revision-keyed
#: (``auto_collapse._ANALYSIS_CACHE``), so keying on its identity gives
#: exact invalidation with zero extra revision walks.
_FINGERPRINTS_ATTR = "_torchlens_member_fingerprints"


def fingerprints_for(trace: Trace, analysis: Any) -> MemberFingerprintCache:
    """Return the shared fingerprint cache for one trace analysis revision.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    analysis:
        The revision-fresh :class:`~.auto_collapse.CollapseAnalysis` the
        caller already holds (every sweep entry point does); its identity is
        the cache key.
    """

    cached = getattr(analysis, _FINGERPRINTS_ATTR, None)
    if cached is not None:
        return cast("MemberFingerprintCache", cached)
    cache = MemberFingerprintCache(trace)
    object.__setattr__(analysis, _FINGERPRINTS_ATTR, cache)
    return cache
