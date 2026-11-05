"""Order-canonical content fingerprints for the op store's write audit.

Moved from :mod:`torchlens._trace_core.op_store` (file cap): the audited
stores there compare these fingerprints before and after a step to tell an
effective container write from a mutate-and-revert.
"""

from __future__ import annotations

from typing import Any


def _value_fingerprint(value: Any) -> int:
    """Order-canonical recursive content fingerprint for audit snapshots.

    Unordered containers (``set``/``frozenset``/``dict``) hash their
    element fingerprints SORTED, so equal content always fingerprints
    equal regardless of iteration order — a ``hash(repr(...))`` snapshot
    was iteration-order sensitive (a mutate-and-revert that resized a
    set's table, or reordered equal dict keys, false-positively read as a
    write of the column). Ordered containers stay positional. Leaves keep
    the ``repr`` basis (in-place mutation anywhere inside still changes
    the fingerprint); mutables nested in NON-builtin custom objects remain
    the audit's disclosed residual. Env-gated enforcement-leg cost only:
    comparable to the former full-``repr`` walk (it recurses the same
    content, minus container string building; leaf ``repr`` — including
    tensor and record cells — still dominates).
    """

    cls = value.__class__
    if cls is dict:
        return hash(
            (
                1,
                tuple(
                    sorted(
                        hash((_value_fingerprint(key), _value_fingerprint(item)))
                        for key, item in value.items()
                    )
                ),
            )
        )
    if cls is list:
        return hash((2, tuple(_value_fingerprint(item) for item in value)))
    if cls is set or cls is frozenset:
        return hash((3, tuple(sorted(_value_fingerprint(item) for item in value))))
    if cls is tuple:
        return hash((4, tuple(_value_fingerprint(item) for item in value)))
    try:
        return hash((0, repr(value)))
    except Exception:
        # Fail CLOSED (R22-8): a shared sentinel here made every pair of
        # fingerprint-refusing values compare EQUAL, collapsing a real write
        # to "no effective write" and laundering the downstream read. The
        # object id keeps same-object comparisons stable within the audit
        # window while distinct refusing objects read as changed -- a false
        # alarm at worst, never a laundering pass.
        return id(value)


def _cell_content_fingerprint(value: Any) -> int | None:
    """Content fingerprint for one exact builtin mutable-container cell.

    Non-container values (and immutable views) return ``None`` — cell
    REPLACEMENT is already caught by the write interception. Containers
    whose content cannot be fingerprinted (cycles) conservatively
    fingerprint as their length so at least size changes are visible.
    """

    cls = value.__class__
    if cls is dict or cls is list or cls is set:
        try:
            return _value_fingerprint(value)
        except Exception:
            return len(value)
    return None
