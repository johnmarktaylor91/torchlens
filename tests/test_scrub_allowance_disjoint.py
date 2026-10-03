"""b5 R50-2: the scrub runtime-only allowance must not shadow declared policy.

``_io/scrub.py`` decides what leaves a Trace on the way into a portable
artifact through TWO authorities, in this order (``scrub.py:612``)::

    if owner_is_trace and _is_runtime_only_trace_field(field_name):
        continue                      # authority 1: the hand-kept allowance
    ...
    policy = _effective_policy(value, field_name, spec[field_name], options)
                                      # authority 2: Trace.PORTABLE_STATE_SPEC

Because the allowance is consulted FIRST, any name in BOTH authorities has a
DEAD declared policy row: flipping that row to ``KEEP`` would change nothing
and the field would still be silently dropped. scrub.py's own comments name
this hazard twice (the B1-02 and B1-17 migrations moved earlier batches out for
exactly this reason), but nothing enforced the endpoint -- eight names are
still in both.

This gate is the enforcement. It is behavioural, not textual: it asks the real
predicate about every declared field, so it holds no matter how the allowance
is spelled. Today's eight overlaps are all ``DROP``, i.e. correct-but-dual;
the gate fails the moment a ninth appears OR one of the eight is declared with
a policy the allowance would silently override. Finishing the migration
(deleting rows from the allowance) belongs to the ``_io/scrub.py`` owner lane;
the seed shrinks as they land.
"""

from __future__ import annotations

from torchlens._io import FieldPolicy
from torchlens._io.scrub import _is_runtime_only_trace_field
from torchlens.data_classes.trace import Trace

#: Names in BOTH the runtime-only allowance and ``Trace.PORTABLE_STATE_SPEC``,
#: seeded from the tree at b5 fixplan time (2026-08-14). Every one is ``DROP``
#: today, so the duplication is currently harmless -- and every one is a
#: declared row that the allowance makes unreachable. SHRINK-ONLY: the fix is
#: to delete the allowance entry, then delete the name here.
_KNOWN_DUAL_AUTHORITY_FIELDS = frozenset(
    {
        "_capture_config",
        "_capture_parent_edge_truth",
        "_defer_streaming_bundle_finalization",
        "_keep_outs_in_memory",
        "_orphan_pruned_func_call_ids",
        "_out_sink",
        "_out_writer",
        "_stop_directive",
    }
)


def _overlap() -> frozenset[str]:
    """Declared Trace fields that the runtime-only allowance shadows."""

    return frozenset(
        name for name in Trace.PORTABLE_STATE_SPEC if _is_runtime_only_trace_field(name)
    )


def test_allowance_and_portable_spec_overlap_only_where_ledgered() -> None:
    """No NEW field may live in both authorities."""

    new = sorted(_overlap() - _KNOWN_DUAL_AUTHORITY_FIELDS)
    assert not new, (
        f"Field(s) in both the scrub runtime-only allowance and "
        f"Trace.PORTABLE_STATE_SPEC: {new}. The allowance is consulted FIRST "
        "(scrub.py:612), so the declared policy row is dead code -- pick ONE "
        "authority: either declare the field with FieldPolicy.DROP and delete "
        "its allowance entry, or leave it undeclared."
    )


def test_dual_authority_ledger_is_shrink_only() -> None:
    """A migrated name must be removed from the seed, keeping the ratchet tight."""

    stale = sorted(_KNOWN_DUAL_AUTHORITY_FIELDS - _overlap())
    assert not stale, (
        f"Dual-authority ledger row(s) no longer overlap: {stale}. The migration "
        "landed -- delete these names here (the ledger is shrink-only). Target "
        "state is an EMPTY ledger: allowance and PORTABLE_STATE_SPEC disjoint."
    )


def test_every_dual_authority_field_is_declared_drop() -> None:
    """While a name is dual, its declared policy must agree with being dropped.

    This is the tripwire half: a ``KEEP`` (or any non-DROP) policy on a
    shadowed field means the artifact silently omits state the schema says it
    persists. That is a load-bearing wrong answer, not a hygiene issue.
    """

    disagreeing = {
        name: Trace.PORTABLE_STATE_SPEC[name].value
        for name in sorted(_overlap())
        if Trace.PORTABLE_STATE_SPEC[name] is not FieldPolicy.DROP
    }
    assert not disagreeing, (
        f"Shadowed field(s) declared with a non-DROP policy: {disagreeing}. The "
        "allowance drops them BEFORE the policy is read, so the declared policy "
        "is a lie. Remove the allowance entry (the policy then takes effect) "
        "rather than leaving the two authorities disagreeing."
    )


def test_probe_is_not_vacuous() -> None:
    """Positive and negative controls on the allowance predicate itself.

    A refactor that made ``_is_runtime_only_trace_field`` always return False
    would make every assertion above pass trivially.
    """

    # Allowance-only name (in the allowance, deliberately NOT declared).
    assert _is_runtime_only_trace_field("_had_unattributed_tensor_args")
    assert "_had_unattributed_tensor_args" not in Trace.PORTABLE_STATE_SPEC
    # A declared, genuinely portable field is never in the allowance.
    assert Trace.PORTABLE_STATE_SPEC["layer_labels"] is FieldPolicy.KEEP
    assert not _is_runtime_only_trace_field("layer_labels")
    assert not _is_runtime_only_trace_field("_no_such_trace_field")
    # The seeded overlap is real in both directions.
    assert set(Trace.PORTABLE_STATE_SPEC) >= _KNOWN_DUAL_AUTHORITY_FIELDS
    assert all(_is_runtime_only_trace_field(name) for name in _KNOWN_DUAL_AUTHORITY_FIELDS)
