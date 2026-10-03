"""M10 ratchet: every Trace field has exactly one owning component.

The declared ownership map (``_trace_components.py``) is the design artifact
the Trace decomposition executes against. This test is the ratchet: a new
Trace field without a declared owner — or a stale map entry — fails here,
so the god object cannot regrow silently while the decomposition proceeds.
"""

from __future__ import annotations

from collections import Counter

from torchlens import constants as tl_constants
from torchlens.data_classes._trace_components import (
    TRACE_COMPONENT_VOCABULARY,
    TRACE_FIELD_OWNERSHIP,
)
from torchlens.data_classes.trace import Trace


def test_ownership_covers_exactly_the_declared_policy() -> None:
    """Ownership keys == Trace.FIELD_POLICY keys, both directions."""

    policy_fields = set(Trace.FIELD_POLICY)
    owned_fields = set(TRACE_FIELD_OWNERSHIP)
    missing = policy_fields - owned_fields
    stale = owned_fields - policy_fields
    assert not missing, f"Trace fields without a declared owner: {sorted(missing)}"
    assert not stale, f"ownership entries for retired fields: {sorted(stale)}"


def test_owners_come_from_the_closed_vocabulary() -> None:
    """Every owner is one of the declared components."""

    unknown = {
        name: owner
        for name, owner in TRACE_FIELD_OWNERSHIP.items()
        if owner not in TRACE_COMPONENT_VOCABULARY
    }
    assert not unknown, unknown


def test_component_sizes_respect_the_design_bound() -> None:
    """No owned component (except the graph core) exceeds ~60 fields.

    The graph plane decomposes into the columnar TraceCore, not a flat
    component, so it is exempt from the per-component bound.
    """

    sizes = Counter(TRACE_FIELD_OWNERSHIP.values())
    for component, size in sizes.items():
        if component == "graph":
            continue
        # 61: fixwave-5 added the _tl_cleaned_up idempotent-cleanup sentinel
        # (b6-opus R25) to the session component; the ~60 design pressure
        # stands -- the next session field should evict or consolidate one.
        # 62: F01 added _tl_injection_state, itself the consolidation of
        # FIVE injection transients into one dict field (the pressure
        # honored); the next session field should evict or consolidate one.
        # 63: the F01 fix cycle ENROLLED _tl_intervene_selector_fire_count
        # (not new state -- the previously-undeclared exemptions-ledger
        # sibling of _tl_save_selector_fire_count, whose missing DROP row
        # refused streamed to_disk + intervene saves at the scrub). Named
        # consolidation target: fold the two fire counters into one dict
        # field the way _tl_injection_state folded five.
        assert size <= 63, f"component {component} holds {size} fields (> 63)"


def test_every_field_order_entry_is_owned() -> None:
    """All 220 public FIELD_ORDER names are covered (subset of policy)."""

    unowned = [
        name for name in tl_constants.MODEL_LOG_FIELD_ORDER if name not in TRACE_FIELD_OWNERSHIP
    ]
    assert not unowned, unowned
