"""Capture-door intervention glue for ``tl.trace(intervene=...)``.

Split out of ``user_funcs.py`` (C03 fix cycle): the backward sticky-spec
builder for backward-only selectors and the capture-door fire-evidence
envelope (ledger memo 3.1) are intervention machinery, not trace-entry
resolution; ``user_funcs`` imports them from here. The live hook-plan
spec builder and its non-mutating merge into an existing spec followed
for the same reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
from torchlens._capture.prepare.resolve import (
    _armed_branch_count as _armed_branch_count,
    _backward_intervention_spec_from_predicate as _backward_intervention_spec_from_predicate,
    _fired_capture_records as _fired_capture_records,
    _intervene_rule_pairs as _intervene_rule_pairs,
    _intervention_spec_from_hook_plan as _intervention_spec_from_hook_plan,
    _merge_intervention_spec_hooks as _merge_intervention_spec_hooks,
    _record_capture_intervention_event as _record_capture_intervention_event,
    _target_spec_for as _target_spec_for,
    activation_only_predicate as activation_only_predicate,
    intervene_selector_fired_nowhere as intervene_selector_fired_nowhere,
)
