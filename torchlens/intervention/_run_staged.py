"""Apply a trace's staged intervention spec on a ``run(inputs=)`` live forward.

The ordinary live provider refreshes a transactional fork from an internal
capture of the new inputs. When the source trace stages user edits (sticky
hooks, value replacements), that capture runs under the same plan the
intervened rerun uses (``rerun.py``), the fork adopts the capture wholesale (an
applied edit inserts ops, so the refresh projector's same-graph contract does
not hold), and the fire receipts settle onto the fork's ``last_run``: how many
entries were planned, how many fired and how many fired nowhere, with the
zero-fire policy applied exactly as the rerun applies it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._fast_live_steer import hooked_run_diagnostic, stages_user_entries
from .._trace_state import TraceState
from .._utils.hashing import compute_graph_shape_hash
from ._helper_fingerprint import refuse_changed_staged_helpers
from ._rerun_predicate import (
    RerunSpecPlan,
    plan_rerun_spec,
    settle_predicate_rerun,
    translate_staged_final_labels,
)
from .hooks import normalize_hooks_from_spec
from .rerun import (
    _assign_unique_plan_ids,
    _reconcile_rerun_hook_fires,
    _refuse_shifted_raw_index_resave,
    _share_source_state_clones,
)
from .runtime import active_intervention_context

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .._runnable import RunnableDiagnostic

#: ``last_run["engine"]`` of a live ``run(inputs=)`` that applied the staged spec.
LIVE_STAGED_ENGINE = "live"


@dataclass(frozen=True)
class StagedRunReceipt:
    """What one staged live run did with its plan entries."""

    hooks: int
    hooks_fired: int
    hooks_unfired: tuple[str, ...]
    divergence_detail: str | None


class StagedRunPlan:
    """The staged spec of one source trace, planned for one live ``run(inputs=)``.

    Built by :func:`build_staged_run_plan` on the run door and threaded through
    the live provider as an opaque object: the refresh capture runs inside
    :meth:`capture_scope` with :meth:`capture_kwargs`, :meth:`adopt` makes the
    fork the refreshed capture, and :meth:`settle` reconciles the fires after
    the forward returned (outside the provider's native-failure classifier, so
    a typed zero-fire refusal is never misread as an input divergence).
    """

    def __init__(
        self,
        source: Trace,
        spec: Any,
        spec_plan: RerunSpecPlan,
        capture_spec: Any,
        hook_plan: list[Any],
    ) -> None:
        self.source = source
        self.spec = spec
        self.spec_plan = spec_plan
        self.capture_spec = capture_spec
        self.hook_plan = hook_plan
        self.refreshed: Trace | None = None
        self.started_at = time.monotonic()

    def capture_kwargs(self) -> dict[str, Any]:
        """Return the capture keywords that install the staged spec on the forward."""

        return {
            "intervention_spec": self.capture_spec,
            "normalized_hook_plan": self.hook_plan,
            "intervene_predicate": self.spec_plan.intervene_predicate,
        }

    @contextmanager
    def capture_scope(self) -> Iterator[None]:
        """Expose the active spec to the hooks and share the source's state clones."""

        with (
            active_intervention_context(
                intervention_spec=self.capture_spec, hook_plan=self.hook_plan
            ),
            _share_source_state_clones(self.source),
        ):
            yield

    def adopt(self, fork: Trace, refreshed: Trace) -> None:
        """Make ``fork`` the refreshed capture; its ops carry the run's fire records."""

        self.refreshed = refreshed
        fork.replace_state_from(refreshed)

    def ceiling(self) -> RunnableDiagnostic | None:
        """Return the unverifiable ceiling an edited forward settles under."""

        return hooked_run_diagnostic(self.source)

    def settle(self, fork: Trace) -> StagedRunReceipt:
        """Reconcile the fires on the adopted fork and stamp its run ledger.

        Raises
        ------
        ControlFlowDivergenceError
            A re-armed ``intervene=`` predicate staged different entries than the
            source trace holds.
        SiteResolutionError
            ``rerun_zero_fire`` under ``on_zero_fire="error"``, or the typed
            same-input and ``tl.where`` zero-fire refusals, exactly as on a rerun.
        """

        refreshed = self.refreshed
        if refreshed is None:
            raise RuntimeError("Internal invariant violation: settle() before adopt().")
        _refuse_shifted_raw_index_resave(self.source, fork)
        settle_predicate_rerun(self.spec_plan, self.spec, refreshed)
        hook_fire_count, unfired = _reconcile_rerun_hook_fires(
            fork, self.hook_plan, source_log=self.source
        )
        receipt = StagedRunReceipt(
            hooks=len(self.hook_plan),
            hooks_fired=hook_fire_count,
            hooks_unfired=unfired,
            divergence_detail=_graph_divergence_detail(self.source, fork),
        )
        _stamp_staged_run(fork, self, receipt)
        return receipt


def build_staged_run_plan(trace: Trace) -> StagedRunPlan | None:
    """Plan the live application of ``trace``'s staged spec, or ``None`` when nothing is staged.

    Sticky hooks the selection-``do()`` plan attached are engine-owned residue
    of an edit already applied to the saved values, so a trace staging only
    those runs plain (the pending-edit disclosure at the run door covers them).
    """

    if not stages_user_entries(trace):
        return None
    spec = trace._intervention_spec
    refuse_changed_staged_helpers(spec, door="run")
    spec_plan = plan_rerun_spec(trace, spec)
    capture_spec = translate_staged_final_labels(trace, spec_plan.capture_spec)
    hook_plan = _assign_unique_plan_ids(normalize_hooks_from_spec(capture_spec))
    return StagedRunPlan(trace, spec, spec_plan, capture_spec, hook_plan)


def _graph_divergence_detail(source: Trace, fork: Trace) -> str | None:
    """Return the rerun divergence detector's finding for the adopted fork, else ``None``.

    Same detector as the intervened rerun: ordered raw operation events with
    normalized parent-edge indices, falling back to the postprocessed graph
    hash when a loaded source has no raw-event hash.
    """

    old_hash = getattr(source, "_raw_event_shape_hash", None)
    new_hash = getattr(fork, "_raw_event_shape_hash", None)
    if old_hash is None or new_hash is None:
        old_hash = compute_graph_shape_hash(source, rerun_signature=True)
        new_hash = compute_graph_shape_hash(fork, rerun_signature=True)
    if old_hash == new_hash:
        return None
    return (
        "the staged run's raw-event shape hash diverged from the captured graph "
        f"(old={old_hash!r}, new={new_hash!r})"
    )


def _stamp_staged_run(fork: Trace, plan: StagedRunPlan, receipt: StagedRunReceipt) -> None:
    """Record the staged live run on the fork: state, ``last_run`` and the operation ledger."""

    now = time.monotonic()
    divergence_count = 0 if receipt.divergence_detail is None else 1
    fork.state = TraceState.RERUN_PROPAGATED
    fork.last_run = {
        "engine": LIVE_STAGED_ENGINE,
        "timestamp": now,
        "started_at": plan.started_at,
        "duration_s": now - plan.started_at,
        "spec_revision": getattr(fork, "_spec_revision", 0),
        "strict": False,
        "append": False,
        "hooks": receipt.hooks,
        "hooks_fired": receipt.hooks_fired,
        "hooks_unfired": len(receipt.hooks_unfired),
        "unfired_plan_ids": list(receipt.hooks_unfired),
        "divergence_count": divergence_count,
        "fast_refresh": False,
        "fast_refused": None,
        "shape_varied": False,
        "old_graph_shape_hash": plan.source._provenance_fingerprint,
        "new_graph_shape_hash": fork._provenance_fingerprint,
    }
    fork._record_operation(
        "run",
        engine=LIVE_STAGED_ENGINE,
        started_at=plan.started_at,
        hook_count=receipt.hooks,
        hook_fire_count=receipt.hooks_fired,
        unfired_hook_count=len(receipt.hooks_unfired),
        divergence_count=divergence_count,
    )
    fork._has_direct_writes = False
    fork._out_recipe_revision = getattr(fork, "_spec_revision", 0)
