"""Trace hook functions reached from the records through ``_data_classes._doors.intervention``.

``attach_hooks`` and ``clear_hooks`` of ``Trace``, moved verbatim as functions
taking the trace first (sticky hook registration on the mutable intervention
spec and its clearing).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from .._data_classes._trace_grad_sites import live_backward_graph_available, sticky_hook_entry
from .._data_classes._trace_intervention_records import (
    history_site_payload,
    mark_intervention_spec_mutated,
    record_operation,
)
from .._errors import InvalidArgumentError
from ._trace_edits import (
    ensure_intervention_spec,
    refuse_chunked_batch_coupling,
    target_spec_from_site,
    validate_intervention_site,
    warn_if_root_mutation,
)

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__: list[str] = []

from .errors import HookSignatureError
from .handles import HookHandle
from .hooks import expand_facet_hook_entries, normalize_hook_plan
from .spec import (
    Recipe as PublicInterventionSpec,
    entries_from_spec,
)


def attach_hooks(
    trace: Trace,
    hooks_or_site: Any,
    hook: Any = None,
    *extra_hooks: Any,
    strict: bool = False,
    prepend: bool = False,
    confirm_mutation: bool = False,
) -> Any:
    """Attach sticky hooks to the current intervention spec. (the ``Trace.attach_hooks`` body)."""

    warn_if_root_mutation(trace, confirm_mutation=confirm_mutation)
    if isinstance(hooks_or_site, PublicInterventionSpec):
        # C03 spec door: the ONE immutable spec is accepted unchanged.
        # Its rules already carry site and action, so extra call
        # arguments have nothing sound to bind to.
        if hook is not None or extra_hooks:
            raise InvalidArgumentError(
                "attach_hooks(spec) takes the InterventionSpec alone: its "
                "rules already carry the site and action, so extra hook "
                "arguments have nothing to bind to",
                code="spec_door_extra_arguments",
                remedy="pass only the spec (build clauses with tl.when and "
                "merge them), or use the (site, hook) call shape without a spec",
                argument="hooks_or_site",
            )
        entries = entries_from_spec(hooks_or_site, door="attach_hooks")
    elif extra_hooks:
        if hook is None:
            raise HookSignatureError("extra hooks require an initial hook argument.")
        entries = normalize_hook_plan(
            [(hooks_or_site, hook_like) for hook_like in (hook, *extra_hooks)],
            allow_replay_only_site_targets=True,
        )
    else:
        entries = normalize_hook_plan(
            hooks_or_site,
            hook,
            allow_replay_only_site_targets=True,
        )
    entries = expand_facet_hook_entries(trace, entries)
    refuse_chunked_batch_coupling(trace, entries)
    for entry in entries:
        validate_intervention_site(trace, entry.site_target, strict=strict)
    spec = ensure_intervention_spec(trace)
    handle_ids: list[str] = []
    arming_group = f"arm-{uuid.uuid4().hex}"
    gradient_entries = 0
    for entry in entries:
        handle_id = f"hook-{uuid.uuid4().hex}"
        handle_ids.append(handle_id)
        metadata, stored_hook, is_gradient = sticky_hook_entry(entry, arming_group)
        gradient_entries += int(is_gradient)
        spec.add_hook(
            target_spec_from_site(trace, entry.site_target, strict=strict),
            stored_hook,
            helper=entry.helper_spec,
            handle=handle_id,
            metadata=metadata,
            prepend=prepend,
        )
    if gradient_entries == len(entries):
        # Gradient rules edit nothing the forward recorded, so the saved
        # activations stay current: bump the recipe revision without
        # marking the outs stale.
        mark_intervention_spec_mutated(trace, outs_stale=False)
    else:
        mark_intervention_spec_mutated(trace)
    history: dict[str, Any] = {}
    if gradient_entries and not live_backward_graph_available(trace):
        history["armed_for"] = "next_rerun"
    record_operation(
        trace,
        "attach_hooks",
        hook_count=len(entries),
        sites=tuple(history_site_payload(entry.site_target) for entry in entries),
        strict=strict,
        prepend=prepend,
        handles=tuple(handle_ids),
        **history,
    )
    trace._last_hook_handle_ids = tuple(handle_ids)
    return HookHandle(trace, tuple(handle_ids), confirm_mutation=confirm_mutation)


def clear_hooks(trace: Trace, *, confirm_mutation: bool = False) -> Trace:
    """Clear all sticky hooks from the current intervention spec. (the ``Trace.clear_hooks`` body)."""

    warn_if_root_mutation(trace, confirm_mutation=confirm_mutation)
    ensure_intervention_spec(trace).clear()
    mark_intervention_spec_mutated(trace)
    record_operation(trace, "clear_hooks")
    return trace
