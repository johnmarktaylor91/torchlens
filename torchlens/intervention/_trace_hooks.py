"""Trace hook functions reached from the records through ``_data_classes._doors.intervention``.

The private ``Trace._attach_hooks`` / ``Trace._clear_hooks`` bodies (sticky hook
registration on the mutable recipe and its clearing). ``Trace.do`` is the public
door: attaching folded into ``do`` and clearing goes through ``clear_edits``.
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
from .._selection.resolver import _label_index_scope
from ._edit_guards import refuse_chunked_batch_coupling, warn_if_root_mutation
from ._live_hook_plan import retire_live_hook_handles, sticky_handle_ids
from ._trace_edits import (
    ensure_intervention_spec,
    target_spec_from_site,
    validate_intervention_site,
)

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__: list[str] = []

from .errors import HookSignatureError
from .hooks import expand_facet_hook_entries, normalize_hook_plan
from .spec import (
    Recipe as PublicInterventionSpec,
    entries_from_spec,
)


def attach_sticky_hooks(  # noqa: PLR0913 - trace stands in for self on the moved method
    trace: Trace,
    hooks_or_site: Any,
    hook: Any = None,
    *extra_hooks: Any,
    strict: bool = False,
) -> tuple[str, ...]:
    """Stage sticky hooks on the recipe (the ``Trace._attach_hooks`` body); return handle ids."""

    warn_if_root_mutation(trace)
    if isinstance(hooks_or_site, PublicInterventionSpec):
        # C03 spec door: the one immutable spec is accepted unchanged; its rules
        # already carry site and action, so extra arguments have nothing to bind to.
        if hook is not None or extra_hooks:
            raise InvalidArgumentError(
                "do(spec) takes the Recipe alone: its "
                "rules already carry the site and action, so extra hook "
                "arguments have nothing to bind to",
                code="spec_door_extra_arguments",
                remedy="pass only the spec (build clauses with tl.when and "
                "merge them), or use the (site, hook) call shape without a spec",
                argument="hooks_or_site",
            )
        entries = entries_from_spec(hooks_or_site, door="do")
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
    spec = ensure_intervention_spec(trace)
    handle_ids: list[str] = []
    arming_group = f"arm-{uuid.uuid4().hex}"
    gradient_entries = 0
    with _label_index_scope():  # one label index for every site; nothing relabels here
        for entry in entries:
            validate_intervention_site(trace, entry.site_target, strict=strict)
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
        handles=tuple(handle_ids),
        **history,
    )
    trace._last_hook_handle_ids = tuple(handle_ids)
    return tuple(handle_ids)


def clear_hooks(trace: Trace) -> Trace:
    """Clear every sticky hook from the recipe (the ``Trace._clear_hooks`` body)."""

    warn_if_root_mutation(trace)
    spec = ensure_intervention_spec(trace)
    handles_before = sticky_handle_ids(spec)
    spec.clear()
    mark_intervention_spec_mutated(trace)
    # Removal takes effect at once, in a rerun or backward pass in progress.
    retire_live_hook_handles(handles_before)
    record_operation(trace, "clear_hooks")
    return trace
