"""Propagation for one ``do()`` transaction: push from the hooks it attached.

``do()`` attaches its edit as sticky hooks and then propagates. The origins
of that propagation are the sites THOSE hooks target; every other rule the
trace carries (a capture-time recipe, an earlier ``attach_hooks``) still
fires wherever the cone re-executes its site, but opens no cone of its own.
Treating every stored rule as an origin re-executed sites nowhere downstream
of the edit; an in-place parameter write re-executed that way reads the
parameter's live (already written) value and changes outside the edit's cone.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ._param_fingerprint import changed_param_scope
from .errors import ReplayPreconditionError
from .hooks import _normalize_sticky_hook_specs
from .replay import (
    _normalize_replay_hooks,
    _origin_sites_for_hooks,
    _preflight_log,
    _refuse_armed_gradient_rules,
    _run_replay,
    _warn_if_direct_writes_will_be_overlaid,
)

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def push_from_handles(trace: Trace, handle_ids: Iterable[str], *, strict: bool) -> Trace:
    """Push one transaction's freshly attached hooks downstream.

    Parameters
    ----------
    trace:
        Model log to mutate in place.
    handle_ids:
        Sticky-hook handle ids this transaction attached. Empty when the
        transaction staged a value instead of hooks; every stored rule is
        then an origin, as for a plain ``push()``.
    strict:
        Whether divergence warnings should raise.

    Returns
    -------
    Trace
        The same model log, mutated in place.
    """

    wanted = frozenset(handle_ids)
    _preflight_log(trace)
    _refuse_armed_gradient_rules(trace, None)
    _warn_if_direct_writes_will_be_overlaid(trace)
    hook_entries = _normalize_replay_hooks(trace, None)
    origin_entries = hook_entries
    if wanted:
        own_specs = [
            spec for spec in trace._ensure_intervention_spec().hook_specs if spec.handle in wanted
        ]
        origin_entries = _normalize_sticky_hook_specs(
            own_specs, allow_replay_only_site_targets=True
        )
    origins = _origin_sites_for_hooks(trace, origin_entries, strict=strict)
    if not origins:
        raise ReplayPreconditionError("push requires at least one hook target")
    with changed_param_scope(False):
        return _run_replay(
            trace, origins, hook_entries=hook_entries, strict=strict, preserve_origins=False
        )
