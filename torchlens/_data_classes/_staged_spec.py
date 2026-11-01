"""Staged intervention spec reads shared by the ``Trace.run()`` door.

Split out of ``_trace_validation.py`` (file cap).
"""

from __future__ import annotations

from typing import Any


def _staged_user_entries(trace: Any) -> tuple[list[Any], Any]:
    """Return the user-staged hook specs and the staged value specs of ``trace``.

    Sticky hooks the selection-do() plan attached are engine-owned residue of
    an edit already applied to the trace's saved values, so they are left out
    of the user hook list; the value specs come back exactly as staged.
    """

    staged_spec = getattr(trace, "_intervention_spec", None)
    staged_user_hooks = [
        hook_spec
        for hook_spec in getattr(staged_spec, "hook_specs", ())
        if not (getattr(hook_spec, "metadata", None) or {}).get("selection_do_engine_owned")
    ]
    return staged_user_hooks, getattr(staged_spec, "target_value_specs", ())
