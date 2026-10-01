"""Trace legacy persisted-state spellings (alias declarations + retirements).

The declarations ``Trace.__setstate__`` still adapts live here so the state
contract's known/unknown partition (ecosystem MEMO 3.3) can enumerate them
without the class body carrying the ledger: every spelling below is a key a
governed release writer actually persisted, each with its disposition --
renamed, folded, or retired-and-dropped. The writer contract publishes the
set as the Trace record's alias rules (alias-or-fail).
"""

from __future__ import annotations

from typing import Any

from .._runnable_seam import LEGACY_RUNNABLE_TRACE_FIELD_MAP

#: Legacy persisted spellings ``Trace.__setstate__`` still adapts (renames
#: and folded families). Declared so the known/unknown partition never
#: mistakes a governed legacy key for an unknown field.
TRACE_PORTABLE_STATE_ALIASES: frozenset[str] = frozenset(
    {
        "_grad_layer_nums_to_save",  # renamed -> _grad_op_nums_to_save
        "_saved_grads_set",  # renamed -> _saved_grad_labels
        "_keep_grads_in_memory",  # retired knob, dropped on restore
        "_grad_stream_retain_in_memory",  # retired knob, dropped on restore
        "conditional_then_entry_edges",  # folded -> conditional_arm_entry_edges
        "conditional_elif_entry_edges",  # folded -> conditional_arm_entry_edges
        "conditional_else_entry_edges",  # folded -> conditional_arm_entry_edges
        # Persisted by released v2.33.0/v2.34.1 writers (harvested-corpus
        # evidence), superseded since: the buffer-write family moved to
        # the capture journal (the live read is a property; a restored
        # trace gets a detached stream), and the scoped-detached-patching
        # stamps were retired with the crawler deletion. Declared and
        # popped explicitly -- these were the silently-absorbed unknown
        # keys of the D-ECO-10 measurement.
        "_buffer_write_events",
        "detached_patch_epoch",
        "detached_patch_policy",
        # One-shot warning flag persisted by the pre-columnar baseline
        # writers (frozen godobject-oracle artifact evidence, tlspec v6),
        # later consolidated into the session-only ``_warned_once`` key
        # set. Dropped on restore.
        "_warned_nonfinite_check_unavailable",
    }
    # The one-field runnable seam's legacy per-key spellings, adapted by
    # normalize_runnable_trace_state on every restore.
    | frozenset(LEGACY_RUNNABLE_TRACE_FIELD_MAP)
)

#: Retired-and-dropped spellings: declared aliases with NO live successor
#: field. Restore pops them explicitly -- the historical silent absorption
#: made an explicit adapter (renamed/folded spellings are adapted separately
#: by ``Trace.__setstate__``'s dedicated rename and fold blocks).
_RETIRED_DROPPED_KEYS: tuple[str, ...] = (
    "_buffer_write_events",
    "detached_patch_epoch",
    "detached_patch_policy",
    "_warned_nonfinite_check_unavailable",
)


def pop_retired_legacy_keys(state: dict[str, Any]) -> None:
    """Drop the retired legacy spellings from an incoming ``Trace`` state.

    Parameters
    ----------
    state:
        Serialized ``Trace`` state mapping, mutated in place.
    """

    for key in _RETIRED_DROPPED_KEYS:
        state.pop(key, None)
