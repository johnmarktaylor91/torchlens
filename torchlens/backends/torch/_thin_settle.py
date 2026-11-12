"""The thin record path's settle side: the per-call defaults table and the plain-session facts.

The defaults tables hold the field values every exhaustive call writes with the
same value whatever the call (the per-call constants). Both field builders,
``_ops_shared_fields._build_shared_fields_dict`` and
``_ops_activations._log_output_tensor_info``, take them from here with one
``dict.update`` instead of one store per field, on today's emit and on the thin
emit alike, so the values every reader (the pending row, its settle, the view)
reads back are identical on both paths. Only immutable values belong here: a
mutable default (``annotations``, ``out_versions_by_child``) is built per call.

:func:`plain_session` is the session half of today's plain-output check
(``_ops_exhaustive.plain_single_output``), shared by that check and the thin emit.

This module imports nothing from the capture import cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

__all__ = ("SHARED_CALL_DEFAULTS", "OUTPUT_CALL_DEFAULTS", "plain_session")

SHARED_CALL_DEFAULTS: MappingProxyType[str, Any] = MappingProxyType(
    {
        "_construction_done": False,
        "container_path": (),
        "container_spec": None,
    }
)
"""The call-wide constants ``_build_shared_fields_dict`` writes on every call."""

OUTPUT_CALL_DEFAULTS: MappingProxyType[str, Any] = MappingProxyType(
    {
        "step_index": None,
        "_tracing_finished": False,
        "out": None,
        "transformed_out": None,
        "has_saved_activation": False,
        "is_edited": False,
        "fire_results": (),
        "has_saved_args": False,
        "saved_args": None,
        "saved_kwargs": None,
        "transformed_out_shape": None,
        "transformed_out_dtype": None,
        "transformed_activation_memory": None,
        "visualizer_path": None,
        # Not measured is None, never a fabricated 0 (observe item 15); the
        # track_device_memory commit-site projection overwrites when sampled.
        "bytes_delta_at_call": None,
        "bytes_peak_at_call": None,
    }
)
"""The per-output constants ``_log_output_tensor_info`` writes for every output."""


def plain_session(trace: Any, keep_alias_contract: Callable[[Any], bool]) -> bool:
    """Return whether the session asks for none of the per-op extras the plain emits skip.

    No device-memory tracking, no predicate save options, no aten values, no
    nonfinite tracking or policy, no echo session and no alias-mutation contract
    (``keep_alias_contract`` is ``_should_keep_alias_mutation_contract``). Read per
    call: a session that switches one on mid-forward leaves the plain emits from
    that call.
    """

    trace_dict = trace.__dict__
    return not (
        getattr(trace, "track_device_memory", False)
        or trace_dict.get("_predicate_save_options") is not None
        or trace._options_aten_ops == "values"
        or trace_dict.get("track_nonfinite")
        or trace_dict.get("_echo_session") is not None
        or trace._options_on_nonfinite
        or keep_alias_contract(trace)
    )
