"""R44: closed-surface lockstep for the LiveOpView field->getter table.

The 253-line requested-column if-chain in ``_event_live_field`` was rewritten to
a closed module-level field->getter table (disputed-r2 b5 #1; both labs measured
the table neutral-to-faster on the real capture access mix). These tests pin the
semantic-authority property sol demanded: the LiveOpView surface is CLOSED — a
new Op field must be explicitly declared as a live getter or as known-late, and
can never silently land in the wrong branch or the ``AttributeError`` tail.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.capture import projections
from torchlens.capture.projections import (
    _LIVE_FIELD_GETTER_PAIRS,
    _LIVE_FIELD_GETTERS,
    _OPLOG_FIELDS_KNOWN_LATE,
    LIVE_OP_VIEW_FIELDS,
    LiveOpView,
    LiveOpViewFieldNotYetWritten,
)
from torchlens.constants import OP_LOG_FIELD_ORDER

# The documented LiveOpView field surface: live getter-table keys plus the
# known-late names. Any change here is a reviewed contract diff — declare a new
# Op field in exactly one branch and update this list in the same change.
_DOCUMENTED_SURFACE = [
    "_construction_done",
    "_edge_uses",
    "_label_raw",
    "_layer_label_raw",
    "_param_barcodes",
    "_tracing_finished",
    "activation_memory",
    "activation_transform",
    "annotations",
    "arg_names",
    "args_template",
    "atomic_module_call",
    "autograd_memory",
    "bool_value",
    "bytes_delta_at_call",
    "bytes_peak_at_call",
    "children",
    "code_context",
    "container_path",
    "container_spec",
    "detach_saved_activations",
    "dtype",
    "equivalence_class",
    "equivalent_ops",
    "final_out",
    "flops_backward",
    "flops_forward",
    "func",
    "func_autocast_state",
    "func_call_id",
    "func_config",
    "func_duration",
    "func_name",
    "func_non_tensor_args",
    "func_qualname",
    "func_rng_states",
    "grad_fn",
    "grad_fn_class_name",
    "grad_fn_class_qualname",
    "grad_fn_handle",
    "grad_fn_object_id",
    "has_children",
    "has_input_ancestor",
    "has_internal_source_ancestor",
    "has_out_variations",
    "has_output_descendant",
    "has_saved_activation",
    "has_saved_args",
    "in_multi_output",
    "input_ancestors",
    "input_to_module_calls",
    "input_was_parameter",
    "internal_source_ancestors",
    "internal_source_parents",
    "intervention_replaced",
    "interventions",
    "io_role",
    "is_atomic_module",
    "is_buffer",
    "is_final_output",
    "is_inplace",
    "is_input",
    "is_internal_sink",
    "is_internal_source",
    "is_module_output",
    "is_orphan",
    "is_output",
    "is_output_parent",
    "is_scalar_bool",
    "kwargs_template",
    "label",
    "label_short",
    "layer_label",
    "layer_label_short",
    "lookup_keys",
    "module",
    "module_call_stack",
    "module_entry_arg_keys",
    "modules",
    "multi_output_index",
    "multi_output_name",
    "non_tensor_kwargs",
    "non_tensor_pos_args",
    "num_args_total",
    "num_autograd_tensors",
    "num_kwargs",
    "num_params",
    "num_passes",
    "num_pos_args",
    "out",
    "out_versions_by_child",
    "output_descendants",
    "output_device",
    "output_of_module_calls",
    "output_of_modules",
    "param_shapes",
    "parent_arg_positions",
    "parent_param_ops",
    "parent_params",
    "parents",
    "pass_index",
    "raw_index",
    "recurrent_ops",
    "root_ancestors",
    "saved_args",
    "saved_kwargs",
    "shape",
    "source_trace",
    "step_index",
    "transformed_activation_memory",
    "transformed_out",
    "transformed_out_dtype",
    "transformed_out_shape",
    "type",
    "type_index",
    "visualizer_path",
]


def test_surface_matches_documented_list_exactly() -> None:
    """Table keys + known-late set == the documented LiveOpView surface."""

    assert sorted(LIVE_OP_VIEW_FIELDS) == _DOCUMENTED_SURFACE


def test_no_duplicate_getter_declarations() -> None:
    """Every field name is declared exactly once in the getter pair list."""

    names = [name for name, _getter in _LIVE_FIELD_GETTER_PAIRS]
    assert len(names) == len(set(names))
    assert set(names) == set(_LIVE_FIELD_GETTERS)


def test_live_and_known_late_are_disjoint() -> None:
    """A field cannot be simultaneously live and postprocess-only."""

    assert not set(_LIVE_FIELD_GETTERS) & _OPLOG_FIELDS_KNOWN_LATE


def test_every_live_field_is_a_declared_op_field() -> None:
    """Live getter names stay in lockstep with the Op field order."""

    stray = sorted(set(_LIVE_FIELD_GETTERS) - set(OP_LOG_FIELD_ORDER))
    assert not stray, f"live getters not in OP_LOG_FIELD_ORDER: {stray}"


def test_getters_are_module_level_two_arg_callables() -> None:
    """The table is built once at import from module-level 2-arg getters."""

    import inspect

    for name, getter in _LIVE_FIELD_GETTER_PAIRS:
        assert len(inspect.signature(getter).parameters) == 2, name


def test_known_late_raises_typed_and_unknown_raises_attribute_error() -> None:
    """The typed known-late refusal and the AttributeError tail are preserved."""

    with pytest.raises(LiveOpViewFieldNotYetWritten):
        projections._event_live_field(None, None, "label")
    with pytest.raises(AttributeError):
        projections._event_live_field(None, None, "definitely_not_a_field")


class _MidForwardReader(nn.Module):
    """Reads every live field from a mid-forward LiveOpView."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.read: dict[str, object] = {}
        self.mutable_fresh: bool | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.lin(x))
        trace = _state._active_trace
        if trace is not None and getattr(trace, "capture_events", None) is not None:
            label = next(reversed(trace.capture_events.live_index.by_raw_label))
            view = LiveOpView(trace, trace.capture_events.live_index.require_event(label))
            for name in _LIVE_FIELD_GETTERS:
                self.read[name] = getattr(view, name)
            # Mutable projections stay fresh per read (adapter contract).
            self.mutable_fresh = getattr(view, "modules") is not getattr(view, "modules")
        return y + 1


def test_every_live_field_reads_mid_forward() -> None:
    """Functional smoke: all 111 live fields project on a real capture event."""

    model = _MidForwardReader()
    tl.trace(model, torch.randn(2, 4))
    assert set(model.read) == set(_LIVE_FIELD_GETTERS)
    assert model.read["_tracing_finished"] is False
    assert model.mutable_fresh is True
