"""M1 keystone lockstep tests: one declared schema, byte-proven views.

The FIELD_ORDER literals in ``constants.py`` remain the hand-written declared
schema; ``_FIELD_POLICY`` tables and the generated ``StorageBinding`` tables
must agree with them exactly, in both directions, for all 11 record classes
(docs/internal/reference/trace_core_design.md section 3.4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from torchlens import _constants as tl_constants
from torchlens._data_classes._schema_bindings import STORAGE_BINDINGS
from torchlens._data_classes.aten_op import AtenOp
from torchlens._data_classes.backward_pass import BackwardPass
from torchlens._data_classes.buffer import Buffer
from torchlens._data_classes.field_policy import (
    field_order_from_policy,
)
from torchlens._data_classes.func_call_location import FuncCallLocation
from torchlens._data_classes.grad_fn import GradFn
from torchlens._data_classes.grad_fn_call import GradFnCall
from torchlens._data_classes.layer import Layer
from torchlens._data_classes.module import Module, ModuleCall
from torchlens._data_classes.op import Op
from torchlens._data_classes.param import Param
from torchlens._data_classes.trace import Trace

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SCHEMA = {
    "primitive_op": (AtenOp, tl_constants.PRIMITIVE_OP_FIELD_ORDER),
    "trace": (Trace, tl_constants.MODEL_LOG_FIELD_ORDER),
    "op": (Op, tl_constants.LAYER_PASS_LOG_FIELD_ORDER),
    "layer": (Layer, tl_constants.LAYER_LOG_FIELD_ORDER),
    "module": (Module, tl_constants.MODULE_LOG_FIELD_ORDER),
    "module_call": (ModuleCall, tl_constants.MODULE_PASS_LOG_FIELD_ORDER),
    "param": (Param, tl_constants.PARAM_LOG_FIELD_ORDER),
    "buffer": (Buffer, tl_constants.BUFFER_LOG_FIELD_ORDER),
    "grad_fn": (GradFn, tl_constants.GRAD_FN_LOG_FIELD_ORDER),
    "grad_fn_call": (GradFnCall, tl_constants.GRAD_FN_PASS_LOG_FIELD_ORDER),
    "backward_pass": (BackwardPass, tl_constants.BACKWARD_PASS_FIELD_ORDER),
    "func_call_location": (
        FuncCallLocation,
        tl_constants.FUNC_CALL_LOCATION_FIELD_ORDER,
    ),
}


@pytest.mark.parametrize("schema_key", sorted(_SCHEMA))
def test_field_order_view_equals_literal(schema_key: str) -> None:
    """Generated FIELD_ORDER view == the hand-written literal, exactly."""

    cls, literal = _SCHEMA[schema_key]
    assert field_order_from_policy(cls._FIELD_POLICY) == list(literal)


# Kept: the ATen-values stack edit added the PAYLOAD-cell assertion after slice 1's
# coverage measurement, so the deletion evidence does not cover it.
def test_primitive_bindings_match_the_promoted_candidate_kinds() -> None:
    """The tlspec v8 bump installed the S3 candidate kinds for real.

    Pre-bump this pinned the installed rows to all-RUNTIME (DROP policies)
    with the candidate table isolated; the bump promoted every primitive
    policy to KEEP, so the installed bindings must now equal the candidate
    classification exactly.
    """

    from tools.generate_record_schema import collect_primitive_candidate_bindings

    candidate = collect_primitive_candidate_bindings()
    installed = {
        name: binding.kind.name for name, binding in STORAGE_BINDINGS["primitive_op"].items()
    }
    assert installed == candidate
    assert candidate["parent_op_refs"] == "EDGE"
    assert candidate["module_call_stack"] == "EDGE"
    assert candidate["input_tensor_facts"] == "SCALAR"
    assert candidate["output_tensor_facts"] == "SCALAR"
    assert candidate["label"] == "INTERNED"
    assert candidate["execution_context"] == "INTERNED"
    # ``out`` (aten_ops="values") is the one payload cell a primitive row carries.
    assert candidate["out"] == "PAYLOAD"
    assert set(candidate.values()) == {"EDGE", "INTERNED", "SCALAR", "PAYLOAD"}
