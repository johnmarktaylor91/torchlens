"""M1 keystone lockstep tests: one declared schema, byte-proven views.

The FIELD_ORDER literals in ``constants.py`` remain the hand-written declared
schema; ``FIELD_POLICY`` tables and the generated ``StorageBinding`` tables
must agree with them exactly, in both directions, for all 11 record classes
(docs/reference/trace_core_design.md section 3.4).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from torchlens import constants as tl_constants
from torchlens.data_classes._schema_bindings import STORAGE_BINDINGS
from torchlens.data_classes.aten_op import AtenOp
from torchlens.data_classes.backward_pass import BackwardPass
from torchlens.data_classes.buffer import Buffer
from torchlens.data_classes.field_policy import (
    StorageKind,
    field_order_from_policy,
)
from torchlens.data_classes.func_call_location import FuncCallLocation
from torchlens.data_classes.grad_fn import GradFn
from torchlens.data_classes.grad_fn_call import GradFnCall
from torchlens.data_classes.layer import Layer
from torchlens.data_classes.module import Module, ModuleCall
from torchlens.data_classes.op import Op
from torchlens.data_classes.param import Param
from torchlens.data_classes.trace import Trace

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


@pytest.mark.smoke_cells(
    "test_field_order_view_equals_literal[buffer]",
    "test_field_order_view_equals_literal[func_call_location]",
)
@pytest.mark.parametrize("schema_key", sorted(_SCHEMA))
def test_field_order_view_equals_literal(schema_key: str) -> None:
    """Generated FIELD_ORDER view == the hand-written literal, exactly."""

    cls, literal = _SCHEMA[schema_key]
    assert field_order_from_policy(cls.FIELD_POLICY) == list(literal)


@pytest.mark.parametrize("schema_key", sorted(_SCHEMA))
def test_every_declared_field_has_exactly_one_binding(schema_key: str) -> None:
    """Binding keys == FIELD_POLICY keys: no orphans, no gaps, all 11 classes."""

    cls, _ = _SCHEMA[schema_key]
    bindings = STORAGE_BINDINGS[schema_key]
    assert set(bindings) == set(cls.FIELD_POLICY)
    for name, policy in cls.FIELD_POLICY.items():
        assert policy.storage is bindings[name], name


@pytest.mark.parametrize("schema_key", sorted(_SCHEMA))
def test_binding_axes_are_coherent(schema_key: str) -> None:
    """Storage kinds respect the declared axes' invariants."""

    from torchlens._io import FieldPolicy

    cls, _ = _SCHEMA[schema_key]
    for name, policy in cls.FIELD_POLICY.items():
        binding = policy.storage
        if binding.kind is StorageKind.COMPUTED:
            resolved = next(
                (mro.__dict__[name] for mro in cls.__mro__ if name in mro.__dict__),
                None,
            )
            assert isinstance(resolved, property), (
                f"{schema_key}.{name} is COMPUTED but not a property"
            )
        if policy.portable_policy is FieldPolicy.DROP:
            assert binding.kind in (StorageKind.RUNTIME, StorageKind.COMPUTED), (
                f"{schema_key}.{name} is DROP but stored as {binding.kind}"
            )
        if binding.mutability == "copy_on_read":
            assert name in ("equivalent_ops", "recurrent_ops"), (
                f"unexpected copy_on_read field {schema_key}.{name}"
            )


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
    assert set(candidate.values()) == {"EDGE", "INTERNED", "SCALAR"}


@pytest.mark.heavy
def test_bindings_module_regenerates_identically() -> None:
    """The generator CLI's ``--check`` contract agrees with the checked-in module.

    The byte-level generated-vs-declared diff itself is smoke-gated in-process by
    ``tests/test_schema_lockstep.py::test_generated_artifact_is_current``; this
    heavier case additionally exercises the developer-facing CLI entry point that
    a contributor actually runs.
    """

    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "tools" / "generate_record_schema.py"), "--check"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"stale _schema_bindings.py:\n{result.stdout}\n{result.stderr}"
