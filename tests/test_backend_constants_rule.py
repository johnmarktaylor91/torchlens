"""The shared constants rule (``torchlens/backends/_constants.py``) on a fake backend.

A NumPy-backed fake backend emits the same ``OpEvent`` records the preview
backends emit, materializes them and runs them through
``finalize_single_pass_trace`` with a ``ConstantLedger``: one closure constant
(a tensor built outside the forward) and one literal producer (an op with no
tensor inputs). No preview framework is needed.

The validation carve-out must stay narrow: a parentless op that lost a parent
edge (``unattributed_tensor_args``) or draws from the RNG is never a constant.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.trace import Trace
from torchlens._errors import CaptureGapError
from torchlens._ir.capture_events import CaptureEvents
from torchlens._ir.events import ArgTemplateRef, FunctionCallRef, OpEvent, OutputRef, ParentEdge
from torchlens._ir.predicate import RecordContext
from torchlens._ir.refs import DtypeRef, ReservedLabel, TensorRef
from torchlens._ir.semantics import BackendSemantics, CapturePolicy
from torchlens._postprocess._materialize import materialize_from_events
from torchlens.backends._backend_protocol import REFUSAL_CODES, VALIDATION_REASONS
from torchlens.backends._constants import (
    CONSTANT_FUNC_NAME,
    CONSTANT_PARENT_PAYLOAD_REASON,
    CONSTANT_PRODUCER_LEAF_REASON,
    CONSTANT_UNREALIZABLE_CODE,
    ConstantLedger,
    apply_constants_rule,
    constant_label,
    constant_leaf_reason,
    constant_op_labels,
    constant_parent_skip_reason,
    is_closure_constant_op,
    is_constant_op,
    is_literal_producer_op,
    is_rng_producer_name,
)
from torchlens.backends._finalize import (
    attach_function_root_module,
    finalize_single_pass_trace,
    mark_output_label,
)
from torchlens.backends.registry import BackendUnsupportedError
from torchlens.recording.types import CaptureSpec


class _NumpyBackend:
    """A fake preview backend whose tensors are NumPy arrays."""

    name = "fake"

    def __init__(self) -> None:
        """Start a labelled tensor store and an empty ledger."""

        self.labels: dict[int, str] = {}
        self.ledger = ConstantLedger(self.name)

    @staticmethod
    def is_tensor(value: object) -> bool:
        """NumPy arrays are this backend's tensors."""

        return isinstance(value, np.ndarray)

    @staticmethod
    def host_copy(value: Any) -> np.ndarray:
        """A copy with shape kept (0-d stays 0-d)."""

        return np.array(value, copy=True)

    def new_trace(self) -> Trace:
        """An empty function-root trace shell with a live event journal."""

        trace = Trace(model_class_name="FakeFunction")
        trace._provenance_backend = "fake"  # type: ignore[assignment]
        trace._provenance_root_entry_point = "function_call:fake_forward"
        trace.capture_events = CaptureEvents()
        return trace

    def emit_input(self, trace: Trace, value: np.ndarray, label: str) -> str:
        """Record one input leaf."""

        events = trace.capture_events
        events.raw_layer_counter += 1
        reserved = ReservedLabel(
            label=label,
            label_raw=label,
            raw_index=events.raw_layer_counter,
            type_index=events.raw_layer_counter,
            layer_type="input",
            site=label,
        )
        self.labels[id(value)] = label
        events.append(
            self._event(trace, "source", reserved, "input", (), value, parents=(), is_input=True)
        )
        return label

    def emit_closure_constant(
        self, trace: Trace, value: np.ndarray, *, consumer_label_raw: str | None = None
    ) -> str:
        """Record one closure constant through the ledger (rule 1)."""

        events = trace.capture_events
        reserved = events.reserve_label("constant")
        record, payload = self.ledger.register_closure(
            value,
            host_copy=self.host_copy,
            label_raw=reserved.label_raw,
            consumer_label_raw=consumer_label_raw,
        )
        self.labels[id(value)] = record.label_raw
        events.append(
            self._event(
                trace,
                "source",
                reserved,
                CONSTANT_FUNC_NAME,
                (),
                payload,
                parents=(),
                is_input=False,
            )
        )
        return record.label_raw

    def emit_op(
        self,
        trace: Trace,
        func_name: str,
        args: tuple[Any, ...],
        output: np.ndarray,
        *,
        unattributed: tuple[str, ...] = (),
    ) -> str:
        """Record one op; tensor args with a label become parents."""

        events = trace.capture_events
        reserved = events.reserve_label(func_name)
        parents = tuple(
            ParentEdge(parent_label_raw=self.labels[id(arg)], arg_position=index, _edge_use="arg")
            for index, arg in enumerate(args)
            if self.is_tensor(arg) and id(arg) in self.labels
        )
        if not parents and not any(self.is_tensor(arg) for arg in args):
            self.ledger.register_literal(
                reserved.label_raw, shape=tuple(output.shape), dtype=str(output.dtype)
            )
        self.labels[id(output)] = reserved.label_raw
        events.append(
            self._event(
                trace,
                "op",
                reserved,
                func_name,
                args,
                output,
                parents=parents,
                is_input=False,
                unattributed=unattributed,
            )
        )
        return reserved.label_raw

    def finish(self, trace: Trace, output: np.ndarray) -> Trace:
        """Mark the output, materialize the journal and finalize."""

        mark_output_label(trace, self.labels[id(output)])
        materialize_from_events(trace, trace.capture_events)
        delattr(trace, "capture_events")
        finalize_single_pass_trace(
            trace,
            backend_name=self.name,
            module_tree=None,
            attach_function_root_module=attach_function_root_module,
            attach_object_module_logs=lambda *_a, **_k: None,
            constants=self.ledger,
        )
        return trace

    def _event(
        self,
        trace: Trace,
        kind: str,
        reserved: ReservedLabel,
        func_name: str,
        args: tuple[Any, ...],
        output: np.ndarray,
        *,
        parents: tuple[ParentEdge, ...],
        is_input: bool,
        unattributed: tuple[str, ...] = (),
    ) -> OpEvent:
        """Build one topology-complete event the way the preview backends do."""

        policy = CapturePolicy(save_payload=True, save_grad=False)
        input_ancestors = frozenset(
            edge.parent_label_raw for edge in parents if edge.parent_label_raw.startswith("input")
        )
        non_tensor_args = tuple(arg for arg in args if not self.is_tensor(arg))
        return OpEvent(
            kind=kind,
            label_raw=reserved.label_raw,
            layer_label_raw=reserved.label_raw,
            layer_type=reserved.layer_type,
            raw_index=reserved.raw_index,
            type_index=reserved.type_index,
            step_index=reserved.raw_index,
            source_trace=trace,
            source_trace_id=None,
            tracing_finished=False,
            construction_done=True,
            function=FunctionCallRef(
                func=None,
                func_name=func_name,
                func_qualname=None,
                func_call_id=reserved.raw_index,
                code_context=(),
                func_duration=None,
                flops_forward=None,
                flops_backward=None,
                func_rng_states=None,
                func_autocast_state=None,
                arg_names=(),
                num_args_total=len(args),
                num_pos_args=len(args),
                num_kwargs=0,
                non_tensor_pos_args=non_tensor_args,
                non_tensor_kwargs=(),
                func_non_tensor_args=non_tensor_args,
                is_inplace=False,
                func_config=(),
            ),
            output=OutputRef(
                tensor=TensorRef(
                    label_raw=reserved.label_raw,
                    shape=tuple(output.shape),
                    dtype=str(output.dtype),
                    device="cpu",
                    requires_grad=False,
                    memory=int(output.nbytes),
                    payload=output,
                    blob_ref=None,
                    backend_handle_id=str(id(output)),
                ),
                transformed_tensor=None,
                has_saved_activation=True,
                output_device="same",
                activation_transform=None,
                detach_saved_activations=False,
                visualizer_path=None,
                multi_output_index=None,
                in_multi_output=False,
                container_path=(),
                container_spec=None,
                child_versions=(),
            ),
            templates=ArgTemplateRef(
                saved_args=None,
                saved_kwargs=None,
                args_template=None,
                kwargs_template=None,
                has_saved_args=False,
            ),
            parents=parents,
            parent_arg_positions={
                "args": {edge.arg_position: edge.parent_label_raw for edge in parents},
                "kwargs": {},
            },
            _edge_uses=tuple(
                (edge.parent_label_raw, edge.arg_position, edge._edge_use) for edge in parents
            ),
            params=(),
            parent_params=(),
            module_stack=(),
            modules=(),
            backend_semantics=BackendSemantics(
                backend_grad_handle=None,
                grad_fn_class_name=None,
                autograd_memory=None,
                num_autograd_tensors=None,
                mutated_input_positions=(),
                aliased_output_inputs=(),
                unknown_aliasing=False,
                bytes_delta_at_call=None,
                bytes_peak_at_call=None,
            ),
            policy=policy,
            predicate_matched=True,
            pass_index=1,
            grad_fn_class_qualname=None,
            grad_fn_handle=None,
            equivalence_class=reserved.layer_type,
            is_transform=False,
            transform_kind=None,
            transform_chain=(),
            transform_config={},
            transform_fn_name=None,
            transform_fn_qualname=None,
            transform_fn_source=None,
            unattributed_tensor_args=unattributed,
            dropped_edge_tensor_args=(),
            is_output_parent=False,
            has_internal_source_ancestor=not is_input and not parents,
            internal_source_ancestors=frozenset(),
            input_ancestors=input_ancestors,
            root_ancestors=input_ancestors or frozenset({reserved.label_raw}),
            func_call_id=reserved.raw_index,
            is_bottom_level=True,
            is_scalar_bool=None,
            bool_value=None,
            intervention_fired=False,
            is_edited=False,
            fire_results=(),
            intervention_template_ref=None,
            record_context=RecordContext(
                kind="op",
                label=reserved.label,
                raw_label=reserved.label_raw,
                pass_number=1,
                event_index=reserved.raw_index,
                step_index=None,
                layer_type=reserved.layer_type,
                type_index=reserved.type_index,
                raw_index=reserved.raw_index,
                func_name=func_name,
                address=None,
                module_type=None,
                module_pass_number=None,
                module_stack=(),
                recent_events=(),
                recent_ops=(),
                parent_labels=(),
                input_output_address=None,
                shape=tuple(output.shape),
                dtype=DtypeRef(backend="fake", name=str(output.dtype)),
                tensor_device=None,
                tensor_requires_grad=False,
                output_index=None,
                is_bottom_level_func=True,
                time_since_pass_start=0.0,
                sample_id=None,
                label_raw=reserved.label_raw,
                label_prefix=reserved.layer_type,
                func_call_id=reserved.raw_index,
                parent_labels_raw=(),
                is_output_parent=False,
                backend_requires_isolation=False,
                is_scalar_bool=False,
                bool_value=None,
            ),
            capture_spec=CaptureSpec(save_out=True, save_metadata=True),
        )


def _capture_planted_forward(*, closure_shape: tuple[int, ...] = (2, 3)) -> tuple[Trace, dict]:
    """``out = (x + W) * ones(2, 3)`` with ``W`` a closure constant, ``ones`` a literal."""

    backend = _NumpyBackend()
    trace = backend.new_trace()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    w = np.full(closure_shape, 0.5, dtype=np.float32)
    labels: dict[str, str] = {}
    labels["x"] = backend.emit_input(trace, x, "input.arg_0")
    labels["w"] = backend.emit_closure_constant(trace, w, consumer_label_raw="add")
    added = x + w
    labels["add"] = backend.emit_op(trace, "add", (x, w), added)
    ones = np.ones((2, 3), dtype=np.float32)
    labels["ones"] = backend.emit_op(trace, "ones", ((2, 3), "float32"), ones)
    out = added * ones
    labels["mul"] = backend.emit_op(trace, "mul", (added, ones), out)
    backend.finish(trace, out)
    return trace, labels


def _op_by_raw(trace: Trace, label_raw: str) -> Any:
    """The finalized op whose raw label is ``label_raw``."""

    for op in trace._layer_list:
        if getattr(op, "_label_raw", None) == label_raw or op.label == label_raw:
            return op
    raise AssertionError(f"no op with raw label {label_raw!r}: {list(trace.layer_labels)}")


@pytest.mark.smoke
def test_closure_constant_is_a_parentless_source_op_with_payload_and_a_consumer_edge() -> None:
    """Rule 1: ``constant`` source op, no parents, payload saved, consumer lists it."""

    trace, labels = _capture_planted_forward()
    constant = _op_by_raw(trace, labels["w"])
    assert constant.func_name == CONSTANT_FUNC_NAME
    assert tuple(constant.parents) == ()
    assert constant.is_input is False and constant.is_buffer is False
    assert constant.out is not None and tuple(constant.out.shape) == (2, 3)
    assert tuple(constant.shape) == (2, 3)
    assert is_closure_constant_op(constant) and is_constant_op(constant)
    assert not is_literal_producer_op(constant)
    consumer = _op_by_raw(trace, labels["add"])
    assert constant.layer_label in tuple(consumer.parents)
    assert consumer.layer_label in tuple(constant.children)


def test_literal_producer_keeps_zero_parents_and_a_payload() -> None:
    """Rule 2: ``ones`` stays an op with zero parents, a payload and its real name."""

    trace, labels = _capture_planted_forward()
    ones = _op_by_raw(trace, labels["ones"])
    assert ones.func_name == "ones"
    assert tuple(ones.parents) == ()
    assert ones.out is not None
    assert is_literal_producer_op(ones) and is_constant_op(ones)
    assert not is_closure_constant_op(ones)
    consumer = _op_by_raw(trace, labels["mul"])
    assert ones.layer_label in tuple(consumer.parents)
    assert set(constant_op_labels(trace)) == {
        _op_by_raw(trace, labels["w"]).label,
        ones.label,
    }
    # The ordinary ops and the input are not constants.
    assert not is_constant_op(consumer)
    assert not is_constant_op(_op_by_raw(trace, labels["x"]))


def test_validation_helpers_name_the_contracted_reasons_and_stay_narrow() -> None:
    """Rule 3: leaf and perturbation reasons come from the closed list; replayable literals replay."""

    trace, labels = _capture_planted_forward()
    constant = _op_by_raw(trace, labels["w"])
    ones = _op_by_raw(trace, labels["ones"])
    add = _op_by_raw(trace, labels["add"])
    assert CONSTANT_PRODUCER_LEAF_REASON in VALIDATION_REASONS
    assert CONSTANT_PARENT_PAYLOAD_REASON in VALIDATION_REASONS
    assert CONSTANT_UNREALIZABLE_CODE in REFUSAL_CODES
    # A closure constant has no function: a leaf whatever the backend can replay.
    assert constant_leaf_reason(constant, replayable=True) == CONSTANT_PRODUCER_LEAF_REASON
    # A literal producer is replayed when its function is available (a mismatch
    # still fails) and is a leaf only when it is not.
    assert constant_leaf_reason(ones, replayable=True) is None
    assert constant_leaf_reason(ones, replayable=False) == CONSTANT_PRODUCER_LEAF_REASON
    # An ordinary op never gets the carve-out.
    assert constant_leaf_reason(add, replayable=False) is None
    assert constant_parent_skip_reason(constant) == CONSTANT_PARENT_PAYLOAD_REASON
    assert constant_parent_skip_reason(ones) == CONSTANT_PARENT_PAYLOAD_REASON
    assert constant_parent_skip_reason(add) is None


def test_a_lost_parent_edge_is_never_a_constant() -> None:
    """A parentless op with an unattributed tensor argument is a capture gap, not a literal."""

    backend = _NumpyBackend()
    trace = backend.new_trace()
    x = np.ones((2,), dtype=np.float32)
    backend.emit_input(trace, x, "input.arg_0")
    stray = np.ones((2,), dtype=np.float32)  # never labelled: the backend missed it
    out = x + stray
    # Emit with only the stray as argument so the op has no parents, and
    # record the miss the way the capture records do.
    backend.emit_op(trace, "add", (stray,), out, unattributed=("arg_0",))
    backend.finish(trace, out)
    gap = next(op for op in trace._layer_list if op.func_name == "add")
    assert tuple(gap.parents) == ()
    assert not is_literal_producer_op(gap)
    assert not is_constant_op(gap)
    assert constant_leaf_reason(gap, replayable=False) is None


@pytest.mark.parametrize(
    "func_name",
    ["rand", "randn", "random.normal", "random_uniform", "bernoulli", "Tensor.rand", "dropout"],
)
def test_rng_producers_are_not_constants(func_name: str) -> None:
    """A parentless RNG op is random, not a constant; norms and normalize are untouched."""

    assert is_rng_producer_name(func_name)
    backend = _NumpyBackend()
    trace = backend.new_trace()
    x = np.ones((2,), dtype=np.float32)
    backend.emit_input(trace, x, "input.arg_0")
    noise = np.zeros((2,), dtype=np.float32)
    backend.emit_op(trace, func_name, ((2,),), noise)
    out = x + noise
    backend.emit_op(trace, "add", (x, noise), out)
    backend.finish(trace, out)
    rng_op = next(op for op in trace._layer_list if op.func_name == func_name)
    assert tuple(rng_op.parents) == ()
    assert not is_constant_op(rng_op)


def test_norm_like_names_are_not_rng_producers() -> None:
    """Token matching keeps ``layer_norm``, ``normalize`` and ``uniform_like`` apart."""

    assert not is_rng_producer_name("layer_norm")
    assert not is_rng_producer_name("normalize")
    assert not is_rng_producer_name("batch_norm")
    assert not is_rng_producer_name(None)
    assert is_rng_producer_name("rand_like")


def test_unreadable_constant_refuses_typed_before_the_consumer_is_recorded() -> None:
    """Rule 4: ``constant_unrealizable`` names the constant and its consumer; nothing is recorded."""

    ledger = ConstantLedger("fake")

    def _no_buffer(value: Any) -> Any:
        raise RuntimeError("need copyin")

    with pytest.raises(BackendUnsupportedError) as excinfo:
        ledger.register_closure(
            np.ones(3),
            host_copy=_no_buffer,
            label_raw="constant_1_3_raw",
            consumer_label_raw="mul_1_4_raw",
        )
    exc = excinfo.value
    assert exc.fields["code"] == CONSTANT_UNREALIZABLE_CODE
    assert exc.fields["op_label"] == "constant_1_3_raw"
    assert exc.fields["consumer_label"] == "mul_1_4_raw"
    assert "constant_1_3_raw" in str(exc) and "mul_1_4_raw" in str(exc)
    assert "need copyin" in str(exc)
    assert len(ledger) == 0

    with pytest.raises(BackendUnsupportedError) as excinfo:
        ledger.register_closure(np.ones(3), host_copy=lambda value: None)
    assert excinfo.value.fields["code"] == CONSTANT_UNREALIZABLE_CODE


def test_typed_refusals_from_host_copy_pass_through_unchanged() -> None:
    """A backend's own typed refusal inside ``host_copy`` is not re-wrapped."""

    ledger = ConstantLedger("fake")

    def _refuse(value: Any) -> Any:
        raise BackendUnsupportedError("fake refuses", code="rng_state_in_graph")

    with pytest.raises(BackendUnsupportedError) as excinfo:
        ledger.register_closure(np.ones(1), host_copy=_refuse)
    assert excinfo.value.fields["code"] == "rng_state_in_graph"


def test_ledger_records_and_default_labels() -> None:
    """The ledger keeps closure and literal records in order with the label convention."""

    ledger = ConstantLedger("fake")
    record, payload = ledger.register_closure(np.float32(2.0), host_copy=np.array)
    assert record.label_raw == constant_label(1) == "constant_1"
    assert record.kind == "closure"
    assert record.shape == () and payload.shape == ()
    literal = ledger.register_literal("fill_1_2_raw", shape=(2, 2), dtype="float32")
    assert literal.kind == "literal"
    assert ledger.labels() == ("constant_1", "fill_1_2_raw")
    assert ledger.labels(kind="closure") == ("constant_1",)
    assert "constant_1" in ledger and ledger.get("missing") is None
    with pytest.raises(ValueError):
        constant_label(0)


def test_finalize_audit_refuses_a_constant_op_recorded_with_parents() -> None:
    """A ``constant`` op that lists parents is a capture bug the audit refuses."""

    backend = _NumpyBackend()
    trace = backend.new_trace()
    x = np.ones((2,), dtype=np.float32)
    backend.emit_input(trace, x, "input.arg_0")
    bad = np.ones((2,), dtype=np.float32)
    # A "constant" op emitted with a parent edge: wrong by construction.
    backend.emit_op(trace, CONSTANT_FUNC_NAME, (x,), bad)
    out = x + bad
    backend.emit_op(trace, "add", (x, bad), out)
    mark_output_label(trace, backend.labels[id(out)])
    materialize_from_events(trace, trace.capture_events)
    delattr(trace, "capture_events")
    with pytest.raises(CaptureGapError, match="with parents"):
        apply_constants_rule(trace, backend_name="fake", ledger=None)


def test_finalize_audit_refuses_a_ledger_entry_with_no_op() -> None:
    """A constant the backend registered but never emitted is a capture gap."""

    backend = _NumpyBackend()
    trace = backend.new_trace()
    x = np.ones((2,), dtype=np.float32)
    backend.emit_input(trace, x, "input.arg_0")
    out = x * 2.0
    backend.emit_op(trace, "mul", (x, 2.0), out)
    backend.ledger.register_closure(np.ones(2), host_copy=np.array, label_raw="constant_9_9_raw")
    mark_output_label(trace, backend.labels[id(out)])
    materialize_from_events(trace, trace.capture_events)
    delattr(trace, "capture_events")
    with pytest.raises(CaptureGapError, match="constant_9_9_raw"):
        apply_constants_rule(trace, backend_name="fake", ledger=backend.ledger)


def test_finalize_audit_refuses_a_constant_without_payload_evidence() -> None:
    """Rule 4 at finalize: a ``constant`` op with neither payload nor shape refuses typed."""

    backend = _NumpyBackend()
    trace = backend.new_trace()
    x = np.ones((2, 3), dtype=np.float32)
    backend.emit_input(trace, x, "input.arg_0")
    w = np.full((2, 3), 0.5, dtype=np.float32)
    label_w = backend.emit_closure_constant(trace, w, consumer_label_raw="add")
    out = x + w
    backend.emit_op(trace, "add", (x, w), out)
    mark_output_label(trace, backend.labels[id(out)])
    materialize_from_events(trace, trace.capture_events)
    delattr(trace, "capture_events")
    raw_op = trace._raw_graph_ws.raw_layer_dict[label_w]
    # Simulate a backend that recorded the op and then lost its payload and shape.
    raw_op._internal_set("out", None)
    raw_op._internal_set("shape", None)
    with pytest.raises(BackendUnsupportedError) as excinfo:
        apply_constants_rule(trace, backend_name="fake", ledger=None)
    assert excinfo.value.fields["code"] == CONSTANT_UNREALIZABLE_CODE
    assert label_w in str(excinfo.value)
    # The same op passes when the ledger vouches for it (payload filtered by save=).
    found = apply_constants_rule(trace, backend_name="fake", ledger=backend.ledger)
    assert label_w in found


def test_finalize_without_a_ledger_still_runs_the_rule() -> None:
    """Backends that keep no ledger yet get the audit and the derived predicates for free."""

    backend = _NumpyBackend()
    trace = backend.new_trace()
    x = np.ones((2,), dtype=np.float32)
    backend.emit_input(trace, x, "input.arg_0")
    ones = np.ones((2,), dtype=np.float32)
    backend.emit_op(trace, "ones", ((2,),), ones)
    out = x + ones
    backend.emit_op(trace, "add", (x, ones), out)
    mark_output_label(trace, backend.labels[id(out)])
    materialize_from_events(trace, trace.capture_events)
    delattr(trace, "capture_events")
    finalize_single_pass_trace(
        trace,
        backend_name="fake",
        module_tree=None,
        attach_function_root_module=attach_function_root_module,
        attach_object_module_logs=lambda *_a, **_k: None,
    )
    assert len(constant_op_labels(trace)) == 1


class _ClosureModel(nn.Module):
    """Torch module using a plain-attribute tensor (not a parameter or buffer) and ``torch.ones``."""

    def __init__(self) -> None:
        """Hold a closure tensor outside the parameter and buffer registries."""

        super().__init__()
        self.scale = torch.full((3,), 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(x * scale) + ones``."""

        return x * self.scale + torch.ones(3)


def test_torch_already_satisfies_the_rule_structure() -> None:
    """Torch records closure tensors as source ops and ``torch.ones`` as a literal producer."""

    trace = tl.trace(_ClosureModel(), torch.arange(3.0))
    ones_ops = [op for op in trace._layer_list if op.func_name == "ones"]
    assert ones_ops, [op.func_name for op in trace._layer_list]
    for op in ones_ops:
        assert tuple(op.parents) == ()
        assert is_literal_producer_op(op) and is_constant_op(op)
    # Torch records the operator form (``__mul__``) for ``x * scale``.
    mul_ops = [op for op in trace._layer_list if op.func_name in {"mul", "__mul__"}]
    assert mul_ops, [op.func_name for op in trace._layer_list]
    mul = mul_ops[0]
    by_layer_label = {op.layer_label: op for op in trace._layer_list}
    parents = [by_layer_label[label] for label in mul.parents]
    assert len(parents) == 2, mul.parents
    # The closure tensor arrives as a parentless buffer source op (torch's existing
    # buffer and parameter source logging): a payload-bearing leaf, never a lost edge.
    sources = [p for p in parents if not p.is_input]
    assert len(sources) == 1, [(p.label, p.is_input, p.is_buffer) for p in parents]
    assert sources[0].is_buffer is True
    assert tuple(sources[0].parents) == ()
    assert sources[0].out is not None
    assert tuple(getattr(mul, "unattributed_tensor_args", ()) or ()) == ()
