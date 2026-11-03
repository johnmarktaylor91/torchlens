"""Concern-group views of an ``Op`` or ``Layer``: ``op.cost``, ``layer.control_flow``, ...

Each group is a light read-only view bound to one record, the way
``trace.cost`` is bound to a Trace. Stored members read the record's private
storage attribute (``_<group>_<leaf>``, same persisted column as before);
derived members compute on read from the record. Nothing is precomputed at
capture. The member table lives in ``_record_group_table``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .._quantities import Bytes, Duration, Flops, Macs
from .._utils.facade import refuse_removed_member
from . import _doors, _op_cost
from ._layer_views import layer_total_macs_forward

if TYPE_CHECKING:
    from .layer import Layer
    from .op import Op


class _RecordGroup:
    """Base for the read-only concern groups of one ``Op`` or ``Layer``."""

    __slots__ = ("_record",)

    def __init__(self, record: Any) -> None:
        """Bind the group to ``record``."""

        self._record = record

    def __repr__(self) -> str:
        """Name the group and its record."""

        record = self._record
        label = getattr(record, "label", None)
        return f"<{type(self).__name__} of {type(record).__name__} {label!r}>"

    def __dir__(self) -> list[str]:
        """List the group's public members."""

        return sorted(n for n in dir(type(self)) if not n.startswith("_"))


class MovedMemberField:
    """Data descriptor that refuses a former flat member spelling, typed.

    Installed on dict-backed records (``Layer``) under the former name, so
    the refusal wins over a restored ``__dict__`` shadow kept under the
    persisted key.
    """

    __slots__ = ("_owner", "_name", "_remedy")

    def __init__(self, owner: str, name: str, remedy: str) -> None:
        """Bind the refusal to its record class name, member and remedy."""

        self._owner = owner
        self._name = name
        self._remedy = remedy

    def __repr__(self) -> str:
        """Name the refused member."""

        return f"<moved member {self._owner}.{self._name}>"

    def _refuse(self) -> Any:
        """Raise the typed ``facade_redirect`` naming the new spelling."""

        refuse_removed_member(self._owner, self._name, {self._name: self._remedy})
        raise AttributeError(self._name)

    def __get__(self, record: Any, objtype: Any = None) -> Any:
        """Refuse a read on an instance; the class attribute is the descriptor."""

        if record is None:
            return self
        return self._refuse()

    def __set__(self, record: Any, value: Any) -> None:
        """Refuse a write through the former spelling."""

        self._refuse()

    def __delete__(self, record: Any) -> None:
        """Refuse a delete through the former spelling."""

        self._refuse()


class OpCost(_RecordGroup):
    """Compute, memory and timing facts of one Op.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> op = trace.ops["linear_1_1"]
    >>> int(op.cost.flops_forward) > 0
    True
    """

    __slots__ = ()
    _record: Op

    @property
    def activation_memory(self) -> Bytes | None:
        """Bytes of this Op's output activation."""

        return self._record._cost_activation_memory

    @property
    def autograd_memory(self) -> Bytes | None:
        """Bytes of the tensors autograd saved for backward at this Op."""

        return self._record._cost_autograd_memory

    @property
    def bytes_delta_at_call(self) -> Bytes | None:
        """Change in allocated bytes at the time of this Op's function call."""

        return self._record._cost_bytes_delta_at_call

    @property
    def bytes_peak_at_call(self) -> Bytes | None:
        """Peak allocated bytes at the time of this Op's function call."""

        return self._record._cost_bytes_peak_at_call

    @property
    def duration(self) -> Duration | None:
        """Time spent in this Op's function call, as a ``Duration``."""

        return self._record._cost_duration

    @property
    def flops_backward(self) -> Flops | None:
        """Approximate backward FLOPs for this Op."""

        return self._record._cost_flops_backward

    @property
    def flops_forward(self) -> Flops | None:
        """Approximate forward FLOPs for this Op."""

        return self._record._cost_flops_forward

    @property
    def input_memory(self) -> Bytes | None:
        """Sum of saved activation bytes across this Op's parent Ops."""

        return self._record._cost_input_memory

    @property
    def param_memory(self) -> Bytes:
        """Bytes used by the parameters consumed by this Op."""

        return self._record._cost_param_memory

    @property
    def transformed_activation_memory(self) -> Bytes | None:
        """Bytes of this Op's transformed output activation, if one was kept."""

        return self._record._cost_transformed_activation_memory

    @property
    def compute_record(self) -> Any | None:
        """Two-term compute record (``fma_macs`` and ``other_flops``), derived on read."""

        return _op_cost.compute_record(self._record)

    @property
    def macs_forward(self) -> Macs | None:
        """True forward multiply-accumulate count for this Op, never ``flops // 2``."""

        return _op_cost.macs_forward(self._record)

    @property
    def flops_total(self) -> Flops:
        """Approximate total FLOPs for this Op."""

        return _op_cost.flops_total(self._record)

    @property
    def bytes_read(self) -> Bytes | None:
        """Theoretical ideal read-once memory traffic for this operation, in bytes."""

        return _doors.debug()._cost.theoretical_op_bytes(self._record)[0]

    @property
    def bytes_written(self) -> Bytes | None:
        """Theoretical ideal write-once memory traffic for this operation, in bytes."""

        return _doors.debug()._cost.theoretical_op_bytes(self._record)[1]

    @property
    def arithmetic_intensity(self) -> float | None:
        """Theoretical forward FLOPs per byte of ideal memory traffic, or ``None``."""

        return _op_cost.arithmetic_intensity(self._record)


class LayerCost(_RecordGroup):
    """Compute, memory and timing facts of one Layer.

    Single-value members describe the representative (first) pass; the
    ``total_*`` members sum over every pass of the layer.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> layer = trace.layers["linear_1"]
    >>> int(layer.cost.total_flops_forward) > 0
    True
    """

    __slots__ = ()
    _record: Layer

    @property
    def activation_memory(self) -> Bytes | None:
        """Bytes of the representative pass's output activation."""

        return self._record._cost_activation_memory

    @property
    def autograd_memory(self) -> Bytes | None:
        """Bytes autograd saved for backward at the representative pass."""

        return self._record._cost_autograd_memory

    @property
    def flops_backward(self) -> Flops | None:
        """Backward FLOPs of this layer's representative (first) pass."""

        return self._record._cost_flops_backward

    @property
    def flops_forward(self) -> Flops | None:
        """Forward FLOPs of this layer's representative (first) pass."""

        return self._record._cost_flops_forward

    @property
    def total_autograd_memory(self) -> Bytes | None:
        """Bytes autograd saved for backward, as carried by this layer."""

        return self._record._cost_total_autograd_memory

    @property
    def total_param_memory(self) -> Bytes:
        """Bytes used by this layer's parameters."""

        return self._record._cost_total_param_memory

    @property
    def transformed_activation_memory(self) -> Bytes | None:
        """Bytes of the representative pass's transformed output activation."""

        return self._record._cost_transformed_activation_memory

    @property
    def duration(self) -> Duration:
        """Execution time of this layer's function call, for single-pass layers."""

        return cast(Duration, self._record._single_pass_or_error("func_duration"))

    @property
    def macs_forward(self) -> Macs | None:
        """True forward MAC count of the representative pass, or ``None`` when unknown."""

        first_op = next(iter(self._record.ops.values()), None)
        return None if first_op is None else first_op.cost.macs_forward

    @property
    def flops_total(self) -> Flops:
        """Representative total FLOPs for this layer."""

        return Flops((self.flops_forward or 0) + (self.flops_backward or 0))

    @property
    def total_activation_memory(self) -> Bytes:
        """Sum of activation memory in bytes across all ops of this layer."""

        ops = self._record.ops.values()
        return Bytes(sum(int(op.cost.activation_memory or 0) for op in ops))

    @property
    def total_flops_forward(self) -> Flops:
        """Sum of forward FLOPs across all ops of this layer."""

        return Flops(sum((op.cost.flops_forward or 0) for op in self._record.ops.values()))

    @property
    def total_flops_backward(self) -> Flops:
        """Sum of backward FLOPs across all ops of this layer."""

        return Flops(sum((op.cost.flops_backward or 0) for op in self._record.ops.values()))

    @property
    def total_flops(self) -> Flops:
        """Sum of forward and backward FLOPs across all ops of this layer."""

        return Flops(self.total_flops_forward + self.total_flops_backward)

    @property
    def total_func_duration(self) -> Duration:
        """Sum of function-call durations across all ops of this layer."""

        ops = self._record.ops.values()
        return Duration(sum(float(op.cost.duration or 0) for op in ops))

    @property
    def total_macs_forward(self) -> Macs | None:
        """Sum of true forward MACs across all ops, or ``None`` when any split is unknown."""

        return layer_total_macs_forward(self._record)

    @property
    def total_macs(self) -> Macs | None:
        """Always ``None``, because total MACs are not derivable."""

        return None


class OpControlFlow(_RecordGroup):
    """Conditional and branch facts of one Op.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> trace.ops["linear_1_1"].control_flow.is_in_conditional
    False
    """

    __slots__ = ()
    _record: Op

    @property
    def arm_children(self) -> Any:
        """Each conditional's arm kinds (then, elif_N, else) mapped to this Op's children inside each arm."""

        return self._record._control_flow_arm_children

    @property
    def bool_value(self) -> bool | None:
        """The Python value of this Op's output when it is a scalar bool."""

        return self._record._control_flow_bool_value

    @property
    def branch_depth(self) -> int:
        """Number of conditional branches this Op is nested inside; 0 outside a conditional body."""

        return self._record._control_flow_branch_depth

    @property
    def branch_stack(self) -> Any:
        """Stack of the conditional branches this Op runs under, outermost first."""

        return self._record._control_flow_branch_stack

    @property
    def context_kind(self) -> str | None:
        """Kind of context (``"if"`` or ``"while"``) of the conditional this terminal bool decides."""

        return self._record._control_flow_context_kind

    @property
    def elif_children(self) -> Any:
        """Each elif arm index mapped to the labels of this Op's children in that arm."""

        return self._record._control_flow_elif_children

    @property
    def else_children(self) -> Any:
        """Labels of this Op's children that sit in an else arm of a conditional."""

        return self._record._control_flow_else_children

    @property
    def entry_children(self) -> Any:
        """Labels of this Op's children whose edge from it enters a conditional arm."""

        return self._record._control_flow_entry_children

    @property
    def in_conditionals(self) -> Any:
        """The conditional roles (evaluation or body) this Op takes part in."""

        return self._record._control_flow_in_conditionals

    @property
    def is_in_conditional_body(self) -> bool:
        """Whether this Op sits inside the body of a conditional arm."""

        return self._record._control_flow_is_in_conditional_body

    @property
    def is_scalar_bool(self) -> bool:
        """Whether this Op's output is a scalar bool."""

        return self._record._control_flow_is_scalar_bool

    @property
    def is_terminal_bool(self) -> bool:
        """Whether this Op is a scalar bool that ends a chain of operations and decides control flow."""

        return self._record._control_flow_is_terminal_bool

    @property
    def is_terminal_conditional_bool(self) -> bool:
        """Whether this terminal bool is the test of a recorded conditional."""

        return self._record._control_flow_is_terminal_conditional_bool

    @property
    def terminal_bool_for(self) -> Any:
        """``(conditional_id, arm_index)`` when this Op is the terminal bool of a conditional arm."""

        return self._record._control_flow_terminal_bool_for

    @property
    def then_children(self) -> Any:
        """Labels of this Op's children that sit in a then arm of a conditional."""

        return self._record._control_flow_then_children

    @property
    def wrapper_kind(self) -> str | None:
        """Wrapper kind of the conditional this terminal bool decides, or ``None``."""

        return self._record._control_flow_wrapper_kind

    @property
    def depth(self) -> int:
        """Number of distinct conditionals this Op takes part in."""

        return len({role.conditional_id for role in self.in_conditionals or []})

    @property
    def is_in_conditional(self) -> bool:
        """Whether this Op takes part in any conditional role."""

        return bool(self.in_conditionals)

    @property
    def is_in_conditional_evaluation(self) -> bool:
        """Whether this Op computes the condition of a conditional arm."""

        return any(role.role == "evaluation" for role in self.in_conditionals or [])


class LayerControlFlow(_RecordGroup):
    """Conditional and branch facts of one Layer.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> trace.layers["linear_1"].control_flow.depth
    0
    """

    __slots__ = ()
    _record: Layer

    @property
    def arm_children(self) -> Any:
        """Each conditional's arm kinds mapped to this layer's child ops inside each arm."""

        return self._record._control_flow_arm_children

    @property
    def bool_value(self) -> bool | None:
        """The Python ``bool`` value of a scalar boolean layer, or ``None``."""

        return self._record._control_flow_bool_value

    @property
    def branch_stack_ops(self) -> Any:
        """Each distinct conditional branch stack mapped to the pass numbers that ran under it."""

        return self._record._control_flow_branch_stack_ops

    @property
    def elif_children(self) -> Any:
        """Each elif arm index mapped to the labels of this layer's children in that arm."""

        return self._record._control_flow_elif_children

    @property
    def else_children(self) -> Any:
        """Labels of this layer's child ops that sit in an else arm of a conditional."""

        return self._record._control_flow_else_children

    @property
    def entry_children(self) -> Any:
        """Labels of this layer's child ops whose edge from it enters a conditional arm."""

        return self._record._control_flow_entry_children

    @property
    def in_conditionals(self) -> Any:
        """The conditional roles (evaluation or body) this layer takes part in."""

        return self._record._control_flow_in_conditionals

    @property
    def is_in_conditional_body(self) -> bool:
        """Whether this layer sits in the body of a conditional arm."""

        return self._record._control_flow_is_in_conditional_body

    @property
    def is_scalar_bool(self) -> bool:
        """Whether this layer's value is a scalar boolean."""

        return self._record._control_flow_is_scalar_bool

    @property
    def is_terminal_bool(self) -> bool:
        """Whether this layer is a scalar boolean that ends inside the model, such as a branch condition."""

        return self._record._control_flow_is_terminal_bool

    @property
    def role_stacks(self) -> Any:
        """The distinct conditional branch stacks this layer's passes ran under."""

        return self._record._control_flow_role_stacks

    @property
    def terminal_bool_for(self) -> Any:
        """The conditional this terminal boolean decides, or ``None``."""

        return self._record._control_flow_terminal_bool_for

    @property
    def then_children(self) -> Any:
        """Labels of this layer's child ops that sit in a then arm of a conditional."""

        return self._record._control_flow_then_children

    @property
    def depth(self) -> int:
        """Number of distinct conditionals this layer takes part in."""

        return len({role.conditional_id for role in self.in_conditionals or []})

    @property
    def is_in_conditional(self) -> bool:
        """Whether this layer takes part in any conditional role."""

        return bool(self.in_conditionals)

    @property
    def is_in_conditional_evaluation(self) -> bool:
        """Whether this layer computes the condition of a conditional arm."""

        return any(role.role == "evaluation" for role in self.in_conditionals or [])


#: Group property name -> group class, per record class.
OP_GROUP_CLASSES: dict[str, type[_RecordGroup]] = {"cost": OpCost, "control_flow": OpControlFlow}
LAYER_GROUP_CLASSES: dict[str, type[_RecordGroup]] = {
    "cost": LayerCost,
    "control_flow": LayerControlFlow,
}


__all__ = [
    "LAYER_GROUP_CLASSES",
    "OP_GROUP_CLASSES",
    "LayerControlFlow",
    "LayerCost",
    "MovedMemberField",
    "OpControlFlow",
    "OpCost",
]
