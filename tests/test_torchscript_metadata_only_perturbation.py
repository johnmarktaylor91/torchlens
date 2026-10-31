"""Perturbation at a TorchScript boundary op whose scripted body reads an argument's metadata only.

A scripted helper that reads only ``x.size(-1)`` (the DeBERTa ``scaled_size_sqrt`` shape)
returns the same value whatever ``x`` holds, so the boundary op is insensitive to that
parent by construction. The validator accepts that edge only through a graph proof that
every use of the argument is a metadata read, and records the proof as its own exemption
reason; a size read inside a nested scripted helper counts (the graph is inlined). Armed
negatives: a helper that reads one element, a helper mixing a size read and an element read,
an element read inside a nested scripted helper, an element read on a branch the call does
not take, and an argument reaching a node kind outside the enumerated list all keep the
perturbation check and still fail it; a parent that fills both a metadata slot and a value
slot gets no proof; plain eager capture is unaffected.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._script_boundary import SCRIPT_FUNCTION_FUNC_NAME
from torchlens.utils._torch_compat import HAS_SCRIPT_FUNCTION_CALL_HOOK
from torchlens.validation import last_validation_failure
from torchlens.validation._script_metadata_proof import (
    METADATA_READ_NODE_KINDS,
    SCRIPT_METADATA_ONLY_REASON,
    metadata_only_arguments,
    script_metadata_only_proof,
)
from torchlens.validation.diagnostics import CHECK_PERTURBATION

pytestmark = pytest.mark.skipif(
    not HAS_SCRIPT_FUNCTION_CALL_HOOK, reason="torch build without a ScriptFunction call hook"
)


def _plain_scaled_size_sqrt(query_layer: torch.Tensor, scale_factor: int) -> torch.Tensor:
    return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)


scaled_size_sqrt = torch.jit.script(_plain_scaled_size_sqrt)


@torch.jit.script
def metadata_reads(x: torch.Tensor) -> torch.Tensor:
    """Every enumerated metadata read of one argument, none of its elements."""

    rank = x.dim() + x.numel() * 0
    return torch.full([x.size(-1)], float(rank), dtype=x.dtype, device=x.device)


@torch.jit.script
def one_element_read(x: torch.Tensor) -> torch.Tensor:
    """Reads one element; finite perturbations happen not to move the output."""

    first = x.view(-1)[0]
    return (first != first).to(torch.float) + 8.0


@torch.jit.script
def size_and_element_read(x: torch.Tensor) -> torch.Tensor:
    """A size read plus an element read whose effect finite perturbations do not show."""

    first = x.view(-1)[0]
    size_term = torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float))
    return size_term + (first != first).to(torch.float)


@torch.jit.script
def nested_size_call(x: torch.Tensor) -> torch.Tensor:
    """Hands the argument to a scripted helper that reads only its size (proved inlined)."""

    return scaled_size_sqrt(x, 2)


@torch.jit.script
def nested_element_call(x: torch.Tensor) -> torch.Tensor:
    """Hands the argument to a scripted helper that reads one element."""

    return one_element_read(x)


@torch.jit.script
def untaken_branch_element_read(x: torch.Tensor) -> torch.Tensor:
    """Reads an element only on a branch that a 2-D argument never takes."""

    if x.dim() > 2:
        return x.view(-1)[0] * 0.0 + 8.0
    return torch.tensor(8.0)


@torch.jit.script
def is_cuda_read(x: torch.Tensor) -> torch.Tensor:
    """Reads ``x.is_cuda`` (``prim::is_cuda``), a metadata read deliberately not enumerated."""

    return torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float)) + float(x.is_cuda)


@torch.jit.script
def scale_by_last_dim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Argument ``x`` is read for its size, ``y`` for its values."""

    return y * x.size(-1)


class _HelperModel(nn.Module):
    """Linear, then ``h / helper(h)``: the helper's only parent is ``h``."""

    def __init__(self, helper) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.helper = helper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        return h / self.helper(h)


class _ScaledModel(_HelperModel):
    """The DeBERTa shape: divide by ``sqrt(size(-1) * scale)``."""

    def __init__(self, helper=None) -> None:
        super().__init__(scaled_size_sqrt if helper is None else helper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        return h / self.helper(h, 2)


class _TwoArgModel(nn.Module):
    """``scale_by_last_dim(a, b)``; ``same`` passes one tensor to both slots."""

    def __init__(self, same: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.same = same

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        g = h if self.same else torch.tanh(h)
        return scale_by_last_dim(h, g)


def _trace_and_validate(model: nn.Module, x: torch.Tensor) -> tuple[tl.Trace, bool]:
    """Capture with every activation saved, then run forward validation on the trace."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            model, x, capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True)
        )
        result = trace.validate_forward_pass([model(x)], validate_metadata=False)
    return trace, result


def _boundary_op(trace: tl.Trace):
    """Return the trace's one ``jitscript`` op."""

    (op,) = [op for op in trace.layer_list if op.func_name == SCRIPT_FUNCTION_FUNC_NAME]
    return op


def _metadata_decisions(trace: tl.Trace) -> list[dict]:
    """Return the recorded metadata-only exemption decisions."""

    return [
        item
        for item in trace.validation_replay_status.decisions
        if item["reason"] == SCRIPT_METADATA_ONLY_REASON
    ]


def _assert_boundary_perturbation_failure(trace: tl.Trace, result: bool) -> None:
    """Assert validation failed perturbation at the boundary op with no metadata proof."""

    assert result is False
    failure = last_validation_failure()
    assert failure is not None and failure.check == CHECK_PERTURBATION
    assert failure.func_name == SCRIPT_FUNCTION_FUNC_NAME
    assert _metadata_decisions(trace) == []


def test_shape_only_helper_passes_with_a_recorded_proof() -> None:
    """The DeBERTa shape validates, and the exemption is recorded with its proof."""

    trace, result = _trace_and_validate(_ScaledModel(), torch.randn(3, 4))

    assert result is True
    (decision,) = _metadata_decisions(trace)
    assert decision["decision"] == "exempted"
    assert decision["phase"] == "perturbation"
    assert decision["op_label"] == _boundary_op(trace).label
    assert scaled_size_sqrt.qualified_name in decision["justification"]
    assert "query_layer only through aten::size" in decision["justification"]
    assert trace.validation_replay_status.exempted_reason_counts[SCRIPT_METADATA_ONLY_REASON] == 1


def test_graph_proof_names_exactly_the_metadata_only_arguments() -> None:
    """The proof covers every enumerated read and nothing that touches elements."""

    assert metadata_only_arguments(scaled_size_sqrt) == {"query_layer": frozenset({"aten::size"})}
    assert metadata_only_arguments(metadata_reads) == {"x": METADATA_READ_NODE_KINDS}
    assert metadata_only_arguments(scale_by_last_dim) == {"x": frozenset({"aten::size"})}
    assert metadata_only_arguments(nested_size_call) == {"x": frozenset({"aten::size"})}
    for function in (
        one_element_read,
        size_and_element_read,
        nested_element_call,
        untaken_branch_element_read,
        is_cuda_read,
    ):
        assert metadata_only_arguments(function) == {}


def test_every_enumerated_metadata_read_passes() -> None:
    """size, dim, numel, dtype and device reads together still give the proof."""

    trace, result = _trace_and_validate(_HelperModel(metadata_reads), torch.randn(3, 4))

    assert result is True
    assert len(_metadata_decisions(trace)) == 1


def test_one_element_read_still_fails_perturbation() -> None:
    """Armed negative: an element read gets no proof, so the insensitive edge fails."""

    trace, result = _trace_and_validate(_HelperModel(one_element_read), torch.randn(3, 4))
    _assert_boundary_perturbation_failure(trace, result)


def test_size_read_mixed_with_an_element_read_still_fails_perturbation() -> None:
    """Armed negative: one value use anywhere voids the proof for that argument."""

    trace, result = _trace_and_validate(_HelperModel(size_and_element_read), torch.randn(3, 4))
    _assert_boundary_perturbation_failure(trace, result)


def test_unknown_node_kind_keeps_the_perturbation_check() -> None:
    """Armed negative: ``x.is_cuda`` is metadata too, but ``prim::is_cuda`` is not enumerated."""

    trace, result = _trace_and_validate(_HelperModel(is_cuda_read), torch.randn(3, 4))
    _assert_boundary_perturbation_failure(trace, result)


def test_size_read_inside_a_nested_scripted_helper_passes() -> None:
    """The nested helper is inlined, so its size-only read gives the proof."""

    trace, result = _trace_and_validate(_HelperModel(nested_size_call), torch.randn(3, 4))

    assert result is True
    (decision,) = _metadata_decisions(trace)
    assert nested_size_call.qualified_name in decision["justification"]


def test_element_read_inside_a_nested_scripted_helper_still_fails_perturbation() -> None:
    """Armed negative: inlining exposes the nested helper's element read."""

    trace, result = _trace_and_validate(_HelperModel(nested_element_call), torch.randn(3, 4))
    _assert_boundary_perturbation_failure(trace, result)


def test_element_read_on_an_untaken_branch_still_fails_perturbation() -> None:
    """Armed negative: the proof is not path-sensitive; any value use voids it."""

    trace, result = _trace_and_validate(
        _HelperModel(untaken_branch_element_read), torch.randn(3, 4)
    )
    _assert_boundary_perturbation_failure(trace, result)


def test_parent_in_a_metadata_slot_and_a_value_slot_gets_no_proof() -> None:
    """A parent that also feeds a value-bearing argument stays perturbation-tested."""

    trace, result = _trace_and_validate(_TwoArgModel(same=True), torch.randn(3, 4))
    assert result is True
    op = _boundary_op(trace)
    slots = op.parent_arg_positions["args"]
    parent = slots[0]
    assert slots[1] == parent
    assert script_metadata_only_proof(op, [parent]) is None
    assert _metadata_decisions(trace) == []


def test_distinct_parents_split_into_proven_and_tested() -> None:
    """With two parents only the size-read one is exempt; the value parent is tested."""

    trace, result = _trace_and_validate(_TwoArgModel(same=False), torch.randn(3, 4))
    assert result is True
    op = _boundary_op(trace)
    size_parent, value_parent = (op.parent_arg_positions["args"][pos] for pos in (0, 1))
    assert script_metadata_only_proof(op, [size_parent]) is not None
    assert script_metadata_only_proof(op, [value_parent]) is None
    (decision,) = _metadata_decisions(trace)
    assert "reads argument(s) x only" in decision["justification"]


def test_unreadable_graph_gives_no_proof() -> None:
    """An object without a readable graph or schema proves nothing."""

    class _NoGraph:
        @property
        def graph(self):
            raise RuntimeError("graph unavailable")

    assert metadata_only_arguments(_NoGraph()) == {}
    assert metadata_only_arguments(object()) == {}


def test_plain_eager_capture_is_unaffected() -> None:
    """The same helper as plain Python: op by op, no boundary, no metadata-only exemption."""

    model = _ScaledModel(helper=_plain_scaled_size_sqrt)
    trace, result = _trace_and_validate(model, torch.randn(3, 4))

    assert result is True
    assert not any(op.func_name == SCRIPT_FUNCTION_FUNC_NAME for op in trace.layer_list)
    assert _metadata_decisions(trace) == []
    for op in trace.layer_list:
        for parent in op.parents:
            assert script_metadata_only_proof(op, [parent]) is None


# Adversarial attacks on the graph proof (review of AGT-2470). Each scripted helper below
# lets the argument's values reach the output, or reaches a node the proof must not trust,
# so the proof must name no argument; the end-to-end attacks check the call-slot mapping.


@torch.jit.script
def view_alias_read(x: torch.Tensor) -> torch.Tensor:
    """``aten::view`` aliases the argument before an element read."""

    return x.view(-1)[0] * 0.0 + float(x.size(-1))


@torch.jit.script
def list_packed_read(x: torch.Tensor) -> torch.Tensor:
    """``prim::ListConstruct`` packs the argument; the list's value is summed."""

    return torch.stack([x, x]).sum() * 0.0 + float(x.size(-1))


@torch.jit.script
def tuple_packed_read(x: torch.Tensor) -> torch.Tensor:
    """``prim::TupleConstruct`` packs the argument; the unpacked member is read for values."""

    packed = (x, x.size(-1))
    return packed[0].sum() * 0.0 + float(packed[1])


@torch.jit.script
def size_fed_back_into_value_op(x: torch.Tensor) -> torch.Tensor:
    """The size read flows into a value op on the same argument."""

    return x.reshape(x.size(0), -1)[0, 0] * 0.0 + float(x.size(-1))


@torch.jit.script
def in_place_then_size(x: torch.Tensor) -> torch.Tensor:
    """An in-place op on the argument, then only a size read."""

    x.add_(0.0)
    return torch.tensor(float(x.size(-1)))


@torch.jit.script
def returned_from_if_block(x: torch.Tensor, flag: bool) -> torch.Tensor:
    """The argument is a ``prim::If`` block output on one branch."""

    if flag:
        return x
    return torch.zeros(x.size())


@torch.jit.script
def loop_carried(x: torch.Tensor) -> torch.Tensor:
    """The argument enters a ``prim::Loop`` as a carried value."""

    y = x
    for _ in range(2):
        y = y * 1.0
    return y


@torch.jit.script
def passthrough(x: torch.Tensor) -> torch.Tensor:
    """The argument is returned as-is (a graph return use)."""

    return x


def _python_peek(x):
    return float(x.sum())


# Real annotation objects: TorchScript cannot resolve this module's postponed (string)
# annotations on an ignored Python function.
_python_peek.__annotations__ = {"x": torch.Tensor, "return": float}
_python_peek = torch.jit.ignore(_python_peek)


@torch.jit.script
def ignored_python_read(x: torch.Tensor) -> torch.Tensor:
    """A ``@torch.jit.ignore`` call (``prim::PythonOp``) receives the argument."""

    return torch.tensor(_python_peek(x) * 0.0 + float(x.size(-1)))


@torch.jit.script
def forked_element_read(x: torch.Tensor) -> torch.Tensor:
    """``prim::fork`` hands the argument to an element-reading helper."""

    future = torch.jit.fork(one_element_read, x)
    return torch.jit.wait(future)


@torch.jit.script
def new_zeros_template(x: torch.Tensor) -> torch.Tensor:
    """``new_zeros`` reads dtype and device only, but it is not an enumerated read."""

    return x.new_zeros(x.size())


def _traced_double(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0


traced_double = torch.jit.trace(_traced_double, torch.randn(2, 4))


@pytest.mark.parametrize(
    "function",
    [
        view_alias_read,
        list_packed_read,
        tuple_packed_read,
        size_fed_back_into_value_op,
        in_place_then_size,
        returned_from_if_block,
        loop_carried,
        passthrough,
        ignored_python_read,
        forked_element_read,
        new_zeros_template,
        traced_double,
    ],
)
def test_attack_value_or_untrusted_use_gives_no_proof(function) -> None:
    """Aliasing, packing, in-place, block, loop, return, Python, fork and trace uses."""

    assert metadata_only_arguments(function) == {}


def test_attack_proof_is_stable_across_executor_warmup() -> None:
    """The proof reads the inlined program, not the executor's profiled graph."""

    before = metadata_only_arguments(scale_by_last_dim)
    graph_before = str(scale_by_last_dim.inlined_graph)
    for _ in range(4):
        scale_by_last_dim(torch.randn(3, 4), torch.randn(3, 4))
    assert metadata_only_arguments(scale_by_last_dim) == before
    assert str(scale_by_last_dim.inlined_graph) == graph_before


@torch.jit.script
def size_then_dict_value(x: torch.Tensor, d: dict[str, torch.Tensor]) -> torch.Tensor:
    """``x`` is read for its size; the dict member is read for values."""

    return d["a"] * x.size(-1)


@torch.jit.script
def size_then_list_value(x: torch.Tensor, xs: list[torch.Tensor]) -> torch.Tensor:
    """``x`` is read for its size; the list member is read for values."""

    return xs[0] * x.size(-1)


@torch.jit.script
def size_then_tuple_value(x: torch.Tensor, pair: tuple[torch.Tensor, int]) -> torch.Tensor:
    """``x`` is read for its size; the tuple member is read for values."""

    return pair[0] * x.size(-1)


@torch.jit.script
def size_then_optional_value(x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
    """``x`` is read for its size; the optional argument is read for values."""

    if y is None:
        return torch.zeros(x.size())
    return y * x.size(-1)


@torch.jit.script
def value_then_size(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Positions reversed: argument 0 is read for values, argument 1 for its size."""

    return y * x.size(-1)


class _CallModel(nn.Module):
    """Linear, then one scripted call built by ``call(h, g)``; ``g = tanh(h)``."""

    def __init__(self, call) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.call = call

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        g = torch.tanh(h)
        return self.call(h, g)


def _parent_label(op, func_name: str) -> str:
    """Return the label of ``op``'s parent produced by ``func_name`` (labels lead with it)."""

    (label,) = [p for p in op.parents if p.split("_")[0] == func_name]
    return label


@pytest.mark.parametrize(
    "call",
    [
        lambda h, g: size_then_dict_value(h, {"a": h}),
        lambda h, g: size_then_list_value(h, [h]),
        lambda h, g: size_then_tuple_value(h, (h, 1)),
        lambda h, g: size_then_optional_value(h, h),
    ],
    ids=["dict", "list", "tuple", "optional"],
)
def test_attack_parent_also_inside_a_container_slot_gets_no_proof(call) -> None:
    """A parent in a size slot AND inside a value-bearing container slot is never proven."""

    trace, result = _trace_and_validate(_CallModel(call), torch.randn(3, 4))
    op = _boundary_op(trace)
    h_label = _parent_label(op, "linear")
    assert script_metadata_only_proof(op, [h_label]) is None
    assert _metadata_decisions(trace) == []
    assert result is True


@pytest.mark.parametrize(
    ("call", "size_func"),
    [
        (lambda h, g: value_then_size(g, h), "linear"),
        (lambda h, g: value_then_size(h, g), "tanh"),
        (lambda h, g: scale_by_last_dim(y=g, x=h), "linear"),
        (lambda h, g: scale_by_last_dim(h, y=g), "linear"),
        (lambda h, g: value_then_size(x=g, y=h), "tanh"),
    ],
    ids=["positional-1", "positional-1-swapped", "keywords", "mixed", "keywords-swapped"],
)
def test_attack_slot_mapping_follows_the_schema(call, size_func: str) -> None:
    """Only the parent bound to the size-read argument is proven, by position or keyword."""

    trace, result = _trace_and_validate(_CallModel(call), torch.randn(3, 4))
    op = _boundary_op(trace)
    value_func = "tanh" if size_func == "linear" else "linear"
    assert script_metadata_only_proof(op, [_parent_label(op, size_func)]) is not None
    assert script_metadata_only_proof(op, [_parent_label(op, value_func)]) is None
    assert len(_metadata_decisions(trace)) == 1
    assert result is True


def test_attack_two_perturbed_parents_get_no_proof() -> None:
    """The proof applies to one perturbed parent at a time only."""

    trace, _ = _trace_and_validate(_TwoArgModel(same=False), torch.randn(3, 4))
    op = _boundary_op(trace)
    assert script_metadata_only_proof(op, list(op.parents)) is None


def test_attack_in_place_call_is_not_collapsed_into_a_boundary_op() -> None:
    """A scripted call that writes its argument in place never becomes a ``jitscript`` op."""

    trace, _ = _trace_and_validate(_HelperModel(in_place_then_size), torch.randn(3, 4))
    assert not any(op.func_name == SCRIPT_FUNCTION_FUNC_NAME for op in trace.layer_list)
    assert _metadata_decisions(trace) == []


@torch.jit.script
def value_times_size(x: torch.Tensor) -> torch.Tensor:
    """The argument is read for its values and its size."""

    return x * x.size(-1)


def test_attack_value_dependent_call_validates_without_the_exemption() -> None:
    """A boundary call whose output depends on the argument's values is perturbation-tested."""

    trace, result = _trace_and_validate(_HelperModel(value_times_size), torch.randn(3, 4))
    assert result is True
    assert any(op.func_name == SCRIPT_FUNCTION_FUNC_NAME for op in trace.layer_list)
    assert _metadata_decisions(trace) == []


def test_attack_aliased_element_read_still_fails_perturbation() -> None:
    """Armed negative: an element read through a view alias keeps the check, which fails."""

    trace, result = _trace_and_validate(_HelperModel(view_alias_read), torch.randn(3, 4))
    _assert_boundary_perturbation_failure(trace, result)


@torch.jit.script
def size_then_nested_list_value(x: torch.Tensor, xss: list[list[torch.Tensor]]) -> torch.Tensor:
    """``x`` is read for its size; a member two container levels down is read for values."""

    return xss[0][0] * x.size(-1)


@torch.jit.script
def size_then_dict_of_lists_value(
    x: torch.Tensor, d: dict[str, list[torch.Tensor]]
) -> torch.Tensor:
    """``x`` is read for its size; a list member inside a dict is read for values."""

    return d["a"][0] * x.size(-1)


@torch.jit.script
def size_with_shallow_list(x: torch.Tensor, xs: list[torch.Tensor]) -> torch.Tensor:
    """``x`` is read for its size; a one-level list (fully slot-mapped) is read for values."""

    return xs[0] * x.size(-1)


@pytest.mark.parametrize(
    "call",
    [
        lambda h, g: size_then_nested_list_value(h, [[h]]),
        lambda h, g: size_then_dict_of_lists_value(h, {"a": [h]}),
    ],
    ids=["list-of-lists", "dict-of-lists"],
)
def test_attack_parent_hidden_below_the_slot_map_gets_no_proof(call) -> None:
    """A parent nested deeper than capture's slot map records is never proven size-only.

    Perturbing the recorded size slot alone cannot reach the unrecorded nested copy, so the
    check fails there: the slot-map gap stays visible instead of being excused.
    """

    trace, result = _trace_and_validate(_CallModel(call), torch.randn(3, 4))
    op = _boundary_op(trace)
    h_label = _parent_label(op, "linear")
    assert script_metadata_only_proof(op, [h_label]) is None
    _assert_boundary_perturbation_failure(trace, result)


def test_attack_shallow_container_keeps_the_proof_for_the_size_argument() -> None:
    """A one-level container argument is fully slot-mapped, so the size argument stays provable."""

    trace, result = _trace_and_validate(
        _CallModel(lambda h, g: size_with_shallow_list(h, [g])), torch.randn(3, 4)
    )
    op = _boundary_op(trace)
    assert script_metadata_only_proof(op, [_parent_label(op, "linear")]) is not None
    assert script_metadata_only_proof(op, [_parent_label(op, "tanh")]) is None
    assert result is True
