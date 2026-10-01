"""Round-28/31 capture-attribution regression pins.

Locks the fixes for the round-31 capture-reseal findings (Sol) and the
validation false-negative hunt (Fable): H2 runtime tensor control-arg parents,
H1/M3 foreach attribution, M4 element-overlap alias propagation, M5 in-place
grad_fn metadata, M6 tensor property setters, and the identity-witness
validation strengthening. Each test reproduces the original defect shape and
pins the corrected capture topology plus the tripwire direction.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.utils._torch_compat import ROLL_TENSOR_SHIFTS_SUPPORTED


def _ops_by_func(trace: object, func_name: str) -> list:
    """Return all ops in ``trace`` whose ``func_name`` matches.

    Parameters
    ----------
    trace:
        Trace to search.
    func_name:
        Exact ``func_name`` to match.

    Returns
    -------
    list
        Matching ops in graph order.
    """

    return [op for op in trace.ops if op.func_name == func_name]


# ---------------------------------------------------------------------------
# H2 -- runtime tensor scalar/control args are real data parents
# ---------------------------------------------------------------------------


class _RollTensorShift(nn.Module):
    """Roll by a runtime tensor shift derived from the input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Roll ``x`` by the number of positive entries.

        Parameters
        ----------
        x:
            1-D input tensor.

        Returns
        -------
        torch.Tensor
            ``x`` rolled by a data-dependent shift.
        """

        shift = (x > 0).sum()
        return torch.roll(x, shift)


class _SoftmaxTensorDim(nn.Module):
    """Softmax over a runtime tensor dim derived from the input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a reshaped view along a data-dependent dim.

        Parameters
        ----------
        x:
            Input tensor with four elements.

        Returns
        -------
        torch.Tensor
            Softmax of the reshaped input along the derived dim.
        """

        dim = (x[0, 0] > 100).long().sum()
        y = x.reshape(2, 2)
        return torch.softmax(y, dim)


class _ArangeTensorEnd(nn.Module):
    """Arange whose endpoint is a runtime tensor scalar."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Build a range sized by the input sum.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ``arange`` output tied back into the graph.
        """

        n = x.sum().long()
        return torch.arange(n) + x.sum() * 0


@pytest.mark.skipif(
    not ROLL_TENSOR_SHIFTS_SUPPORTED,
    reason="torch.roll rejects a bare 0-dim tensor shifts on this torch (capability probe)",
)
def test_h2_roll_tensor_shift_is_parent() -> None:
    """``roll(x, shifts=tensor)`` records the shift producer as a data parent."""

    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    trace = tl.trace(_RollTensorShift().eval(), x)
    roll_op = _ops_by_func(trace, "roll")[0]
    sum_label = _ops_by_func(trace, "sum")[0].layer_label

    assert sum_label in roll_op.parents
    assert roll_op.parent_arg_positions["args"].get(1) == sum_label
    assert roll_op.unattributed_tensor_args == ()
    assert tl.validate(_RollTensorShift().eval(), x, scope="forward")


def test_h2_softmax_tensor_dim_is_parent_and_validates() -> None:
    """``softmax(y, dim=tensor)`` records the dim producer and still validates.

    The dim perturbation initially draws an out-of-range value; the
    step-retry must recover an adjacent valid dim so a CORRECT capture
    validates instead of reporting ``perturbation_execution_exception``.
    """

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 4)
    trace = tl.trace(_SoftmaxTensorDim().eval(), x)
    softmax_op = _ops_by_func(trace, "softmax")[0]
    sum_label = _ops_by_func(trace, "sum")[0].layer_label

    assert sum_label in softmax_op.parents
    assert softmax_op.parent_arg_positions["args"].get(1) == sum_label
    assert tl.validate(_SoftmaxTensorDim().eval(), x, scope="forward")


def test_h2_arange_tensor_end_is_parent_not_internal_source() -> None:
    """``arange(t)`` is parented to the endpoint producer, not a raw source."""

    x = torch.tensor([2.0, 4.0])
    trace = tl.trace(_ArangeTensorEnd().eval(), x)
    arange_op = _ops_by_func(trace, "arange")[0]
    long_label = _ops_by_func(trace, "long")[0].layer_label

    assert arange_op.parents == (long_label,)
    assert arange_op.parent_arg_positions["args"].get(0) == long_label
    assert not arange_op.is_internal_source
    assert arange_op.unattributed_tensor_args == ()
    assert tl.validate(_ArangeTensorEnd().eval(), x, scope="forward")


def test_h2_literal_control_args_stay_clean() -> None:
    """Literal (non-tensor) control args must not grow parents or markers."""

    class _LiteralRoll(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Roll by a Python literal shift.

            Parameters
            ----------
            x:
                1-D input tensor.

            Returns
            -------
            torch.Tensor
                ``x`` rolled by one.
            """

            return torch.roll(x, 1)

    trace = tl.trace(_LiteralRoll().eval(), torch.tensor([0.0, 1.0, 2.0]))
    roll_op = _ops_by_func(trace, "roll")[0]

    assert roll_op.parents == ("input_1",)
    assert roll_op.unattributed_tensor_args == ()


# ---------------------------------------------------------------------------
# H1 -- list-returning in-place foreach threads mutation labels to live members
# ---------------------------------------------------------------------------


class _ForeachInplace(nn.Module):
    """Mutate two derived tensors through ``torch._foreach_add_``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum both foreach-mutated tensors.

        Parameters
        ----------
        x:
            1-D input tensor.

        Returns
        -------
        torch.Tensor
            Combined sum of the two mutated tensors.
        """

        left = x + 1
        right = x + 3
        torch._foreach_add_([left, right], 1)
        return left.sum() + right.sum()


class _ForeachOutOfPlace(nn.Module):
    """Zip two tensor lists through out-of-place ``torch._foreach_add``."""

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the zipped pairs and keep both members alive.

        Parameters
        ----------
        a:
            First member of the first list.
        b:
            Second member of the first list.
        c:
            First member of the second list.
        d:
            Second member of the second list.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The two zipped sums.
        """

        outs = torch._foreach_add([a, b], [c, d])
        return outs[0] * 1, outs[1] * 1


def test_h1_foreach_inplace_threads_mutation_to_consumers() -> None:
    """Consumers of foreach-mutated members parent the mutation nodes."""

    x = torch.tensor([1.0, 2.0, 3.0])
    trace = tl.trace(_ForeachInplace().eval(), x)
    foreach_ops = _ops_by_func(trace, "_foreach_add_")
    sum_ops = _ops_by_func(trace, "sum")

    assert len(foreach_ops) == 2
    # Zipped parents: each mutation node consumes exactly its own member.
    assert [len(op.parents) for op in foreach_ops] == [1, 1]
    assert foreach_ops[0].parents != foreach_ops[1].parents
    # Mutation nodes are NOT dead ends: each sum consumes its mutation node.
    assert sum_ops[0].parents == (foreach_ops[0].layer_label,)
    assert sum_ops[1].parents == (foreach_ops[1].layer_label,)
    assert all(op.children for op in foreach_ops)
    assert tl.validate(_ForeachInplace().eval(), x, scope="forward")


def test_m3_foreach_out_of_place_parents_are_zipped_without_duplicates() -> None:
    """Out-of-place foreach member ``i`` parents exactly ``(a[i], b[i])``."""

    inputs = [
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([10.0]),
        torch.tensor([20.0]),
    ]
    trace = tl.trace(_ForeachOutOfPlace().eval(), inputs)
    foreach_ops = _ops_by_func(trace, "_foreach_add")

    assert len(foreach_ops) == 2
    assert foreach_ops[0].parents == ("input_3", "input_1")
    assert foreach_ops[1].parents == ("input_4", "input_2")
    assert foreach_ops[0].parent_arg_positions["args"] == {
        (1, 0): "input_3",
        (0, 0): "input_1",
    }
    assert foreach_ops[1].parent_arg_positions["args"] == {
        (1, 1): "input_4",
        (0, 1): "input_2",
    }
    assert tl.validate(_ForeachOutOfPlace().eval(), inputs, scope="forward")


def test_m3_dropped_zipped_edge_still_fails_validation() -> None:
    """The sibling-slot exemption cannot mask a genuinely dropped zipped edge.

    Surgically removes one member's own zipped parent edge from the trace
    (exactly what an attribution regression would leave behind) and asserts
    the orphan-arg sweep still fails: the sibling attributes a DIFFERENT slot,
    so the narrow foreach exemption does not apply.
    """

    from torchlens.validation.core import validate_saved_outs

    inputs = [
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([10.0]),
        torch.tensor([20.0]),
    ]
    model = _ForeachOutOfPlace().eval()
    ground_truth = model(*inputs)
    trace = tl.trace(
        model,
        inputs,
        capture=tl.options.CaptureOptions(save_arg_values=True, save_rng_states=True),
    )
    member0 = _ops_by_func(trace, "_foreach_add")[0]
    member0.parents = [p for p in member0.parents if p != "input_3"]
    member0.parent_arg_positions["args"] = {
        key: label
        for key, label in member0.parent_arg_positions["args"].items()
        if label != "input_3"
    }

    result = validate_saved_outs(trace, list(ground_truth))

    assert not bool(result)


# ---------------------------------------------------------------------------
# M4 -- element-exact storage-alias mutation propagation
# ---------------------------------------------------------------------------


class _InterleavedDisjointViews(nn.Module):
    """Mutate the even-strided view; consume the element-disjoint odd view."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the odd elements, untouched by the even-view mutation.

        Parameters
        ----------
        x:
            1-D input tensor.

        Returns
        -------
        torch.Tensor
            Odd-position elements times one.
        """

        base = x + 0
        even = base[::2]
        odd = base[1::2]
        even.add_(100.0)
        return odd * 1


class _GenuinelyOverlappingViews(nn.Module):
    """Mutate a view that truly shares elements with the consumed view."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the overlapping slice after the head mutation.

        Parameters
        ----------
        x:
            1-D input tensor with at least six elements.

        Returns
        -------
        torch.Tensor
            Overlapping slice times one.
        """

        base = x + 0
        head = base[:4]
        overlap = base[2:6]
        head.add_(100.0)
        return overlap * 1


def test_m4_element_disjoint_interleaved_views_get_no_mutation_edge() -> None:
    """``base[1::2]`` consumers must not be parented to a ``base[::2]`` mutation."""

    x = torch.arange(6, dtype=torch.float32)
    trace = tl.trace(_InterleavedDisjointViews().eval(), x)
    mutation_op = _ops_by_func(trace, "add_")[0]
    mul_op = _ops_by_func(trace, "__mul__")[0]
    odd_view_label = _ops_by_func(trace, "__getitem__")[1].layer_label

    assert mul_op.parents == (odd_view_label,)
    assert mutation_op.layer_label not in mul_op.parents
    assert tl.validate(_InterleavedDisjointViews().eval(), x, scope="forward")


def test_m4_genuinely_overlapping_views_keep_mutation_edge() -> None:
    """True element overlap still propagates the mutation label (W3 F1 kept)."""

    x = torch.arange(8, dtype=torch.float32)
    trace = tl.trace(_GenuinelyOverlappingViews().eval(), x)
    mutation_op = _ops_by_func(trace, "add_")[0]
    mul_op = _ops_by_func(trace, "__mul__")[0]

    assert mul_op.parents == (mutation_op.layer_label,)
    assert tl.validate(_GenuinelyOverlappingViews().eval(), x, scope="forward")


def test_m4_exact_overlap_helper_matrix() -> None:
    """Unit matrix for the element-exact overlap decision."""

    from torchlens.backends.torch.wrappers import _strided_views_share_storage_elements

    base = torch.arange(12, dtype=torch.float32)
    assert _strided_views_share_storage_elements(base[::2], base[1::2]) is False
    assert _strided_views_share_storage_elements(base[::2], base[::3]) is True
    assert _strided_views_share_storage_elements(base[:4], base[2:6]) is True
    assert _strided_views_share_storage_elements(base[:4], base[4:]) is False
    assert _strided_views_share_storage_elements(base[::4], base[2::4]) is False
    grid = base.reshape(3, 4)
    assert _strided_views_share_storage_elements(grid[:, 0], grid[:, 1]) is False
    assert _strided_views_share_storage_elements(grid[0], grid[:, 0]) is True
    expanded = base[:1].expand(5)
    assert _strided_views_share_storage_elements(expanded, base[:1]) is True
    assert _strided_views_share_storage_elements(expanded, base[1:2]) is False


# ---------------------------------------------------------------------------
# M5 -- in-place grad_fn metadata is the user op's, not TorchLens's clone
# ---------------------------------------------------------------------------


class _InplaceGradModel(nn.Module):
    """In-place add on a derived tensor with autograd live."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate a derived value in place and return a consumer.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Double the mutated value.
        """

        value = x + 1
        value.add_(1)
        return value * 2


def test_m5_inplace_grad_fn_metadata_is_user_op() -> None:
    """``add_`` records ``AddBackward0``, never the safe copy's clone node."""

    x = torch.tensor([1.0, 3.0], requires_grad=True)
    trace = tl.trace(
        _InplaceGradModel().eval(),
        x,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    inplace_op = _ops_by_func(trace, "add_")[0]

    assert inplace_op.grad_fn_class_name == "AddBackward0"
    assert "CloneBackward" not in str(inplace_op.grad_fn_class_qualname)
    trace.cleanup()


# ---------------------------------------------------------------------------
# M6 -- mutating tensor property setters emit ops with correct parents
# ---------------------------------------------------------------------------


class _RealSetterModel(nn.Module):
    """Replace the real part of a complex tensor via ``.real =``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the magnitude after the real-part replacement.

        Parameters
        ----------
        x:
            Real input tensor.

        Returns
        -------
        torch.Tensor
            Magnitudes reflecting the replacement values.
        """

        z = torch.complex(x, x * 0)
        replacement = x + 5
        z.real = replacement
        return z.abs()


class _DataSetterModel(nn.Module):
    """Rebind a derived tensor's storage via ``.data =``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a consumer of the rebound tensor.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Double the rebound values.
        """

        target = x + 1
        replacement = x + 5
        target.data = replacement
        return target * 2


def test_m6_real_setter_emits_op_with_receiver_and_rhs_parents() -> None:
    """``z.real = rhs`` emits an op; the consumer binds to it, not stale z."""

    x = torch.tensor([1.0, 2.0])
    trace = tl.trace(_RealSetterModel().eval(), x)
    setter_op = _ops_by_func(trace, "real")[0]
    complex_label = _ops_by_func(trace, "complex")[0].layer_label
    rhs_label = _ops_by_func(trace, "__add__")[0].layer_label
    abs_op = _ops_by_func(trace, "__abs__")[0]

    assert set(setter_op.parents) == {complex_label, rhs_label}
    assert abs_op.parents == (setter_op.layer_label,)
    assert setter_op.children == (abs_op.layer_label,)
    assert tl.validate(_RealSetterModel().eval(), x, scope="forward")


def test_m6_data_setter_emits_op_and_threads_consumers() -> None:
    """``t.data = rhs`` emits an op parented to the RHS producer."""

    x = torch.tensor([1.0, 2.0])
    trace = tl.trace(_DataSetterModel().eval(), x)
    setter_op = _ops_by_func(trace, "data")[0]
    rhs_label = next(
        op.layer_label for op in _ops_by_func(trace, "__add__") if float(op.out[0]) == 6.0
    )
    mul_op = _ops_by_func(trace, "__mul__")[0]

    assert setter_op.parents == (rhs_label,)
    assert mul_op.parents == (setter_op.layer_label,)
    assert torch.equal(setter_op.out, torch.tensor([6.0, 7.0]))
    assert tl.validate(_DataSetterModel().eval(), x, scope="forward")


def test_m6_data_getter_still_records_as_detach() -> None:
    """The ``.data`` GETTER keeps its canonical detach identity."""

    class _DataGetter(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Read ``.data`` and consume it.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                The detached alias plus one.
            """

            return (x + 1).data + 1

    trace = tl.trace(_DataGetter().eval(), torch.tensor([1.0]))

    assert len(_ops_by_func(trace, "detach")) == 1
    assert not _ops_by_func(trace, "data")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
