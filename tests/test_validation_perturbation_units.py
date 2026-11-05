"""Perturbation and deep-clone unit tests for the validation subpackage.

Moved verbatim from ``tests/test_validation.py`` (at its size ceiling): the
replay-input helpers that validation leans on -- ``_perturb_layer_outs``
(every dtype family, saturation, scale, empty and scalar edge cases) plus
``_deep_clone_tensors`` / ``_copy_validation_args`` -- tested in isolation.
"""

from collections import namedtuple

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._utils.tensor_utils import tensor_nanequal
from torchlens.options import CaptureOptions
from torchlens.validation import validate_forward_pass
from torchlens.validation._core import (
    _copy_validation_args,
    _deep_clone_tensors,
    validate_saved_outs,
)
from torchlens.validation._diagnostics import get_validation_failure
from torchlens.validation._perturb import _perturb_layer_outs

# =============================================================================
# Perturbation unit tests
# =============================================================================


def test_perturbation_response_gate_treats_one_ulp_change_as_unequal() -> None:
    """The perturbation gate accepts only exact, NaN-aware output equality."""

    saved = torch.tensor([1.0], dtype=torch.float32)
    recomputed = torch.nextafter(saved, torch.tensor([float("inf")], dtype=torch.float32))

    assert not tensor_nanequal(recomputed, saved, allow_tolerance=False)
    assert tensor_nanequal(saved, saved.clone(), allow_tolerance=False)


def test_perturbation_changes_float_tensor() -> None:
    """Floating-point perturbation changes ordinary tensor values."""

    parent = torch.randn(10, 10)
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_outs(parent, output)
    assert not torch.equal(perturbed, parent)
    assert perturbed.shape == parent.shape


def test_perturbation_scales_near_constant_float_to_large_output() -> None:
    """Near-constant float perturbations scale up when child outputs are huge."""

    parent = torch.zeros(16, dtype=torch.float32)
    output = torch.full((16,), 1.0e30, dtype=torch.float32)

    perturbed = _perturb_layer_outs(parent, output)

    assert not torch.equal(perturbed, parent)
    assert perturbed.abs().max() > 1.0e20
    assert not torch.equal(output - perturbed, output)


def test_perturbation_scales_tiny_float_range_to_large_output() -> None:
    """Tiny float ranges scale up when otherwise swallowed by huge operands."""

    parent = torch.linspace(-0.25, 0.25, 16, dtype=torch.float32)
    output = torch.full((16,), 1.0e30, dtype=torch.float32)

    perturbed = _perturb_layer_outs(parent, output)

    assert not torch.equal(perturbed, parent)
    assert perturbed.abs().max() > 1.0e20
    assert not torch.equal(output + perturbed, output)


def test_validation_perturbs_zero_parent_at_large_float_scale() -> None:
    """Replay validation detects sensitivity when a zero parent meets a huge operand."""

    class HugeSubZero(nn.Module):
        """Model whose subtraction parent is zero but value-sensitive."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Subtract a data-derived zero tensor from a huge float tensor.

            Parameters
            ----------
            x:
                Input tensor used to shape the zero-valued parent.

            Returns
            -------
            torch.Tensor
                Huge float tensor with a zero-valued subtraction parent.
            """

            huge = torch.ones_like(x) * 1.0e30
            zero = x * 0.0
            return huge - zero

    assert validate_forward_pass(HugeSubZero(), torch.ones(2, 3), seed=123)


def test_perturbation_changes_int_tensor() -> None:
    """Integer perturbation changes tensor values while preserving dtype."""

    parent = torch.randint(0, 100, (10, 10))
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_outs(parent, output)
    assert not torch.equal(perturbed, parent)
    assert perturbed.dtype == parent.dtype


def test_perturbation_int64_saturated_max_does_not_overflow() -> None:
    """C1 regression: an int64 parent holding INT64_MAX must not crash randint.

    A legitimately captured int64 tensor that contains ``iinfo(int64).max``
    (common as PyG sentinel/cluster index values) used to make
    ``parent_outs.max() + 1`` wrap to ``INT64_MIN`` and raise
    ``RuntimeError("random_ expects 'from' to be less than 'to'...")``. The
    perturbation must run cleanly *and* still meaningfully perturb the tensor --
    a no-op would silently disarm the validation tripwire.
    """

    imax = torch.iinfo(torch.int64).max
    parent = torch.tensor([0, 5, imax, 100, imax, 42], dtype=torch.int64)
    output = torch.randn(parent.shape)

    perturbed = _perturb_layer_outs(parent, output)

    # (a) no raise (reached here), (b) genuinely perturbed, (c) dtype/shape kept.
    assert not torch.equal(perturbed, parent)
    assert perturbed.dtype == torch.int64
    assert perturbed.shape == parent.shape


def test_perturbation_uint8_saturated_max_does_not_overflow() -> None:
    """C1 regression: a uint8 parent at its dtype max must clamp, not overflow."""

    umax = torch.iinfo(torch.uint8).max  # 255
    parent = torch.tensor([0, 5, umax, 10, umax], dtype=torch.uint8)
    output = torch.randn(parent.shape)

    perturbed = _perturb_layer_outs(parent, output)

    assert not torch.equal(perturbed, parent)
    assert perturbed.dtype == torch.uint8
    assert perturbed.shape == parent.shape


def test_sagpooling_validates_end_to_end_without_perturb_overflow() -> None:
    """C1 golden-model gate: a PyG SAGPooling graph validates green end-to-end.

    SAGPooling emits an int64 index tensor carrying ``INT64_MAX`` during its
    top-k selection. Before the fix this made the perturbation helper raise on a
    successfully-captured trace (a validation-machinery crash masking 21 PyG
    models). The tripwire must now run to a real pass/fail with no crash.
    """

    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import GCNConv, SAGPooling

    torch.manual_seed(0)

    class SAGPoolNet(nn.Module):
        """Minimal GCN + SAGPooling graph that exercises the int64 sentinel path."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = GCNConv(8, 16)
            self.pool = SAGPooling(16, ratio=0.5)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Embed nodes, then pool the graph and return the pooled features."""

            x = self.conv(x, edge_index).relu()
            x, edge_index, _, _, _, _ = self.pool(x, edge_index)
            return x

    n_nodes = 10
    x = torch.randn(n_nodes, 8)
    edge_index = torch.randint(0, n_nodes, (2, 30), dtype=torch.long)

    # Returns True: the perturbation tripwire ran end-to-end with no overflow
    # crash and the replay/perturbation checks genuinely passed.
    assert validate_forward_pass(SAGPoolNet(), (x, edge_index), seed=1)


def test_perturbation_changes_bool_tensor():
    parent = torch.ones(10, 10, dtype=torch.bool)
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_outs(parent, output)
    # With 100 elements all True, random should differ
    assert not torch.equal(perturbed, parent)
    assert perturbed.dtype == torch.bool


def test_perturbation_changes_complex_tensor():
    parent = torch.complex(torch.randn(5, 5), torch.randn(5, 5))
    output = torch.randn(5, 5)
    perturbed = _perturb_layer_outs(parent, output)
    assert not torch.equal(perturbed, parent)
    assert perturbed.is_complex()


def test_perturbation_respects_dtype():
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]:
        if dtype in (torch.int32, torch.int64):
            parent = torch.randint(0, 100, (5, 5), dtype=dtype)
        elif dtype == torch.bool:
            parent = torch.ones(5, 5, dtype=torch.bool)
        else:
            parent = torch.randn(5, 5, dtype=dtype)
        output = torch.randn(5, 5)
        perturbed = _perturb_layer_outs(parent, output)
        assert perturbed.dtype == dtype


def test_perturbation_handles_empty_tensor():
    parent = torch.tensor([])
    output = torch.tensor([])
    perturbed = _perturb_layer_outs(parent, output)
    assert perturbed.numel() == 0
    assert torch.equal(perturbed, parent)


def test_perturbation_terminates_on_scalar():
    """MAX_PERTURB_ATTEMPTS guard prevents infinite loop on single-element tensors."""
    # Single-element bool tensor: 50% chance each attempt matches original.
    # With MAX_PERTURB_ATTEMPTS=100, it should terminate regardless.
    parent = torch.tensor([True])
    output = torch.tensor([1.0])
    perturbed = _perturb_layer_outs(parent, output)
    assert perturbed.dtype == torch.bool
    assert perturbed.shape == parent.shape


# =============================================================================
# Deep clone tests
# =============================================================================


def test_deep_clone_nested_list_of_tensors():
    original = [torch.tensor([1.0, 2.0]), [torch.tensor([3.0]), torch.tensor([4.0])]]
    cloned = _deep_clone_tensors(original)
    assert isinstance(cloned, list)
    assert isinstance(cloned[1], list)
    assert torch.equal(cloned[0], original[0])
    assert torch.equal(cloned[1][0], original[1][0])


def test_deep_clone_nested_dict_of_tensors():
    original = {"a": torch.tensor([1.0]), "b": {"c": torch.tensor([2.0])}}
    cloned = _deep_clone_tensors(original)
    assert isinstance(cloned, dict)
    assert isinstance(cloned["b"], dict)
    assert torch.equal(cloned["a"], original["a"])
    assert torch.equal(cloned["b"]["c"], original["b"]["c"])


def test_deep_clone_independence():
    """Modifying clone doesn't affect original."""
    original = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    cloned = _deep_clone_tensors(original)
    cloned[0][0] = 999.0
    assert original[0][0].item() == 1.0


def test_deep_clone_preserves_non_tensors():
    original = [42, "hello", None, (1, 2)]
    cloned = _deep_clone_tensors(original)
    assert cloned == original


def test_deep_clone_preserves_namedtuple_type() -> None:
    """Namedtuple containers are reconstructed with positional fields."""

    point_type = namedtuple("Point", ["x", "y"])
    original = point_type(torch.tensor([1.0]), torch.tensor([2.0]))

    cloned = _deep_clone_tensors(original)

    assert isinstance(cloned, point_type)
    assert torch.equal(cloned.x, original.x)
    assert torch.equal(cloned.y, original.y)


def test_copy_validation_args():
    """_copy_validation_args deep-clones tensors in args and kwargs."""
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([3.0])
    input_args = {
        "args": [t1, [t2, 42]],
        "kwargs": {"key": torch.tensor([5.0])},
    }
    copied = _copy_validation_args(input_args)

    # Independence
    copied["args"][0][0] = 999.0
    assert t1[0].item() == 1.0

    copied["kwargs"]["key"][0] = 999.0
    assert input_args["kwargs"]["key"][0].item() == 5.0


_WIDE_UNSIGNED = tuple(name for name in ("uint16", "uint32", "uint64") if hasattr(torch, name))
_SIGNED_OF = {"uint16": torch.int16, "uint32": torch.int32, "uint64": torch.int64}


@pytest.mark.parametrize("name", _WIDE_UNSIGNED)
@pytest.mark.parametrize("values", ([7, 7], [7, -3, 0, 12]), ids=("constant", "spread"))
def test_perturbation_changes_wide_unsigned_tensor(name: str, values: list[int]) -> None:
    """A uint16/32/64 parent (a philox key) is perturbed as an integer, never through finfo."""

    signed = _SIGNED_OF[name]
    parent = torch.tensor(values, dtype=signed).view(getattr(torch, name))
    perturbed = _perturb_layer_outs(parent, torch.zeros(2))
    assert perturbed.dtype == parent.dtype and perturbed.shape == parent.shape
    assert not torch.equal(perturbed.view(signed), parent.view(signed))


class _Uint64KeyConsumer(nn.Module):
    """Reads a uint64 key's bits as signed integers and adds them to ``x``."""

    def forward(self, x: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return x + key.view(torch.int64).to(x.dtype)


def _uint64_key_inputs(dtype_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(2), torch.tensor([7, -3], dtype=torch.int64).view(getattr(torch, dtype_name))


_UINT64 = tuple(name for name in _WIDE_UNSIGNED if name == "uint64")


@pytest.mark.parametrize("dtype_name", _UINT64)
def test_validation_perturbs_uint64_parent(dtype_name: str) -> None:
    """A forward with a uint64 parent validates instead of crashing in the float branch."""

    assert validate_forward_pass(_Uint64KeyConsumer(), _uint64_key_inputs(dtype_name)) is True


@pytest.mark.parametrize("dtype_name", _UINT64)
def test_planted_key_blind_uint64_consumer_fails_perturbation(dtype_name: str) -> None:
    """A replay that ignores its uint64 parent is caught by perturbing that parent."""

    model, inputs = _Uint64KeyConsumer(), _uint64_key_inputs(dtype_name)
    ground_truth = model(*inputs)
    trace = tl.trace(model, inputs, capture=CaptureOptions(save_args=True, seed=0), save="all")
    try:
        view_op = next(op for op in trace._layer_list if op.func_name == "view")
        key_label = view_op.call.parent_arg_positions["args"][0]
        frozen = view_op.out.detach().clone()

        def key_blind_view(*args: object, **kwargs: object) -> torch.Tensor:
            return frozen.clone()

        key_blind_view.__name__ = view_op.func.__name__
        view_op.func = key_blind_view
        status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
        assert status.state == "failed"
        failure = get_validation_failure(trace)
        assert failure is not None and failure.check_name == "perturbation"
        assert key_label in failure.extra["perturbed_parents"]
    finally:
        trace.cleanup()
