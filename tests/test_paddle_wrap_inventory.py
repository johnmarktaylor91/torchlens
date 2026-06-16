"""Static Paddle wrapper inventory snapshot tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from torchlens.backends.paddle import wrappers as paddle_wrappers
from torchlens.backends.paddle.wrappers import PaddleInventory, _PaddleWrapperRegistry

paddle = pytest.importorskip("paddle")

SNAPSHOT_MESSAGE = (
    "Paddle wrapper inventory changed. Classify the new/moved/removed op in "
    "torchlens.backends.paddle.wrappers, then update tests/test_paddle_wrap_inventory.py. "
    "This static snapshot is the correctness guard for same-object no-op and scalar-escape "
    "coverage gaps that dynamic validation cannot see."
)

EXPECTED_WRAPPED = (
    "abs",
    "add",
    "argmax",
    "argmin",
    "assign",
    "cast",
    "clip",
    "concat",
    "cos",
    "divide",
    "einsum",
    "equal",
    "exp",
    "flatten",
    "floor",
    "full",
    "full_like",
    "functional.avg_pool1d",
    "functional.avg_pool2d",
    "functional.batch_norm",
    "functional.conv1d",
    "functional.conv2d",
    "functional.dropout",
    "functional.gelu",
    "functional.hardswish",
    "functional.layer_norm",
    "functional.leaky_relu",
    "functional.linear",
    "functional.max_pool1d",
    "functional.max_pool2d",
    "functional.relu",
    "functional.sigmoid",
    "functional.silu",
    "functional.softmax",
    "functional.tanh",
    "greater_equal",
    "greater_than",
    "less_equal",
    "less_than",
    "linspace",
    "log",
    "matmul",
    "max",
    "mean",
    "min",
    "mm",
    "multiply",
    "negative",
    "ones",
    "ones_like",
    "pow",
    "prod",
    "reshape",
    "sin",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "subtract",
    "sum",
    "tanh",
    "tensor.__add__",
    "tensor.__getitem__",
    "tensor.__matmul__",
    "tensor.__mul__",
    "tensor.__neg__",
    "tensor.__pow__",
    "tensor.__radd__",
    "tensor.__rmatmul__",
    "tensor.__rmul__",
    "tensor.__rpow__",
    "tensor.__rsub__",
    "tensor.__rtruediv__",
    "tensor.__sub__",
    "tensor.__truediv__",
    "tensor.abs",
    "tensor.add",
    "tensor.astype",
    "tensor.cast",
    "tensor.clip",
    "tensor.contiguous",
    "tensor.divide",
    "tensor.exp",
    "tensor.flatten",
    "tensor.log",
    "tensor.matmul",
    "tensor.max",
    "tensor.mean",
    "tensor.min",
    "tensor.multiply",
    "tensor.pow",
    "tensor.prod",
    "tensor.reshape",
    "tensor.rsqrt",
    "tensor.scale",
    "tensor.sqrt",
    "tensor.square",
    "tensor.squeeze",
    "tensor.std",
    "tensor.subtract",
    "tensor.sum",
    "tensor.t",
    "tensor.tile",
    "tensor.transpose",
    "tensor.unsqueeze",
    "tensor.var",
    "tile",
    "to_tensor",
    "transpose",
    "unsqueeze",
    "var",
    "where",
    "zeros",
    "zeros_like",
)

EXPECTED_DENIED = (
    "abs_",
    "acos_",
    "acosh_",
    "addmm_",
    "asin_",
    "asinh_",
    "async_save",
    "atan_",
    "atanh_",
    "baddbmm_",
    "bernoulli",
    "bernoulli_",
    "bitwise_and_",
    "bitwise_invert_",
    "bitwise_left_shift_",
    "bitwise_not_",
    "bitwise_or_",
    "bitwise_right_shift_",
    "bitwise_xor_",
    "cast_",
    "cauchy_",
    "clear_async_save_task_queue",
    "copysign_",
    "cos_",
    "cosh_",
    "cumprod_",
    "cumsum_",
    "digamma_",
    "disable_static",
    "div_",
    "divide_",
    "enable_static",
    "equal_",
    "erf_",
    "expm1_",
    "flatten_",
    "floor_divide_",
    "floor_mod_",
    "frac_",
    "functional.elu_",
    "functional.embedding_renorm_",
    "functional.hardtanh_",
    "functional.leaky_relu_",
    "functional.relu_",
    "functional.softmax_",
    "functional.tanh_",
    "functional.thresholded_relu_",
    "gammainc_",
    "gammaincc_",
    "gammaln_",
    "gcd_",
    "geometric_",
    "greater_equal_",
    "greater_than_",
    "hypot_",
    "i0_",
    "index_add_",
    "index_fill_",
    "index_put_",
    "lcm_",
    "ldexp_",
    "less_",
    "less_equal_",
    "less_than_",
    "lgamma_",
    "load",
    "log10_",
    "log1p_",
    "log2_",
    "log_",
    "log_normal_",
    "logical_and_",
    "logical_not_",
    "logical_or_",
    "logical_xor_",
    "logit_",
    "manual_seed",
    "masked_fill_",
    "masked_scatter_",
    "mod_",
    "multigammaln_",
    "multiply_",
    "nan_to_num_",
    "neg_",
    "normal",
    "normal_",
    "not_equal_",
    "poisson",
    "polygamma_",
    "pow_",
    "rand",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "remainder_",
    "renorm_",
    "reshape_",
    "save",
    "scatter_",
    "scatter_add_",
    "seed",
    "set_device",
    "sin_",
    "sinc_",
    "sinh_",
    "square_",
    "squeeze_",
    "sub_",
    "subtract_",
    "t_",
    "tan_",
    "tanh_",
    "tensor.__array__",
    "tensor.__bool__",
    "tensor.__dlpack__",
    "tensor.__float__",
    "tensor.__index__",
    "tensor.__int__",
    "tensor.__setitem__",
    "tensor._apply_",
    "tensor._to_dist_",
    "tensor._to_static_var",
    "tensor.abs_",
    "tensor.acos_",
    "tensor.acosh_",
    "tensor.add_",
    "tensor.addmm_",
    "tensor.apply_",
    "tensor.asin_",
    "tensor.asinh_",
    "tensor.atan_",
    "tensor.atanh_",
    "tensor.baddbmm_",
    "tensor.bernoulli_",
    "tensor.bitwise_and_",
    "tensor.bitwise_invert_",
    "tensor.bitwise_left_shift_",
    "tensor.bitwise_not_",
    "tensor.bitwise_or_",
    "tensor.bitwise_right_shift_",
    "tensor.bitwise_xor_",
    "tensor.cast_",
    "tensor.cauchy_",
    "tensor.ceil_",
    "tensor.clamp_",
    "tensor.clip_",
    "tensor.copy_",
    "tensor.copysign_",
    "tensor.cos_",
    "tensor.cosh_",
    "tensor.cumprod_",
    "tensor.cumsum_",
    "tensor.detach_",
    "tensor.digamma_",
    "tensor.div_",
    "tensor.divide_",
    "tensor.equal_",
    "tensor.erfinv_",
    "tensor.exp_",
    "tensor.exponential_",
    "tensor.fill_",
    "tensor.fill_diagonal_",
    "tensor.fill_diagonal_tensor_",
    "tensor.flatten_",
    "tensor.floor_",
    "tensor.floor_divide_",
    "tensor.floor_mod_",
    "tensor.frac_",
    "tensor.gammainc_",
    "tensor.gammaincc_",
    "tensor.gammaln_",
    "tensor.gcd_",
    "tensor.geometric_",
    "tensor.greater_equal_",
    "tensor.greater_than_",
    "tensor.hypot_",
    "tensor.i0_",
    "tensor.index_add_",
    "tensor.index_fill_",
    "tensor.index_put_",
    "tensor.item",
    "tensor.lcm_",
    "tensor.ldexp_",
    "tensor.lerp_",
    "tensor.less_",
    "tensor.less_equal_",
    "tensor.less_than_",
    "tensor.lgamma_",
    "tensor.log10_",
    "tensor.log1p_",
    "tensor.log2_",
    "tensor.log_",
    "tensor.log_normal_",
    "tensor.logical_and_",
    "tensor.logical_not_",
    "tensor.logical_or_",
    "tensor.logical_xor_",
    "tensor.logit_",
    "tensor.masked_fill_",
    "tensor.masked_scatter_",
    "tensor.mod_",
    "tensor.mul_",
    "tensor.multigammaln_",
    "tensor.multiply_",
    "tensor.nan_to_num_",
    "tensor.neg_",
    "tensor.normal_",
    "tensor.not_equal_",
    "tensor.numpy",
    "tensor.polygamma_",
    "tensor.pow_",
    "tensor.put_along_axis_",
    "tensor.random_",
    "tensor.reciprocal_",
    "tensor.reconstruct_from_",
    "tensor.remainder_",
    "tensor.renorm_",
    "tensor.requires_grad_",
    "tensor.reshape_",
    "tensor.resize_",
    "tensor.round_",
    "tensor.rsqrt_",
    "tensor.scale_",
    "tensor.scatter_",
    "tensor.scatter_add_",
    "tensor.set_",
    "tensor.set_value",
    "tensor.sigmoid_",
    "tensor.sin_",
    "tensor.sinc_",
    "tensor.sinh_",
    "tensor.sqrt_",
    "tensor.square_",
    "tensor.squeeze_",
    "tensor.sub_",
    "tensor.subtract_",
    "tensor.t_",
    "tensor.tan_",
    "tensor.tanh_",
    "tensor.tolist",
    "tensor.transpose_",
    "tensor.tril_",
    "tensor.triu_",
    "tensor.trunc_",
    "tensor.uniform_",
    "tensor.unsqueeze_",
    "tensor.where_",
    "tensor.zero_",
    "tolist",
    "transpose_",
    "tril_",
    "triu_",
    "trunc_",
    "uniform",
    "unsqueeze_",
    "where_",
)

ALIAS_NO_OP_APIS = frozenset(
    (
        "functional.dropout",
        "reshape",
        "tensor.astype",
        "tensor.cast",
        "tensor.contiguous",
        "tensor.reshape",
    )
)
TENSOR_ESCAPE_APIS = frozenset(
    (
        "tensor.__array__",
        "tensor.__bool__",
        "tensor.__dlpack__",
        "tensor.__float__",
        "tensor.__index__",
        "tensor.__int__",
        "tensor.item",
        "tensor.numpy",
        "tensor.tolist",
        "tolist",
    )
)
MUTATOR_APIS = frozenset(("tensor.__setitem__", "tensor.copy_", "tensor.set_value"))
RNG_APIS = frozenset(("bernoulli", "normal", "poisson", "rand", "randn", "uniform"))
COARSE_COMPOSITE_APIS = frozenset(
    (
        "functional.avg_pool1d",
        "functional.avg_pool2d",
        "functional.batch_norm",
        "functional.conv1d",
        "functional.conv2d",
        "functional.dropout",
        "functional.layer_norm",
        "functional.linear",
        "functional.max_pool1d",
        "functional.max_pool2d",
        "functional.softmax",
    )
)
STOCHASTIC_COMPOSITE_DENY_APIS = frozenset(("functional.dropout",))


class _NoopBackend:
    """Backend placeholder used only to build wrapper inventory."""


@contextmanager
def _installed_inventory() -> Iterator[PaddleInventory]:
    """Install a private registry and yield its inventory.

    Yields
    ------
    PaddleInventory
        Sorted wrapper inventory built from the live Paddle runtime.
    """

    registry = _PaddleWrapperRegistry()
    registry.wrap(_NoopBackend())
    try:
        yield registry.inventory()
    finally:
        registry.unwrap()


def _assert_inventory_matches_snapshot(inventory: PaddleInventory) -> None:
    """Assert a live inventory matches the pinned static snapshot.

    Parameters
    ----------
    inventory
        Live inventory to compare against the committed snapshot.
    """

    assert inventory.wrapped == EXPECTED_WRAPPED, SNAPSHOT_MESSAGE
    assert inventory.denied == EXPECTED_DENIED, SNAPSHOT_MESSAGE


def test_paddle_wrapper_inventory_matches_static_snapshot() -> None:
    """Pin the exact wrapped and denied Paddle wrapper inventory."""

    with _installed_inventory() as inventory:
        _assert_inventory_matches_snapshot(inventory)


def test_paddle_wrapper_inventory_load_bearing_membership() -> None:
    """Assert static coverage classes remain classified as intended."""

    with _installed_inventory() as inventory:
        wrapped = set(inventory.wrapped)
        denied = set(inventory.denied)

    assert {"std", "var", "tensor.std", "tensor.var"} <= wrapped
    assert ALIAS_NO_OP_APIS <= wrapped
    assert MUTATOR_APIS <= denied
    assert RNG_APIS <= denied
    assert TENSOR_ESCAPE_APIS <= denied
    assert COARSE_COMPOSITE_APIS <= wrapped
    assert STOCHASTIC_COMPOSITE_DENY_APIS <= wrapped
    for op_name in (
        "functional.dropout",
        "functional.identity",
        "reshape",
        "tensor.astype",
        "tensor.cast",
        "tensor.contiguous",
        "tensor.reshape",
    ):
        assert paddle_wrappers.is_alias_allowed_op(op_name)


def test_paddle_same_object_gap_fails_static_inventory_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate an unwrapped same-object API gap and require the snapshot to fail."""

    patched_tensor_methods = paddle_wrappers._TENSOR_CORE_METHODS - {"astype", "reshape"}
    monkeypatch.setattr(paddle_wrappers, "_TENSOR_CORE_METHODS", patched_tensor_methods)

    with (
        _installed_inventory() as inventory,
        pytest.raises(AssertionError, match="inventory changed"),
    ):
        _assert_inventory_matches_snapshot(inventory)
