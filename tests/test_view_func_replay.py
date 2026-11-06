"""Capture and replay of the Tensor view-function methods.

``_view_func(new_base)`` and ``_view_func_unsafe(new_base)`` replay a view's chain onto a new
base; ``_rev_view_func_unsafe(new_view)`` maps a view-shaped tensor back to the base shape.
Torch reads only the receiver's view chain, never its values, and returns ``None`` for a
receiver that is not a view. Capture saves the receiver rebuilt on a private copy of its base,
so replay calls the method on a real view instead of a compact clone that lost its view
relation. The receiver slot is a structural position (read for its view chain only); the new
base stays a value input under the perturbation check.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens import _state
from torchlens._constants import get_orig_torch_funcs
from torchlens._utils._torch_compat import HAS_REV_VIEW_FUNC_UNSAFE
from torchlens.backends.torch._tl import is_decorated_function
from torchlens.options import CaptureOptions
from torchlens.validation._core import validate_saved_outs
from torchlens.validation._diagnostics import get_validation_failure

_VIEW_FUNCS = tuple(
    name
    for name in ("_view_func", "_view_func_unsafe", "_rev_view_func_unsafe")
    if hasattr(torch.Tensor, name)
)


_REQUIRES_REV_VIEW_FUNC = pytest.mark.skipif(
    not HAS_REV_VIEW_FUNC_UNSAFE,
    reason="Tensor._rev_view_func_unsafe postdates the torch 2.1 floor (2.3+)",
)


def _transpose(m: torch.Tensor) -> torch.Tensor:
    """A one-step view chain."""

    return m.t()


def _transpose_row(m: torch.Tensor) -> torch.Tensor:
    """A two-step view chain (transpose, then select a row)."""

    return m.t()[0]


_CHAINS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "transpose": _transpose,
    "transpose_row": _transpose_row,
}


def _tracked_view(chain: Callable[[torch.Tensor], torch.Tensor], m: torch.Tensor) -> torch.Tensor:
    """``chain(m)`` made under original-view tracking, so it keeps its view functions."""

    with torch.autograd._force_original_view_tracking(True):
        return chain(m)


def _call_view_func(
    view: torch.Tensor,
    method: str,
    chain: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """Call ``method`` on ``view`` with a new base (the reverse method: a new view) from ``x``."""

    new_base = x * 2
    arg = chain(new_base) if method == "_rev_view_func_unsafe" else new_base
    return getattr(view, method)(arg)


class _ViewFuncModel(nn.Module):
    """Makes a tracked view inside the forward, calls the method on it, consumes the result."""

    def __init__(self, method: str, chain: str) -> None:
        """Keep the method name and the view chain to build."""

        super().__init__()
        self.method = method
        self.chain = _CHAINS[chain]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the view-function method on a view of ``relu(x)``."""

        view = _tracked_view(self.chain, torch.relu(x))
        return _call_view_func(view, self.method, self.chain, x) * 2 + x


class _OutsideViewModel(nn.Module):
    """Calls the method on a view made in ``__init__``, outside any forward."""

    def __init__(self, method: str) -> None:
        """Build the view of a fixed base once, at construction."""

        super().__init__()
        self.method = method
        gen = torch.Generator().manual_seed(1)
        self.view = _tracked_view(_transpose, torch.randn(4, 4, generator=gen))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the view-function method on the stored view."""

        return _call_view_func(self.view, self.method, _transpose, x) * 2 + x


def _input_4x4() -> torch.Tensor:
    """A seeded 4x4 input."""

    torch.manual_seed(0)
    return torch.randn(4, 4)


def _assert_validates_and_replays(model: nn.Module, x: torch.Tensor) -> None:
    """Forward validation passes and a live replay on fresh input matches a plain forward."""

    assert tl.validate(model, x, scope="forward", validate_metadata=True) is True
    trace = tl.trace(model, x)
    try:
        fresh = torch.randn_like(x)
        result = trace.run_transaction(inputs=fresh)
        try:
            assert torch.equal(result.output, model(fresh))
        finally:
            result.trace.cleanup()
    finally:
        trace.cleanup()


def _view_func_op(trace: tl.Trace, method: str) -> object:
    """The one recorded op of ``method`` in ``trace``."""

    (op,) = [op for op in trace._layer_list if op.func_name == method]
    return op


@pytest.mark.parametrize("method", _VIEW_FUNCS)
def test_view_func_method_is_decorated(method: str) -> None:
    """The view-function method is in TorchLens's decorated set."""

    assert ("torch.Tensor", method) in set(get_orig_torch_funcs())


@pytest.mark.parametrize("chain", tuple(_CHAINS))
@pytest.mark.parametrize("method", _VIEW_FUNCS)
def test_view_func_method_validates_and_replays(method: str, chain: str) -> None:
    """The method records with the new base as a parent and a consumer, validates and replays."""

    model, x = _ViewFuncModel(method, chain), _input_4x4()
    trace = tl.trace(model, x)
    try:
        op = _view_func_op(trace, method)
        positions = op.call.parent_arg_positions["args"]
        assert 0 in positions and 1 in positions and op.children, (positions, op.children)
    finally:
        trace.cleanup()
    _assert_validates_and_replays(model, x)


@pytest.mark.parametrize("method", _VIEW_FUNCS)
def test_view_func_on_view_made_outside_forward_validates(method: str) -> None:
    """A receiver built before the forward replays through its saved view context.

    ``tl.validate`` runs its ground truth on a deep copy of the model, and torch's
    ``Tensor.__deepcopy__`` rebuilds a tensor with ``set_``, which drops the view relation, so
    this case validates the trace of the model itself against its own plain forward.
    """

    trace, ground_truth = _traced_with_args(_OutsideViewModel(method), _input_4x4())
    try:
        op = _view_func_op(trace, method)
        assert op.call.saved_args[0]._is_view()
        status = validate_saved_outs(trace, [ground_truth], validate_metadata=True)
        assert status.state == "passed", get_validation_failure(trace)
    finally:
        trace.cleanup()


@pytest.mark.parametrize("chain", tuple(_CHAINS))
@pytest.mark.parametrize("method", _VIEW_FUNCS)
def test_receiver_values_never_reach_the_output(method: str, chain: str) -> None:
    """Same view chain, different receiver values: the output is bit-identical.

    This is the premise of the structural row for the receiver slot on the installed torch.
    """

    gen = torch.Generator().manual_seed(2)
    first = _tracked_view(_CHAINS[chain], torch.randn(4, 4, generator=gen))
    second = _tracked_view(_CHAINS[chain], torch.randn(4, 4, generator=gen) * 100)
    assert not torch.equal(first, second)
    x = _input_4x4()
    left = _call_view_func(first, method, _CHAINS[chain], x)
    right = _call_view_func(second, method, _CHAINS[chain], x)
    assert left.dtype == right.dtype and torch.equal(left, right)


def _traced_with_args(model: nn.Module, x: torch.Tensor) -> tuple[tl.Trace, torch.Tensor]:
    """A trace that keeps saved arguments, and the plain forward's output."""

    ground_truth = model(x)
    trace = tl.trace(model, x, capture=CaptureOptions(save_args=True), save="all")
    return trace, ground_truth


@pytest.mark.parametrize("method", _VIEW_FUNCS)
def test_planted_new_base_blind_replay_fails_perturbation(method: str) -> None:
    """A replay that ignores the new base fails: only the receiver slot is structural."""

    trace, ground_truth = _traced_with_args(_ViewFuncModel(method, "transpose"), _input_4x4())
    try:
        op = _view_func_op(trace, method)
        new_base_label = op.call.parent_arg_positions["args"][1]
        frozen = op.out.detach().clone()

        def new_base_blind(*args: object, **kwargs: object) -> torch.Tensor:
            """Return the saved output whatever the new base is."""

            return frozen.clone()

        new_base_blind.__name__ = op.func.__name__
        op.func = new_base_blind
        status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
        assert status.state == "failed"
        failure = get_validation_failure(trace)
        assert failure is not None and failure.check_name == "perturbation"
        assert new_base_label in failure.extra["perturbed_parents"]
    finally:
        trace.cleanup()


@pytest.mark.parametrize("method", _VIEW_FUNCS)
def test_planted_wrong_new_base_value_fails_replay(method: str) -> None:
    """A replay handed a shifted new base does not reproduce the saved output."""

    trace, ground_truth = _traced_with_args(_ViewFuncModel(method, "transpose"), _input_4x4())
    try:
        op = _view_func_op(trace, method)
        real = op.func

        def shifted_new_base(receiver: torch.Tensor, arg: torch.Tensor) -> torch.Tensor:
            """Call the real method with the new base moved by one."""

            return real(receiver, arg + 1)

        shifted_new_base.__name__ = real.__name__
        op.func = shifted_new_base
        status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
        assert status.state == "failed"
        failure = get_validation_failure(trace)
        assert failure is not None and failure.op_label.split(":")[0] == op.layer_label
        assert failure.check_name == "forward_replay"
    finally:
        trace.cleanup()


@pytest.mark.parametrize("method", _VIEW_FUNCS)
def test_compact_receiver_is_a_replay_error(method: str) -> None:
    """A receiver saved without its view relation is refused, never replayed to ``None``."""

    trace, ground_truth = _traced_with_args(_ViewFuncModel(method, "transpose"), _input_4x4())
    try:
        op = _view_func_op(trace, method)
        assert op.call.saved_args[0]._is_view()
        op._call_saved_args = (op.call.saved_args[0].detach().clone(), *op.call.saved_args[1:])
        with pytest.warns(UserWarning, match="Validation replay raised an exception") as caught:
            status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
        replay_warnings = [w for w in caught if "Validation replay raised" in str(w.message)]
        assert replay_warnings[0].filename == __file__
        assert status.state == "failed"
        failure = get_validation_failure(trace)
        assert failure is not None and failure.op_label.split(":")[0] == op.layer_label
        assert "raised an exception" in failure.message
    finally:
        trace.cleanup()


def _rev_view_func_two_step_trace() -> tuple[tl.Trace, torch.Tensor, object]:
    """A trace of the reverse method on a ``t()[0]`` view, its ground truth and the op."""

    model = _ViewFuncModel("_rev_view_func_unsafe", "transpose_row")
    trace, ground_truth = _traced_with_args(model, _input_4x4())
    return trace, ground_truth, _view_func_op(trace, "_rev_view_func_unsafe")


@_REQUIRES_REV_VIEW_FUNC
def test_rev_view_func_new_view_keeps_its_storage() -> None:
    """The reverse of ``select`` reads the new view's whole storage, so capture keeps it."""

    trace, ground_truth, op = _rev_view_func_two_step_trace()
    try:
        new_view = op.call.saved_args[1]
        assert new_view.numel() == 4
        assert new_view.untyped_storage().nbytes() == 16 * new_view.element_size()
        status = validate_saved_outs(trace, [ground_truth], validate_metadata=True)
        assert status.state == "passed", get_validation_failure(trace)
    finally:
        trace.cleanup()


@_REQUIRES_REV_VIEW_FUNC
def test_compact_new_view_is_a_replay_error() -> None:
    """A new view saved without its storage cannot be mapped back to the base: replay fails."""

    trace, ground_truth, op = _rev_view_func_two_step_trace()
    try:
        op._call_saved_args = (
            op.call.saved_args[0],
            op.call.saved_args[1].detach().clone(),
            *op.call.saved_args[2:],
        )
        with pytest.warns(UserWarning, match="Validation replay raised an exception") as caught:
            status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
        replay_warnings = [w for w in caught if "Validation replay raised" in str(w.message)]
        assert replay_warnings[0].filename == __file__
        assert status.state == "failed"
        failure = get_validation_failure(trace)
        assert failure is not None and failure.op_label.split(":")[0] == op.layer_label
        assert "raised an exception" in failure.message
    finally:
        trace.cleanup()


def test_reverse_view_func_is_decorated_exactly_where_torch_has_it() -> None:
    """Graceful floor test for ``HAS_REV_VIEW_FUNC_UNSAFE``: decoration follows torch.

    Torch before 2.3 has no ``Tensor._rev_view_func_unsafe``; TorchLens decorates the
    view-function methods this torch has and leaves the missing one out, so ``_view_func``
    still decorates and replays (``test_view_func_method_validates_and_replays``).
    """

    decorated = set(get_orig_torch_funcs())
    assert (("torch.Tensor", "_rev_view_func_unsafe") in decorated) is HAS_REV_VIEW_FUNC_UNSAFE
    assert ("torch.Tensor", "_view_func") in decorated


def _wrapped_view_and_base() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a transposed view and its base after a capture installed the wrappers."""

    tl.trace(nn.Identity(), torch.ones(2))
    base = torch.arange(6.0).reshape(2, 3)
    return _tracked_view(_transpose, base), base


@pytest.mark.parametrize("method", _VIEW_FUNCS)
def test_bound_view_func_repr_never_renders_the_receiver(method: str) -> None:
    """A wrapped view-function method bound to a view reprs like torch's builtin method.

    Torch's fake-tensor describer stores ``t._view_func_unsafe`` and reprs it when
    structured-trace logging is on (pytest's log capture turns it on). A plain bound
    method's repr renders the receiver's values, which runs tensor ops inside scan's
    backward ``make_fx`` and hits torch's PythonFallbackKernel internal assert.
    """

    view, _ = _wrapped_view_and_base()
    assert is_decorated_function(getattr(torch.Tensor, method))
    text = repr(getattr(view, method))
    assert "tensor(" not in text
    assert text == f"<built-in method {method} of Tensor object at {id(view):#x}>"


@pytest.mark.parametrize("method", _VIEW_FUNCS)
def test_bound_view_func_keeps_the_bound_method_contract(method: str) -> None:
    """The quiet bound form compares, hashes, copies and calls like a bound method."""

    view, base = _wrapped_view_and_base()
    first, second = getattr(view, method), getattr(view, method)
    assert first == second and hash(first) == hash(second)
    assert first != getattr(_tracked_view(_transpose, base), method)
    assert first.__self__ is view and first.__func__ is getattr(torch.Tensor, method)
    assert copy.copy(first) == first
    assert copy.deepcopy(first).__func__ is first.__func__
    stock = _state._decorated_to_orig[id(first.__func__)]
    new_base = base * 2
    argument = _transpose(new_base) if method == "_rev_view_func_unsafe" else new_base
    torch.testing.assert_close(first(argument), stock(view, argument))
