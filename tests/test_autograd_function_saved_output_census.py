"""The alias autograd takes of a custom Function's own output is credited, nothing else.

A custom ``torch.autograd.Function`` whose forward saves its own output for backward
(entmoid in NODE-GAM, entmax-style Functions) makes ``Function.apply`` dispatch one
``aten.detach`` of that output in C++ after the Python forward returns. No wrapper owns
it, so the completeness census counted it as a dropped op and a correct model failed
``bfs_completeness`` by one per call. The credit covers only a pure alias dispatched
by the C++ ``apply`` frame whose operand is one of that call's own returned tensors; a
pure alias of the same output taken inside the forward, an alias of any other tensor,
and every other dispatch still count.
"""

from __future__ import annotations

import dataclasses
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._user_funcs import _validate_forward_pass_torch
from torchlens._utils import _torch_compat
from torchlens.backends.torch import _function_epilogue
from torchlens.backends.torch._completeness_types import _DispatchEvent
from torchlens.backends.torch._function_epilogue import (
    credit_saved_output_aliases,
    epilogue_credit_refs,
)


class _SaveOut(torch.autograd.Function):
    """Sigmoid whose backward reads its saved output."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        y = x.sigmoid()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        (y,) = ctx.saved_tensors
        return grad * y * (1 - y)


class _AliasInForward(torch.autograd.Function):
    """Planted: an untraced pure alias of its own output INSIDE the forward."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        y = x.sigmoid()
        # A raw TensorBase.detach (no TorchLens wrapper): an untraced aten.detach of y
        # dispatched from the forward frame, not from the C++ apply epilogue.
        _torch_compat.get_tensorbase_class().detach(y)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        y = x.sigmoid()
        return grad * y * (1 - y)


class _Model(nn.Module):
    """A linear layer (so the Function's input requires grad), then the Function."""

    def __init__(self, function: type[torch.autograd.Function]) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.function = function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.function.apply(self.lin(x)) * 2


def _validate(model: nn.Module) -> bool:
    """Run forward validation with metadata checks on one fixed input."""

    torch.manual_seed(0)
    return _validate_forward_pass_torch(
        model, [torch.randn(3, 4)], {}, seed=0, validate_metadata=True
    )


@pytest.mark.parametrize("train", (False, True))
def test_function_saving_its_output_validates(train: bool) -> None:
    """The epilogue detach of the Function's own output is credited in eval and train."""

    model = _Model(_SaveOut).train(train)
    assert _validate(model), tl.validation.last_validation_failure()


def test_alias_of_own_output_inside_forward_still_fails() -> None:
    """The same pure alias taken inside the forward is not the epilogue and still counts."""

    assert not _validate(_Model(_AliasInForward))
    failure = tl.validation.last_validation_failure()
    assert failure is not None
    assert "bfs_completeness" in failure.summary()


# Unit checks on minimal witness records: what the epilogue credit refuses.

_MODULE_OWNER = SimpleNamespace(boundary_credit_is_output_scoped=True)


@dataclasses.dataclass
class _Token:
    """Stands in for the capture-accounted owner token the boundary builds."""

    census_scope: str = "expected_opaque"


def _event(
    operator: str, operand: torch.Tensor, result: torch.Tensor, **fields: Any
) -> _DispatchEvent:
    """Build one finished witness event with the given operand and result."""

    event = _DispatchEvent(operator, fields.pop("owner", _MODULE_OWNER), None)
    event.outcome = "returned_tensor"
    event.apply_epilogue_alias = fields.pop("marked", True)
    event.mutates = fields.pop("mutates", False)
    event.credit_refs = (weakref.ref(result), weakref.ref(operand))
    return event


def test_credit_takes_only_epilogue_aliases_of_the_calls_own_outputs() -> None:
    """Unmarked, non-output, output-returning, mutating and wrapper-owned events keep owners."""

    output, other, alias = torch.ones(2), torch.zeros(2), torch.ones(2)
    token = _Token()
    credited = _event("aten.detach.default", output, alias)
    refused = [
        _event("aten.detach.default", output, alias, marked=False),
        _event("aten.detach.default", other, alias),
        _event("aten.alias.default", output, output),
        _event("aten.detach.default", output, alias, mutates=True),
        _event("aten.detach.default", output, alias, owner=SimpleNamespace()),
    ]
    unowned = _event("aten.alias.default", output, alias, owner=None)
    state = SimpleNamespace(events=[credited, *refused, unowned])
    credit_saved_output_aliases(state, 0, (output, 3), token)
    assert credited.owner.census_scope == "owned"
    assert unowned.owner is credited.owner
    assert all(event.owner is not credited.owner for event in refused)
    before = _event("aten.detach.default", output, alias)
    late = SimpleNamespace(events=[before])
    credit_saved_output_aliases(late, 1, output, token)
    assert before.owner is _MODULE_OWNER


def _dispatch_from(event: _DispatchEvent, operand: torch.Tensor) -> None:
    """Stand in for the witness callback: frame 2 of the refs call is the caller of this."""

    event.credit_refs = epilogue_credit_refs(event, None, (operand,), operand.detach())


def _fake_apply(event: _DispatchEvent, operand: torch.Tensor) -> None:
    """Stand in for the ``Function.apply`` frame that enters the dispatcher."""

    _dispatch_from(event, operand)


def test_only_a_pure_alias_entered_from_the_apply_frame_is_marked() -> None:
    """A registered apply frame marks a detach; another frame or a non-alias op does not."""

    operand = torch.ones(2)
    code = _fake_apply.__code__
    _function_epilogue._APPLY_CODES.add(code)
    try:
        from_apply = _DispatchEvent("aten.detach.default", _MODULE_OWNER, None)
        _fake_apply(from_apply, operand)
        assert from_apply.apply_epilogue_alias is True
        assert len(from_apply.credit_refs) == 2
        from_forward = _DispatchEvent("aten.detach.default", _MODULE_OWNER, None)
        _dispatch_from(from_forward, operand)
        assert from_forward.apply_epilogue_alias is False
        clone = _DispatchEvent("aten.clone.default", _MODULE_OWNER, None)
        _fake_apply(clone, operand)
        assert clone.apply_epilogue_alias is False
        assert len(clone.credit_refs) == 1
    finally:
        _function_epilogue._APPLY_CODES.discard(code)


@dataclasses.dataclass(eq=False)
class _Code:
    """A code stand-in: a file under ``site-packages/torch`` and a function name."""

    co_filename: str
    co_name: str


@dataclasses.dataclass
class _Frame:
    """A frame stand-in linked to its caller."""

    f_code: _Code
    f_back: Any = None


def test_frame_walk_steps_over_only_dynamo_disable_wrappers() -> None:
    """``inner`` and ``_fn`` are stepped over; a user frame or too many wrappers end the walk."""

    torch_dir = "/venv/site-packages/torch/"
    apply_code = _Code(torch_dir + "autograd/function.py", "apply")
    inner = _Code(torch_dir + "_compile.py", "inner")
    disable = _Code(torch_dir + "_dynamo/eval_frame.py", "_fn")
    forward = _Code("/models/odst.py", "forward")
    _function_epilogue._APPLY_CODES.add(apply_code)  # type: ignore[arg-type]
    try:
        walk = _function_epilogue._dispatching_frame_is_apply
        apply_frame = _Frame(apply_code)
        assert walk(_Frame(disable, _Frame(inner, apply_frame)))  # type: ignore[arg-type]
        assert walk(apply_frame)  # type: ignore[arg-type]
        assert not walk(_Frame(forward, apply_frame))  # type: ignore[arg-type]
        assert not walk(_Frame(disable, _Frame(forward, apply_frame)))  # type: ignore[arg-type]
        deep: Any = apply_frame
        for _ in range(5):
            deep = _Frame(disable, deep)
        assert not walk(deep)
        assert not walk(None)
    finally:
        _function_epilogue._APPLY_CODES.discard(apply_code)  # type: ignore[arg-type]
