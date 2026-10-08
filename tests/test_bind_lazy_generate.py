"""``spec.bind(model)`` over lazy outputs: generators, lazy tails, async refusals.

A bound ``generate`` whose model returns a generator runs its forwards while
the CALLER iterates, after ``generate()`` has returned. The binding must keep
steering every one of those forwards, settle ``.last_report`` only when the
iteration ends (exhaustion, ``close()``, or an exception), and leave no hook
on the model afterward. Async lazy outputs cannot be held armed across an
event loop and refuse typed (``bind_lazy_output``). Every case uses a
non-idempotent action (``tl.add``) so a skipped or doubled step changes the
numbers.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import BindingRuntimeError
from torchlens.intervention.steering import steer_generate


class _Doubler(nn.Module):
    """``y = 2 * relu(x)``; a relu ``+1`` steer gives ``2 * (relu(x) + 1)``."""

    def __init__(self) -> None:
        """Build the single relu block."""

        super().__init__()
        self.block = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run relu then double."""

        return self.block(x) * 2


class _PureGen(_Doubler):
    """``generate`` is itself a generator: no forward runs inside the call."""

    def generate(self, x: torch.Tensor, steps: int = 2) -> Iterator[torch.Tensor]:
        """Yield ``steps`` chained forwards."""

        for _ in range(steps):
            x = self(x)
            yield x


class _LazyTail(_Doubler):
    """One eager forward inside ``generate``, then a lazy generator tail."""

    def generate(self, x: torch.Tensor) -> Iterator[torch.Tensor]:
        """Run the first forward now and return a generator for the second."""

        first = self(x)

        def _later() -> Iterator[torch.Tensor]:
            """Yield the eager result, then run one more forward lazily."""

            yield first
            yield self(first)

        return _later()


class _Failing(_Doubler):
    """A generator that raises after its first step."""

    def generate(self, x: torch.Tensor) -> Iterator[torch.Tensor]:
        """Yield one forward, then raise."""

        yield self(x)
        raise ValueError("mid-iteration failure")


class _NoForwardGen(_Doubler):
    """A generator that yields without running any forward."""

    def generate(self, x: torch.Tensor) -> Iterator[torch.Tensor]:
        """Yield the input untouched."""

        yield x


class _StepIterator:
    """A plain (non-generator) lazy iterator that runs one forward per step."""

    def __init__(self, model: nn.Module, x: torch.Tensor, steps: int) -> None:
        """Hold the model, the running value, and the remaining step count."""

        self.model = model
        self.x = x
        self.remaining = steps

    def __iter__(self) -> _StepIterator:
        """Return this iterator."""

        return self

    def __next__(self) -> torch.Tensor:
        """Run one forward, or stop."""

        if self.remaining == 0:
            raise StopIteration
        self.remaining -= 1
        self.x = self.model(self.x)
        return self.x


class _IterGen(_Doubler):
    """``generate`` returns a plain lazy iterator object."""

    def generate(self, x: torch.Tensor) -> _StepIterator:
        """Return the lazy step iterator."""

        return _StepIterator(self, x, 2)


class _AsyncGen(_Doubler):
    """``generate`` returns an async generator."""

    def generate(self, x: torch.Tensor) -> Any:
        """Return an async generator over one forward."""

        async def _agen() -> Any:
            """Yield one forward asynchronously."""

            yield self(x)

        return _agen()


class _CoroGen(_Doubler):
    """``generate`` is a coroutine function."""

    async def generate(self, x: torch.Tensor) -> torch.Tensor:
        """Run one forward when awaited."""

        return self(x)


def _add_one() -> Any:
    """The non-idempotent steer: relu output ``+1``."""

    return tl.when(tl.func("relu"), tl.add(1))


def _values(items: Any) -> list[list[float]]:
    """Round every yielded tensor into a plain list for exact comparison."""

    return [[round(float(v), 4) for v in t.detach().flatten().tolist()] for t in items]


def _assert_clean(model: nn.Module) -> None:
    """No TorchLens hook or torch-function mode survives on the model/thread."""

    for name, sub in model.named_modules():
        assert len(sub._forward_hooks) == 0, f"forward hook left on {name!r}"
        assert len(sub._forward_pre_hooks) == 0, f"forward pre-hook left on {name!r}"
    assert torch.overrides._get_current_function_mode_stack() == []


def test_pure_generator_steers_every_step() -> None:
    """Every forward run during iteration is steered; the report counts them."""

    model = _PureGen()
    bound = _add_one().bind(model)
    gen = bound.generate(torch.ones(1))
    assert bound.last_report is None, "no settled report while the output is open"
    assert _values(gen) == [[4.0], [10.0]]
    report = bound.last_report
    assert report is not None
    assert report.status == "fired"
    assert report.door == "generate"
    assert report.fire_count == 2
    assert report.zero_fire_rule_ids == ()
    assert report.cleanup == "removed"
    _assert_clean(model)


def test_lazy_tail_steers_eager_and_lazy_forwards() -> None:
    """The eager forward and the lazily run forward are both steered once."""

    model = _LazyTail()
    bound = _add_one().bind(model)
    assert _values(bound.generate(torch.ones(1))) == [[4.0], [10.0]]
    assert bound.last_report is not None
    assert bound.last_report.fire_count == 2
    assert bound.last_report.status == "fired"
    _assert_clean(model)


def test_steering_is_scoped_to_the_generator_body() -> None:
    """Between steps no hook is armed: the caller's own forward runs unsteered."""

    model = _PureGen()
    bound = _add_one().bind(model)
    gen = bound.generate(torch.ones(1), steps=2)
    assert _values([next(gen)]) == [[4.0]]
    _assert_clean(model)
    assert _values([model(torch.ones(1))]) == [[2.0]]
    assert _values([next(gen)]) == [[10.0]]
    with pytest.raises(StopIteration):
        next(gen)
    assert bound.last_report is not None
    assert bound.last_report.fire_count == 2


def test_open_lazy_output_holds_the_binding() -> None:
    """The binding stays serial while its lazy output is open, then frees."""

    model = _PureGen()
    bound = _add_one().bind(model)
    gen = bound.generate(torch.ones(1))
    next(gen)
    with pytest.raises(BindingRuntimeError) as excinfo:
        bound(torch.ones(1))
    assert excinfo.value.fields["code"] == "binding_reentrant_call"
    list(gen)
    assert _values([bound(torch.ones(1))]) == [[4.0]]
    assert bound.last_report is not None
    assert bound.last_report.fire_count == 1
    assert bound.last_report.door == "call"


def test_early_close_settles_and_removes_hooks() -> None:
    """``close()`` after one step settles a one-fire report and frees the binding."""

    model = _PureGen()
    bound = _add_one().bind(model)
    gen = bound.generate(torch.ones(1), steps=3)
    assert _values([next(gen)]) == [[4.0]]
    gen.close()
    report = bound.last_report
    assert report is not None
    assert report.status == "fired"
    assert report.fire_count == 1
    assert report.error is None
    _assert_clean(model)
    with pytest.raises(StopIteration):
        next(gen)
    assert _values([bound(torch.ones(1))]) == [[4.0]]


def test_exception_mid_iteration_settles_error_report() -> None:
    """A generator failure propagates; the report records it and the fires so far."""

    model = _Failing()
    bound = _add_one().bind(model)
    gen = bound.generate(torch.ones(1))
    assert _values([next(gen)]) == [[4.0]]
    with pytest.raises(ValueError, match="mid-iteration failure"):
        next(gen)
    report = bound.last_report
    assert report is not None
    assert report.status == "error"
    assert report.fire_count == 1
    assert report.error is not None and "ValueError" in report.error
    _assert_clean(model)
    assert _values([bound(torch.ones(1))]) == [[4.0]]


def test_throw_into_lazy_output_settles_error_report() -> None:
    """``throw()`` reaches the model's generator and settles an error report."""

    model = _PureGen()
    bound = _add_one().bind(model)
    gen = bound.generate(torch.ones(1), steps=3)
    next(gen)
    with pytest.raises(KeyError):
        gen.throw(KeyError("stop"))
    assert bound.last_report is not None
    assert bound.last_report.status == "error"
    assert bound.last_report.fire_count == 1
    _assert_clean(model)


def test_zero_fire_fails_closed_at_exhaustion() -> None:
    """A lazy output that never fires refuses ``bind_zero_fire`` when it ends."""

    model = _NoForwardGen()
    bound = _add_one().bind(model)
    gen = bound.generate(torch.ones(1))
    next(gen)
    with pytest.raises(BindingRuntimeError) as excinfo:
        next(gen)
    assert excinfo.value.fields["code"] == "bind_zero_fire"
    assert bound.last_report is not None
    assert bound.last_report.status == "no_fire"
    _assert_clean(model)


def test_zero_fire_disclose_settles_quietly() -> None:
    """Under ``on_zero_fire='disclose'`` exhaustion settles a ``no_fire`` report."""

    model = _NoForwardGen()
    bound = _add_one().bind(model, on_zero_fire="disclose")
    assert _values(bound.generate(torch.ones(1))) == [[1.0]]
    assert bound.last_report is not None
    assert bound.last_report.status == "no_fire"
    assert bound.last_report.fire_count == 0


def test_plain_lazy_iterator_steers_every_step() -> None:
    """A non-generator lazy iterator is held armed per step like a generator."""

    model = _IterGen()
    bound = _add_one().bind(model)
    assert _values(bound.generate(torch.ones(1))) == [[4.0], [10.0]]
    assert bound.last_report is not None
    assert bound.last_report.fire_count == 2
    _assert_clean(model)


def test_abandoned_lazy_output_releases_the_binding() -> None:
    """Dropping an unstarted lazy output settles the report and frees the binding."""

    model = _PureGen()
    bound = _add_one().bind(model, on_zero_fire="disclose")
    gen = bound.generate(torch.ones(1))
    del gen
    gc.collect()
    assert bound.last_report is not None
    assert bound.last_report.fire_count == 0
    _assert_clean(model)
    assert _values([bound(torch.ones(1))]) == [[4.0]]


@pytest.mark.parametrize("model_cls", [_AsyncGen, _CoroGen])
def test_async_lazy_output_refuses_typed(model_cls: type[_Doubler]) -> None:
    """Async lazy outputs cannot be held armed across an event loop: typed refusal."""

    model = model_cls()
    bound = _add_one().bind(model, on_zero_fire="disclose")
    with pytest.raises(BindingRuntimeError) as excinfo:
        bound.generate(torch.ones(1))
    assert excinfo.value.fields["code"] == "bind_lazy_output"
    assert bound.last_report is not None
    assert bound.last_report.status == "error"
    _assert_clean(model)
    assert _values([bound(torch.ones(1))]) == [[4.0]]


def test_steer_generate_refuses_lazy_generate() -> None:
    """The turnkey wrapper needs a settled report at return: lazy refuses typed."""

    model = _PureGen()
    with pytest.raises(BindingRuntimeError) as excinfo:
        steer_generate(model, torch.ones(1), _add_one())
    assert excinfo.value.fields["code"] == "bind_lazy_output"
    _assert_clean(model)
