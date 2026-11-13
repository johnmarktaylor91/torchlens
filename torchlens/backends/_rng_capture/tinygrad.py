"""tinygrad random-state capture and replay (private; contract in the package docstring).

tinygrad keeps, per device, a seed tensor and a lazy two-word counter tensor
(``Tensor._device_seeds``, ``Tensor._device_rng_counters``). Every draw calls
``Tensor._next_counter(device, num)``, which lazily assigns ``counter + num``
and draws from the old value. Realizing the counter before tensors that were
drawn from an older value (random-init weights not yet realized) would change
those tensors, so this module never realizes, assigns or reads the caller's
counter while anything may still depend on it.

Capture records, for each outermost public random call, the device and the
draw's offset from the counter value at scope entry (the sum of ``num`` over
earlier draws on that device, kept on the host). The entry value itself is
read when the scope opens only if the counter is already realized (a buffer,
no pending assign, so reading computes nothing); otherwise
:func:`finalize_counters` reads it after the capture's single realize and
subtracts the scope's total offset. Until then those draws carry a reason.

Replay swaps fresh, private seed and counter tensors holding the recorded
values into the per-device tables, calls the same public function, and puts
the caller's tensor objects back in a ``finally`` block, so the caller's
counter is neither read nor advanced.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np

from . import (
    REASON_UNENCODABLE,
    RngDraw,
    RngLog,
    attach_log,
    decode_call,
    draw_from_recipe,
    encode_call,
    patched_attributes,
    session_log,
)

__all__ = ("can_replay_draw", "draw_for", "finalize_counters", "replay_draw", "rng_scope")

#: Public ``Tensor`` random functions (static and instance methods).
TENSOR_RANDOM_NAMES: tuple[str, ...] = (
    "dropout",
    "glorot_uniform",
    "kaiming_normal",
    "kaiming_uniform",
    "multinomial",
    "normal",
    "rand",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "randperm",
    "scaled_uniform",
    "uniform",
)

REASON_COUNTER_PENDING = "tinygrad_counter_unread: call finalize_counters after the realize"
REASON_NO_DRAW = "tinygrad_call_drew_nothing"
WORD = 1 << 32


def _tensor_cls() -> Any:
    """Import ``tinygrad.Tensor`` lazily."""

    from tinygrad import Tensor

    return Tensor


def _is_realized(tensor: Any) -> bool:
    """Whether ``tensor`` is a realized buffer (reading it computes nothing)."""

    uop = getattr(tensor, "uop", None)
    base = getattr(uop, "base", None)
    return getattr(base, "realized", None) is not None


def _words(tensor: Any) -> tuple[int, int]:
    """A realized two-word uint32 tensor as Python ints."""

    low, high = (int(v) for v in np.asarray(tensor.numpy()).reshape(-1)[:2])
    return low, high


def _add(words: tuple[int, int], offset: int) -> tuple[int, int]:
    """Two-word counter plus ``offset`` with 64-bit wraparound."""

    total = (words[0] + words[1] * WORD + offset) % (WORD * WORD)
    return total % WORD, total // WORD


def _sub(words: tuple[int, int], offset: int) -> tuple[int, int]:
    """Two-word counter minus ``offset`` with 64-bit wraparound."""

    return _add(words, -offset)


class _Recorder:
    """Per-scope counters: entry values, host offsets and the open call."""

    def __init__(self, log: RngLog) -> None:
        tensor = _tensor_cls()
        self.log = log
        self.entry: dict[str, tuple[int, int] | None] = {}
        self.offsets: dict[str, int] = {}
        self.open_call: list[dict[str, Any]] = []
        for device, counter in dict(tensor._device_rng_counters).items():
            self.entry[device] = _words(counter) if _is_realized(counter) else None

    def on_next_counter(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``Tensor._next_counter`` to keep host offsets per device."""

        tensor = _tensor_cls()

        def wrapper(device: str, num: int) -> Any:
            """Note the offset this draw consumes, then advance as usual."""

            if device not in tensor._device_seeds and device not in self.entry:
                self.entry[device] = (0, 0)
            offset = self.offsets.get(device, 0)
            self.offsets[device] = offset + int(num)
            if self.open_call and "device" not in self.open_call[-1]:
                self.open_call[-1].update(device=device, offset=offset)
            return original(device, num)

        return wrapper

    def make_wrapper(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap one public random function to log the outermost call."""

        if name == "_next_counter":
            return self.on_next_counter(original)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Log the call and the counter offset its first draw consumed."""

            with self.log.outermost() as outermost:
                if not outermost:
                    return original(*args, **kwargs)
                self.open_call.append({})
                try:
                    output = original(*args, **kwargs)
                finally:
                    call = self.open_call.pop()
            self.log.add(self._draw(name, args, kwargs, call), output)
            return output

        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        return wrapper

    def _draw(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any], call: dict[str, Any]
    ) -> RngDraw:
        """Build the draw record (state completed by :meth:`finish`)."""

        tensor = _tensor_cls()
        encoded = encode_call(args, kwargs, is_array=lambda v: isinstance(v, tensor))
        index = len(self.log.draws)
        if "device" not in call:
            return RngDraw("tinygrad", "counter", name, index, reason=REASON_NO_DRAW)
        state = (call["device"], call["offset"])
        if encoded is None:
            return RngDraw("tinygrad", "counter", name, index, state, reason=REASON_UNENCODABLE)
        enc_args, enc_kwargs, _ = encoded
        return RngDraw("tinygrad", "counter", name, index, state, enc_args, enc_kwargs)

    def finish(self) -> None:
        """Resolve draws whose entry counter is known; mark the rest pending."""

        tensor = _tensor_cls()
        for index, draw in enumerate(self.log.draws):
            if draw.reason is not None or len(draw.state) != 2:
                continue
            device, offset = draw.state
            entry = self.entry.get(device)
            seed = tensor._device_seeds.get(device)
            if entry is None or seed is None:
                self.log.replace_draw(index, replace(draw, reason=REASON_COUNTER_PENDING))
                continue
            self.log.replace_draw(
                index, replace(draw, state=(device, _words(seed), _add(entry, offset)))
            )


@contextmanager
def rng_scope(session: object) -> Iterator[RngLog]:
    """Log every tinygrad random call made inside the ``with`` block.

    Parameters
    ----------
    session:
        The capture session; the log is attached to it as well as yielded.

    Yields
    ------
    RngLog
        The draw log. ``finished`` is set only when the block exits normally.
    """

    tensor = _tensor_cls()
    log = RngLog("tinygrad")
    attach_log(session, log)
    recorder = _Recorder(log)
    targets = [(tensor, name) for name in (*TENSOR_RANDOM_NAMES, "_next_counter")]
    with patched_attributes(targets, recorder.make_wrapper):
        yield log
    log.pending_recorder = recorder  # type: ignore[attr-defined]
    recorder.finish()
    log.finished = True


def finalize_counters(session_or_log: object) -> None:
    """Resolve pending draws after the capture has realized its outputs.

    Reads each pending device's counter (realized by then, so reading
    computes nothing new) and subtracts the scope's total offset on that device.
    """

    log = session_or_log if isinstance(session_or_log, RngLog) else session_log(session_or_log)
    recorder = getattr(log, "pending_recorder", None)
    if log is None or recorder is None:
        return
    tensor = _tensor_cls()
    for device, entry in list(recorder.entry.items()):
        counter = tensor._device_rng_counters.get(device)
        if entry is None and counter is not None and _is_realized(counter):
            recorder.entry[device] = _sub(_words(counter), recorder.offsets.get(device, 0))
    for index, draw in enumerate(log.draws):
        if draw.reason == REASON_COUNTER_PENDING:
            log.replace_draw(index, replace(draw, reason=None))
    recorder.finish()


def draw_for(session: object, value: Any) -> RngDraw | None:
    """The logged draw that produced ``value`` in ``session``'s scope."""

    log = session_log(session)
    return None if log is None else log.draw_for(value)


def can_replay_draw(draw: RngDraw) -> str | None:
    """``None`` when ``draw`` can be replayed, else the reason."""

    if draw.reason is not None:
        return draw.reason
    if draw.kind != "counter" or len(draw.state) != 3:
        return f"tinygrad_rng_state_incomplete:{draw.kind}"
    return None


@contextmanager
def _swapped_state(device: str, seed: Sequence[int], counter: Sequence[int]) -> Iterator[None]:
    """Swap private seed and counter tensors in for ``device``; restore after."""

    tensor = _tensor_cls()
    from tinygrad import dtypes

    seeds, counters = tensor._device_seeds, tensor._device_rng_counters
    had = device in seeds
    old = (seeds.get(device), counters.get(device))
    seeds[device] = tensor(list(seed), device=device, dtype=dtypes.uint32)
    counters[device] = tensor(list(counter), device=device, dtype=dtypes.uint32)
    try:
        yield
    finally:
        if had:
            seeds[device], counters[device] = old
        else:
            seeds.pop(device, None)
            counters.pop(device, None)


def replay_draw(recipe: object, parent_values: Sequence[Any] | None) -> tuple[Any, ...]:
    """Re-run one tinygrad random call from its recorded seed and counter.

    Parameters
    ----------
    recipe:
        ``OpRecipe`` whose ``rng`` field holds the draw, or the draw itself.
    parent_values:
        Tensor arguments in call order (``self`` first for a method).

    Returns
    -------
    tuple
        The replayed output, realized while the private state is in place.
    """

    draw = draw_from_recipe(recipe)
    reason = can_replay_draw(draw)
    if reason is not None:
        raise ValueError(f"tinygrad draw {draw.op_name} cannot be replayed: {reason}")
    tensor = _tensor_cls()
    device, seed, counter = draw.state
    args, kwargs = decode_call(draw, parent_values)
    function = tensor.__dict__[draw.op_name]
    function = function.__func__ if isinstance(function, staticmethod) else function
    function = getattr(function, "__wrapped__", function)
    with _swapped_state(device, seed, counter):
        output = function(*args, **kwargs).realize()
    return (output,)
