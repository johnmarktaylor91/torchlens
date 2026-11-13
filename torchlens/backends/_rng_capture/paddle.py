"""Paddle random-state capture and replay (private; contract in the package docstring).

A Paddle random op draws from the generator of the place it runs on (the CPU
generator, plus the device generators on an accelerator place), unless it was
given a non-zero ``seed`` argument, in which case it builds its own engine
from that seed and leaves the generators alone. The scope records, before each
outermost draw, the CPU generator state and the current place's generator
states (``"global_state"``), or notes the explicit seed (``"explicit_seed"``).
States are recorded in draw order, so the sequence of state transitions within
the forward is kept.

Replay sets the recorded generator states, draws, and restores the caller's
generator states in a ``finally`` block, so the caller's random state is the
same after a replay as before it, also when the replay raises.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

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

__all__ = (
    "can_replay_draw",
    "data_reason",
    "draw_for",
    "replay_draw",
    "rng_scope",
    "snapshot_state",
)

#: Top-level ``paddle`` random functions.
PADDLE_RANDOM_NAMES: tuple[str, ...] = (
    "bernoulli",
    "binomial",
    "log_normal",
    "multinomial",
    "normal",
    "poisson",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "standard_gamma",
    "standard_normal",
    "uniform",
)

#: ``paddle.nn.functional`` functions that draw in training mode.
FUNCTIONAL_RANDOM_NAMES: tuple[str, ...] = (
    "alpha_dropout",
    "dropout",
    "dropout2d",
    "dropout3d",
    "feature_alpha_dropout",
    "gumbel_softmax",
    "rrelu",
)

#: Functions whose ``seed`` argument (at this positional index) selects a
#: private engine when non-zero.
SEED_POSITIONS: tuple[tuple[str, int], ...] = (("uniform", 4), ("gaussian", 3))

#: Why a generator-state draw cannot leave the process.
REASON_STATE_NOT_DATA = (
    "paddle_generator_state_not_data: Paddle exposes no data form of a generator state "
    "(GeneratorState cannot be pickled and shows only current_seed)"
)

#: Array arguments up to this many bytes are also kept by value.
SMALL_PARENT_BYTES = 1 << 16


def _paddle() -> Any:
    """Import ``paddle`` lazily."""

    import paddle

    return paddle


def _cpu_generator() -> Any:
    """The process CPU generator."""

    return _paddle().framework.core.default_cpu_generator()


def snapshot_state() -> tuple[Any, ...]:
    """Return ``(cpu_state, place_states)`` for the current place.

    ``place_states`` is ``None`` on a CPU place (the CPU state covers it).
    """

    paddle = _paddle()
    cpu_state = _cpu_generator().get_state()
    on_cpu = str(paddle.device.get_device()).startswith("cpu")
    return (cpu_state, None if on_cpu else tuple(paddle.get_rng_state()))


def restore_state(state: Sequence[Any]) -> None:
    """Restore a :func:`snapshot_state` result."""

    paddle = _paddle()
    cpu_state, place_states = state
    _cpu_generator().set_state(cpu_state)
    if place_states is not None:
        paddle.set_rng_state(list(place_states))


def _explicit_seed(name: str, args: Sequence[Any], kwargs: dict[str, Any]) -> int:
    """The non-zero ``seed`` a call passed, else ``0``."""

    position = dict(SEED_POSITIONS).get(name)
    if position is None:
        return 0
    seed = kwargs.get("seed", args[position] if len(args) > position else 0)
    try:
        return int(seed or 0)
    except (TypeError, ValueError):
        return 0


def _encode_other(value: Any) -> Any:
    """Encode a Paddle dtype or place as data."""

    paddle = _paddle()
    dtype_types = tuple(
        t for t in (getattr(paddle, "dtype", None), type(paddle.float32)) if isinstance(t, type)
    )
    if dtype_types and isinstance(value, dtype_types):
        return ("__framework__", "dtype", str(value).rsplit(".", 1)[-1])
    raise TypeError(type(value).__name__)


def _decode_other(value: tuple[Any, ...]) -> Any:
    """Decode :func:`_encode_other` output."""

    _, _, name = value
    return getattr(_paddle(), name)


def _record(log: RngLog, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> RngDraw:
    """Build the draw record for one call, with the state it will consume."""

    paddle = _paddle()
    seed = _explicit_seed(name.rpartition(".")[2], args, kwargs)
    kind = "explicit_seed" if seed else "global_state"
    state = (seed,) if seed else snapshot_state()
    encoded = encode_call(
        args, kwargs, is_array=lambda v: isinstance(v, paddle.Tensor), encode_other=_encode_other
    )
    if encoded is None:
        return RngDraw("paddle", kind, name, len(log.draws), state, reason=REASON_UNENCODABLE)
    enc_args, enc_kwargs, arrays = encoded
    small = all(int(a.numel()) * a.element_size() <= SMALL_PARENT_BYTES for a in arrays)
    return RngDraw(
        "paddle",
        kind,
        name,
        len(log.draws),
        state,
        enc_args,
        enc_kwargs,
        recorded_parents=tuple(a.clone() for a in arrays) if small else None,
    )


def _make_wrapper(
    log: RngLog, owner_name: str
) -> Callable[[str, Callable[..., Any]], Callable[..., Any]]:
    """Return the wrapper factory that logs each outermost draw."""

    def make(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap one Paddle random function."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Log the consumed state, then draw as the caller asked."""

            with log.outermost() as outermost:
                draw = _record(log, f"{owner_name}.{name}", args, kwargs) if outermost else None
                output = original(*args, **kwargs)
            if draw is not None:
                log.add(draw, output)
            return output

        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        return wrapper

    return make


@contextmanager
def rng_scope(session: object) -> Iterator[RngLog]:
    """Log every Paddle random draw made inside the ``with`` block.

    Parameters
    ----------
    session:
        The capture session; the log is attached to it as well as yielded.

    Yields
    ------
    RngLog
        The draw log. ``finished`` is set only when the block exits normally.
    """

    paddle = _paddle()
    log = RngLog("paddle")
    attach_log(session, log)
    functional = paddle.nn.functional
    with (
        patched_attributes(
            [(paddle, name) for name in PADDLE_RANDOM_NAMES], _make_wrapper(log, "paddle")
        ),
        patched_attributes(
            [(functional, name) for name in FUNCTIONAL_RANDOM_NAMES],
            _make_wrapper(log, "paddle.nn.functional"),
        ),
    ):
        yield log
    log.finished = True


def draw_for(session: object, value: Any) -> RngDraw | None:
    """The logged draw that produced ``value`` in ``session``'s scope."""

    log = session_log(session)
    return None if log is None else log.draw_for(value)


def can_replay_draw(draw: RngDraw) -> str | None:
    """``None`` when ``draw`` can be replayed, else the reason."""

    if draw.reason is not None:
        return draw.reason
    if draw.kind not in {"explicit_seed", "global_state"}:
        return f"paddle_rng_kind_unknown:{draw.kind}"
    return None


def data_reason(draw: RngDraw) -> str | None:
    """``None`` when ``draw`` can be saved as data and replayed after load."""

    if draw.kind == "global_state":
        return REASON_STATE_NOT_DATA
    return can_replay_draw(draw)


def _function(op_name: str) -> Callable[..., Any]:
    """Resolve ``paddle.<name>`` or ``paddle.nn.functional.<name>`` unwrapped."""

    paddle = _paddle()
    owner_name, _, name = op_name.rpartition(".")
    owner = paddle.nn.functional if owner_name == "paddle.nn.functional" else paddle
    function = getattr(owner, name)
    return getattr(function, "__wrapped__", function)


def replay_draw(recipe: object, parent_values: Sequence[Any] | None) -> tuple[Any, ...]:
    """Re-run one Paddle draw from its recorded generator state.

    Parameters
    ----------
    recipe:
        ``OpRecipe`` whose ``rng`` field holds the draw, or the draw itself.
    parent_values:
        Tensor arguments in call order, or ``None`` for the recorded ones.

    Returns
    -------
    tuple
        The replayed output.
    """

    draw = draw_from_recipe(recipe)
    reason = can_replay_draw(draw)
    if reason is not None:
        raise ValueError(f"paddle draw {draw.op_name} cannot be replayed: {reason}")
    args, kwargs = decode_call(draw, parent_values, decode_other=_decode_other)
    function = _function(draw.op_name)
    caller = snapshot_state()
    try:
        if draw.kind == "global_state":
            restore_state(draw.state)
        return (function(*args, **kwargs),)
    finally:
        restore_state(caller)
