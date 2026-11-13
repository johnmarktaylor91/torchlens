"""MLX random-state capture and replay (private; contract in the package docstring).

An MLX random function either takes an explicit ``key`` array or draws from the
global key in ``mx.random.state``: it splits that key, stores the first half
back as the new global key and draws with the second half. The scope records
the explicit key, or the global key array as it was just before the call. The
snapshot is the lazy array object itself, so recording it evaluates nothing
and cannot force a pending draw or weight; it is converted to NumPy only when
the scope closes, after the forward has produced its outputs.

Replay passes the recorded key (or the second half of the split of the
recorded global key) as the explicit ``key`` argument, which never reads or
writes ``mx.random.state``, so the caller's random state is untouched.
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

__all__ = ("can_replay_draw", "draw_for", "replay_draw", "rng_scope")

#: ``mlx.core.random`` functions that draw, with the positional index of ``key``.
KEY_POSITIONS: tuple[tuple[str, int], ...] = (
    ("bernoulli", 2),
    ("categorical", 4),
    ("gumbel", 2),
    ("laplace", 4),
    ("multivariate_normal", 4),
    ("normal", 4),
    ("permutation", 2),
    ("randint", 4),
    ("truncated_normal", 4),
    ("uniform", 4),
)

#: Array arguments up to this many bytes are also kept by value.
SMALL_PARENT_BYTES = 1 << 16


def _mx() -> Any:
    """Import ``mlx.core`` lazily."""

    import mlx.core as mx

    return mx


def _key_position(name: str) -> int:
    """Positional index of ``key`` for the random function ``name``."""

    return dict(KEY_POSITIONS)[name]


def _split_key(args: Sequence[Any], kwargs: dict[str, Any], name: str) -> tuple[list[Any], Any]:
    """Return ``(args without key, key)``, moving a positional key out."""

    position = _key_position(name)
    args_list = list(args)
    key = kwargs.pop("key", None)
    if len(args_list) > position:
        key = args_list[position]
        if len(args_list) > position + 1:
            kwargs.setdefault("stream", args_list[position + 1])
        args_list = args_list[:position]
    return args_list, key


def _encode_other(value: Any) -> Any:
    """Encode an MLX dtype or device as data."""

    mx = _mx()
    if isinstance(value, mx.Dtype):
        return ("__framework__", "dtype", str(value).rsplit(".", 1)[-1])
    if isinstance(value, mx.Device):
        return ("__framework__", "device", str(value.type).rsplit(".", 1)[-1])
    raise TypeError(type(value).__name__)


def _decode_other(value: tuple[Any, ...]) -> Any:
    """Decode :func:`_encode_other` output."""

    mx = _mx()
    _, kind, name = value
    if kind == "dtype":
        return getattr(mx, name)
    return mx.cpu if name == "cpu" else mx.gpu


def _record(log: RngLog, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> RngDraw:
    """Build the draw record for one call (state still a live array)."""

    mx = _mx()
    kwargs = dict(kwargs)
    args_list, key = _split_key(args, kwargs, name)
    kind = "explicit_key" if key is not None else "global_state"
    state = (key if key is not None else mx.random.state[0],)
    encoded = encode_call(
        args_list, kwargs, is_array=lambda v: isinstance(v, mx.array), encode_other=_encode_other
    )
    if encoded is None:
        return RngDraw("mlx", kind, name, len(log.draws), state, reason=REASON_UNENCODABLE)
    enc_args, enc_kwargs, arrays = encoded
    small = all(int(a.nbytes) <= SMALL_PARENT_BYTES for a in arrays)
    return RngDraw(
        "mlx",
        kind,
        name,
        len(log.draws),
        state,
        enc_args,
        enc_kwargs,
        recorded_parents=tuple(arrays) if small else None,
    )


def _make_wrapper(log: RngLog) -> Callable[[str, Callable[..., Any]], Callable[..., Any]]:
    """Return the wrapper factory that logs each outermost draw."""

    def make(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap one ``mlx.core.random`` function."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Log the consumed key, then draw as the caller asked."""

            with log.outermost() as outermost:
                draw = _record(log, name, args, kwargs) if outermost else None
                output = original(*args, **kwargs)
            if draw is not None:
                log.add(draw, output)
            return output

        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        return wrapper

    return make


def _to_data(draw: RngDraw) -> RngDraw:
    """Convert a draw's live arrays to NumPy (evaluates keys and small parents only)."""

    state = tuple(np.array(item) for item in draw.state)
    parents = (
        None
        if draw.recorded_parents is None
        else tuple(np.array(item) for item in draw.recorded_parents)
    )
    return replace(draw, state=state, recorded_parents=parents)


@contextmanager
def rng_scope(session: object) -> Iterator[RngLog]:
    """Log every MLX random draw made inside the ``with`` block.

    Parameters
    ----------
    session:
        The capture session; the log is attached to it as well as yielded.

    Yields
    ------
    RngLog
        The draw log. ``finished`` is set only when the block exits normally.
    """

    mx = _mx()
    log = RngLog("mlx")
    attach_log(session, log)
    targets = [(mx.random, name) for name, _ in KEY_POSITIONS]
    with patched_attributes(targets, _make_wrapper(log)):
        yield log
    for index, draw in enumerate(log.draws):
        log.replace_draw(index, _to_data(draw))
    log.finished = True


def draw_for(session: object, value: Any) -> RngDraw | None:
    """The logged draw that produced ``value`` in ``session``'s scope."""

    log = session_log(session)
    return None if log is None else log.draw_for(value)


def can_replay_draw(draw: RngDraw) -> str | None:
    """``None`` when ``draw`` can be replayed, else the reason."""

    if draw.reason is not None:
        return draw.reason
    if draw.kind not in {"explicit_key", "global_state"}:
        return f"mlx_rng_kind_unknown:{draw.kind}"
    return None


def replay_key(draw: RngDraw) -> Any:
    """The explicit key that reproduces ``draw``."""

    mx = _mx()
    key = mx.array(np.asarray(draw.state[0]))
    if draw.kind == "explicit_key":
        return key
    return mx.random.split(key)[1]


def replay_draw(recipe: object, parent_values: Sequence[Any] | None) -> tuple[Any, ...]:
    """Re-run one MLX draw from its recorded key.

    Parameters
    ----------
    recipe:
        ``OpRecipe`` whose ``rng`` field holds the draw, or the draw itself.
    parent_values:
        Array arguments in call order, or ``None`` for the recorded ones.

    Returns
    -------
    tuple
        The replayed output.
    """

    draw = draw_from_recipe(recipe)
    reason = can_replay_draw(draw)
    if reason is not None:
        raise ValueError(f"mlx draw {draw.op_name} cannot be replayed: {reason}")
    mx = _mx()
    args, kwargs = decode_call(draw, parent_values, decode_other=_decode_other)
    kwargs["key"] = replay_key(draw)
    function = getattr(mx.random, draw.op_name)
    original = getattr(function, "__wrapped__", function)
    return (original(*args, **kwargs),)
