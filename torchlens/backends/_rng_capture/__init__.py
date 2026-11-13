"""Random-state capture and replay for the non-PyTorch backends (private).

One module per backend (``mlx``, ``paddle``, ``tf``, ``tinygrad``), looked up by
backend name through :func:`rng_module`, each exposing the same functions:

``rng_scope(session)``
    Context manager the capture path enters around the forward. While it is
    open, every random draw the framework makes through its public entry points
    is logged with the random state it consumed, in draw order. The caller's
    random state advances exactly as an eager forward would advance it.
``draw_for(session, value)``
    The :class:`RngDraw` whose output is ``value`` (``None`` for a value no
    logged draw produced). The capture path stores it with the op record.
``can_replay_draw(draw)``
    ``None``, or the reason this draw cannot be replayed.
``replay_draw(recipe, parent_values)``
    Re-run one draw from its recorded state. ``recipe`` is an ``OpRecipe``
    whose ``rng`` field holds the draw (or the draw itself, or its
    :meth:`RngDraw.to_data` mapping). ``parent_values`` are the op's array
    arguments in call order (``None`` uses the values recorded at capture where
    the draw kept them). Returns a tuple of outputs. The caller's random state
    is the same after the call as before it, also when the replay raises.

Everything recorded here is data: numbers, strings, tuples and NumPy arrays.
Framework handles are kept only until :meth:`RngLog.close` converts them.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import Any

import numpy as np

__all__ = (
    "ParentSlot",
    "RngDraw",
    "RngLog",
    "decode_call",
    "draw_from_recipe",
    "encode_call",
    "patched_attributes",
    "rng_module",
    "session_log",
)

#: Backends with a random-state capture module.
RNG_CAPTURE_BACKENDS: frozenset[str] = frozenset({"mlx", "paddle", "tf", "tinygrad"})

#: Session attribute that holds the open or finished :class:`RngLog`.
SESSION_LOG_ATTR = "_torchlens_rng_log"

#: Reason given when a call argument has no data-only encoding.
REASON_UNENCODABLE = "rng_argument_unencodable"


def rng_module(backend: str) -> ModuleType:
    """Return the random-state capture module for ``backend``.

    Parameters
    ----------
    backend:
        Backend name (``"mlx"``, ``"paddle"``, ``"tf"``, ``"tinygrad"``).

    Returns
    -------
    ModuleType
        The backend's capture module.
    """

    if backend not in RNG_CAPTURE_BACKENDS:
        raise KeyError(f"no random-state capture module for backend {backend!r}")
    return importlib.import_module(f"{__name__}.{backend}")


@dataclass(frozen=True)
class ParentSlot:
    """Placeholder for the ``index``-th array argument of a recorded call."""

    index: int


@dataclass(frozen=True)
class RngDraw:
    """One random draw and the random state it consumed (data only).

    Parameters
    ----------
    backend:
        Backend name.
    kind:
        How the state was obtained: ``"global_state"``, ``"explicit_key"``,
        ``"explicit_seed"``, ``"generator_state"``, ``"stateless"``,
        ``"legacy_seeded"`` or ``"counter"``.
    op_name:
        Framework function or raw op that drew.
    ordinal:
        Position of the draw within its scope (0-based).
    state:
        The consumed state as NumPy arrays and numbers.
    args, kwargs:
        Call arguments with array arguments replaced by :class:`ParentSlot`.
    recorded_parents:
        Array arguments kept by value when they are small, else ``None``.
    reason:
        ``None`` when the draw can be replayed, else why not.
    """

    backend: str
    kind: str
    op_name: str
    ordinal: int
    state: tuple[Any, ...] = ()
    args: tuple[Any, ...] = ()
    kwargs: tuple[tuple[str, Any], ...] = ()
    recorded_parents: tuple[Any, ...] | None = None
    reason: str | None = None

    def to_data(self) -> dict[str, Any]:
        """Return a plain mapping (for an ``OpRecipe.rng`` payload)."""

        return {
            "backend": self.backend,
            "kind": self.kind,
            "op_name": self.op_name,
            "ordinal": self.ordinal,
            "state": tuple(self.state),
            "args": tuple(self.args),
            "kwargs": tuple(self.kwargs),
            "recorded_parents": self.recorded_parents,
            "reason": self.reason,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> RngDraw:
        """Rebuild a draw from :meth:`to_data` output."""

        return cls(
            backend=str(data["backend"]),
            kind=str(data["kind"]),
            op_name=str(data["op_name"]),
            ordinal=int(data["ordinal"]),
            state=tuple(data.get("state", ())),
            args=tuple(data.get("args", ())),
            kwargs=tuple(tuple(item) for item in data.get("kwargs", ())),
            recorded_parents=data.get("recorded_parents"),
            reason=data.get("reason"),
        )


def draw_from_recipe(recipe: object) -> RngDraw:
    """Return the :class:`RngDraw` an ``OpRecipe`` (or draw, or mapping) carries."""

    payload = getattr(recipe, "rng", recipe)
    if isinstance(payload, RngDraw):
        return payload
    if isinstance(payload, Mapping):
        return RngDraw.from_data(payload)
    raise TypeError(f"recipe carries no random draw: {type(recipe).__name__}")


@dataclass
class RngLog:
    """Draws logged by one ``rng_scope`` in draw order.

    ``finished`` is ``False`` while the scope is open and stays ``False`` when
    the forward raised, so a partial log is never mistaken for a complete one.
    """

    backend: str
    draws: list[RngDraw] = field(default_factory=list)
    finished: bool = False
    _by_output: dict[int, int] = field(default_factory=dict)
    _outputs: list[Any] = field(default_factory=list)
    _depth: int = 0

    def add(self, draw: RngDraw, output: Any) -> RngDraw:
        """Append ``draw`` and index it by the identity of ``output``."""

        index = len(self.draws)
        self.draws.append(replace(draw, ordinal=index))
        for item in output if isinstance(output, (tuple, list)) else (output,):
            self._by_output[id(item)] = index
            self._outputs.append(item)
        return self.draws[index]

    def replace_draw(self, index: int, draw: RngDraw) -> None:
        """Replace the draw at ``index`` (used when the state is finalized)."""

        self.draws[index] = draw

    def draw_for(self, value: Any) -> RngDraw | None:
        """The draw whose output is ``value``, or ``None``."""

        index = self._by_output.get(id(value))
        return None if index is None else self.draws[index]

    def output_of(self, index: int) -> Any:
        """The first live output recorded for draw ``index`` (``None`` if none)."""

        for key, position in self._by_output.items():
            if position == index:
                for item in self._outputs:
                    if id(item) == key:
                        return item
        return None

    @contextmanager
    def outermost(self) -> Iterator[bool]:
        """Yield whether this is the outermost intercepted draw call."""

        self._depth += 1
        try:
            yield self._depth == 1
        finally:
            self._depth -= 1


def session_log(session: object) -> RngLog | None:
    """Return the :class:`RngLog` an ``rng_scope`` attached to ``session``."""

    log = getattr(session, SESSION_LOG_ATTR, None)
    return log if isinstance(log, RngLog) else None


def attach_log(session: object, log: RngLog) -> None:
    """Attach ``log`` to ``session`` when the session accepts attributes."""

    if session is None:
        return
    try:
        setattr(session, SESSION_LOG_ATTR, log)
    except (AttributeError, TypeError):
        return


def _encode_value(value: Any, is_array: Callable[[Any], bool], arrays: list[Any]) -> Any:
    """Encode one argument; arrays become :class:`ParentSlot` entries."""

    if is_array(value):
        arrays.append(value)
        return ParentSlot(len(arrays) - 1)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return tuple(_encode_value(item, is_array, arrays) for item in value)
    raise TypeError(type(value).__name__)


def encode_call(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    is_array: Callable[[Any], bool],
    encode_other: Callable[[Any], Any] | None = None,
) -> tuple[tuple[Any, ...], tuple[tuple[str, Any], ...], list[Any]] | None:
    """Encode a call as data, replacing arrays with :class:`ParentSlot`.

    Parameters
    ----------
    args, kwargs:
        The call's arguments.
    is_array:
        Whether a value is a framework array (a parent).
    encode_other:
        Encoder for framework objects such as dtypes; raises ``TypeError``
        for a value it cannot encode.

    Returns
    -------
    tuple | None
        ``(args, kwargs, arrays)`` or ``None`` when an argument has no data
        encoding.
    """

    arrays: list[Any] = []

    def encode(value: Any) -> Any:
        """Encode with the framework hook first."""

        if encode_other is not None and not is_array(value):
            try:
                return encode_other(value)
            except TypeError:
                pass
        return _encode_value(value, is_array, arrays)

    try:
        enc_args = tuple(encode(item) for item in args)
        enc_kwargs = tuple((key, encode(kwargs[key])) for key in sorted(kwargs))
    except TypeError:
        return None
    return enc_args, enc_kwargs, arrays


def decode_call(
    draw: RngDraw,
    parent_values: Sequence[Any] | None,
    *,
    decode_other: Callable[[Any], Any] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Rebuild ``(args, kwargs)`` for ``draw`` from ``parent_values``.

    ``parent_values`` of ``None`` uses the arrays recorded at capture.
    """

    parents = draw.recorded_parents if parent_values is None else tuple(parent_values)

    def decode(value: Any) -> Any:
        """Substitute parents and decode framework objects."""

        if isinstance(value, ParentSlot):
            if parents is None or value.index >= len(parents):
                raise ValueError(
                    f"{draw.op_name} replay needs array argument {value.index}, "
                    f"got {0 if parents is None else len(parents)} parent values"
                )
            return parents[value.index]
        if isinstance(value, tuple):
            if decode_other is not None and value[:1] == ("__framework__",):
                return decode_other(value)
            return tuple(decode(item) for item in value)
        return value

    return [decode(item) for item in draw.args], {key: decode(val) for key, val in draw.kwargs}


@contextmanager
def patched_attributes(
    targets: Sequence[tuple[object, str]],
    make_wrapper: Callable[[str, Callable[..., Any]], Callable[..., Any]],
) -> Iterator[None]:
    """Replace each ``(owner, name)`` with a wrapper for the ``with`` block.

    Missing attributes are skipped. On exit each attribute is restored only if
    it still holds this block's wrapper, so a wrapper installed after ours by
    someone else is never clobbered, and restoration runs when the block
    raises.
    """

    installed: list[tuple[object, str, Any, Any]] = []
    try:
        for owner, name in targets:
            original = _raw_attribute(owner, name)
            function = original.__func__ if isinstance(original, staticmethod) else original
            if function is None or not callable(function):
                continue
            wrapper: Any = make_wrapper(name, function)
            if isinstance(original, staticmethod):
                wrapper = staticmethod(wrapper)
            setattr(owner, name, wrapper)
            installed.append((owner, name, original, wrapper))
        yield
    finally:
        for owner, name, original, wrapper in reversed(installed):
            if _raw_attribute(owner, name) is wrapper:
                setattr(owner, name, original)


def _raw_attribute(owner: object, name: str) -> Any:
    """``owner.name`` as stored: a class's own descriptor, else the attribute."""

    if isinstance(owner, type):
        return owner.__dict__.get(name)
    return getattr(owner, name, None)
