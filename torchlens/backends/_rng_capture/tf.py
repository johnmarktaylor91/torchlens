"""TensorFlow random-state capture and replay (private; contract in the package docstring).

TensorFlow eager random ops come in three families, told apart by op type:

``"stateless"``
    ``Stateless*`` kernels take their seed (or key and counter) as inputs. They
    are pure: replay runs the raw op on the parent values.
``"generator_state"``
    ``Stateful*`` kernels read and advance a ``tf.random.Generator`` state
    variable. The scope wraps the generated op functions to read the variable
    just before the kernel runs; replay runs the raw op against a fresh
    variable holding that state, so the caller's generator is never touched.
``"legacy_seeded"``
    ``RandomStandardNormal``, ``RandomUniform``, ``TruncatedNormal`` and
    ``RandomUniformInt`` carry ``seed``/``seed2`` attributes. Each eager kernel
    keeps a hidden Philox position that advances on every call; the scope
    counts the calls it sees per kernel, and replay runs the matching
    ``Stateless*V2`` kernel at key ``seed``, counter ``(position, seed2)``,
    which leaves the cached kernel (and so the caller's sequence) alone. When
    the scope closes, each legacy draw is re-run once at its counted position;
    a draw whose kernel had already run before the scope (its position is
    hidden) does not reproduce and is marked unreplayable rather than guessed.
    A legacy op with ``seed == seed2 == 0`` took an OS seed: TensorFlow sets
    that when no global seed is set, and it cannot be replayed.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np

from ... import _state
from . import (
    RngDraw,
    RngLog,
    attach_log,
    draw_from_recipe,
    patched_attributes,
    session_log,
)

__all__ = ("can_replay_draw", "draw_for", "replay_draw", "rng_scope")

#: Legacy seeded kernels and the stateless kernel that reproduces each.
LEGACY_TO_STATELESS: tuple[tuple[str, str], ...] = (
    ("RandomStandardNormal", "StatelessRandomNormalV2"),
    ("RandomUniform", "StatelessRandomUniformV2"),
    ("TruncatedNormal", "StatelessTruncatedNormalV2"),
    ("RandomUniformInt", "StatelessRandomUniformIntV2"),
)

#: Philox samples a legacy kernel reserves per output element.
LEGACY_SAMPLES_PER_ELEMENT = 256

#: Philox algorithm id of ``Stateless*V2`` kernels.
PHILOX_ALG = 1

#: Generated stateful op functions whose first argument is the generator state.
STATEFUL_FUNCTIONS: tuple[tuple[str, str], ...] = (
    ("stateful_standard_normal_v2", "StatefulStandardNormalV2"),
    ("stateful_truncated_normal", "StatefulTruncatedNormal"),
    ("stateful_uniform", "StatefulUniform"),
    ("stateful_uniform_full_int", "StatefulUniformFullInt"),
    ("stateful_uniform_int", "StatefulUniformInt"),
    ("rng_read_and_skip", "RngReadAndSkip"),
)

REASON_UNSEEDED = "tf_legacy_unseeded: set a global seed (tf.random.set_seed) to replay"
REASON_POSITION = "tf_legacy_kernel_ran_before_trace: its Philox position is hidden"
REASON_GENERATOR = "tf_generator_state_not_observed"


def _tf() -> Any:
    """Import ``tensorflow`` lazily."""

    import tensorflow as tf

    return tf


def _gen_stateful_ops() -> Any | None:
    """TensorFlow's generated stateful random op module, if present."""

    try:
        from tensorflow.python.ops import gen_stateful_random_ops
    except ImportError:
        return None
    return gen_stateful_random_ops


def _attr_map(attrs: Any) -> dict[str, Any]:
    """Normalize op-callback attributes into a mapping."""

    if isinstance(attrs, dict):
        return dict(attrs)
    items = tuple(attrs or ())
    return {str(items[i]): items[i + 1] for i in range(0, len(items) - 1, 2)}


def _attr_data(value: Any) -> int | float | str | None:
    """An attribute value as data (a ``DType`` becomes its enum), else ``None``."""

    enum = getattr(value, "as_datatype_enum", None)
    if enum is not None:
        return int(enum)
    if isinstance(value, bytes):
        return value.decode()
    return value if isinstance(value, (int, float, str)) else None


def _op_text(op_type: Any) -> str:
    """Op type as ``str``."""

    return op_type.decode() if isinstance(op_type, bytes) else str(op_type)


def family(op_type: str) -> str | None:
    """``"stateless"``, ``"generator_state"``, ``"legacy_seeded"`` or ``None``."""

    if op_type in dict(LEGACY_TO_STATELESS):
        return "legacy_seeded"
    if op_type in {op for _, op in STATEFUL_FUNCTIONS}:
        return "generator_state"
    random_like = "Random" in op_type or "TruncatedNormal" in op_type
    if op_type.startswith("Stateless") and random_like:
        return "stateless"
    return None


def _numpy_inputs(inputs: Sequence[Any]) -> tuple[Any, ...] | None:
    """Small non-resource inputs as NumPy (``None`` if any is unreadable)."""

    tf = _tf()
    values = []
    for item in inputs:
        if getattr(item, "dtype", None) == tf.resource:
            values.append(None)
            continue
        try:
            values.append(np.array(item))
        except (TypeError, ValueError):
            return None
    return tuple(values)


class _Recorder:
    """Per-scope state: kernel positions, pending generator states, the log."""

    def __init__(self, log: RngLog) -> None:
        self.log = log
        self.positions: dict[tuple[Any, ...], int] = {}
        self.pending: list[tuple[str, Any, int]] = []

    def on_op(
        self, op_type: Any, inputs: Any, attrs: Any, outputs: Any, *_: Any, **__: Any
    ) -> None:
        """Op callback: log one random kernel execution."""

        name = _op_text(op_type)
        kind = family(name)
        if kind is None or not outputs:
            return None
        attr = _attr_map(attrs)
        with _state.pause_logging():
            parents = _numpy_inputs(tuple(inputs or ()))
        clean = tuple(
            (k, _attr_data(v)) for k, v in sorted(attr.items()) if _attr_data(v) is not None
        )
        draw = RngDraw("tf", kind, name, len(self.log.draws), (), kwargs=clean)
        draw = replace(draw, recorded_parents=parents)
        if kind == "legacy_seeded":
            draw = self._legacy(draw, attr, tuple(outputs))
        elif kind == "generator_state":
            draw = self._generator(draw)
        self.log.add(draw, tuple(outputs)[0])
        return None

    def _legacy(self, draw: RngDraw, attr: dict[str, Any], outputs: tuple[Any, ...]) -> RngDraw:
        """Attach ``(seed, seed2, position)`` and advance the kernel position."""

        seed, seed2 = int(attr.get("seed", 0)), int(attr.get("seed2", 0))
        if seed == 0 and seed2 == 0:
            return replace(draw, reason=REASON_UNSEEDED)
        kernel = (draw.op_name, seed, seed2, attr.get("dtype"), attr.get("T"), attr.get("Tout"))
        position = self.positions.get(kernel, 0)
        size = int(np.prod(outputs[0].shape))
        self.positions[kernel] = position + LEGACY_SAMPLES_PER_ELEMENT * size
        return replace(draw, state=(seed, seed2, position))

    def _generator(self, draw: RngDraw) -> RngDraw:
        """Attach the generator state read just before this kernel ran."""

        for index in range(len(self.pending) - 1, -1, -1):
            op_type, state, algorithm = self.pending[index]
            if op_type == draw.op_name:
                del self.pending[index]
                return replace(draw, state=(state, algorithm))
        return replace(draw, reason=REASON_GENERATOR)

    def make_wrapper(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap one generated stateful op function to read its state first."""

        op_type = dict(STATEFUL_FUNCTIONS)[name]

        def wrapper(resource: Any, algorithm: Any, *args: Any, **kwargs: Any) -> Any:
            """Read the generator state, then run the kernel."""

            tf = _tf()
            with _state.pause_logging():
                state = np.array(tf.raw_ops.ReadVariableOp(resource=resource, dtype=tf.int64))
            entry = (op_type, state, int(np.array(algorithm)))
            self.pending.append(entry)
            try:
                return original(resource, algorithm, *args, **kwargs)
            finally:
                if entry in self.pending:
                    self.pending.remove(entry)

        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        return wrapper


def _confirm_legacy(log: RngLog) -> None:
    """Re-run each legacy draw at its counted position; mark misses unreplayable."""

    for index, draw in enumerate(log.draws):
        if draw.kind != "legacy_seeded" or draw.reason is not None:
            continue
        output = log.output_of(index)
        with _state.pause_logging():
            again = replay_draw(draw, None)[0]
            same = output is not None and np.array_equal(np.array(again), np.array(output))
        if not same:
            log.replace_draw(index, replace(draw, reason=REASON_POSITION))


@contextmanager
def rng_scope(session: object) -> Iterator[RngLog]:
    """Log every TensorFlow random kernel run inside the ``with`` block.

    Parameters
    ----------
    session:
        The capture session; the log is attached to it as well as yielded.

    Yields
    ------
    RngLog
        The draw log. ``finished`` is set only when the block exits normally.
    """

    from ..tf._tf_compat import get_op_callbacks_module

    callbacks = get_op_callbacks_module()
    log = RngLog("tf")
    attach_log(session, log)
    recorder = _Recorder(log)
    gen_ops = _gen_stateful_ops()
    targets = [] if gen_ops is None else [(gen_ops, name) for name, _ in STATEFUL_FUNCTIONS]
    callbacks.add_op_callback(recorder.on_op)
    try:
        with patched_attributes(targets, recorder.make_wrapper):
            yield log
    finally:
        callbacks.remove_op_callback(recorder.on_op)
    _confirm_legacy(log)
    log.finished = True


def draw_for(session: object, value: Any) -> RngDraw | None:
    """The logged draw that produced ``value`` in ``session``'s scope."""

    log = session_log(session)
    return None if log is None else log.draw_for(value)


def can_replay_draw(draw: RngDraw) -> str | None:
    """``None`` when ``draw`` can be replayed, else the reason."""

    if draw.reason is not None:
        return draw.reason
    if draw.kind not in {"stateless", "generator_state", "legacy_seeded"}:
        return f"tf_rng_kind_unknown:{draw.kind}"
    return None


def _call_raw(op_type: str, inputs: Sequence[Any], attrs: dict[str, Any]) -> Any:
    """Run ``tf.raw_ops.<op_type>`` with inputs in op-definition order."""

    tf = _tf()
    function = getattr(tf.raw_ops, op_type)
    names = [p for p in inspect.signature(function).parameters if p != "name"]
    kwargs = dict(zip(names, inputs))
    kwargs.update({k: v for k, v in attrs.items() if k in names and k not in kwargs})
    with _state.pause_logging():
        return function(**kwargs)


def _replay_legacy(draw: RngDraw, inputs: Sequence[Any], attrs: dict[str, Any]) -> Any:
    """Run the stateless kernel that reproduces a legacy seeded draw."""

    tf = _tf()
    seed, seed2, position = draw.state
    stateless = dict(LEGACY_TO_STATELESS)[draw.op_name]
    key = tf.constant([seed], tf.uint64)
    counter = tf.constant([position, seed2], tf.uint64)
    alg = tf.constant(PHILOX_ALG, tf.int32)
    dtype = tf.dtypes.as_dtype(attrs.get("dtype", attrs.get("Tout", tf.float32)))
    extra = list(inputs[1:3]) if draw.op_name == "RandomUniformInt" else []
    return _call_raw(stateless, [inputs[0], key, counter, alg, *extra], {"dtype": dtype})


def _replay_generator(draw: RngDraw, inputs: Sequence[Any], attrs: dict[str, Any]) -> Any:
    """Run a stateful kernel against a fresh variable holding the recorded state."""

    tf = _tf()
    state, algorithm = draw.state
    with _state.pause_logging():
        variable = tf.Variable(np.asarray(state, dtype=np.int64), trainable=False)
    return _call_raw(draw.op_name, [variable.handle, algorithm, *inputs[2:]], attrs)


def replay_draw(recipe: object, parent_values: Sequence[Any] | None) -> tuple[Any, ...]:
    """Re-run one TensorFlow random kernel from its recorded state.

    Parameters
    ----------
    recipe:
        ``OpRecipe`` whose ``rng`` field holds the draw, or the draw itself.
    parent_values:
        The kernel's inputs in op-definition order (a resource input may be
        ``None``), or ``None`` for the values recorded at capture.

    Returns
    -------
    tuple
        The replayed output.
    """

    draw = draw_from_recipe(recipe)
    reason = can_replay_draw(draw)
    if reason is not None:
        raise ValueError(f"tf draw {draw.op_name} cannot be replayed: {reason}")
    inputs = list(draw.recorded_parents or ()) if parent_values is None else list(parent_values)
    attrs = dict(draw.kwargs)
    if draw.kind == "legacy_seeded":
        return (_replay_legacy(draw, inputs, attrs),)
    if draw.kind == "generator_state":
        return (_replay_generator(draw, inputs, attrs),)
    return (_call_raw(draw.op_name, inputs, attrs),)
