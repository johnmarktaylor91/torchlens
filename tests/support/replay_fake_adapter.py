"""NumPy fake replay adapter for coding against the replay contract.

It fills every function of
:class:`torchlens.backends._replay_adapters._contract.ReplayAdapter` over a
tiny fake trace, so engine lanes (finished-trace edits, validation after
load, the reverse engine, gradient sites) can test without a framework.

Use it as a module: ``from tests.support import replay_fake_adapter as fake``;
``fake.build_trace(...)`` makes a trace, and the module itself is the adapter
object passed to :class:`ReplaySession` or checked against the protocol.

Op identities (the closed allowlist :data:`ALLOWLIST`):
``input`` (parentless, replays its payload), ``add``, ``mul``, ``neg``,
``scale`` (literal ``factor``), ``sum`` (attr ``axis``), ``split2``
(multi-output, attr ``index``), ``opaque`` (``can_replay`` refuses it).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from torchlens.backends._replay_adapters._contract import InputOccurrence, OpRecipe

BACKEND = "fake"
ALLOWLIST = frozenset({"input", "add", "mul", "neg", "scale", "sum", "split2", "opaque"})


@dataclass
class FakeOp:
    """A recorded op of the fake trace; ``label`` is its first output slot."""

    label: str
    op_identity: str
    parents: tuple[str, ...]
    output_slots: tuple[str, ...]
    literals: tuple[tuple[tuple[str | int, ...], Any], ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class FakeTrace:
    """Ops in graph order, saved payloads per output label, and the runtime
    capture list (label to recipe) that ``bind_recipes`` rebuilds."""

    ops: dict[str, FakeOp]
    payloads: dict[str, np.ndarray]
    backend: str = BACKEND
    captures: dict[str, OpRecipe] = field(default_factory=dict)
    param_payloads: dict[str, Any] = field(default_factory=dict)

    @property
    def op_labels(self) -> tuple[str, ...]:
        return tuple(self.ops)


def build_trace(ops: Sequence[FakeOp], inputs: Mapping[str, np.ndarray]) -> FakeTrace:
    """Build a fake trace and fill its payloads by running each op forward."""

    trace = FakeTrace(ops={op.label: op for op in ops}, payloads={})
    for op in ops:
        if op.op_identity == "input":
            trace.payloads[op.label] = np.asarray(inputs[op.label], dtype=np.float64)
            continue
        outputs = forward(trace, op, [trace.payloads[p] for p in op.parents])
        trace.payloads.update(zip(op.output_slots, outputs))
    return trace


def op_for_label(trace: FakeTrace, label: str) -> FakeOp:
    """Return the op that produced output ``label`` (any slot)."""

    for op in trace.ops.values():
        if label in op.output_slots:
            return op
    raise KeyError(label)


def can_replay(trace: FakeTrace, op: FakeOp) -> str | None:
    if op.op_identity == "opaque":
        return "opaque region"
    return None


def _literal(op: FakeOp, name: str) -> Any:
    for path, value in op.literals:
        if path == (name,):
            return value
    raise KeyError(name)


def _split_vjp(op: FakeOp, vals: list[Any], cts: list[Any]) -> tuple[Any, ...]:
    index = op.attrs["index"]
    upper = cts[0] if cts[0] is not None else np.zeros_like(vals[0][:index])
    lower = cts[1] if cts[1] is not None else np.zeros_like(vals[0][index:])
    return (np.concatenate([upper, lower]),)


def _sum_vjp(op: FakeOp, vals: list[Any], cts: list[Any]) -> tuple[Any, ...]:
    grad = np.broadcast_to(np.expand_dims(cts[0], op.attrs["axis"]), vals[0].shape)
    return (grad.copy(),)


# op identity -> (forward(op, parent values), vjp(op, parent values, cotangents))
_RULES: dict[str, tuple[Any, Any]] = {
    "add": (lambda op, v: (v[0] + v[1],), lambda op, v, c: (c[0], c[0])),
    "mul": (lambda op, v: (v[0] * v[1],), lambda op, v, c: (c[0] * v[1], c[0] * v[0])),
    "neg": (lambda op, v: (-v[0],), lambda op, v, c: (-c[0],)),
    "scale": (
        lambda op, v: (v[0] * _literal(op, "factor"),),
        lambda op, v, c: (c[0] * _literal(op, "factor"),),
    ),
    "sum": (lambda op, v: (np.sum(v[0], axis=op.attrs["axis"]),), _sum_vjp),
    "split2": (
        lambda op, v: (v[0][: op.attrs["index"]], v[0][op.attrs["index"] :]),
        _split_vjp,
    ),
}


def forward(trace: FakeTrace, op: FakeOp, parent_values: Sequence[Any]) -> tuple[Any, ...]:
    """Recompute ``op`` from ``parent_values`` (one per occurrence)."""

    if op.op_identity == "input":
        return (trace.payloads[op.label],)
    if op.op_identity not in _RULES:
        raise ValueError(f"fake adapter cannot replay {op.op_identity!r}")
    return tuple(_RULES[op.op_identity][0](op, list(parent_values)))


def vjp(
    trace: FakeTrace,
    op: FakeOp,
    parent_values: Sequence[Any],
    cotangents: Sequence[Any],
) -> tuple[Any | None, ...]:
    """Hand-written derivative per op identity; one cotangent per occurrence."""

    if op.op_identity == "input":
        return ()
    if op.op_identity not in _RULES:
        raise ValueError(f"fake adapter has no derivative for {op.op_identity!r}")
    return tuple(_RULES[op.op_identity][1](op, list(parent_values), list(cotangents)))


def recipe(trace: FakeTrace, op: FakeOp) -> OpRecipe | None:
    """Data-only recipe; parents keep their call order and repeats."""

    if op.op_identity == "opaque":
        return None
    return OpRecipe(
        backend=BACKEND,
        op_identity=op.op_identity,
        inputs=tuple(InputOccurrence(p, (i,)) for i, p in enumerate(op.parents)),
        literals=op.literals,
        attrs=dict(op.attrs),
        output_slots=op.output_slots,
        framework_version=np.__version__,
    )


def bind_recipes(
    trace: FakeTrace,
    recipes: Mapping[str, OpRecipe],
    *,
    param_payloads: Mapping[str, Any] | None = None,
) -> None:
    """Rebuild ops and the capture list from recipes (as after a load)."""

    trace.captures = dict(recipes)
    trace.param_payloads = dict(param_payloads or {})
    for label, rec in recipes.items():
        trace.ops[label] = FakeOp(
            label=label,
            op_identity=rec.op_identity,
            parents=rec.parent_labels,
            output_slots=rec.output_slots,
            literals=rec.literals,
            attrs=dict(rec.attrs),
        )


def zeros_like(value: Any) -> Any:
    return np.zeros_like(value)


def add(left: Any, right: Any) -> Any:
    return left + right


def to_host(value: Any) -> Any:
    return np.asarray(value)


def close(expected: Any, actual: Any) -> bool:
    return bool(np.array_equal(np.asarray(expected), np.asarray(actual)))
