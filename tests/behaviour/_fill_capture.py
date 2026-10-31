"""Capture-verb fills for the behaviour net: fuller promises and new toy rows.

The features that change how a forward is captured (``halt``, ``multipass``,
``backward``, ``record``, ``fast_run``, ``rerun``, ``index``, ``recipes``) get
an ``after`` hook that checks the feature's whole promise through the public
surface, and the toy registry gains rows for model shapes the corpus lacks
(nested containers in and out, forward hooks installed by the model itself,
and the census gaps: ``bernoulli``, ``eq``, ``hardsigmoid``, ``mish`` and
``nn.Mish``, ``tensor_split`` and a graph of 3,000 or more ops).

``features.py`` and ``cases.py`` each import one name from here under their
``# wave-3 fills`` marker: ``FEATURES`` (the base rows with the claimed rows
re-pointed) and ``TOY_CASES`` (the base toys plus these). Both are computed on
first import through the module ``__getattr__``, so neither registry imports
the other while it is still loading.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

import torchlens as tl
from torchlens.types import CaptureStatus

if TYPE_CHECKING:
    from tests.behaviour.cases import Case
    from tests.behaviour.features import Feature
    from tests.behaviour.harness import Cell

SEED = 0


def _not_applicable(reason: str) -> Exception:
    from tests.behaviour.features import NotApplicable

    return NotApplicable(reason)


def _order(trace: Any) -> list[Any]:
    from tests.behaviour.harness import execution_order

    return execution_order(trace)


def _sequence(trace: Any) -> list[tuple[str, tuple[int, ...] | None]]:
    """Return ``(func_name, shape)`` per op in capture order."""

    return [
        (op.func_name, tuple(op.shape) if op.shape is not None else None) for op in _order(trace)
    ]


def _first_difference(got: list[Any], want: list[Any]) -> str:
    index = next((i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b), None)
    if index is None:
        return f"lengths {len(got)} vs {len(want)}"
    return f"first difference at {index}: {got[index]!r} vs {want[index]!r}"


def _base_after(name: str) -> Any:
    """Return the base row's ``after`` hook (the thin promise this fill extends)."""

    from tests.behaviour import features

    return getattr(features, f"_after_{name}")


# --- second capture: the structure is a function of the model and inputs ------


def _retrace(cell: Cell) -> Any:
    """Capture the cell's model again with the cell's own configuration."""

    kwargs = dict(cell.trace_kwargs)
    if "storage" in kwargs:
        kwargs["storage"] = tl.to_disk(str(cell.tmp / "stream-second.tlspec"))
    if cell.capture_fields:
        kwargs["capture"] = tl.options.CaptureOptions(**cell.capture_fields)
    torch.manual_seed(SEED)
    return tl.trace(cell.model, cell.fresh_inputs(), **kwargs)


def _expect_stable_second_capture(cell: Cell) -> None:
    """A second capture of the same model gives the same labels, functions and shapes.

    Leaves no wrapper or hook state behind that changes the next capture. The
    cell's own trace is already the model's second capture (the case facts
    were the first), so a first-call cache cannot explain a difference here.
    """

    if cell.case.has("stochastic"):
        return
    trace = cell.require_trace()
    again = _retrace(cell)
    first = [op.label for op in _order(trace)]
    second = [op.label for op in _order(again)]
    cell.expect(
        first == second, f"second capture changed the op labels: {_first_difference(second, first)}"
    )
    cell.expect(
        _sequence(again) == _sequence(trace),
        f"second capture changed functions or shapes: "
        f"{_first_difference(_sequence(again), _sequence(trace))}",
    )
    cell.expect(again.outcome.status is trace.outcome.status, "second capture changed the outcome")


# --- halt ---------------------------------------------------------------------


def _after_halt(cell: Cell) -> None:
    """A halted capture is the plain capture's prefix up to the halt site, live and loaded.

    The halt predicate is ``tl.func(mid_func)``: it fires at the first op running
    ``mid_func``, so every op the plain capture ran before that op is kept, and
    nothing after the halting op is recorded.
    """

    _base_after("halt")(cell)
    trace = cell.require_trace()
    plain = cell.facts.func_names
    got = [op.func_name for op in _order(trace)]
    site = plain.index(cell.facts.mid_func)
    cell.expect(
        site <= len(got) <= site + 1,
        f"halt at the first {cell.facts.mid_func!r} (plain position {site}) kept {len(got)} ops",
    )
    if cell.loaded is not None:
        cell.expect(
            cell.loaded.outcome.status is CaptureStatus.STOPPED,
            f"loaded outcome {cell.loaded.outcome.status!r}",
        )
        cell.expect(
            [op.label for op in _order(cell.loaded)] == [op.label for op in _order(trace)],
            "the loaded halted trace holds other ops",
        )
    _expect_stable_second_capture(cell)


# --- multipass ----------------------------------------------------------------


def _layer_key(layer: Any) -> str:
    return str(layer.ops[0].label).split(":")[0]


def _expect_pass_census(cell: Cell, name: str, trace: Any) -> None:
    """Pass indexes run 1..num_passes per layer and the recurrent filter is exact."""

    layers = list(trace.layers)
    covered = sum(len(layer.ops) for layer in layers)
    cell.expect(covered == len(trace.ops), f"{name}: layers hold {covered} of {len(trace.ops)} ops")
    repeated = set()
    for layer in layers:
        passes = [op.pass_index for op in layer.ops]
        want = list(range(1, layer.num_passes + 1))
        cell.expect(
            passes == want, f"{name}: layer {_layer_key(layer)} passes {passes[:8]} != {want[:8]}"
        )
        if layer.num_passes > 1:
            repeated.add(_layer_key(layer))
    recurrent = {_layer_key(layer) for layer in trace.layers.recurrent}
    cell.expect(
        recurrent == repeated,
        f"{name}: layers.recurrent {sorted(recurrent)[:6]} != multi-pass layers "
        f"{sorted(repeated)[:6]}",
    )
    cell.expect(bool(trace.graph.is_recurrent) == bool(repeated), f"{name}: is_recurrent disagrees")


def _after_multipass(cell: Cell) -> None:
    """The model's repeated layers show as multi-pass, live and loaded, with exact pass indexes."""

    for name, trace in (("live", cell.trace), ("loaded", cell.loaded)):
        if trace is None:
            continue
        cell.expect(bool(trace.graph.is_recurrent), f"{name} trace is not recurrent")
        cell.expect(len(trace.layers.recurrent) > 0, f"{name} trace has no recurrent layer")
        _expect_pass_census(cell, name, trace)


# --- backward -----------------------------------------------------------------


def _graph_loss(trace: Any) -> torch.Tensor:
    """Sum the trace's graph-attached floating-point output payloads into one scalar.

    ``reconstruct_output()`` returns detached copies, so the loss is built from
    the output ops' own payloads, as the backward guide does.
    """

    terms = []
    for op in trace.output_ops:
        out = op.out
        if isinstance(out, torch.Tensor) and out.is_floating_point() and out.requires_grad:
            terms.append(out.float().sum())
    if not terms:
        raise _not_applicable("no output payload carries an autograd graph")
    return torch.stack(terms).sum()


def _after_backward(cell: Cell) -> None:
    """Backward pairs every GradFn with a forward op, and a False verdict says why.

    Also checks the inference-only contract: the same capture entered under
    ``torch.no_grad()`` records the same ops, reports ``inference_only`` and
    refuses a backward with a typed error.
    """

    from tests.behaviour.features import _loss_of
    from torchlens.errors import TorchLensError
    from torchlens.validation import last_validation_failure

    trace = cell.require_trace()
    if cell.changes("outcome") or cell.changes("payloads"):
        raise _not_applicable("the output payloads are not held in memory with their graph")
    trace.backward(_graph_loss(trace))
    passes = len(trace.autograd.backward_passes)
    cell.expect(passes == 1, f"one backward call recorded {passes} backward passes")
    grad_fns = list(trace.autograd.grad_fns)
    cell.expect(bool(grad_fns), "backward recorded no GradFn")
    paired = [g for g in grad_fns if g.op_label is not None]
    cell.expect(bool(paired), "no GradFn pairs with a forward op")
    dangling = [g.label for g in paired if g.op is None]
    cell.expect(not dangling, f"GradFns name forward ops the trace lacks: {dangling[:6]}")
    verdict = tl.validate(
        cell.model, cell.fresh_inputs(), scope="backward", seed=SEED, loss_fn=_loss_of
    )
    if not bool(verdict):
        cell.expect(
            last_validation_failure() is not None,
            "backward validation False with no failure record",
        )
    cell.expect(bool(verdict), cell.validation_failure_summary())
    with torch.no_grad():
        torch.manual_seed(SEED)
        frozen = tl.trace(cell.model, cell.fresh_inputs(), **cell.trace_kwargs)
    cell.expect(frozen.inference_only is True, "a capture under no_grad is not inference_only")
    if not cell.case.has("stochastic"):
        cell.expect(
            _sequence(frozen) == _sequence(trace),
            f"no_grad capture changed the ops: {_first_difference(_sequence(frozen), _sequence(trace))}",
        )
    try:
        frozen.backward(torch.zeros((), requires_grad=True))
    except TorchLensError:
        return
    cell.expect(False, "backward on an inference-only capture did not refuse with a typed error")


# --- record -------------------------------------------------------------------


def _after_record(cell: Cell) -> None:
    """``tl.record(...).to_trace()`` reproduces the traced op sequence, functions and shapes."""

    _base_after("record")(cell)
    model, _ = cell.case.build()
    torch.manual_seed(SEED)
    cooked = tl.record(model, cell.fresh_inputs(), default_op=True).to_trace()
    cell.expect(
        cooked.outcome.status is CaptureStatus.COMPLETE, f"record outcome {cooked.outcome.status!r}"
    )
    plain = cell.facts.plain
    cell.expect(
        _sequence(cooked) == _sequence(plain),
        f"record shapes differ from trace: {_first_difference(_sequence(cooked), _sequence(plain))}",
    )


# --- fast run and rerun ---------------------------------------------------------


def _eager_reference(cell: Cell) -> list[Any] | None:
    """Return the plain forward's leaves when the trace's output should equal them."""

    if cell.changes("forward_match") or cell.changes("outcome"):
        return None
    if any(not isinstance(leaf, torch.Tensor) for leaf in cell.plain_output):
        return None
    return cell.plain_output


def _after_fast_run(cell: Cell) -> None:
    """The fast run equals the captured output and eager bit for bit, twice, verified."""

    _base_after("fast_run")(cell)
    trace = cell.require_trace()
    runs = []
    for _ in range(2):
        torch.manual_seed(SEED)
        result = trace.run(inputs=cell.fresh_inputs(), fast=True)
        cell.expect(not result.report.poisoned, f"fast run poisoned: {result.report!r}")
        runs.append(
            [
                leaf.detach() if isinstance(leaf, torch.Tensor) else leaf
                for leaf in cell.leaves(result.output)
            ]
        )
    cell.expect_same_leaves(runs[0], runs[1], "second fast run")
    eager = _eager_reference(cell)
    if eager is not None:
        cell.expect_same_leaves(eager, runs[0], "fast run against eager")


def _after_rerun(cell: Cell) -> None:
    """A rerun keeps the op structure and, without an edit, equals eager."""

    _base_after("rerun")(cell)
    trace = cell.require_trace()
    labels = [op.label for op in _order(trace)]
    torch.manual_seed(SEED)
    trace.run(cell.model, cell.fresh_inputs())
    cell.expect([op.label for op in _order(trace)] == labels, "rerun changed the op labels")
    eager = _eager_reference(cell)
    if eager is not None:
        cell.expect_same_leaves(eager, cell.output_leaves(trace), "rerun against eager")


# --- index --------------------------------------------------------------------


def _after_index(cell: Cell) -> None:
    """Every label and module address resolves to its own record; a second capture agrees."""

    _base_after("index")(cell)
    trace = cell.require_trace()
    wrong = [op.label for op in trace.ops if trace[op.label].label != op.label]
    cell.expect(not wrong, f"trace[label] returns another record for {wrong[:6]}")
    for layer in trace.layers:
        for k, op in enumerate(layer.ops, start=1):
            cell.expect(
                op.pass_index == k, f"layer {_layer_key(layer)} pass {k} is {op.pass_index}"
            )
    _expect_stable_second_capture(cell)


# --- recipes ------------------------------------------------------------------


def _noop_facet(module: Any) -> dict[str, Any]:
    """A per-trace facet recipe that adds no facet."""

    return {}


def _prepare_recipes(cell: Cell) -> None:
    cell.trace_kwargs["facets"] = [_noop_facet]


# --- the claimed rows -----------------------------------------------------------

_AFTER = {
    "halt": _after_halt,
    "multipass": _after_multipass,
    "backward": _after_backward,
    "record": _after_record,
    "fast_run": _after_fast_run,
    "rerun": _after_rerun,
    "index": _after_index,
}
_PREPARE = {"recipes": _prepare_recipes}


def _filled_features() -> tuple[Feature, ...]:
    from tests.behaviour import features

    rows = []
    for row in features.FEATURES:
        changes: dict[str, Any] = {}
        if row.name in _AFTER:
            changes["after"] = _AFTER[row.name]
        if row.name in _PREPARE:
            changes["prepare"] = _PREPARE[row.name]
        rows.append(dataclasses.replace(row, **changes) if changes else row)
    return tuple(rows)


# --- toy rows ---------------------------------------------------------------------


class NestedContainers(nn.Module):
    """Takes a dict holding a list holding a tuple; returns the same nesting."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        first, (second, third) = batch["seq"]
        mixed = self.proj(first) + second * third
        return {"out": [mixed, (torch.relu(mixed), batch["side"] * 2)], "aux": {"sum": mixed.sum()}}


def _nested_inputs() -> list[dict[str, Any]]:
    return [
        {
            "seq": [torch.rand(3, 4), (torch.rand(3, 4), torch.rand(3, 4))],
            "side": torch.rand(2),
        }
    ]


def _double_output(module: nn.Module, args: Any, output: torch.Tensor) -> torch.Tensor:
    return output * 2


def _shift_input(module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
    return (args[0] + 1,)


class SelfHooked(nn.Module):
    """Installs a forward pre-hook and a forward hook on a child in ``__init__``."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(4, 4)
        self.outer = nn.Linear(4, 2)
        self.inner.register_forward_pre_hook(_shift_input)
        self.inner.register_forward_hook(_double_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outer(torch.tanh(self.inner(x)))


class CensusGaps(nn.Module):
    """Runs the ops the corpus census lacks: eq, hardsigmoid, mish, nn.Mish, tensor_split."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 6)
        self.mish = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mish(self.lin(x))
        a, b, c = torch.tensor_split(h, 3, dim=1)
        gate = torch.nn.functional.hardsigmoid(a) + torch.nn.functional.mish(b)
        same = torch.eq(torch.round(c), 0).to(h.dtype)
        return torch.cat([gate * same, c], dim=1)


class BernoulliGate(nn.Module):
    """Samples a Bernoulli mask in the forward (stochastic)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.sigmoid(self.lin(x))
        return h * torch.bernoulli(h)


class _Block(nn.Module):
    """Ten cheap ops on a small tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        h = torch.tanh(h) + x
        h = h * 0.5
        h = torch.relu(h) - 0.1
        h = torch.sigmoid(h) + h
        return h / 2


class WideGraph(nn.Module):
    """Three hundred distinct blocks of ten ops: a graph of 3,000+ ops with no repeated layer."""

    def __init__(self, blocks: int = 300) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _rows(n: int, width: int) -> Any:
    return lambda: torch.rand(n, width)


def _filled_toys() -> tuple[Case, ...]:
    from tests.behaviour import cases

    return (
        *cases.TOY_CASES,
        cases._toy(
            "nested_containers", NestedContainers, _nested_inputs, "containers", "child_module"
        ),
        cases._toy("self_hooked", SelfHooked, _rows(3, 4), "child_module"),
        cases._toy("census_gaps", CensusGaps, _rows(4, 6), "child_module"),
        cases._toy("bernoulli_gate", BernoulliGate, _rows(4, 6), "child_module", "stochastic"),
        cases._toy("wide_graph_3000", WideGraph, _rows(2, 4), "child_module", seconds=20.0),
    )


def __getattr__(name: str) -> Any:
    """Compute ``FEATURES`` and ``TOY_CASES`` when a registry imports them."""

    if name == "FEATURES":
        return _filled_features()
    if name == "TOY_CASES":
        return _filled_toys()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
