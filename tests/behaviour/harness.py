"""Run one cell of the behaviour net: a case under a set of features.

``run_cell(case, features, tmp)`` traces, validates (default validation, never
loosened), saves to a temp dir, loads, checks the loaded trace against the
live one through the public surface, checks the plain forward output (or the
replay or prefix a feature declares instead), then runs each feature's own
check. Every step records an :class:`Outcome`: ``pass``, ``fail`` (a finding),
``refused`` (a typed TorchLens error, with its class and code),
``not_applicable`` (the case cannot take the feature by design) or ``not_run``.

Everything goes through the public surface. The one documented seam is
``torchlens.validation.last_validation_failure`` (a public submodule name) to
report *why* validation failed; the verdict itself is ``tl.validate``.
"""

from __future__ import annotations

import copy
import inspect
import shutil
import warnings
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from tests.behaviour.cases import Case
from tests.behaviour.features import SEED, Feature, NotApplicable
from torchlens.errors import TorchLensError
from torchlens.types import CaptureStatus
from torchlens.validation import last_validation_failure

BASE_ORACLES: tuple[str, ...] = ("trace", "validate", "save", "load", "equality", "forward_match")
_SCALARS = (int, float, bool, str, bytes, type(None))


@dataclass(frozen=True)
class Outcome:
    """The result of one oracle in one cell."""

    status: str
    detail: str = ""
    error_class: str = ""
    code: str = ""

    @property
    def ok(self) -> bool:
        """Whether the oracle counts as satisfied (pass, refused or not applicable)."""

        return self.status in {"pass", "refused", "not_applicable"}


class OracleFailure(AssertionError):
    """An oracle's expectation did not hold (a finding)."""


@dataclass(frozen=True)
class CaseFacts:
    """What a plain, feature-free capture of the case shows.

    Features use these to pick sites that exist (a function name to save, a
    label to edit, a module to extract).
    """

    plain: Any
    labels: tuple[str, ...]
    func_names: tuple[str, ...]
    common_func: str
    common_float_func: str | None
    mid_func: str
    single_pass_float_label: str | None
    module_addresses: tuple[str, ...]
    single_call_module: str | None


@dataclass
class Cell:
    """Mutable state of one running cell; features read and extend it."""

    case: Case
    features: tuple[Feature, ...]
    tmp: Path
    model: nn.Module
    inputs: Any
    facts: CaseFacts
    trace_kwargs: dict[str, Any] = field(default_factory=dict)
    capture_fields: dict[str, Any] = field(default_factory=dict)
    trace: Any = None
    loaded: Any = None
    plain_output: list[Any] = field(default_factory=list)
    outcomes: dict[str, Outcome] = field(default_factory=dict)

    def changes(self, oracle: str) -> bool:
        """Whether any active feature changes the meaning of ``oracle``."""

        return any(oracle in feature.changes for feature in self.features)

    def require_trace(self) -> Any:
        """Return the live trace, or refuse when capture did not finish."""

        if self.trace is None:
            raise NotApplicable("no live trace (capture refused or failed)")
        return self.trace

    def expect(self, condition: bool, detail: str) -> None:
        """Raise an :class:`OracleFailure` carrying ``detail`` unless ``condition``."""

        if not condition:
            raise OracleFailure(detail)

    def fresh_inputs(self) -> Any:
        """Return a deep copy of the inputs (a forward may write into them in place)."""

        return copy.deepcopy(self.inputs)

    def positional_inputs(self, inputs: Any = None) -> list[Any]:
        """Return the inputs as the positional arguments ``model(*args)`` takes."""

        return positional_arguments(self.inputs if inputs is None else inputs, self.model)

    def leaves(self, output: Any) -> Iterator[Any]:
        """Flatten a model output into its leaves."""

        return flatten_leaves(output)

    def output_leaves(self, trace: Any) -> list[Any]:
        """Return detached copies of a trace's output leaves.

        Disk-streamed payloads are not materialized implicitly, so under a
        feature that moves payloads off memory the output ops are read through
        ``materialize_out``.
        """

        leaves = [_detached(leaf) for leaf in flatten_leaves(trace.reconstruct_output())]
        if self.changes("payloads") and any(leaf is None for leaf in leaves):
            leaves = [_detached(op.materialize_out()) for op in trace.output_ops]
        return leaves

    def expect_same_leaves(self, want: Sequence[Any], got: Sequence[Any], what: str) -> None:
        """Expect two leaf sequences to match exactly (NaN equal to NaN)."""

        diff = first_leaf_difference(want, got)
        self.expect(diff is None, f"{what}: {diff}")

    def validation_failure_summary(self) -> str:
        """Return the structured summary of the last validation failure."""

        failure = last_validation_failure()
        return failure.summary() if failure is not None else "no failure record"


@dataclass(frozen=True)
class CellResult:
    """Outcomes of one cell by oracle name (base oracles first, then ``feature:<name>``)."""

    cell_id: str
    outcomes: dict[str, Outcome]

    def oracle(self, name: str) -> Outcome:
        """Return the outcome of ``name``, ``not_run`` when the cell never reached it."""

        return self.outcomes.get(name, Outcome("not_run", "the cell did not reach this oracle"))


# --- public-surface helpers --------------------------------------------------


def positional_arguments(inputs: Any, model: nn.Module) -> list[Any]:
    """Return ``inputs`` as the positional arguments the model call takes.

    Mirrors the documented ``tl.trace`` convention: a bare value is one
    argument; a list or tuple is several, except that a model whose forward
    takes exactly one positional parameter receives a multi-element list or
    tuple as that one argument.
    """

    if isinstance(inputs, (list, tuple)):
        if len(inputs) > 1 and _forward_takes_one_positional(model):
            return [inputs]
        return list(inputs)
    if inputs is None:
        return []
    return [inputs]


def _forward_takes_one_positional(model: nn.Module) -> bool:
    """Whether ``model.forward`` declares exactly one positional parameter and no ``*args``."""

    params = [
        p
        for p in inspect.signature(model.forward).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
    ]
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return False
    return len(params) == 1


def flatten_leaves(value: Any) -> Iterator[Any]:
    """Yield the leaves of a nested tensor container in a stable order."""

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for key in value:
            yield from flatten_leaves(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_leaves(item)
    else:
        yield value


def _detached(leaf: Any) -> Any:
    return leaf.detach().clone() if isinstance(leaf, torch.Tensor) else leaf


def _same_leaf(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.is_complex():
            return bool(torch.equal(torch.view_as_real(a), torch.view_as_real(b)))
        if a.is_floating_point():
            return bool(torch.allclose(a, b, rtol=0, atol=0, equal_nan=True))
        return bool(torch.equal(a, b))
    return bool(a == b)


def first_leaf_difference(want: Sequence[Any], got: Sequence[Any]) -> str | None:
    """Return a description of the first differing leaf, or ``None`` when all match."""

    if len(want) != len(got):
        return f"{len(want)} leaves expected, {len(got)} found"
    for index, (a, b) in enumerate(zip(want, got, strict=True)):
        if not _same_leaf(a, b):
            return f"leaf {index} differs: {_describe(a)} vs {_describe(b)}{_gap(a, b)}"
    return None


def _gap(a: Any, b: Any) -> str:
    """Describe how far two same-shaped real tensors are apart."""

    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return ""
    if a.shape != b.shape or a.dtype != b.dtype or a.is_complex() or not a.is_floating_point():
        return ""
    delta = (a.double() - b.double()).abs()
    finite = torch.isfinite(delta)
    if not bool(finite.any()):
        return " (no finite difference)"
    return (
        f" (max abs diff {delta[finite].max().item():.3e} over {int(finite.sum())} "
        f"of {delta.numel()} elements, {int((delta[finite] > 0).sum())} differ)"
    )


def _describe(leaf: Any) -> str:
    if isinstance(leaf, torch.Tensor):
        return f"tensor{tuple(leaf.shape)} {leaf.dtype}"
    return repr(leaf)[:80]


# --- case facts ------------------------------------------------------------


def execution_order(trace: Any) -> list[Any]:
    """Return a trace's ops in capture order (``raw_index``), not label order."""

    return sorted(trace.ops, key=lambda op: (op.raw_index is None, op.raw_index or 0))


def _is_float_dtype(dtype: Any) -> bool:
    text = str(dtype)
    return "float" in text or "bfloat" in text


def _module_calls(trace: Any) -> Counter[str]:
    """Count module calls by address from the ops' module context."""

    calls: Counter[str] = Counter()
    seen: set[tuple[str, Any]] = set()
    for op in trace.ops:
        for entry in op.modules or ():
            address, call = (entry[0], entry[1]) if isinstance(entry, tuple) else (str(entry), 0)
            address = str(address).split(":")[0]
            if (address, call) not in seen:
                seen.add((address, call))
                calls[address] += 1
    return calls


_FACTS: OrderedDict[str, CaseFacts] = OrderedDict()
_FACTS_KEEP = 2


def case_facts(case: Case, model: nn.Module, inputs: Any) -> CaseFacts:
    """Capture the case plainly once and derive the sites features need (cached per case)."""

    cached = _FACTS.get(case.name)
    if cached is not None:
        return cached
    facts = _compute_facts(model, copy.deepcopy(inputs))
    _FACTS[case.name] = facts
    while len(_FACTS) > _FACTS_KEEP:
        _FACTS.popitem(last=False)
    return facts


def _compute_facts(model: nn.Module, inputs: Any) -> CaseFacts:
    torch.manual_seed(SEED)
    plain = tl.trace(model, inputs)
    ops = execution_order(plain)
    labels = tuple(op.label for op in ops)
    func_names = tuple(op.func_name for op in ops)
    counts = Counter(func_names)
    common_func = counts.most_common(1)[0][0]
    float_funcs = Counter(
        op.func_name for op in ops if _is_float_dtype(op.dtype) and op.func_name != "identity"
    )
    common_float_func = float_funcs.most_common(1)[0][0] if float_funcs else None
    mid_func = func_names[len(func_names) // 2] if len(func_names) > 1 else func_names[0]
    single_pass = next(
        (
            op.label
            for op in ops
            if _is_float_dtype(op.dtype) and getattr(op.layer, "num_passes", 1) == 1
        ),
        None,
    )
    calls = _module_calls(plain)
    addresses = tuple(sorted(calls))
    single_call = next((address for address in addresses if calls[address] == 1), None)
    return CaseFacts(
        plain=plain,
        labels=labels,
        func_names=func_names,
        common_func=common_func,
        common_float_func=common_float_func,
        mid_func=mid_func,
        single_pass_float_label=single_pass,
        module_addresses=addresses,
        single_call_module=single_call,
    )


# --- the cell runner -------------------------------------------------------


def _guard(cell: Cell, oracle: str, step: Callable[[], None]) -> bool:
    """Run ``step`` and record its outcome under ``oracle``; return whether it passed."""

    try:
        step()
    except NotApplicable as exc:
        cell.outcomes[oracle] = Outcome("not_applicable", str(exc))
        return False
    except OracleFailure as exc:
        cell.outcomes[oracle] = Outcome("fail", str(exc))
        return False
    except TorchLensError as exc:
        code = str(getattr(exc, "fields", {}).get("code", "") or getattr(exc, "code", ""))
        cell.outcomes[oracle] = Outcome("refused", str(exc)[:300], type(exc).__name__, code)
        return False
    except Exception as exc:  # noqa: BLE001 - an untyped error is a finding, recorded as one
        cell.outcomes[oracle] = Outcome("fail", f"{type(exc).__name__}: {str(exc)[:300]}")
        return False
    cell.outcomes[oracle] = Outcome("pass")
    return True


def _prepare(cell: Cell) -> None:
    for feature in cell.features:
        if feature.prepare is not None:
            feature.prepare(cell)


def _trace(cell: Cell) -> None:
    kwargs = dict(cell.trace_kwargs)
    if cell.capture_fields:
        kwargs["capture"] = tl.options.CaptureOptions(**cell.capture_fields)
    torch.manual_seed(SEED)
    trace = tl.trace(cell.model, cell.fresh_inputs(), **kwargs)
    cell.expect(len(trace.ops) > 0, "empty trace")
    want = CaptureStatus.STOPPED if cell.changes("outcome") else CaptureStatus.COMPLETE
    cell.expect(trace.outcome.status is want, f"outcome {trace.outcome.status!r}, wanted {want!r}")
    cell.trace = trace


def _plain_forward(cell: Cell) -> list[Any]:
    """Run the model plainly, in grad mode as the capture ran, and keep detached leaves."""

    torch.manual_seed(SEED)
    output = cell.model(*cell.positional_inputs(cell.fresh_inputs()))
    return [_detached(leaf) for leaf in flatten_leaves(output)]


def _forward_match(cell: Cell) -> None:
    trace = cell.require_trace()
    if cell.changes("outcome"):
        raise NotApplicable(
            "a halted capture has no final output; the halt feature checks the prefix"
        )
    if cell.changes("forward_match"):
        before = cell.output_leaves(trace)
        torch.manual_seed(SEED)
        trace.run(cell.model, cell.fresh_inputs())
        cell.expect_same_leaves(before, cell.output_leaves(trace), "replay output")
        return
    foreign = [type(leaf).__name__ for leaf in cell.plain_output if not _comparable(leaf)]
    if foreign:
        raise NotApplicable(f"the output holds non-tensor objects {sorted(set(foreign))}")
    diff = first_leaf_difference(cell.plain_output, cell.output_leaves(trace))
    if diff is not None and cell.case.has("stochastic"):
        raise NotApplicable(f"the forward draws random numbers; {diff}")
    if diff is not None and first_leaf_difference(cell.plain_output, _plain_forward(cell)):
        raise NotApplicable(
            f"two plain forwards differ, so the forward is not deterministic; {diff}"
        )
    cell.expect(diff is None, f"plain forward output: {diff}")


def _comparable(leaf: Any) -> bool:
    return isinstance(leaf, (torch.Tensor, *_SCALARS))


def _validate(cell: Cell) -> None:
    verdict = tl.validate(
        cell.model, cell.fresh_inputs(), scope="forward", seed=SEED, validate_metadata=True
    )
    cell.expect(bool(verdict), f"forward validation failed: {cell.validation_failure_summary()}")


def _save(cell: Cell) -> None:
    cell.require_trace().save(cell.tmp / "cell.tlspec", level="portable")


def _load(cell: Cell) -> None:
    cell.loaded = tl.load(cell.tmp / "cell.tlspec")


def _op_facts(op: Any) -> tuple[Any, ...]:
    return (
        op.label,
        op.func_name,
        tuple(op.shape) if op.shape is not None else None,
        str(op.dtype),
        tuple(op.parents),
        tuple(op.children),
        op.pass_index,
    )


def _equality(cell: Cell) -> None:
    live, loaded = cell.require_trace(), cell.loaded
    if loaded is None:
        raise NotApplicable("nothing loaded")
    live_ops, loaded_ops = list(live.ops), list(loaded.ops)
    cell.expect(
        [op.label for op in live_ops] == [op.label for op in loaded_ops],
        "loaded op labels differ from the live trace",
    )
    cell.expect(
        tl.fingerprint.trace(live) == tl.fingerprint.trace(loaded),
        "structural hash differs after load",
    )
    for a, b in zip(live_ops, loaded_ops, strict=True):
        cell.expect(_op_facts(a) == _op_facts(b), f"op {a.label}: {_op_facts(a)} != {_op_facts(b)}")
    cell.expect(
        set(live.modules.keys()) == set(loaded.modules.keys()), "module addresses differ after load"
    )
    live_saved = {op.label for op in live.saved_ops}
    loaded_saved = {op.label for op in loaded.saved_ops}
    cell.expect(
        live_saved == loaded_saved,
        f"saved op sets differ after load: {sorted(live_saved ^ loaded_saved)[:8]}",
    )
    if cell.changes("payloads"):
        return
    for a, b in zip(live_ops, loaded_ops, strict=True):
        if a.label in live_saved:
            cell.expect(_same_leaf(a.out, b.out), f"op {a.label}: saved payload differs after load")


def _after(cell: Cell) -> None:
    for feature in cell.features:
        if feature.after is None:
            continue
        _guard(cell, f"feature:{feature.name}", lambda f=feature: f.after(cell))


def cell_id(case: Case, features: Iterable[Feature]) -> str:
    """Return the test id of a cell: ``case-featureA-featureB``."""

    return "-".join([case.name.replace(" ", "_"), *(feature.name for feature in features)])


def run_cell(case: Case, features: Sequence[Feature], tmp: Path) -> CellResult:
    """Run one cell and return every oracle's outcome.

    Parameters
    ----------
    case:
        The model row.
    features:
        The features active in this cell (usually a pair).
    tmp:
        A fresh directory for artifacts.
    """

    features = tuple(features)
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        return _run_cell(case, features, tmp)


def run_cell_and_clean(
    case: Case, features: Sequence[Feature], tmp: Path, *, keep_failed: bool = False
) -> CellResult:
    """Run one cell in ``tmp``, then remove ``tmp``, pass or fail.

    Every oracle has its outcome when :func:`run_cell` returns (the result
    holds outcomes only, never a path), so nothing reads ``tmp`` afterwards.
    With ``keep_failed`` the directory stays when an oracle failed or the cell
    raised, for inspection; a passing cell's directory is removed either way.
    """

    failed = True
    try:
        result = run_cell(case, features, tmp)
        failed = any(outcome.status == "fail" for outcome in result.outcomes.values())
    finally:
        if not (failed and keep_failed):
            shutil.rmtree(tmp)
    return result


def _run_cell(case: Case, features: tuple[Feature, ...], tmp: Path) -> CellResult:
    outcomes: dict[str, Outcome] = {}
    reasons = [f"{f.name}: {f.applies(case.tags)}" for f in features if f.applies(case.tags)]
    if reasons:
        for oracle in BASE_ORACLES:
            outcomes[oracle] = Outcome("not_applicable", "; ".join(reasons))
        return CellResult(cell_id(case, features), outcomes)
    model, inputs = case.build()
    facts = case_facts(case, model, inputs)
    cell = Cell(case, features, tmp, model, inputs, facts)
    if not _guard(cell, "trace", lambda: _prepare(cell)):
        return _finish(cell)
    if not _guard(cell, "trace", lambda: _trace(cell)):
        return _finish(cell)
    cell.plain_output = _plain_forward(cell)
    _guard(cell, "forward_match", lambda: _forward_match(cell))
    _guard(cell, "validate", lambda: _validate(cell))
    if _guard(cell, "save", lambda: _save(cell)) and _guard(cell, "load", lambda: _load(cell)):
        _guard(cell, "equality", lambda: _equality(cell))
    _after(cell)
    return _finish(cell)


def _finish(cell: Cell) -> CellResult:
    outcomes = dict(cell.outcomes)
    gate = outcomes.get("trace")
    if gate is not None and gate.status in {"refused", "not_applicable"}:
        for oracle in BASE_ORACLES:
            outcomes.setdefault(
                oracle, Outcome(gate.status, f"capture: {gate.detail}", gate.error_class, gate.code)
            )
    result = CellResult(cell_id(cell.case, cell.features), outcomes)
    cell.trace = None
    cell.loaded = None
    return result
