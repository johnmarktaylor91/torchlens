"""The 21 features of the interaction matrix as capture configurations.

Each feature is one row: a name, a cost class, the oracles it changes, the
case tags it needs, a ``prepare`` hook that configures the capture (trace
keywords or ``CaptureOptions`` fields) and an ``after`` hook that checks the
feature's own promise on the live and loaded traces. Hooks raise a typed
TorchLens error to record a refusal and ``NotApplicable`` when the case cannot
take the feature by design. Add a feature by adding a row, never a file.

The feature axis is pinned to ``tests/interaction_matrix.json`` by
``test_matrix.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

import torchlens as tl
from torchlens.types import CaptureStatus

if TYPE_CHECKING:
    from tests.behaviour.harness import Cell

SEED = 0
CHEAP = "cheap"
MEDIUM = "medium"
HEAVY = "heavy"


class NotApplicable(Exception):
    """The case cannot take this feature by design (recorded, never a failure)."""


Hook = Callable[["Cell"], None]


@dataclass(frozen=True)
class Feature:
    """One feature row.

    Attributes
    ----------
    name:
        The interaction matrix's feature key.
    cost:
        ``"cheap"``, ``"medium"`` or ``"heavy"``: the extra work per cell.
    summary:
        One line, from the matrix.
    changes:
        Oracle names whose meaning this feature changes: ``forward_match``
        (replaced by a replay or prefix match), ``outcome`` (HALTED instead
        of COMPLETE), ``payloads`` (saved payloads are not compared).
    requires:
        Case tags the feature needs; a case without one is not applicable.
    prepare:
        Configures the capture before ``tl.trace``.
    after:
        Checks the feature's promise after trace, validate, save and load.
    """

    name: str
    cost: str
    summary: str
    changes: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()
    prepare: Hook | None = None
    after: Hook | None = None

    def applies(self, tags: frozenset[str]) -> str | None:
        """Return why the feature does not apply to a case with ``tags``, or ``None``."""

        missing = sorted(self.requires - tags)
        if missing:
            return f"case lacks tag(s) {missing}"
        return None


def _no_stochastic(cell: Cell) -> None:
    """Refuse a value-comparison feature on a case whose forward draws random numbers."""

    if cell.case.has("stochastic"):
        raise NotApplicable("the forward draws random numbers; a second run cannot match")


# --- capture-time features -------------------------------------------------


def _prepare_save(cell: Cell) -> None:
    cell.trace_kwargs["save"] = tl.func(cell.facts.common_func)


def _after_save(cell: Cell) -> None:
    """The selected ops are saved; anything else saved is an output or a buffer node."""

    for name, trace in (("live", cell.trace), ("loaded", cell.loaded)):
        if trace is None:
            continue
        want = {op.label for op in trace.ops if op.func_name == cell.facts.common_func}
        got = {op.label for op in trace.saved_ops}
        allowed = {op.label for op in trace.output_ops}
        allowed |= {op.label for op in trace.ops if op.is_buffer_source}
        allowed |= {label for label in got if label.startswith("buffer_")}
        missing, extra = sorted(want - got), sorted(got - want - allowed)
        cell.expect(not missing, f"{name}: selected ops not saved: {missing[:8]}")
        cell.expect(not extra, f"{name}: unselected ops saved: {extra[:8]}")


def _prepare_intervene(cell: Cell) -> None:
    target = cell.facts.common_float_func
    if target is None:
        raise NotApplicable("no floating-point op to scale")
    cell.trace_kwargs["intervene"] = tl.when(tl.func(target), tl.scale(0.5))
    cell.trace_kwargs["on_zero_fire"] = "error"


def _after_intervene(cell: Cell) -> None:
    """Every selected op records a replaced value (``on_zero_fire="error"`` guards firing)."""

    trace = cell.require_trace()
    target = cell.facts.common_float_func
    want = {op.label for op in trace.ops if op.func_name == target}
    got = {op.label for op in trace.ops if op.intervention_replaced}
    cell.expect(
        got == want,
        f"ops recording a replaced value {sorted(got)[:8]} != selected {sorted(want)[:8]}",
    )


def _prepare_halt(cell: Cell) -> None:
    cell.trace_kwargs["halt"] = tl.func(cell.facts.mid_func)


def _after_halt(cell: Cell) -> None:
    trace = cell.require_trace()
    cell.expect(trace.outcome.status is CaptureStatus.HALTED, f"outcome {trace.outcome.status!r}")
    from tests.behaviour.harness import execution_order

    got = tuple(op.func_name for op in execution_order(trace))
    plain = cell.facts.func_names
    cell.expect(
        0 < len(got) < len(plain) and plain[: len(got)] == got,
        f"halted ops are not a proper prefix of the plain capture ({len(got)}/{len(plain)}): "
        f"halted {list(got[:12])} vs plain {list(plain[:12])}",
    )


def _prepare_storage(cell: Cell) -> None:
    cell.trace_kwargs["storage"] = tl.to_disk(str(cell.tmp / "stream.tlspec"))


def _after_storage(cell: Cell) -> None:
    trace = cell.require_trace()
    if cell.changes("outcome"):
        raise NotApplicable("a halted capture holds only the ops that ran; no plain count to match")
    cell.expect(len(trace.ops) == len(cell.facts.labels), "streamed capture changed the op count")


def _prepare_backward(cell: Cell) -> None:
    if not any(p.requires_grad for p in cell.model.parameters()):
        raise NotApplicable("no trainable parameter to receive a gradient")
    cell.capture_fields["backward_ready"] = True


def _loss_of(output: Any) -> torch.Tensor:
    """Sum every floating-point leaf of a model output into one scalar."""

    from tests.behaviour.harness import flatten_leaves

    terms = [
        leaf.float().sum()
        for leaf in flatten_leaves(output)
        if isinstance(leaf, torch.Tensor) and leaf.is_floating_point() and leaf.requires_grad
    ]
    if not terms:
        raise NotApplicable("the output has no differentiable floating-point leaf")
    return torch.stack(terms).sum()


def _after_backward(cell: Cell) -> None:
    trace = cell.require_trace()
    trace.backward(_loss_of(trace.reconstruct_output()))
    verdict = tl.validate(
        cell.model, cell.fresh_inputs(), scope="backward", seed=SEED, loss_fn=_loss_of
    )
    cell.expect(bool(verdict), cell.validation_failure_summary())


def _prepare_episode(cell: Cell) -> None:
    raise NotApplicable("episode capture needs a stepped generation root; no case declares one")


def _prepare_backend(cell: Cell) -> None:
    cell.trace_kwargs["backend"] = "torch"


def _after_backend(cell: Cell) -> None:
    for name, trace in (("live", cell.trace), ("loaded", cell.loaded)):
        if trace is not None:
            cell.expect(trace.backend == "torch", f"{name} backend {trace.backend!r}")


def _prepare_reporting(cell: Cell) -> None:
    cell.trace_kwargs["profile"] = True
    cell.trace_kwargs["grouping"] = "structural"


def _after_reporting(cell: Cell) -> None:
    trace = cell.require_trace()
    cell.expect(trace.grouping == "structural", f"grouping {trace.grouping!r}")
    cell.expect(trace.capture_duration is not None, "profiled capture records no duration")


def _noop_recipe(module: Any) -> dict[str, Any]:
    """A per-trace facet recipe that adds no facet."""

    return {}


def _prepare_recipes(cell: Cell) -> None:
    cell.trace_kwargs["recipes"] = [_noop_recipe]


# --- post-capture features -------------------------------------------------


def _after_draw(cell: Cell) -> None:
    for name, trace in (("live", cell.trace), ("loaded", cell.loaded)):
        if trace is None:
            continue
        out = cell.tmp / f"graph-{name}"
        trace.draw(vis_outpath=str(out), vis_save_only=True, vis_fileformat="svg")
        rendered = list(cell.tmp.glob(f"graph-{name}*"))
        cell.expect(bool(rendered), f"{name} draw wrote nothing under {out}")


def _after_rerun(cell: Cell) -> None:
    _no_stochastic(cell)
    trace = cell.require_trace()
    before = cell.output_leaves(trace)
    torch.manual_seed(SEED)
    trace.run(cell.model, cell.fresh_inputs())
    cell.expect_same_leaves(before, cell.output_leaves(trace), "rerun output")


def _after_fast_run(cell: Cell) -> None:
    _no_stochastic(cell)
    trace = cell.require_trace()
    before = cell.output_leaves(trace)
    torch.manual_seed(SEED)
    result = trace.run(inputs=cell.fresh_inputs(), fast=True)
    got = [
        leaf.detach() if isinstance(leaf, torch.Tensor) else leaf
        for leaf in cell.leaves(result.output)
    ]
    cell.expect_same_leaves(before, got, "fast run output")


def _prepare_fork_do(cell: Cell) -> None:
    if cell.facts.single_pass_float_label is None:
        raise NotApplicable("no single-pass floating-point op to edit")
    cell.capture_fields["intervention_ready"] = True


def _editable_site(cell: Cell, trace: Any) -> str:
    """Return a saved, single-pass, floating-point op label on ``trace``."""

    from tests.behaviour.harness import _is_float_dtype

    for op in trace.saved_ops:
        if (
            _is_float_dtype(op.dtype)
            and getattr(op.layer, "num_passes", 1) == 1
            and op.out is not None
        ):
            return op.label
    raise NotApplicable("no saved single-pass floating-point op to edit on this capture")


def _after_fork_do(cell: Cell) -> None:
    trace = cell.require_trace()
    label = _editable_site(cell, trace)
    original = trace[label].out.detach().clone()
    fork = trace.fork()
    fork.do(tl.label(label), tl.scale(0.5))
    edited = fork[label].out
    cell.expect(
        torch.allclose(edited, original * 0.5, rtol=0, atol=0, equal_nan=True),
        f"fork.do(scale 0.5) did not halve {label}",
    )
    cell.expect(len(fork.ops) == len(trace.ops), "the fork changed the op count")


def _after_index(cell: Cell) -> None:
    trace = cell.require_trace()
    for op in trace.ops:
        got = trace[op.label]
        cell.expect(got is not None, f"trace[{op.label!r}] is None")
    addresses = cell.facts.module_addresses
    if cell.changes("outcome"):
        # A halted capture holds only the module calls that ran before the halt.
        from tests.behaviour.harness import _module_calls

        addresses = tuple(sorted(_module_calls(trace)))
    for address in addresses:
        cell.expect(trace.modules[address] is not None, f"modules[{address!r}] is None")


def _after_multipass(cell: Cell) -> None:
    for name, trace in (("live", cell.trace), ("loaded", cell.loaded)):
        if trace is None:
            continue
        cell.expect(bool(trace.graph.is_recurrent), f"{name} trace is not recurrent")
        cell.expect(len(trace.recurrent_layers) > 0, f"{name} trace has no recurrent layer")


def _after_record(cell: Cell) -> None:
    _no_stochastic(cell)
    torch.manual_seed(SEED)
    recording = tl.record(cell.model, cell.fresh_inputs(), default_op=True)
    cooked = recording.to_trace()
    from tests.behaviour.harness import execution_order

    got = tuple(op.func_name for op in execution_order(cooked))
    want = cell.facts.func_names
    first = next((i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b), None)
    cell.expect(
        got == want,
        f"record.to_trace() gives {len(got)} ops against {len(want)} traced; first difference at "
        f"{first}: {got[first] if first is not None and first < len(got) else None!r} vs "
        f"{want[first] if first is not None and first < len(want) else None!r}",
    )


def _single_call_module(cell: Cell) -> str:
    address = cell.facts.single_call_module
    if address is None:
        raise NotApplicable("no child module called exactly once")
    return address


def _after_extract(cell: Cell) -> None:
    _no_stochastic(cell)
    if cell.case.has("containers"):
        raise NotApplicable("extract takes one tensor input")
    address = _single_call_module(cell)
    cell.require_trace()
    torch.manual_seed(SEED)
    features = tl.extract(cell.model, cell.fresh_inputs(), [address])
    plain = cell.facts.plain[address].out
    cell.expect_same_leaves([plain], [features[address]], f"extract({address!r})")


def _after_bind(cell: Cell) -> None:
    _no_stochastic(cell)
    address = _single_call_module(cell)
    spec = tl.when(tl.module(address), tl.scale(0.5))
    bound = spec.bind(cell.model)
    torch.manual_seed(SEED)
    with torch.no_grad():
        bound_out = bound(*cell.positional_inputs(cell.fresh_inputs()))
    bound_leaves = list(cell.leaves(bound_out))
    foreign = sorted(
        {type(leaf).__name__ for leaf in bound_leaves if not isinstance(leaf, torch.Tensor)}
    )
    if foreign:
        raise NotApplicable(f"the model returns non-tensor objects {foreign}")
    torch.manual_seed(SEED)
    steered = tl.trace(cell.model, cell.fresh_inputs(), intervene=spec, on_zero_fire="error")
    cell.expect_same_leaves(cell.output_leaves(steered), bound_leaves, "bound executor output")


def _after_bundle(cell: Cell) -> None:
    trace = cell.require_trace()
    members = [trace] if cell.loaded is None else [trace, cell.loaded]
    bundle = tl.bundle(members)
    cell.expect(len(bundle) == len(members), f"bundle has {len(bundle)} members")


def _after_validate(cell: Cell) -> None:
    verdict = tl.validate(cell.model, cell.fresh_inputs(), scope="saved", seed=SEED)
    cell.expect(bool(verdict), cell.validation_failure_summary())


def _after_export(cell: Cell) -> None:
    for name, trace in (("live", cell.trace), ("loaded", cell.loaded)):
        if trace is None:
            continue
        frame = trace.to_pandas()
        cell.expect(
            len(frame) == len(trace), f"{name} to_pandas has {len(frame)} rows for {len(trace)} ops"
        )


FEATURES: tuple[Feature, ...] = (
    Feature("save", CHEAP, "Selective capture (save=)", prepare=_prepare_save, after=_after_save),
    Feature(
        "intervene",
        CHEAP,
        "Capture-time edit (intervene=tl.when(site, action))",
        changes=frozenset({"forward_match"}),
        requires=frozenset({"float_output"}),
        prepare=_prepare_intervene,
        after=_after_intervene,
    ),
    Feature(
        "halt",
        CHEAP,
        "Stop the forward early (halt=)",
        changes=frozenset({"forward_match", "outcome"}),
        prepare=_prepare_halt,
        after=_after_halt,
    ),
    Feature(
        "storage",
        MEDIUM,
        "Payloads streamed to disk (storage=tl.to_disk)",
        changes=frozenset({"payloads"}),
        prepare=_prepare_storage,
        after=_after_storage,
    ),
    Feature(
        "backward",
        HEAVY,
        "Gradient capture (backward_ready, trace.backward)",
        requires=frozenset({"float_output", "child_module"}),
        prepare=_prepare_backward,
        after=_after_backward,
    ),
    Feature(
        "episode",
        HEAVY,
        "Multi-step episode capture (episode=EpisodeOptions)",
        requires=frozenset({"episode_root"}),
        prepare=_prepare_episode,
    ),
    Feature(
        "backend",
        CHEAP,
        "Explicit backend (backend=)",
        prepare=_prepare_backend,
        after=_after_backend,
    ),
    Feature(
        "reporting",
        CHEAP,
        "Capture-time reporting (profile, grouping)",
        prepare=_prepare_reporting,
        after=_after_reporting,
    ),
    Feature("recipes", CHEAP, "Facet recipes at capture (recipes=)", prepare=_prepare_recipes),
    Feature("draw", MEDIUM, "Graph rendering (Trace.draw)", after=_after_draw),
    Feature("rerun", MEDIUM, "Rerun on the live model (Trace.run)", after=_after_rerun),
    Feature("fast_run", MEDIUM, "Guarded fast rerun (run(fast=True))", after=_after_fast_run),
    Feature(
        "fork_do",
        MEDIUM,
        "Post-hoc edit on a fork (Trace.fork, do)",
        requires=frozenset({"float_output"}),
        prepare=_prepare_fork_do,
        after=_after_fork_do,
    ),
    Feature("index", CHEAP, "Lookup by label and module (trace[...])", after=_after_index),
    Feature(
        "multipass",
        CHEAP,
        "Layers run more than once (passes)",
        requires=frozenset({"recurrent"}),
        after=_after_multipass,
    ),
    Feature("record", MEDIUM, "Sparse recording (tl.record, to_trace)", after=_after_record),
    Feature(
        "extract",
        MEDIUM,
        "Feature extraction (tl.extract)",
        requires=frozenset({"child_module"}),
        after=_after_extract,
    ),
    Feature(
        "bind",
        MEDIUM,
        "Capture-free executor (spec.bind(model))",
        requires=frozenset({"child_module", "float_output"}),
        after=_after_bind,
    ),
    Feature("bundle", CHEAP, "Traces together (tl.bundle)", after=_after_bundle),
    Feature("validate", MEDIUM, "Saved-scope validation (tl.validate)", after=_after_validate),
    Feature("export", CHEAP, "Tabular export (to_pandas)", after=_after_export),
)

FEATURES_BY_NAME: dict[str, Feature] = {feature.name: feature for feature in FEATURES}


def feature(name: str) -> Feature:
    """Return the feature called ``name``."""

    return FEATURES_BY_NAME[name]
