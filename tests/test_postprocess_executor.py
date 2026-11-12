"""Behaviour tests for the postprocess pipeline and its step executor.

The named steps run in ``PIPELINE`` order through ``run_pipeline``; each step
adapter calls its function through the function's home module
(``graph_traversal``, ``labeling``, ``finalization``, ...). These tests pin
the order, the record contract (names, home modules, declared writes), the
one-evaluation-per-step gate invariant, the home-module seams, and the
observable effects of the container-adoption and streamed-out eviction steps.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._postprocess._pipeline import PIPELINE
from torchlens._utils._record_group_table import record_member
from torchlens.errors import TraceCaptureError

#: The pipeline in run order. Reordering steps is a reviewed semantic change.
PIPELINE_ORDER = (
    "add_output_layers",
    "find_output_ancestors",
    "remove_orphan_nodes",
    "mark_layer_depths",
    "mark_conditional_branches",
    "fix_buffer_layers",
    "group_recurrent_layers",
    "map_final_labels",
    "log_final_layer_info",
    "build_lookup_keys",
    "populate_source_var_names",
    "resolve_deferred_retention",
    "undecorate_saved_tensors",
    "clear_cuda_cache",
    "log_time_elapsed",
    "finalize_param_logs",
    "build_layer_logs",
    "build_module_logs",
    "hash_graph_shape",
    "adopt_containers_and_drop_workspaces",
    "seal_op_store",
    "mark_tracing_finished",
    "finalize_streamed_bundle",
    "evict_streamed_outs",
    "release_param_refs",
)

#: Fields every timed step rewrites through ``_vtimed``; not a per-step write.
_TIMING_FIELDS = frozenset({"_phase_timings"})


class _TinyModel(nn.Module):
    """Minimal linear+relu model."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def test_pipeline_runs_steps_in_the_pinned_order() -> None:
    """``PIPELINE`` lists the named steps in the pinned run order."""

    from torchlens._postprocess import _pipeline

    assert tuple(step.name for step in _pipeline.PIPELINE) == PIPELINE_ORDER


def test_pipeline_step_names_are_unique_snake_case_without_ranks() -> None:
    """Steps are identified by name alone: unique snake_case, no digits, no ranks.

    No record carries a rank or fractional id; the tuple position is the order.
    """

    from torchlens._postprocess import _pipeline

    names = [step.name for step in _pipeline.PIPELINE]
    assert len(set(names)) == len(names)
    for name in names:
        assert name.isidentifier() and name == name.lower(), name
        assert not any(ch.isdigit() for ch in name), name
    assert {f.name for f in _pipeline.PipelineStep.__dataclass_fields__.values()} == {
        "name",
        "module",
        "run",
        "reads",
        "writes",
        "should_run",
        "postcondition",
    }


def test_pipeline_step_modules_resolve_and_own_the_adapter_or_body() -> None:
    """Every record's home module imports under ``torchlens``.

    A step implemented inline names the executor; every other step's module is
    the home module its adapter calls through.
    """

    from torchlens._postprocess import _pipeline

    for step in _pipeline.PIPELINE:
        module = importlib.import_module(f"torchlens.{step.module}")
        assert module is not None, step.name
        assert step.run.__name__ == f"_run_{step.name}", step.name
        assert step.run.__module__ == "torchlens._postprocess._executor", step.name
        if step.should_run is not _pipeline._always:
            assert step.should_run.__name__ == f"_should_run_{step.name}", step.name
        if step.postcondition is not None:
            assert step.postcondition.__name__ == f"_{step.name}_violation", step.name
        assert step.reads == tuple(sorted(set(step.reads))), step.name
        assert step.writes == tuple(sorted(set(step.writes))), step.name


def test_pipeline_declared_writes_cover_observed_trace_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each step's ``writes`` covers every trace attribute it adds, drops or rebinds.

    The check watches ``trace.__dict__``: keys added or removed by a step and
    keys whose value object was replaced (``is`` identity) must be declared.
    In-place mutation of a list or dict is not observable this way, so the
    declared set may be wider than what this test sees, never narrower.
    """

    import dataclasses

    from torchlens._postprocess import _pipeline

    undeclared: dict[str, set[str]] = {}

    def observing(step: _pipeline.PipelineStep) -> _pipeline.PipelineStep:
        inner = step.run

        def run(ctx: _pipeline.StepContext) -> None:
            before = dict(ctx.trace.__dict__)
            inner(ctx)
            after = ctx.trace.__dict__
            changed = {
                key
                for key in set(before) | set(after)
                if key not in before or key not in after or before[key] is not after[key]
            }
            extra = changed - set(step.writes) - _TIMING_FIELDS
            if extra:
                undeclared[step.name] = extra

        return dataclasses.replace(step, run=run)

    monkeypatch.setattr(
        _pipeline, "PIPELINE", tuple(observing(step) for step in _pipeline.PIPELINE)
    )
    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    try:
        assert undeclared == {}, undeclared
    finally:
        trace.cleanup()


def test_executor_seam_patched_step_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A step function replaced on its home module executes.

    The executor calls each step through its home module (here
    ``graph_traversal``) at call time; a registry of imported references
    would silently break every home-module seam.
    """

    import torchlens._postprocess as pp

    calls: list[str] = []
    real_step2 = pp.graph_traversal._find_output_ancestors

    def patched_step2(trace: object) -> None:
        calls.append("ran")
        real_step2(trace)

    monkeypatch.setattr(pp.graph_traversal, "_find_output_ancestors", patched_step2)
    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    try:
        assert calls == ["ran"], "the patched step body must have executed"
    finally:
        trace.cleanup()


def test_executor_should_run_called_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor invariant: should_run evaluates once per step, in order.

    The ``finalize_streamed_bundle`` gate is deliberately context-writing (THE
    streaming snapshot point); a re-evaluated ``evict_streamed_outs`` gate
    after that step cleared _out_writer would always be false and streamed
    outs would never be evicted.
    """

    import dataclasses

    from torchlens._postprocess import _pipeline

    counts: dict[str, int] = {}
    seen_order: list[str] = []
    original = _pipeline.PIPELINE

    def counting(step: _pipeline.PipelineStep) -> _pipeline.PipelineStep:
        inner = step.should_run

        def counted(ctx: _pipeline.StepContext) -> bool:
            counts[step.name] = counts.get(step.name, 0) + 1
            seen_order.append(step.name)
            return inner(ctx)

        return dataclasses.replace(step, should_run=counted)

    monkeypatch.setattr(_pipeline, "PIPELINE", tuple(counting(step) for step in original))
    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    try:
        assert counts == {step.name: 1 for step in original}
        assert seen_order == list(PIPELINE_ORDER)
    finally:
        trace.cleanup()


def test_executor_failing_step_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising step propagates its exception out of ``tl.trace``."""

    import torchlens._postprocess as pp

    class _StepBoom(RuntimeError):
        pass

    def exploding_step9(trace: object) -> None:
        raise _StepBoom("step 9 exploded")

    monkeypatch.setattr(pp.labeling, "_log_final_info_for_layers", exploding_step9)
    with pytest.raises(TraceCaptureError) as _tce_info:
        tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    assert isinstance(_tce_info.value.__cause__, _StepBoom)


def test_phase_timing_bucket_names_default_capture() -> None:
    """The _vtimed bucket-name set for a default capture is frozen (§5.5, io.5 step-name keys).

    The surface oracle normalizes timing fields, so a drift in WHICH
    buckets exist would pass the byte gate silently; this pins the set
    (clear_cuda_cache and adopt_containers_and_drop_workspaces are deliberately
    unwrapped; conditional steps absent when skipped).
    """

    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    try:
        step_names = {step.name for step in PIPELINE}
        buckets = {name for name in trace._phase_timings if name in step_names}
        assert set(trace._phase_timings) >= {"materialize_capture_events"}
        assert buckets == {
            "add_output_layers",
            "find_output_ancestors",
            "remove_orphan_nodes",
            "mark_layer_depths",
            "mark_conditional_branches",
            "fix_buffer_layers",
            "group_recurrent_layers",
            "map_final_labels",
            "log_final_layer_info",
            "build_lookup_keys",
            "populate_source_var_names",
            "resolve_deferred_retention",
            "undecorate_saved_tensors",
            "log_time_elapsed",
            "finalize_param_logs",
            "build_layer_logs",
            "build_module_logs",
            "hash_graph_shape",
            "mark_tracing_finished",
            "release_param_refs",
        }
    finally:
        trace.cleanup()


class _DictOutputModel(nn.Module):
    """Model whose forward returns a dict output (populates container records)."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a two-key dict so the output container registry has a record."""

        return {"a": x.relu(), "b": x.sigmoid()}


def test_adopt_containers_step_adopts_records_before_the_registry_clears() -> None:
    """The adopt_containers_and_drop_workspaces step's one NON-redundant effect: adopt container records onto the trace.

    Mutation-margin arming (W2, lane L3 triage step 21): a first version of
    this test asserted the per-phase workspace attrs it also pops
    (``_raw_graph_ws`` etc.) were gone -- but ``_drop_transient_capture_state``
    (``_postprocess/__init__.py``) unconditionally pops those SAME fields right
    after ``run_pipeline`` returns, for every capture, the adopt step included or
    not. That made the first version vacuously true either way (empirically
    proven: a whole-function return-None disarm of ``_run_adopt_containers_and_drop_workspaces``
    survived it). The one thing ONLY this step does is copy
    ``container_registry.records`` into ``trace._containers`` BEFORE that
    later scrub clears the registry's live state; skip it and the
    records are lost, never adopted, with no second chance to recover them.
    """

    trace = tl.trace(
        _DictOutputModel().eval(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(capture_container_structure=True),
    )
    try:
        containers = trace.__dict__.get("_containers")
        assert containers, "container records were never adopted onto the trace"
    finally:
        trace.cleanup()


def test_evict_streamed_outs_step_and_gate_evict_from_memory(tmp_path: Path) -> None:
    """evict_streamed_outs (and its gate) must drop in-memory ``out`` once streamed.

    Mutation-margin arming (W2): ``_run_evict_streamed_outs`` return-None and
    ``_should_run_evict_streamed_outs`` forced-False both leave every streamed op's
    ``out`` resident in memory despite ``out_ref`` already pointing at the
    on-disk blob -- the exact observable effect this test pins. Neither
    mutant is visible to the phase-timing bucket golden above (the step is
    conditional and absent from that axis's expected set either way).
    """

    trace = tl.trace(
        _TinyModel().eval(),
        torch.randn(2, 3),
        storage=tl.to_disk(tmp_path / "step19.tlspec"),
    )
    try:
        streamed = [
            op for op in trace._layer_list if record_member(op, "out_ref", None) is not None
        ]
        assert streamed, (
            "no op streamed an out_ref; the axis setup is not exercising the eviction step"
        )
        streamed = [
            op for op in trace._layer_list if record_member(op, "out_ref", None) is not None
        ]
        assert streamed, "no op streamed an out_ref; the axis setup is not exercising step 19"
        assert all(op.out is None for op in streamed), (
            "a streamed op's out survived the eviction step"
        )
    finally:
        trace.cleanup()


def test_buffer_duplicate_axis_actually_merges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 6 merges duplicate buffers and repoints ``buffer_source`` at the survivor."""

    from support.postprocess_axes import _axis_buffer_duplicate

    import torchlens._postprocess.control_flow as cf

    merges: list[tuple[str, str]] = []
    real_merge = cf._merge_buffer_entries

    def spying_merge(
        trace: Any,
        survivor: Any,
        removed: Any,
        *,
        deferred_removals: Any = None,
        survivor_edge_shadows: Any = None,
    ) -> None:
        merges.append((survivor._label_raw, removed._label_raw))
        real_merge(
            trace,
            survivor,
            removed,
            deferred_removals=deferred_removals,
            survivor_edge_shadows=survivor_edge_shadows,
        )

    monkeypatch.setattr(cf, "_merge_buffer_entries", spying_merge)
    trace = _axis_buffer_duplicate()
    try:
        assert merges, "the buffer_duplicate axis must reach _merge_buffer_entries"
        survivor_label, removed_label = merges[0]
        # buffer_source now rides the raw->final label rename (L4's
        # _SCALAR_LABEL_FIELDS_TO_RENAME fix), so resolve the survivor's FINAL
        # label for the comparison; the removed buffer never gets a final label.
        # Final labels follow the lookup convention parents and buffer_source
        # use: the bare layer label for a single-pass layer, pass-qualified
        # only for a pass of a multi-pass one. Under call-site grouping the
        # buffer's versions are grouped by the place that wrote them
        # (docs/guides/grouping.md), so the survivor can be single-pass.
        survivor = next(op for op in trace._layer_list if op._label_raw == survivor_label)
        survivor_final = survivor.layer_label if survivor.num_passes == 1 else survivor.label
        repointed = [
            op
            for op in trace._layer_list
            if op.graph.is_buffer and op.graph.buffer_source == survivor_final
        ]
        assert repointed, "the scalar buffer_source repoint must have fired"
        assert all(op.graph.buffer_source not in (removed_label,) for op in trace._layer_list), (
            "no surviving op may still reference the merged-away buffer"
        )
    finally:
        trace.cleanup()


def _wrap_then(real: Any, after: Any) -> Any:
    """Return a step function that runs ``real`` and then breaks its result."""

    def patched(trace: Any, *args: Any) -> Any:
        result = real(trace, *args)
        after(trace)
        return result

    return patched


def _empty_output_layers(trace: Any) -> None:
    # Before step 1, ``output_layers`` already holds the output parents' raw
    # labels, so a no-op step 1 would not leave it empty; empty it explicitly.
    trace.output_layers = []


def _duplicate_output_layer(trace: Any) -> None:
    trace.output_layers = [*trace.output_layers, trace.output_layers[0]]


def _clear_layer_label_map(trace: Any) -> None:
    trace._raw_to_final_layer_labels.clear()


def _clear_op_label_map(trace: Any) -> None:
    trace._raw_to_final_op_labels.clear()


def _drop_first_lookup_keys(trace: Any) -> None:
    trace._layer_list[0].lookup_keys = []


def _clear_layer_logs(trace: Any) -> None:
    trace._layer_logs.clear()


def _drop_module_pending(trace: Any) -> None:
    trace.__dict__.pop("_module_pending", None)


def _postcondition_breakers() -> list[Any]:
    """(step name, patched attribute, factory of the breaking replacement, message)."""

    import torchlens._postprocess as pp

    def noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    return [
        pytest.param(
            "add_output_layers",
            "_add_output_layers",
            lambda: _wrap_then(pp.graph_traversal._add_output_layers, _empty_output_layers),
            "registered 0, expected 1",
            id="1-missing",
        ),
        pytest.param(
            "add_output_layers",
            "_add_output_layers",
            lambda: _wrap_then(pp.graph_traversal._add_output_layers, _duplicate_output_layer),
            "registered 2, expected 1",
            id="1-extra",
        ),
        pytest.param(
            "map_final_labels",
            "_map_raw_labels_to_final_labels",
            lambda: _wrap_then(pp.labeling._map_raw_labels_to_final_labels, _clear_layer_label_map),
            "raw-to-final layer labels",
            id="8-layer",
        ),
        pytest.param(
            "map_final_labels",
            "_map_raw_labels_to_final_labels",
            lambda: _wrap_then(pp.labeling._map_raw_labels_to_final_labels, _clear_op_label_map),
            "raw-to-final op labels",
            id="8-op",
        ),
        pytest.param(
            "build_lookup_keys",
            "_build_lookup_keys_and_finalize_retained_layers",
            lambda: _wrap_then(
                pp.labeling._build_lookup_keys_and_finalize_retained_layers, _drop_first_lookup_keys
            ),
            "without lookup keys",
            id="11",
        ),
        pytest.param(
            "build_layer_logs",
            "_build_layer_logs",
            lambda: _wrap_then(pp.finalization._build_layer_logs, _clear_layer_logs),
            "aggregate layer logs",
            id="15.5",
        ),
        pytest.param(
            "build_module_logs",
            # The step's whole body: the seal leaves the records pending for first lookup.
            "_seal_module_logs",
            lambda: _wrap_then(pp.finalization._seal_module_logs, _drop_module_pending),
            "module logs",
            id="16",
        ),
        pytest.param(
            "hash_graph_shape",
            "compute_graph_shape_hash",
            lambda: lambda *_args, **_kwargs: None,
            "graph_shape_hash",
            id="16.5",
        ),
        pytest.param(
            "mark_tracing_finished",
            "_set_tracing_finished",
            lambda: noop,
            "tracing finished",
            id="17",
        ),
    ]


@pytest.mark.parametrize(
    ("step", "attribute", "make_breaker", "message"), _postcondition_breakers()
)
def test_unmet_step_postcondition_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
    step: str,
    attribute: str,
    make_breaker: Any,
    message: str,
) -> None:
    """A step that leaves its postcondition unmet fails the capture, naming the step.

    The check runs on the default path (no env flag) and is a typed raise, not
    an ``assert``, so ``python -O`` cannot strip it.
    """

    import re

    import torchlens._postprocess as pp
    from torchlens._utils import hashing
    from torchlens.errors import MetadataInvariantError

    homes = {
        "_add_output_layers": pp.graph_traversal,
        "_map_raw_labels_to_final_labels": pp.labeling,
        "_build_lookup_keys_and_finalize_retained_layers": pp.labeling,
        "_build_layer_logs": pp.finalization,
        "_build_module_logs": pp.finalization,
        "_seal_module_logs": pp.finalization,
        "compute_graph_shape_hash": hashing,
        "_set_tracing_finished": pp.finalization,
    }
    monkeypatch.setattr(homes[attribute], attribute, make_breaker())
    with pytest.raises(MetadataInvariantError) as excinfo:
        tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    assert excinfo.value.check_name == "postprocess_step_postcondition"
    assert excinfo.value.fields["code"] == "postprocess_step_postcondition"
    text = str(excinfo.value)
    assert re.search(rf"\b{re.escape(step)} ", text), text
    assert message in text, text


def test_step_postconditions_cover_the_seven_steps() -> None:
    """Exactly the seven postcondition steps declare a postcondition."""

    from torchlens._postprocess import _pipeline

    checked = {step.name for step in _pipeline.PIPELINE if step.postcondition is not None}
    assert checked == {
        "add_output_layers",
        "map_final_labels",
        "build_lookup_keys",
        "build_layer_logs",
        "build_module_logs",
        "hash_graph_shape",
        "mark_tracing_finished",
    }
