"""Behaviour tests for the postprocess step executor.

Steps 1-20 run in registry order through ``run_pipeline``; each step body
resolves its callable through the ``torchlens.postprocess`` namespace at call
time. These tests pin the order, the one-evaluation-per-step gate invariant,
the patch seams, and the observable effects of steps 17.5 and 19.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

#: Steps 1-20 in run order. Reordering steps is a reviewed semantic change.
REGISTRY_ORDER = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "11.5",
    "11.75",
    "12",
    "13",
    "14",
    "15",
    "15.5",
    "16",
    "16.5",
    "17",
    "17.5",
    "18",
    "19",
    "20",
)


class _TinyModel(nn.Module):
    """Minimal linear+relu model."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def test_registry_runs_steps_in_the_pinned_order() -> None:
    """The executor registry lists steps 1-20 in the pinned run order."""

    from torchlens.postprocess import _executor as ex

    assert tuple(spec.step for spec in ex.STEP_REGISTRY) == REGISTRY_ORDER


def test_executor_seam_patched_step_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A monkeypatched step function still executes.

    The executor resolves step callables through the module namespace at CALL
    time; a registry of imported references would silently break every
    monkeypatch seam.
    """

    import torchlens.postprocess as pp

    calls: list[str] = []
    real_step2 = pp._find_output_ancestors

    def patched_step2(trace: object) -> None:
        calls.append("ran")
        real_step2(trace)

    monkeypatch.setattr(pp, "_find_output_ancestors", patched_step2)
    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    try:
        assert calls == ["ran"], "the patched step body must have executed"
    finally:
        trace.cleanup()


def test_executor_should_run_called_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor invariant: should_run evaluates once per step, in order.

    Step 18's predicate is deliberately context-writing (THE streaming
    snapshot point); a re-evaluated step-19 predicate after 18 cleared
    _out_writer would always be false and streamed outs would never be
    evicted.
    """

    from torchlens.postprocess import _executor as ex

    counts: dict[str, int] = {}
    seen_order: list[str] = []
    original_registry = ex.STEP_REGISTRY

    def counting(spec: ex.StepSpec) -> ex.StepSpec:
        inner = spec.should_run

        def counted(ctx: ex.StepContext) -> bool:
            counts[spec.step] = counts.get(spec.step, 0) + 1
            seen_order.append(spec.step)
            return inner(ctx)

        return ex.StepSpec(spec.step, spec.run, counted)

    monkeypatch.setattr(ex, "STEP_REGISTRY", tuple(counting(spec) for spec in original_registry))
    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    try:
        assert counts == {spec.step: 1 for spec in original_registry}
        assert seen_order == list(REGISTRY_ORDER)
    finally:
        trace.cleanup()


def test_executor_failing_step_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising step propagates its exception out of ``tl.trace``."""

    import torchlens.postprocess as pp

    class _StepBoom(RuntimeError):
        pass

    def exploding_step9(trace: object) -> None:
        raise _StepBoom("step 9 exploded")

    monkeypatch.setattr(pp, "_log_final_info_for_layers", exploding_step9)
    with pytest.raises(_StepBoom):
        tl.trace(_TinyModel().eval(), torch.randn(2, 3))


def test_phase_timing_bucket_names_default_capture() -> None:
    """The _vtimed bucket-name set for a default capture is frozen (§5.5).

    The surface oracle normalizes timing fields, so a drift in WHICH
    buckets exist would pass the byte gate silently; this pins the set
    (step 13 and 17.5 are deliberately unwrapped; conditional steps absent
    when skipped).
    """

    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    try:
        buckets = {name for name in trace._phase_timings if name.startswith("postprocess:Step")}
        assert buckets == {
            "postprocess:Step 0: Materialize capture events",
            "postprocess:Step 1: Add output layers",
            "postprocess:Step 2: Trace output ancestors",
            "postprocess:Step 3: Remove orphan nodes",
            "postprocess:Step 4: Input/output distances",
            "postprocess:Step 5: Mark conditional branches",
            "postprocess:Step 6: Fix buffer layers",
            "postprocess:Step 7: Loop detection (full)",
            "postprocess:Step 8: Map labels",
            "postprocess:Step 9: Log final info",
            "postprocess:Step 10: Rename labels",
            "postprocess:Step 11: Build lookup keys",
            "postprocess:Step 11.5: Populate source var names",
            "postprocess:Step 11.75: Resolve deferred retention",
            "postprocess:Step 12: Undecorate tensors",
            "postprocess:Step 14: Log timing",
            "postprocess:Step 15: Finalize params",
            "postprocess:Step 15.5: Build layer logs",
            "postprocess:Step 16: Build module logs",
            "postprocess:Step 16.5: Graph shape hash",
            "postprocess:Step 17: Mark pass finished",
            "postprocess:Step 20: Release param refs",
        }
    finally:
        trace.cleanup()


class _DictOutputModel(nn.Module):
    """Model whose forward returns a dict output (populates container records)."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a two-key dict so the output container registry has a record."""

        return {"a": x.relu(), "b": x.sigmoid()}


@pytest.mark.smoke
def test_step17_5_adopts_container_records_before_the_registry_clears() -> None:
    """Step 17.5's one NON-redundant effect: adopt container records onto the trace.

    Mutation-margin arming (W2, lane L3 triage step 21): a first version of
    this test asserted the per-phase workspace attrs it also pops
    (``_raw_graph_ws`` etc.) were gone -- but ``_drop_transient_capture_state``
    (``postprocess/__init__.py``) unconditionally pops those SAME fields right
    after ``run_pipeline`` returns, for every capture, step 17.5 included or
    not. That made the first version vacuously true either way (empirically
    proven: a whole-function return-None disarm of ``_run_step_17_5``
    survived it). The one thing ONLY step 17.5 does is copy
    ``container_registry.records`` into ``trace._containers`` BEFORE that
    later scrub clears the registry's live state; skip step 17.5 and the
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


@pytest.mark.smoke
def test_step19_and_gate_evict_streamed_outs_from_memory(tmp_path: Path) -> None:
    """Step 19 (and its gate) must drop in-memory ``out`` once streamed.

    Mutation-margin arming (W2): ``_run_step_19`` return-None and
    ``_should_run_step_19`` forced-False both leave every streamed op's
    ``out`` resident in memory despite ``out_ref`` already pointing at the
    on-disk blob -- the exact observable effect this test pins. Neither
    mutant is visible to the phase-timing bucket golden above (step 19 is
    conditional and absent from that axis's expected set either way).
    """

    trace = tl.trace(
        _TinyModel().eval(),
        torch.randn(2, 3),
        storage=tl.to_disk(tmp_path / "step19.tlspec"),
    )
    try:
        streamed = [op for op in trace.layer_list if getattr(op, "out_ref", None) is not None]
        assert streamed, "no op streamed an out_ref; the axis setup is not exercising step 19"
        assert all(op.out is None for op in streamed), (
            "a streamed op's out survived step 19's eviction"
        )
    finally:
        trace.cleanup()


def test_buffer_duplicate_axis_actually_merges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 6 merges duplicate buffers and repoints ``buffer_source`` at the survivor."""

    from support.postprocess_axes import _axis_buffer_duplicate

    import torchlens.postprocess.control_flow as cf

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
        survivor = next(op for op in trace.layer_list if op._label_raw == survivor_label)
        survivor_final = survivor.layer_label if survivor.num_passes == 1 else survivor.label
        repointed = [
            op for op in trace.layer_list if op.is_buffer and op.buffer_source == survivor_final
        ]
        assert repointed, "the scalar buffer_source repoint must have fired"
        assert all(op.buffer_source not in (removed_label,) for op in trace.layer_list), (
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


def _clear_layer_label_map(trace: Any) -> None:
    trace._raw_to_final_layer_labels.clear()


def _clear_op_label_map(trace: Any) -> None:
    trace._raw_to_final_op_labels.clear()


def _drop_first_lookup_keys(trace: Any) -> None:
    trace.layer_list[0].lookup_keys = []


def _clear_layer_logs(trace: Any) -> None:
    trace.layer_logs.clear()


def _null_module_logs(trace: Any) -> None:
    trace._module_logs = None


def _postcondition_breakers() -> list[Any]:
    """(step, patched attribute, factory of the breaking replacement, message)."""

    import torchlens.postprocess as pp

    def noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    return [
        pytest.param("1", "_add_output_layers", lambda: noop, "must register output", id="1"),
        pytest.param(
            "8",
            "_map_raw_labels_to_final_labels",
            lambda: _wrap_then(pp._map_raw_labels_to_final_labels, _clear_layer_label_map),
            "raw-to-final layer labels",
            id="8-layer",
        ),
        pytest.param(
            "8",
            "_map_raw_labels_to_final_labels",
            lambda: _wrap_then(pp._map_raw_labels_to_final_labels, _clear_op_label_map),
            "raw-to-final op labels",
            id="8-op",
        ),
        pytest.param(
            "11",
            "_build_lookup_keys_and_finalize_retained_layers",
            lambda: _wrap_then(
                pp._build_lookup_keys_and_finalize_retained_layers, _drop_first_lookup_keys
            ),
            "without lookup keys",
            id="11",
        ),
        pytest.param(
            "15.5",
            "_build_layer_logs",
            lambda: _wrap_then(pp._build_layer_logs, _clear_layer_logs),
            "aggregate layer logs",
            id="15.5",
        ),
        pytest.param(
            "16",
            "_build_module_logs",
            lambda: _wrap_then(pp._build_module_logs, _null_module_logs),
            "module logs",
            id="16",
        ),
        pytest.param(
            "16.5",
            "compute_graph_shape_hash",
            lambda: lambda *_args, **_kwargs: None,
            "graph_shape_hash",
            id="16.5",
        ),
        pytest.param("17", "_set_tracing_finished", lambda: noop, "tracing finished", id="17"),
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

    import torchlens.postprocess as pp
    from torchlens.validation import MetadataInvariantError

    monkeypatch.setattr(pp, attribute, make_breaker())
    with pytest.raises(MetadataInvariantError) as excinfo:
        tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    assert excinfo.value.check_name == "postprocess_step_postcondition"
    text = str(excinfo.value)
    assert re.search(rf"\bStep {re.escape(step)} ", text), text
    assert message in text, text


def test_step_postconditions_cover_the_seven_steps() -> None:
    """The executor checks exactly the seven postcondition steps, all registered."""

    from torchlens.postprocess import _executor as ex

    assert set(ex.STEP_POSTCONDITIONS) == {"1", "8", "11", "15.5", "16", "16.5", "17"}
    assert set(ex.STEP_POSTCONDITIONS) <= {spec.step for spec in ex.STEP_REGISTRY}
