"""Regression tests for round-14 postprocess metadata hardening."""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._data_classes.trace import ConditionalAccessor, ConditionalEvent, Trace
from torchlens._postprocess.finalization import _build_conditional_records
from torchlens.validation.invariants import MetadataInvariantError


class _EquivalentOpsModel(nn.Module):
    """Small model that creates repeated linear and tanh equivalence groups."""

    def __init__(self) -> None:
        """Initialize the repeated linear block."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two repeated linear+tanh blocks.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated model output.
        """

        y = torch.tanh(self.lin(x))
        return torch.tanh(self.lin(y))


class _ConditionalBuildTrace:
    """Minimal trace stand-in for `_build_conditional_records` tests."""

    def __init__(self) -> None:
        """Populate only the fields used by conditional finalization."""

        self.conditional_records = [
            _make_conditional_event(0, "gt_1_3", 7),
            _make_conditional_event(1, "gt_1_3", 12),
        ]
        self.conditional_arm_entry_edges = {
            (0, "then"): [("gt_1_3", "then_1")],
            (1, "then"): [("gt_1_3", "then_2")],
        }
        self.layer_list = [
            _make_layer_stub("gt_1_3:1", "gt_1_3", terminal_conditional_id=0),
            _make_layer_stub("then_1:1", "then_1"),
            _make_layer_stub("gt_1_3:2", "gt_1_3", terminal_conditional_id=1),
            _make_layer_stub("then_2:1", "then_2"),
        ]
        self.layer_dict_all_keys = {
            "gt_1_3": SimpleNamespace(
                bool_value=True,
                parents=[],
                has_output_descendant=True,
            )
        }


def _make_conditional_event(event_id: int, bool_label: str, source_line: int) -> ConditionalEvent:
    """Build a small conditional event fixture.

    Parameters
    ----------
    event_id:
        Dense conditional-event id.
    bool_label:
        Final layer label for the terminal bool layer.
    source_line:
        Source line for the owning ``if`` statement.

    Returns
    -------
    ConditionalEvent
        Event with stable metadata for finalization tests.
    """

    return ConditionalEvent(
        id=event_id,
        kind="if_chain",
        source_file="test_file.py",
        function_qualname="Tiny.forward",
        function_span=(1, 20),
        if_stmt_span=(source_line, source_line + 1),
        test_span=(source_line, 0, source_line, 8),
        branch_ranges={"then": (source_line, 9, source_line + 1, 12)},
        branch_test_spans={"then": (source_line, 0, source_line, 8)},
        call_depth=0,
        parent_conditional_id=None,
        parent_branch_kind=None,
        bool_layers=[bool_label],
    )


def _make_layer_stub(
    label: str,
    layer_label: str,
    terminal_conditional_id: int | None = None,
) -> SimpleNamespace:
    """Create a minimal layer-like object for conditional finalization tests.

    Parameters
    ----------
    label:
        Op label stored on the stub.
    layer_label:
        Pass-stripped layer label.
    terminal_conditional_id:
        Dense conditional-event id for terminal bool ops.

    Returns
    -------
    SimpleNamespace
        Layer-like object exposing the fields touched by finalization.
    """

    return SimpleNamespace(
        label=label,
        layer_label=layer_label,
        in_conditionals=[],
        terminal_conditional_id=terminal_conditional_id,
        terminal_bool_for=None,
        parents=[],
        has_output_descendant=True,
    )


def _load_module_from_code(module_name: str, path: Path, code: str) -> ModuleType:
    """Write and import a Python module from source code.

    Parameters
    ----------
    module_name:
        Import name for the temporary module.
    path:
        Filesystem path to write.
    code:
        Python source code to load.

    Returns
    -------
    ModuleType
        Imported module object.
    """

    path.write_text(textwrap.dedent(code), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace_file_backed_model(
    tmp_path: Path,
    module_name: str,
    code: str,
    input_tensor: torch.Tensor,
) -> Trace:
    """Trace a file-backed model so conditional source metadata is available.

    Parameters
    ----------
    tmp_path:
        Per-test temporary directory.
    module_name:
        Import name for the module.
    code:
        Python source code containing class ``M``.
    input_tensor:
        Input tensor for the trace.

    Returns
    -------
    Trace
        Finished TorchLens trace for the temporary model.
    """

    module = _load_module_from_code(module_name, tmp_path / f"{module_name}.py", code)
    return tl.trace(module.M(), input_tensor)


def test_equivalent_ops_are_renamed_to_final_labels() -> None:
    """Per-op and per-layer equivalent ops should use final op labels."""

    trace = tl.trace(_EquivalentOpsModel(), torch.randn(1, 4))
    repeated_ops = [
        op
        for op in trace.layer_list
        if len(trace.graph.op_equivalence_classes.get(op.equivalence_class, set())) > 1
    ]

    assert repeated_ops
    for op in repeated_ops:
        expected_group = trace.graph.op_equivalence_classes[op.equivalence_class]
        assert op.equivalent_ops == expected_group
        assert all(not label.endswith("_raw") for label in op.equivalent_ops)

    repeated_layers = [
        layer
        for layer in trace.layer_logs.values()
        if len(getattr(layer, "equivalent_ops", set())) > 1
    ]
    assert repeated_layers
    for layer in repeated_layers:
        first_pass = next(iter(layer.ops.values()))
        assert layer.equivalent_ops == first_pass.equivalent_ops
        assert all(not label.endswith("_raw") for label in layer.equivalent_ops)


def test_equivalence_invariant_rejects_stale_per_op_views() -> None:
    """Equivalence invariants should catch stale per-op and per-layer views."""

    trace = tl.trace(_EquivalentOpsModel(), torch.randn(1, 4))
    target_op = next(
        op
        for op in trace.layer_list
        if len(trace.graph.op_equivalence_classes.get(op.equivalence_class, set())) > 1
    )
    target_op.equivalent_ops = {"linear_1_2_raw"}
    trace.layer_logs[target_op.layer_label].equivalent_ops = {"linear_1_2_raw"}

    with pytest.raises(MetadataInvariantError, match="equivalence_symmetry"):
        trace.check_metadata_invariants()


def test_synthetic_output_node_resets_equivalent_ops() -> None:
    """Synthetic output nodes should not inherit their parent equivalence set."""

    trace = tl.trace(_EquivalentOpsModel(), torch.randn(1, 4))
    output_op = next(op for op in trace.layer_list if op.is_output)

    assert output_op.equivalent_ops == {output_op.label}


def test_build_conditional_records_disambiguates_colliding_bool_layers() -> None:
    """Conditional public ids should stay unique when bool layers collide."""

    trace = _ConditionalBuildTrace()
    _build_conditional_records(trace)

    conditionals = list(trace.graph.conditionals)
    assert [conditional.id for conditional in conditionals] == [
        "cond_gt_1_3__event_0",
        "cond_gt_1_3__event_1",
    ]
    assert trace.layer_list[0].terminal_bool_for == ("cond_gt_1_3__event_0", 0)
    assert trace.layer_list[2].terminal_bool_for == ("cond_gt_1_3__event_1", 0)
    assert {role.conditional_id for role in trace.layer_list[1].in_conditionals} == {
        "cond_gt_1_3__event_0"
    }
    assert {role.conditional_id for role in trace.layer_list[3].in_conditionals} == {
        "cond_gt_1_3__event_1"
    }


def test_conditional_invariant_rejects_duplicate_public_ids(tmp_path: Path) -> None:
    """Conditional invariants should reject duplicate public ids."""

    trace = _trace_file_backed_model(
        tmp_path,
        "two_ifs",
        """
        import torch
        from torch import nn

        class M(nn.Module):
            def forward(self, x):
                y = x + 1
                if y.sum() > 0:
                    a = torch.tanh(y)
                else:
                    a = y
                if y.sum() > 0:
                    b = torch.tanh(y)
                else:
                    b = y
                return a + b
        """,
        torch.ones(2, 2),
    )
    conditionals = list(trace.graph.conditionals)
    conditionals[1].id = conditionals[0].id
    trace._graph_conditionals = ConditionalAccessor(conditionals)

    with pytest.raises(MetadataInvariantError, match="conditional_invariants"):
        trace.check_metadata_invariants()


def test_loop_conditionals_clear_ambiguous_fired_arm_summary(tmp_path: Path) -> None:
    """Looped conditionals should not claim one unique fired arm when two fired."""

    trace = _trace_file_backed_model(
        tmp_path,
        "loop_branch_model",
        """
        import torch
        from torch import nn

        class M(nn.Module):
            def forward(self, x):
                outs = []
                for i in range(x.shape[0]):
                    if x[i, 0] > 0:
                        outs.append(x[i] + 1)
                    else:
                        outs.append(x[i] - 1)
                return torch.stack(outs)
        """,
        torch.tensor([[1.0], [-1.0]]),
    )
    conditional = trace.graph.conditionals[0]

    assert [arm.fired for arm in conditional.arms] == [True, True]
    assert conditional.fired_arm_index is None
    assert conditional.fired_arm_kind is None
    assert trace.check_metadata_invariants() is True


def test_conditional_invariant_rejects_ambiguous_fired_arm_summary(tmp_path: Path) -> None:
    """Conditional invariants should reject a single-arm summary for multi-arm fires."""

    trace = _trace_file_backed_model(
        tmp_path,
        "loop_branch_model_invalid",
        """
        import torch
        from torch import nn

        class M(nn.Module):
            def forward(self, x):
                outs = []
                for i in range(x.shape[0]):
                    if x[i, 0] > 0:
                        outs.append(x[i] + 1)
                    else:
                        outs.append(x[i] - 1)
                return torch.stack(outs)
        """,
        torch.tensor([[1.0], [-1.0]]),
    )
    trace.graph.conditionals[0].fired_arm_index = 0
    trace.graph.conditionals[0].fired_arm_kind = "then"

    with pytest.raises(MetadataInvariantError, match="conditional_invariants"):
        trace.check_metadata_invariants()


def test_internal_sink_logging_never_rescans_the_sink_ledgers() -> None:
    """Sink logging must use shadow sets, not per-node list scans (R52-2).

    ``_log_internally_terminated_tensor`` guarded its appends with
    ``label not in self.internal_sink_ops`` / ``... not in
    self.internally_terminated_bool_ops`` -- list membership once per visited
    childless node, O(k^2) in sink count across the reachability flood. The
    guards now read the flood's persistent shadow sets; the lists stay the
    portable source of truth and stay duplicate-free.
    """

    from types import SimpleNamespace

    from torchlens._postprocess.graph_traversal import _log_internally_terminated_tensor

    class _CountingList(list):
        """List that counts membership scans."""

        contains_calls = 0

        def __contains__(self, item: object) -> bool:
            type(self).contains_calls += 1
            return super().__contains__(item)

    layers = {
        f"op_{i}_raw": SimpleNamespace(
            is_internal_sink=False,
            is_scalar_bool=(i % 2 == 0),
            is_terminal_bool=False,
        )
        for i in range(200)
    }

    class _TraceStub:
        """Minimal trace surface for the sink logger."""

        def __init__(self) -> None:
            self.internal_sink_ops = _CountingList()
            self.internally_terminated_bool_ops = _CountingList()

        def __getitem__(self, label: str) -> SimpleNamespace:
            return layers[label]

    trace = _TraceStub()

    seen_sinks: set[str] = set(trace.internal_sink_ops)
    seen_bools: set[str] = set(trace.internally_terminated_bool_ops)
    _CountingList.contains_calls = 0
    for label in list(layers) + list(layers):  # revisits exercise the dedupe guard
        _log_internally_terminated_tensor(
            trace,
            label,
            seen_sink_labels=seen_sinks,
            seen_terminated_bool_labels=seen_bools,
        )

    assert _CountingList.contains_calls == 0, (
        f"sink logging scanned the list ledgers {_CountingList.contains_calls} "
        "times -- the O(k^2) membership guard is back"
    )
    assert list(trace.internal_sink_ops) == list(layers)
    assert list(trace.internally_terminated_bool_ops) == [
        label for i, label in enumerate(layers) if i % 2 == 0
    ]
    for i, label in enumerate(layers):
        assert layers[label].is_internal_sink is True
        assert layers[label].is_terminal_bool is (i % 2 == 0)
