"""Re-check contracts for metadata-only display cards and collection identities."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.intervention
from torchlens._data_classes.func_call_location import FuncCallLocation
from torchlens._ir.events import GradFnDiscovered
from torchlens._selection.resolver import _SiteTable
from torchlens._utils import _display_card
from torchlens._utils._display_bounded import RecordTuple, short_dataclass_repr
from torchlens._utils._display_errors import lookup_message
from torchlens.intervention.types import CapturedArgTemplate, LiteralValue, ParentRef
from torchlens.validation.consolidated import InterventionValidationReport
from torchlens.validation.diagnostics import ValidationFailure, record_validation_failure


@pytest.fixture
def trace():
    captured = tl.trace(
        nn.Sequential(nn.Linear(4, 4), nn.ReLU()),
        torch.ones(2, 4),
        capture=tl.options.CaptureOptions(aten_ops=True),
    )
    yield captured
    captured.cleanup()


def test_ordinary_card_labels_fit_shared_column(trace, monkeypatch):
    """Cover every ordinary card builder in this lane through real objects."""

    labels = []
    original = _display_card.card_row

    def record(label, value, *, width=_display_card.LABEL_WIDTH):
        labels.append(label)
        return original(label, value, width=width)

    monkeypatch.setattr(_display_card, "card_row", record)
    trace = tl.trace(
        nn.Sequential(nn.Linear(4, 4), nn.ReLU()),
        torch.ones(2, 4),
        capture=tl.options.CaptureOptions(save_grads=True, backward_ready=True, aten_ops=True),
    )
    trace.backward(trace[trace.output_layers[0]].out.sum())
    recipe = tl.when(tl.func("relu"), tl.zero_ablate())
    selection = trace[tl.func("relu")]
    objects = [
        trace.event_stream,
        trace.op_events,
        trace.aten_ops,
        *trace.aten_ops,
        trace.autograd.grad_fns[0],
        trace.autograd.grad_fns[0].calls[0],
        trace.autograd.backward_passes[0],
        trace.cost,
        trace.graph,
        trace.health,
        trace.autograd,
        trace.source,
        trace.options,
        trace.provenance,
        trace.validation_replay_status,
        ValidationFailure("forward_replay", op_label="relu_1_7"),
        InterventionValidationReport(True, True, False, True, True, {}),
        selection,
        selection[0],
        _SiteTable(tuple(trace._layer_list)),
        trace["relu_1"].edge_uses[0],
        recipe,
        recipe.rules[0],
        recipe.bind(nn.ReLU()),
        trace.recipe,
        CapturedArgTemplate(
            args=(LiteralValue(1),), kwargs=(("long_keyword_name", LiteralValue(2)),)
        ),
        FuncCallLocation(file="missing.py", line_number=1, func_name="forward"),
    ]
    for obj in objects:
        text = str(obj)
        assert text.splitlines()[0] == repr(obj), type(obj).__name__
        assert "metadata unavailable" not in text, type(obj).__name__
        assert text.splitlines()[-1].lstrip().startswith("more "), type(obj).__name__
    trace.cleanup()
    assert labels
    assert all(len(label) <= 9 for label in labels), labels


def test_option_and_error_columns_align_to_longest_label():
    options = tl.options.CaptureOptions(save_grads=True)
    rows = str(options).splitlines()[1:]
    widths = [len(line) - len(line.lstrip()) + len(line.lstrip().split()[0]) for line in rows]
    value_column = max(widths) + 1
    for line in rows:
        assert len(line) > value_column and line[value_column] != " "
    assert rows[-1].lstrip().startswith("more ")
    assert "None" not in str(options)
    assert "  save_grads" in str(options) and "enabled" in str(options)
    error = lookup_message("Missing key.", ["far", "near"], "choose a key", query="nea")
    assert error.splitlines()[1:] == [
        "  candidates 'near', 'far'",
        "  remedy     choose a key",
    ]


def test_group_display_guards_the_report_door(trace, monkeypatch):
    """A failed deferred report import cannot break notebook display."""

    from torchlens._data_classes import _trace_groups

    def unavailable():
        raise ImportError("report unavailable")

    monkeypatch.setattr(_trace_groups._doors, "report", unavailable)
    assert repr(trace.graph) == "TraceGraph: display unavailable"
    assert str(trace.graph) == repr(trace.graph)


def test_candidates_rank_before_cut_and_remedy_survives():
    candidates = ["zzzzzz", "yyyyyy", "xxxxxx", "wwwwww", "vvvvvv", "tanh_6", "tanh_7"]
    text = lookup_message("Layer not found.", candidates, "use a key.", query="tanh_6")
    assert text.index("'tanh_6'") < text.index("'tanh_7'") < text.index("'zzzzzz'")
    assert "'vvvvvv'" not in text and "; and 2 more" in text
    assert tl.errors.TorchLensError(text).fields["remedy"] == "use a key"
    assert candidates[0] == "zzzzzz"


def test_template_literals_and_parent_keys_are_readable(trace):
    template = trace["relu_1"].args_template
    assert "linear_1" in str(template)
    assert "<ParentRef>" not in str(template)
    nested = CapturedArgTemplate(
        args=((LiteralValue(2), LiteralValue(3)), ParentRef("stored_label_9")),
    )
    text = str(nested)
    assert "(2, 3)" in text and "stored_label_9" in text
    assert "<LiteralValue>" not in text


def test_validation_locations_resolve_without_rewriting_stored_label(trace):
    op = trace["relu_1"]
    failure = ValidationFailure("forward_replay", op_label=op.label)
    record_validation_failure(trace, failure)
    assert failure.op_label == op.label
    assert "  location  relu_1" in str(failure)
    detached = ValidationFailure("forward_replay", op_label="unknown_88_99")
    assert "  location  unknown_88_99" in str(detached)
    report = InterventionValidationReport(True, True, False, True, True, {})
    assert repr(report) == "InterventionValidationReport: failed, 4 of 5 axes passed"
    assert "  complete  failed" in str(report)
    assert "False" not in str(report) and "True" not in str(report)


def test_family_identity_and_singular_counts():
    event = GradFnDiscovered(
        123456, "ReluBackward0", "ReluBackward0", False, "relu_1", None, 1, 987654, {}, (), 2
    )
    text = repr(event)
    assert text.startswith("GradFnDiscovered(class_name='ReluBackward0'")
    assert "object_id" not in text and "123456" not in text and "987654" not in text

    @dataclass
    class Row:
        payload: object
        label: str
        items: tuple

    row = Row(SimpleNamespace(secret="nested"), "key_1", (1,))
    assert short_dataclass_repr(row) == "Row(label='key_1', items=<1 item>)"
    rows = RecordTuple((row,), "Rows", "rows")
    assert repr(rows) == "Rows: 1 row"
    assert str(rows).splitlines()[0] == repr(rows)
    assert "key_1" in str(rows).splitlines()[1]
    assert str(rows).endswith("  more      .count()  .index()")


def test_aten_inputs_have_separate_rows(trace):
    row = next(row for row in trace.aten_ops if len(row.input_tensor_facts) >= 3)
    original = row.input_tensor_facts
    row.input_tensor_facts = (*original[:3], *original[:3])
    try:
        text = str(row)
        facts = text.split("  inputs", 1)[1].split("  outputs", 1)[0].splitlines()
        assert len(facts) == 4
        assert facts[-1].strip() == "and 3 more tensors"
        assert all(line.count("stride") <= 1 for line in facts)
        row.algorithmic_flops = None
        row.flop_status = "unsupported"
        assert "  flops     not counted for this operator" in str(row)
    finally:
        row.input_tensor_facts = original
