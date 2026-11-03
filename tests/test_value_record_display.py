"""Record displays share bounded facts without implicit parameter or device scans."""

from __future__ import annotations

from dataclasses import replace
from html import escape
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._value_params import buffer_card
from torchlens._data_classes._value_state import LAZY_DISK, payload_facts, value_line
from torchlens.stats import _record_values, _tensor_stats
from torchlens.stats._stats_kernel import SAMPLE_CAP
from torchlens.stats._tensor_stats import FamilyEvidence, tensor_stats


class _Reuse(nn.Module):
    """A shared module gives three passes and a live parameter record."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(self.fc(x))
        return x


@pytest.fixture
def captured():
    """Keep the source model alive while inspecting live parameter metadata."""

    model = _Reuse()
    trace = tl.trace(model, torch.arange(8, dtype=torch.float32).reshape(2, 4))
    yield trace, model
    trace.cleanup()


def test_saved_cpu_statistics_compute_once_and_invalidate(monkeypatch) -> None:
    """The existing versioned cache bounds repeat rendering to one scan."""

    tensor = torch.tensor([-1.0, 0.5, 1.0])
    calls = []
    kernel = _tensor_stats.run_kernel

    def counted(*args, **kwargs):
        calls.append(1)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(_tensor_stats, "run_kernel", counted)
    first = _record_values.record_stats(tensor, compute=True)
    assert _record_values.record_stats(tensor, compute=True) is first
    assert len(calls) == 1
    tensor.add_(1)
    second = _record_values.record_stats(tensor, compute=True)
    assert second is not first and second.finite_max == 2
    assert len(calls) == 2


def test_live_parameter_never_scans_and_only_reuses_current_version(captured, monkeypatch) -> None:
    """Even a tiny live parameter is cold until explicitly summarized."""

    trace, model = captured
    param = trace.params["fc.weight"]
    _tensor_stats._STATS_CACHE.pop(id(model.fc.weight), None)
    assert "not computed" in str(param)
    tensor_stats(model.fc.weight)
    assert "min " in str(param)
    with torch.no_grad():
        model.fc.weight.add_(1)

    def no_scan(*args, **kwargs):
        pytest.fail("printing a live parameter scanned its values")

    monkeypatch.setattr(_record_values, "tensor_stats", no_scan)
    assert "not computed" in str(param)
    assert "card unavailable" not in param._repr_html_()


def test_device_and_over_gate_tensors_never_scan(monkeypatch) -> None:
    """The cold-compute gate checks device and count before kernel entry."""

    def no_scan(*args, **kwargs):
        pytest.fail("cold device or over-gate scan")

    monkeypatch.setattr(_record_values, "tensor_stats", no_scan)
    assert _record_values.record_stats(torch.empty(4, device="meta"), compute=True) is None
    assert _record_values.record_stats(torch.empty(SAMPLE_CAP + 1), compute=True) is None


def test_lazy_payload_and_buffer_do_not_materialize() -> None:
    """Both Op probes and Buffer snapshot probes use resident slots only."""

    class Lazy:
        dtype = torch.float32
        shape = (4,)
        device = "cpu"

        def _slot(self, name):
            return object() if name == "out_ref" else None

        @property
        def out(self):
            pytest.fail("a repr materialized a lazy payload")

    assert payload_facts(Lazy())[2] == LAZY_DISK
    buffer = SimpleNamespace(
        address="running_mean", dtype=torch.float32, shape=(4,), device="cpu", versions=[Lazy()]
    )
    assert "metadata only" in buffer_card(buffer)
    assert "not computed" in buffer_card(buffer)


def test_reused_layer_has_per_pass_values_without_a_pooled_summary(captured) -> None:
    """A Layer cannot claim one statistic for three different output tensors."""

    trace, _ = captured
    layer = trace.layers["tanh_1"]
    assert repr(layer) == "Layer tanh_1: 3 passes, float32 [2, 4] on cpu"
    assert repr(layer.ops[1]).startswith("Op tanh_1:2: float32 [2, 4] on cpu")
    card = str(layer)
    assert "min " not in card.splitlines()[0]
    assert all(f"pass {index}" in card for index in (1, 2, 3))
    assert len(card.splitlines()) <= 11


def test_html_and_text_cards_have_identical_rows(captured) -> None:
    """Notebook cards escape exactly the text facts and never add a value grid."""

    trace, _ = captured
    for record in (trace["tanh_1:2"], trace.layers["tanh_1"], trace.params["fc.weight"]):
        text = str(record)
        html = record._repr_html_()
        assert "card unavailable" not in html
        assert escape(repr(record)) in html
        for line in text.splitlines()[1:]:
            assert f'<th scope="row">{escape(line[2:11].strip())}</th>' in html
            assert f"<td>{escape(line[12:])}</td>" in html
        assert '<table class="tl-grid-table"' not in html
        assert html.count("<style>") == 1
        assert len(html) < 20_000


def test_nested_record_cards_share_one_stylesheet(captured) -> None:
    """A notebook card tree includes the record stylesheet only once."""

    from torchlens.report._notebook._value_cards import op_value_card, param_value_card
    from torchlens.report._notebook.cardtree import CardCollection, render_card_html

    trace, _ = captured
    tree = CardCollection(
        children=(op_value_card(trace["tanh_1:2"]), param_value_card(trace.params["fc.weight"]))
    )
    html = render_card_html(tree)
    assert html.count("<style>") == 1
    assert html.count('<table class="tl-record-facts">') == 2


def test_value_words_keep_sampled_and_nonfinite_evidence() -> None:
    """Dropping a sparkline never drops uncertainty or nonfinite counts."""

    stats = tensor_stats(torch.tensor([1.0, 2.0, float("nan"), float("inf")]))
    sample = FamilyEvidence("sampled", population=100, sample_size=2)
    sampled = replace(stats, mean_evidence=sample, mean_se=0.1)
    words = _record_values.render_values(sampled)
    assert "NaN 1" in words and "+Inf 1" in words and "mean ~" in words
    assert "min 1.000  max 2.000" in words
    assert "▁" not in words and "mean=" not in words


def test_width_ladder_preserves_metadata_and_honesty() -> None:
    """Moments are optional; device, geometry, key and state are required."""

    stats = tensor_stats(torch.tensor([-1.0, 2.0]))
    for key in ("relu_1", "some_module_activation_1", "some_longer_module_activation_1"):
        line = value_line(f"Op {key}", "float32 [1, 5, 3072] on cpu", stats, "(edited)")
        assert len(line) <= 100
        assert f"Op {key}: float32 [1, 5, 3072] on cpu (edited)" in line
        if "sd " in line:
            assert "min " in line and "max " in line and "mean " in line


def test_source_row_never_reads_current_source(captured, monkeypatch) -> None:
    """Source cards use captured frame metadata, even if files are unavailable."""

    from torchlens._data_classes.func_call_location import FuncCallLocation

    def no_source(*args, **kwargs):
        pytest.fail("record rendering read source text")

    monkeypatch.setattr(FuncCallLocation, "_source_text_state", no_source)
    trace, _ = captured
    card = str(trace["tanh_1:2"])
    assert "source    " in card and " in " in card
    assert "(capture not finished)" not in card


def test_graph_count_names_output_descendants(captured) -> None:
    """Reachable model outputs are not a count of every downstream operation."""

    trace, _ = captured
    op = trace["tanh_1:2"]
    count = len(op.output_descendants)
    noun = "output" if count == 1 else "outputs"
    assert f"({count} {noun} downstream)" in str(op)
