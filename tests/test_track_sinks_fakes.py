"""MLflowSink, CometSink and the MLflow/Aim export parity features against fake objects.

No tracker package is needed: each fake records exactly what the sink or the
exporter sends, so the contracts (caller step axis, typed refusals, no run
start or end, refuse-before-write) run everywhere. The real-package legs live
in ``test_track_sinks_mlflow_comet_real.py`` and ``test_track_export_parity_real.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.trackers as trk
from torchlens.trackers._errors import SinkProtocolError, TagGrammarError


class _FakeMlflowClient:
    """The ``MlflowClient`` shape: every method takes the run id first."""

    def __init__(self) -> None:
        self.metrics: list[tuple[str, str, float, int | None]] = []
        self.texts: dict[tuple[str, str], str] = {}
        self.dicts: dict[tuple[str, str], dict[str, Any]] = {}

    def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        timestamp: int | None = None,
        step: int | None = None,
    ) -> None:
        self.metrics.append((run_id, key, value, step))

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        self.texts[(run_id, artifact_file)] = text

    def log_dict(self, run_id: str, dictionary: dict[str, Any], artifact_file: str) -> None:
        self.dicts[(run_id, artifact_file)] = dictionary


class _FakeComet:
    """The slice of a ``comet_ml`` experiment the sink uses."""

    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, int | None]] = []
        self.texts: list[tuple[str, int | None, dict[str, Any] | None]] = []
        self.ended = False

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        self.metrics.append((name, value, step))

    def log_text(
        self, text: str, step: int | None = None, metadata: dict[str, Any] | None = None
    ) -> None:
        self.texts.append((text, step, metadata))

    def end(self) -> None:
        self.ended = True


def _mlp() -> tuple[nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    return model, torch.optim.SGD(model.parameters(), lr=0.1)


def _train(session: trk.WatchSession, model: nn.Module, opt: torch.optim.Optimizer) -> None:
    for step in (100, 102, 104):
        with session.step(step):
            opt.zero_grad(set_to_none=True)
            model(torch.randn(3, 4)).sum().backward()
            opt.step()


def test_mlflow_sink_keeps_the_callers_step_axis_and_never_ends_the_run() -> None:
    client = _FakeMlflowClient()
    sink = trk.MLflowSink(client, run_id="run-1")
    memory = trk.MemorySink()
    model, opt = _mlp()
    session = trk.watch(model, to=(sink, memory), optimizer=opt, every=1, hist_every=1)
    try:
        _train(session, model, opt)
    finally:
        session.close()
    assert memory.scalars and memory.histograms and memory.texts
    assert {run_id for run_id, *_ in client.metrics} == {"run-1"}
    assert {(key, step) for _r, key, _v, step in client.metrics} == {
        (point.tag, point.step) for point in memory.scalars
    }
    assert {step for *_x, step in client.metrics} == {100, 102, 104}
    for point in memory.histograms:
        stored = client.dicts[("run-1", f"{point.tag}/step_{point.step:010d}.json")]
        assert stored["counts"] == list(point.counts)
        assert stored["edges"] == list(point.edges)
        assert stored["summary"] == point.summary
    for point in memory.texts:
        assert client.texts[("run-1", f"{point.tag}/step_{point.step:010d}.txt")] == point.text
    assert sink.run_id == "run-1"


def test_mlflow_sink_over_the_fluent_module_uses_the_active_run() -> None:
    client = _FakeMlflowClient()
    module = SimpleNamespace(
        active_run=lambda: SimpleNamespace(info=SimpleNamespace(run_id="active-9")),
        MlflowClient=lambda: client,
    )
    sink = trk.MLflowSink(module)
    sink.emit_scalar(trk.ScalarPoint(tag="parameters/mean/w", step=7, value=0.5))
    assert client.metrics == [("active-9", "parameters/mean/w", 0.5, 7)]
    pinned = trk.MLflowSink(module, run_id="other")
    assert pinned.run_id == "other"


def test_mlflow_sink_refuses_typed_without_a_live_run() -> None:
    no_run = SimpleNamespace(active_run=lambda: None, MlflowClient=_FakeMlflowClient)
    with pytest.raises(SinkProtocolError) as no_active:
        trk.MLflowSink(no_run)
    assert no_active.value.fields["code"] == "tracker_sink_invalid"
    with pytest.raises(SinkProtocolError) as no_id:
        trk.MLflowSink(_FakeMlflowClient())
    assert no_id.value.fields["code"] == "tracker_sink_invalid"
    with pytest.raises(SinkProtocolError) as bare:
        trk.MLflowSink(object(), run_id="r")
    assert bare.value.fields["code"] == "tracker_sink_invalid"


def test_mlflow_sink_without_the_package_refuses_naming_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(SinkProtocolError) as info:
        trk.MLflowSink()
    assert info.value.fields["code"] == "tracker_sink_unavailable"
    assert "pip install mlflow" in info.value.fields["remedy"]


def test_mlflow_sink_refuses_a_tag_mlflow_cannot_carry() -> None:
    sink = trk.MLflowSink(_FakeMlflowClient(), run_id="r")
    with pytest.raises(TagGrammarError) as info:
        sink.emit_scalar(trk.ScalarPoint(tag="parameters/mean/blocks[0].w", step=1, value=1.0))
    assert info.value.fields["code"] == "tracker_tag_unsafe"


def test_sinks_advertise_what_they_carry() -> None:
    mlflow_caps = trk.MLflowSink(_FakeMlflowClient(), run_id="r").capabilities()
    assert mlflow_caps == {"scalar", "raw_histogram", "text_manifest", "run_metadata"}
    comet_caps = trk.CometSink(_FakeComet()).capabilities()
    assert comet_caps == {"scalar", "text_manifest", "run_metadata"}
    trk.require_capabilities(
        trk.MLflowSink(_FakeMlflowClient(), run_id="r"), ("scalar", "raw_histogram")
    )
    with pytest.raises(SinkProtocolError) as info:
        trk.require_capabilities(trk.CometSink(_FakeComet()), ("raw_histogram",))
    assert info.value.fields["code"] == "tracker_sink_capability_missing"


def test_comet_sink_keeps_the_callers_step_axis_and_never_ends_the_experiment() -> None:
    experiment = _FakeComet()
    sink = trk.CometSink(experiment)
    memory = trk.MemorySink()
    model, opt = _mlp()
    session = trk.watch(model, to=(sink, memory), optimizer=opt, every=1)
    try:
        _train(session, model, opt)
    finally:
        session.close()
    assert {(name, step) for name, _v, step in experiment.metrics} == {
        (point.tag, point.step) for point in memory.scalars
    }
    assert {step for *_x, step in experiment.metrics} == {100, 102, 104}
    assert [(text, step, meta) for text, step, meta in experiment.texts] == [
        (point.text, point.step, {"tag": point.tag}) for point in memory.texts
    ]
    assert experiment.ended is False


def test_comet_sink_histograms_refuse_by_name_at_attach() -> None:
    model, opt = _mlp()
    with pytest.raises(SinkProtocolError) as info:
        trk.watch(model, to=trk.CometSink(_FakeComet()), optimizer=opt, hist_every=1)
    assert info.value.fields["code"] == "tracker_sink_capability_missing"
    assert info.value.fields["missing"] == ("raw_histogram",)
    point = trk.HistogramPoint(tag="t", step=1, counts=(1,), edges=(0.0, 1.0))
    with pytest.raises(SinkProtocolError):
        trk.CometSink(_FakeComet()).emit_histogram(point)


def test_comet_sink_refuses_a_non_experiment() -> None:
    with pytest.raises(SinkProtocolError) as info:
        trk.CometSink(object())
    assert info.value.fields["code"] == "tracker_sink_invalid"


@pytest.fixture
def trace() -> Any:
    torch.manual_seed(0)
    return tl.trace(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)), torch.randn(3, 4))


def test_mlflow_export_refuses_a_client_without_parity_methods_before_logging(trace: Any) -> None:
    class _MetricsOnly:
        def __init__(self) -> None:
            self.logged: list[str] = []

        def log_metric(self, key: str, value: float) -> None:
            self.logged.append(key)

    client = _MetricsOnly()
    with pytest.raises(TypeError, match="log_table"):
        tl.export.mlflow(trace, client=client)
    assert client.logged == []


def test_aim_export_refuses_a_run_without_parity_methods_before_tracking(trace: Any) -> None:
    class _TrackOnly:
        def __init__(self) -> None:
            self.tracked: list[str] = []

        def track(self, value: float, name: str) -> None:
            self.tracked.append(name)

    run = _TrackOnly()
    with pytest.raises(TypeError, match="add_tag"):
        tl.export.aim(trace, to=run)
    assert run.tracked == []
