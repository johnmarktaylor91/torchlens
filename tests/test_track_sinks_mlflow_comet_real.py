"""MLflowSink and CometSink under ``watch`` against the real packages (local only).

MLflow logs into a local file store; Comet runs as an ``OfflineExperiment`` that
only writes an archive under ``tmp_path`` (no account, no key, no network). Both
sinks must keep the caller's step axis (steps 100, 102, 104, never 0, 1, 2) and
leave the user's run or experiment open. The two groups import their package
inside a fixture so one missing package never hides the other's tests.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch

import torchlens.trackers as trk

pytestmark = [pytest.mark.optional, pytest.mark.heavy]

_STEPS = (100, 102, 104)


def _mlp() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 2))
    return model, torch.optim.SGD(model.parameters(), lr=0.1)


def _train(session: trk.WatchSession, model: torch.nn.Module, opt: torch.optim.Optimizer) -> None:
    for step in _STEPS:
        with session.step(step):
            opt.zero_grad(set_to_none=True)
            model(torch.randn(3, 4)).sum().backward()
            opt.step()


@pytest.fixture
def mlflow_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The mlflow module and a local FILE-store URI (not sqlite)."""

    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return mlflow, f"file://{tmp_path}/mlruns"


def _watch_into_mlflow(client: Any, sink: trk.MLflowSink, run_id: str) -> trk.MemorySink:
    """Watch a tiny model into ``sink`` (plus a memory sink that records the truth)."""

    memory = trk.MemorySink()
    model, opt = _mlp()
    session = trk.watch(model, to=(sink, memory), optimizer=opt, every=1, hist_every=1)
    try:
        _train(session, model, opt)
    finally:
        session.close()
    assert client.get_run(run_id).info.status == "RUNNING", "close() must not end the run"
    return memory


def _assert_mlflow_matches(
    client: Any, run_id: str, memory: trk.MemorySink, tmp_path: Path
) -> None:
    assert memory.scalars and memory.histograms and memory.texts
    (tmp_path / "read").mkdir(parents=True, exist_ok=True)  # download_artifacts needs it
    run = client.get_run(run_id)
    assert set(run.data.metrics) == {point.tag for point in memory.scalars}
    emitted_steps: set[int] = set()
    for tag in sorted(run.data.metrics):
        history = client.get_metric_history(run_id, tag)
        expected = [(p.step, p.value) for p in memory.scalars if p.tag == tag]
        assert sorted((m.step, m.value) for m in history) == sorted(expected), tag
        emitted_steps.update(m.step for m in history)
    # The caller's axis, not MLflow's own counter (0, 1, 2).
    assert emitted_steps == set(_STEPS)
    for point in memory.texts:
        path = client.download_artifacts(
            run_id, f"{point.tag}/step_{point.step:010d}.txt", str(tmp_path / "read")
        )
        assert Path(path).read_text() == point.text
    for point in memory.histograms:
        path = client.download_artifacts(
            run_id, f"{point.tag}/step_{point.step:010d}.json", str(tmp_path / "read")
        )
        stored = json.loads(Path(path).read_text())
        assert stored["counts"] == list(point.counts)
        assert stored["edges"] == list(point.bin_edges)
        assert stored["summary"] == point.summary


def test_mlflow_sink_over_a_client_keeps_the_callers_steps(
    mlflow_store: Any, tmp_path: Path
) -> None:
    mlflow, uri = mlflow_store
    client = mlflow.MlflowClient(tracking_uri=uri)
    run_id = client.create_run(client.create_experiment("watch")).info.run_id
    memory = _watch_into_mlflow(client, trk.MLflowSink(client, run_id=run_id), run_id)
    _assert_mlflow_matches(client, run_id, memory, tmp_path)


def test_mlflow_sink_over_the_fluent_module_logs_into_the_active_run(
    mlflow_store: Any, tmp_path: Path
) -> None:
    mlflow, uri = mlflow_store
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        with mlflow.start_run() as run:
            sink = trk.MLflowSink(mlflow)
            assert sink.run_id == run.info.run_id
            client = mlflow.MlflowClient(tracking_uri=uri)
            memory = _watch_into_mlflow(client, sink, run.info.run_id)
            assert mlflow.active_run() is not None
        _assert_mlflow_matches(client, run.info.run_id, memory, tmp_path)
    finally:
        mlflow.set_tracking_uri(previous)


@pytest.fixture
def comet_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """An ``OfflineExperiment`` with every auto-logging and environment feature off."""

    comet_ml = pytest.importorskip("comet_ml")
    monkeypatch.delenv("COMET_API_KEY", raising=False)
    monkeypatch.setenv("COMET_DISABLE_AUTO_LOGGING", "1")
    return comet_ml.OfflineExperiment(
        offline_directory=str(tmp_path / "comet"),
        project_name="torchlens-sink-test",
        auto_param_logging=False,
        auto_metric_logging=False,
        auto_output_logging=False,
        auto_log_co2=False,
        parse_args=False,
        log_code=False,
        log_graph=False,
        log_env_details=False,
        log_git_metadata=False,
        log_git_patch=False,
        log_env_gpu=False,
        log_env_host=False,
        log_env_cpu=False,
        log_env_network=False,
        log_env_disk=False,
        display_summary_level=0,
    )


def _archive_messages(directory: Path) -> tuple[zipfile.ZipFile, list[dict[str, Any]]]:
    archives = sorted(directory.glob("*.zip"))
    assert len(archives) == 1, archives
    archive = zipfile.ZipFile(archives[0])
    lines = archive.read("messages.json").decode("utf-8").splitlines()
    return archive, [json.loads(line) for line in lines if line.strip()]


def test_comet_sink_offline_archive_holds_the_callers_steps(
    comet_offline: Any, tmp_path: Path
) -> None:
    memory = trk.MemorySink()
    model, opt = _mlp()
    session = trk.watch(model, to=(trk.CometSink(comet_offline), memory), optimizer=opt, every=1)
    try:
        _train(session, model, opt)
    finally:
        session.close()
    comet_offline.end()

    archive, messages = _archive_messages(tmp_path / "comet")
    metrics = [m["payload"]["metric"] for m in messages if m["type"] == "metric_msg"]
    arrived = {(m["metricName"], m["step"]): m["metricValue"] for m in metrics}
    assert memory.scalars
    for point in memory.scalars:
        assert arrived[(point.tag, point.step)] == pytest.approx(point.value)
    assert {step for (name, step) in arrived if name in {p.tag for p in memory.scalars}} == set(
        _STEPS
    )
    uploads = [
        m["payload"]
        for m in messages
        if m["type"] == "file_upload" and m["payload"]["upload_type"] == "text-sample"
    ]
    assert memory.texts
    expected = {(p.step, p.tag, p.text) for p in memory.texts}
    assert {
        (
            u["additional_params"]["step"],
            u["metadata"]["tag"],
            archive.read(u["file_path"]).decode(),
        )
        for u in uploads
    } == expected
