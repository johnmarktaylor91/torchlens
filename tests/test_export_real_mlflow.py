"""MLflow tracker export against the real mlflow package (sqlite backend).

``tl.export.mlflow`` must work with both MLflow client shapes: the fluent
``mlflow`` module (active run) and an ``MlflowClient`` (``run_id=``), and log
exactly what logging the same metrics with MLflow directly records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

mlflow = pytest.importorskip("mlflow")

pytestmark = [pytest.mark.optional, pytest.mark.slow]

_KEYS = ("num_layers", "num_saved_ops", "total_activation_memory")


@pytest.fixture
def trace() -> Any:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    return tl.trace(model, torch.randn(3, 4))


@pytest.fixture
def client(tmp_path: Path) -> Any:
    return mlflow.MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")


def test_mlflowclient_with_run_id_matches_direct(trace: Any, client: Any) -> None:
    experiment = client.create_experiment("torchlens")
    bridge_run = client.create_run(experiment).info.run_id
    direct_run = client.create_run(experiment).info.run_id

    metrics = tl.export.mlflow(trace, client=client, run_id=bridge_run)
    for key in _KEYS:
        client.log_metric(direct_run, f"torchlens.{key}", metrics[key])

    bridged = client.get_run(bridge_run).data.metrics
    direct = client.get_run(direct_run).data.metrics
    assert bridged == direct
    assert bridged["torchlens.num_layers"] == len(trace.layer_list)
    print(f"\nbridge={bridged} direct={direct}")


def test_mlflowclient_without_run_id_is_refused(trace: Any, client: Any) -> None:
    with pytest.raises(TypeError, match="without run_id"):
        tl.export.mlflow(trace, client=client)


def test_fluent_module_still_logs_to_the_active_run(trace: Any, tmp_path: Path) -> None:
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/fluent.db")
    try:
        with mlflow.start_run() as run:
            metrics = tl.export.mlflow(trace, client=mlflow)
        read = mlflow.MlflowClient().get_run(run.info.run_id).data.metrics
    finally:
        mlflow.set_tracking_uri(previous)
    assert read == {f"torchlens.{key}": float(metrics[key]) for key in _KEYS}
