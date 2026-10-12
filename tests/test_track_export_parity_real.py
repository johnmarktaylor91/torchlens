"""MLflow and Aim tracker exports at parity with the wandb export, on real stores.

``tl.export.mlflow`` and ``tl.export.aim`` must write, beside the three summary
metrics, a per-layer statistics table, the report (with the capture-honesty
preamble) and every capture-honesty fact as a string tag. MLflow runs against a
local file store and Aim against a local repo; nothing leaves the temp folder.
The MLflow and Aim groups import their package inside a fixture so one missing
package never hides the other's tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_honesty import capture_honesty_facts, honesty_preamble_lines

pytestmark = [pytest.mark.optional, pytest.mark.heavy]

_KEYS = ("num_layers", "num_saved_ops", "total_activation_memory")


@pytest.fixture
def trace() -> Any:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    return tl.trace(model, torch.randn(3, 4))


def _fact_text(value: Any) -> str:
    """The documented string form of one honesty fact (strings verbatim, else sorted JSON)."""

    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


@pytest.fixture
def mlflow_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The mlflow module and a local FILE-store URI (not sqlite)."""

    mlflow = pytest.importorskip("mlflow")
    # Newer MLflow gates the file store behind this opt-in; older ones ignore it.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return mlflow, f"file://{tmp_path}/mlruns"


def _assert_mlflow_run(
    client: Any, run_id: str, trace: Any, result: dict[str, Any], prefix: str, tmp_path: Path
) -> None:
    """Read the run back through the client and check every parity feature."""

    tmp_path.mkdir(parents=True, exist_ok=True)  # download_artifacts needs an existing dir
    run = client.get_run(run_id)
    assert run.data.metrics == {f"{prefix}.{key}": float(result[key]) for key in _KEYS}
    assert run.data.metrics[f"{prefix}.num_layers"] == len(trace.layer_list)

    artifacts = {item.path for item in client.list_artifacts(run_id, prefix)}
    assert artifacts == {f"{prefix}/layers.json", f"{prefix}/report.txt"}
    assert result["artifacts"] == [f"{prefix}/layers.json", f"{prefix}/report.txt"]

    layers = json.loads(
        Path(client.download_artifacts(run_id, f"{prefix}/layers.json", str(tmp_path))).read_text()
    )
    assert len(layers["data"]) == len(trace.layer_list) == len(result["table"])
    assert list(layers["columns"]) == list(result["table"].columns)

    report = Path(
        client.download_artifacts(run_id, f"{prefix}/report.txt", str(tmp_path))
    ).read_text()
    assert report == result["report"]
    for line in honesty_preamble_lines(trace):
        assert line in report

    facts = capture_honesty_facts(trace)
    tags = {key: value for key, value in run.data.tags.items() if key.startswith(f"{prefix}.")}
    assert tags == {
        f"{prefix}.capture_honesty.{fact}": _fact_text(value) for fact, value in facts.items()
    }
    assert result["tags"] == tags
    assert tags[f"{prefix}.capture_honesty.capture_status"] == "complete"


def test_mlflow_client_on_a_file_store_has_parity(
    trace: Any, mlflow_store: Any, tmp_path: Path
) -> None:
    mlflow, uri = mlflow_store
    client = mlflow.MlflowClient(tracking_uri=uri)
    run_id = client.create_run(client.create_experiment("torchlens")).info.run_id
    result = tl.export.mlflow(trace, client=client, prefix="tl", run_id=run_id)
    _assert_mlflow_run(client, run_id, trace, result, "tl", tmp_path / "read")


def test_mlflow_fluent_module_on_a_file_store_has_parity(
    trace: Any, mlflow_store: Any, tmp_path: Path
) -> None:
    mlflow, uri = mlflow_store
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        with mlflow.start_run() as run:
            result = tl.export.mlflow(trace, client=mlflow, prefix="tl")
        client = mlflow.MlflowClient(tracking_uri=uri)
        _assert_mlflow_run(client, run.info.run_id, trace, result, "tl", tmp_path / "read")
    finally:
        mlflow.set_tracking_uri(previous)


def test_mlflow_fluent_module_with_run_id_targets_that_run(
    trace: Any, mlflow_store: Any, tmp_path: Path
) -> None:
    mlflow, uri = mlflow_store
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        client = mlflow.MlflowClient(tracking_uri=uri)
        target = client.create_run(client.create_experiment("target")).info.run_id
        with mlflow.start_run() as active:
            result = tl.export.mlflow(trace, client=mlflow, prefix="tl", run_id=target)
        _assert_mlflow_run(client, target, trace, result, "tl", tmp_path / "read")
        # The active run received nothing: the explicit run id won for every write.
        assert client.get_run(active.info.run_id).data.metrics == {}
        assert not any(
            key.startswith("tl.") for key in client.get_run(active.info.run_id).data.tags
        )
    finally:
        mlflow.set_tracking_uri(previous)


@pytest.fixture
def aim_run(tmp_path: Path) -> Any:
    """An Aim repo in ``tmp_path`` and the module; runs are opened by the tests."""

    aim = pytest.importorskip("aim")
    aim.Repo.from_path(str(tmp_path), init=True)
    return aim


def _open_run(aim: Any, tmp_path: Path) -> Any:
    return aim.Run(
        repo=str(tmp_path),
        log_system_params=False,
        capture_terminal_logs=False,
        system_tracking_interval=None,
    )


def _aim_series(run: Any, name: str, context: dict[str, Any]) -> tuple[list[int], list[float]]:
    """Read one tracked series back through the Aim SDK as ``(steps, values)``."""

    for metric in run.metrics():
        found = metric.context
        found = found.to_dict() if hasattr(found, "to_dict") else dict(found)
        if metric.name == name and found == context:
            steps, values = metric.values.sparse_numpy()
            return list(steps), list(values)
    raise AssertionError(f"series {name!r} with context {context!r} was not tracked")


def _aim_tag_names(run: Any) -> set[str]:
    tags = run.props.tags
    return {tag.name for tag in tags}


def test_aim_run_on_a_local_repo_has_parity(trace: Any, aim_run: Any, tmp_path: Path) -> None:
    aim = aim_run
    run = _open_run(aim, tmp_path)
    try:
        result = tl.export.aim(trace, run=run, prefix="tl")
        run_hash = run.hash
    finally:
        run.close()

    read = aim.Run(run_hash=run_hash, repo=str(tmp_path), read_only=True)
    # Summary metrics (existing behavior, kept).
    for key in _KEYS:
        steps, values = _aim_series(read, f"tl.{key}", {})
        assert values == [float(result[key])]
    # Per-layer table: lossless run param, one row per layer, same columns.
    rows = read["tl.layers"]
    assert len(rows) == len(trace.layer_list) == len(result["table"])
    assert set(rows[0]) == set(result["table"].columns)
    # ... plus one chartable series per numeric column, step = row index.
    column = next(
        name
        for name in rows[0]
        if all(
            isinstance(row[name], int | float) and not isinstance(row[name], bool) for row in rows
        )
    )
    steps, values = _aim_series(read, f"tl.layers.{column}", {"table": "layers"})
    assert steps == list(range(len(rows)))
    assert values == [float(row[column]) for row in rows]
    # Report: the honesty preamble rides the run param.
    report = read["tl.report"]
    assert report == result["report"]
    for line in honesty_preamble_lines(trace):
        assert line in report
    # Honesty facts: full dict as a param, and one string tag per fact.
    facts = capture_honesty_facts(trace)
    assert read["tl.capture_honesty"] == json.loads(json.dumps(facts))
    expected = {f"tl.capture_honesty.{fact}={_fact_text(value)}" for fact, value in facts.items()}
    assert _aim_tag_names(read) == expected
    assert "tl.capture_honesty.capture_status=complete" in expected
