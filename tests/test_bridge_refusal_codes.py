"""Typed refusal codes of the depyf and MLflow bridges.

Every refusal these bridges added raises a ``TorchLensError`` subclass that
keeps its historical built-in lineage (``ValueError``/``TypeError``/
``RuntimeError``) and carries a stable ``code`` plus a non-empty ``remedy``
(docs/reference/error_refusal_contract.md). These tests provoke each code with
fakes, so they need none of the bridged packages installed.
"""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Any

import pytest
import torch

from torchlens._errors import (
    ArgumentTypeError,
    CaptureNotActiveError,
    KeywordConflictError,
)
from torchlens.bridge import depyf as bdepyf
from torchlens.export import _trackers


def _assert_code(exc: BaseException, code: str) -> None:
    """The refusal carries the code and a non-empty remedy ending its message."""

    fields = getattr(exc, "fields", {})
    assert fields["code"] == code
    assert fields["remedy"]
    assert str(exc).rstrip().endswith(f"{fields['remedy']}.")


def _fake_depyf(monkeypatch: pytest.MonkeyPatch, *, prepare_debug: Any | None) -> None:
    """Install a stub ``depyf`` module, optionally without ``prepare_debug``."""

    module = types.ModuleType("depyf")
    if prepare_debug is not None:
        module.prepare_debug = prepare_debug  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "depyf", module)


def test_depyf_prepare_debug_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _fake_depyf(monkeypatch, prepare_debug=None)
    with pytest.raises(CaptureNotActiveError) as info:
        bdepyf.dump(torch.nn.Linear(2, 2), torch.randn(1, 2), tmp_path)
    assert isinstance(info.value, RuntimeError)
    _assert_code(info.value, "bridge_depyf_prepare_debug_missing")


def test_depyf_nothing_dumped(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _fake_depyf(monkeypatch, prepare_debug=lambda path, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(torch, "compile", lambda model: model)
    with pytest.raises(CaptureNotActiveError) as info:
        bdepyf.dump(torch.nn.Linear(2, 2), torch.randn(1, 2), tmp_path)
    _assert_code(info.value, "bridge_depyf_nothing_dumped")


class _FluentClient:
    """The fluent ``mlflow`` module shape: ``log_metric(key, value)``."""

    def log_metric(self, key: str, value: float) -> None:
        """Accept one metric."""


class _RunClient:
    """The ``MlflowClient`` shape: ``log_metric(run_id, key, value)``."""

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        """Accept one metric for a run."""


def _summary_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the trace summary so a bare object stands in for a Trace."""

    monkeypatch.setattr(_trackers, "_summary_metrics", lambda log: {"num_layers": 1})
    monkeypatch.setattr(_trackers, "capture_honesty_facts", lambda log: {})


def test_mlflow_run_id_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _summary_only(monkeypatch)
    with pytest.raises(KeywordConflictError) as info:
        _trackers.mlflow(object(), run_id="r1")
    assert isinstance(info.value, TypeError)
    _assert_code(info.value, "tracker_mlflow_run_id_without_client")


def test_mlflow_run_id_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _summary_only(monkeypatch)
    with pytest.raises(ArgumentTypeError) as info:
        _trackers.mlflow(object(), client=_RunClient())
    _assert_code(info.value, "tracker_mlflow_run_id_missing")


def test_mlflow_run_id_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    _summary_only(monkeypatch)
    with pytest.raises(KeywordConflictError) as info:
        _trackers.mlflow(object(), client=_FluentClient(), run_id="r1")
    _assert_code(info.value, "tracker_mlflow_run_id_unsupported")
