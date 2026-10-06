"""Typed refusal codes of the contrastive, depyf, nnsight and MLflow bridges.

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

import numpy as np
import pytest
import torch

from torchlens._errors import (
    ArgumentTypeError,
    CaptureContextError,
    InvalidArgumentError,
    KeywordConflictError,
)
from torchlens.bridge import _contrastive, depyf as bdepyf, nnsight as bnnsight
from torchlens.export import _trackers


def _site(out: torch.Tensor, label: str = "site") -> types.SimpleNamespace:
    """A layer-like site: ``resolve_one_site`` passes it through unchanged."""

    return types.SimpleNamespace(out=out, layer_label=label)


def _log() -> types.SimpleNamespace:
    """A trace-like object with no saved attention mask and no live model."""

    return types.SimpleNamespace(layer_list=[])


def _assert_code(exc: BaseException, code: str) -> None:
    """The refusal carries the code and a non-empty remedy ending its message."""

    fields = getattr(exc, "fields", {})
    assert fields["code"] == code
    assert fields["remedy"]
    assert str(exc).rstrip().endswith(f"{fields['remedy']}.")


def _rows(positive: torch.Tensor, negative: torch.Tensor, **kwargs: Any) -> Any:
    """Call the shared contrastive reader with explicit sites in one trace."""

    return _contrastive._contrastive_rows(
        _log(),
        _site(positive, "pos"),
        _site(negative, "neg"),
        negative_log=None,
        read_token_index=kwargs.pop("read_token_index", -1),
        **kwargs,
    )


def test_contrastive_rows_mismatch() -> None:
    with pytest.raises(InvalidArgumentError) as info:
        _rows(torch.randn(2, 3, 4), torch.randn(3, 3, 4))
    assert isinstance(info.value, ValueError)
    _assert_code(info.value, "bridge_contrastive_rows_mismatch")


def test_contrastive_rows_identical() -> None:
    rows = torch.randn(2, 3, 4)
    with pytest.raises(InvalidArgumentError) as info:
        _rows(rows, rows.clone())
    _assert_code(info.value, "bridge_contrastive_rows_identical")


def test_contrastive_negative_missing() -> None:
    with pytest.raises(InvalidArgumentError) as info:
        _contrastive._negative_source(_log(), _site(torch.randn(2, 3, 4)), None, None)
    _assert_code(info.value, "bridge_contrastive_negative_missing")


def test_contrastive_read_token_rank() -> None:
    with pytest.raises(InvalidArgumentError) as info:
        _contrastive._read_rows(torch.randn(2, 4), -1, None, "positive")
    _assert_code(info.value, "bridge_contrastive_read_token_rank")


def test_contrastive_read_token_count() -> None:
    with pytest.raises(InvalidArgumentError) as info:
        _contrastive._read_rows(torch.randn(2, 3, 4), [0, 1, 2], None, "positive")
    _assert_code(info.value, "bridge_contrastive_read_token_count")


def test_contrastive_mask_shape() -> None:
    with pytest.raises(InvalidArgumentError) as info:
        _contrastive._read_rows(torch.randn(2, 3, 4), -1, torch.ones(2, 5), "positive")
    _assert_code(info.value, "bridge_contrastive_mask_shape")


def test_contrastive_mask_empty_row() -> None:
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
    with pytest.raises(InvalidArgumentError) as info:
        _contrastive._read_rows(torch.randn(2, 3, 4), -1, mask, "negative")
    _assert_code(info.value, "bridge_contrastive_mask_empty_row")


def test_contrastive_method_unknown() -> None:
    with pytest.raises(InvalidArgumentError) as info:
        _contrastive._read_directions(
            np.zeros((4, 3), dtype=np.float32),
            "no_such_method",
            diff_methods=("pca_diff",),
            center_in_place=False,
            mean_diff=False,
        )
    _assert_code(info.value, "bridge_contrastive_method_unknown")


def test_contrastive_model_type_unknown() -> None:
    with pytest.raises(InvalidArgumentError) as info:
        _contrastive._model_type(_log(), None)
    _assert_code(info.value, "bridge_contrastive_model_type_unknown")


def _fake_depyf(monkeypatch: pytest.MonkeyPatch, *, prepare_debug: Any | None) -> None:
    """Install a stub ``depyf`` module, optionally without ``prepare_debug``."""

    module = types.ModuleType("depyf")
    if prepare_debug is not None:
        module.prepare_debug = prepare_debug  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "depyf", module)


def test_depyf_prepare_debug_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _fake_depyf(monkeypatch, prepare_debug=None)
    with pytest.raises(CaptureContextError) as info:
        bdepyf.dump(torch.nn.Linear(2, 2), torch.randn(1, 2), tmp_path)
    assert isinstance(info.value, RuntimeError)
    _assert_code(info.value, "bridge_depyf_prepare_debug_missing")


def test_depyf_nothing_dumped(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _fake_depyf(monkeypatch, prepare_debug=lambda path, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(torch, "compile", lambda model: model)
    with pytest.raises(CaptureContextError) as info:
        bdepyf.dump(torch.nn.Linear(2, 2), torch.randn(1, 2), tmp_path)
    _assert_code(info.value, "bridge_depyf_nothing_dumped")


def test_nnsight_trace_unsupported() -> None:
    with pytest.raises(ArgumentTypeError) as info:
        bnnsight.from_trace(object())
    assert isinstance(info.value, TypeError)
    _assert_code(info.value, "bridge_nnsight_trace_unsupported")


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
