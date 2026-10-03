"""Serve one-liner tests (memo D18, B5).

The full-package launch is a scheduled leg (it needs the vendor app's
server stack); here the contract is pinned: the missing-dependency refusal
teaches the exact install, and the call routes through the PUBLIC vendor
``visualize`` API only -- exported file first, no private attribute of the
pinned dependency.
"""

from __future__ import annotations

import builtins
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


@pytest.fixture()
def tiny_log() -> Any:
    """Trace a two-layer toy."""

    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
    return tl.trace(model, torch.randn(1, 3))


def test_serve_without_vendor_refuses_with_install_remedy(
    tiny_log: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent app package -> MissingDependencyError naming the pip install."""

    real_import = builtins.__import__

    def deny_vendor(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "model_explorer":
            raise ImportError("gated for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "model_explorer", raising=False)
    monkeypatch.setattr(builtins, "__import__", deny_vendor)
    with pytest.raises(Exception) as excinfo:
        tl.export.model_explorer_serve(tiny_log)
    assert excinfo.value.fields["code"] == "model_explorer_serve_unavailable"
    assert "pip install ai-edge-model-explorer" in excinfo.value.fields["install"]


def test_serve_exports_then_calls_public_visualize(
    tiny_log: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Trace source exports first; visualize() gets the file path."""

    calls: list[dict[str, Any]] = []
    fake_vendor = types.ModuleType("model_explorer")

    def visualize(model_paths: list[str], **kwargs: Any) -> None:
        """Record the public-API call."""

        calls.append({"model_paths": model_paths, **kwargs})

    fake_vendor.visualize = visualize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "model_explorer", fake_vendor)
    destination = tl.export.model_explorer_serve(
        tiny_log, path=tmp_path / "served.json", reuse_server=True
    )
    assert destination == tmp_path / "served.json"
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert set(payload) == {"label", "graphs", "graphSorting"}
    assert calls == [{"model_paths": [str(destination)], "reuse_server": True}]


@pytest.mark.smoke
def test_serve_accepts_an_existing_export_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path source is served as-is, no re-export."""

    calls: list[list[str]] = []
    fake_vendor = types.ModuleType("model_explorer")

    def visualize(model_paths: list[str], **kwargs: Any) -> None:
        """Record the served paths."""

        calls.append(model_paths)

    fake_vendor.visualize = visualize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "model_explorer", fake_vendor)
    existing = tmp_path / "already.json"
    existing.write_text("{}", encoding="utf-8")
    served = tl.export.model_explorer_serve(existing)
    assert served == existing
    assert calls == [[str(existing)]]
