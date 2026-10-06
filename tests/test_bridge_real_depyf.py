"""depyf bridge against the real depyf package (no sys.modules fakes).

``tl.bridge.depyf.dump(model, x, path)`` must write the same compiled-graph
sources as ``with depyf.prepare_debug(path): torch.compile(model)(x)``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl

depyf = pytest.importorskip("depyf")

pytestmark = [pytest.mark.optional, pytest.mark.slow]

_UUID = re.compile(r"[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}")
_COUNTER = re.compile(r"(__compiled_fn|__transformed_code|_for_inner|full_code_for_inner)_(\d+)")


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 14 * 14, 2)
    ).eval()


def _shape(names: list[str]) -> set[str]:
    """File names with per-process counters and uuids normalized away."""

    return {_COUNTER.sub(r"\1_N", _UUID.sub("UUID", name)) for name in names}


def test_dump_writes_what_depyf_writes_directly(tmp_path: Path) -> None:
    model, x = _model(), torch.randn(2, 3, 16, 16)
    torch._dynamo.reset()
    with depyf.prepare_debug(str(tmp_path / "direct")):
        torch.compile(model)(x)
    direct = sorted(p.name for p in (tmp_path / "direct").rglob("*") if p.is_file())

    torch._dynamo.reset()
    written = tl.bridge.depyf.dump(model, x, tmp_path / "bridge")

    assert written, "dump returned no files"
    assert all(p.is_file() and p.parent == tmp_path / "bridge" for p in written)
    assert any(p.name.startswith("__compiled_fn") for p in written)
    assert _shape([p.name for p in written]) == _shape(direct)
    print(f"\ndirect files={len(direct)} bridge files={len(written)}")


def test_cached_compile_is_never_an_empty_list(tmp_path: Path) -> None:
    model, x = _model(), torch.randn(2, 3, 16, 16)
    torch._dynamo.reset()
    assert tl.bridge.depyf.dump(model, x, tmp_path / "first")
    try:
        again = tl.bridge.depyf.dump(model, x, tmp_path / "second")
    except RuntimeError as exc:
        assert "torch._dynamo.reset" in str(exc)
    else:
        assert again
    torch._dynamo.reset()
