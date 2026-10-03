"""The netron serve/widget one-liner (lane F14, netron memo D-20).

Gotchas locked by execution during the panel: ``bytearray`` not ``bytes``
(bytes falls into an experimental sniffing branch), the served route is
``/data/<basename>``, the name must end ``.json``, loopback-only default
with an ephemeral port, clean stop. The round-trip below runs the REAL
installed netron server and fetches the artifact back byte-exactly.
Compo row C10: the one-liner works on a forked/intervened trace with its
provenance marks intact.
"""

from __future__ import annotations

import builtins
import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

netron_package = pytest.importorskip(
    "netron", reason="serve round-trip runs the real pinned netron server"
)


def _serve_roundtrip(log: Any, tmp_path: Path, **kwargs: Any) -> tuple[bytes, bytes]:
    """Export with open=True against a patched browser, fetch the bytes back."""

    import webbrowser

    opened: list[str] = []
    original = webbrowser.open

    def _capture(url: str, *args: Any, **kw: Any) -> bool:
        """Record the URL instead of opening a browser."""

        opened.append(url)
        return True

    webbrowser.open = _capture
    try:
        path = tl.export.netron(log, tmp_path / "served.json", open=True, **kwargs)
        assert opened, "the one-liner opens exactly one browser tab"
        with urllib.request.urlopen(opened[0].rstrip("/") + "/data/served.json") as reply:
            served = reply.read()
        return path.read_bytes(), served
    finally:
        webbrowser.open = original
        netron_package.stop()


def test_serve_round_trip_bytes_exact(tmp_path: Path) -> None:
    """The served /data/<basename> route returns the written artifact bytes."""

    log = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    written, served = _serve_roundtrip(log, tmp_path)
    assert served == written


def test_c10_serve_works_on_forked_intervened_trace(tmp_path: Path) -> None:
    """Compo row C10: any Trace serves, provenance marks intact."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.trace(
        model,
        torch.randn(2, 4),
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    written, served = _serve_roundtrip(log, tmp_path)
    assert served == written
    payload = json.loads(served.decode("utf-8"))
    props = {row["key"]: row["value"] for row in payload["metadataProps"]}
    assert props["torchlens.intervened"] == "true"


def test_missing_netron_package_refuses_teaching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the extra, open=True refuses typed and names the remedy."""

    real_import = builtins.__import__

    def _no_netron(name: str, *args: Any, **kwargs: Any) -> Any:
        """Simulate the missing optional dependency."""

        if name == "netron":
            raise ImportError("No module named 'netron'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_netron)
    log = tl.trace(nn.Sequential(nn.ReLU()), torch.randn(1, 2))
    with pytest.raises(tl.errors.ConfigurationError) as excinfo:
        tl.export.netron(log, tmp_path / "x.json", open=True)
    assert excinfo.value.fields["code"] == "netron_serve_unavailable"
    assert "torchlens[netron]" in str(excinfo.value)
    assert (tmp_path / "x.json").exists(), "the FILE lands before the serve refusal"
