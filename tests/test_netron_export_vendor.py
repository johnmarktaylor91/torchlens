"""Tier T3: the installed netron wheel's parser executes on the artifact.

The harness (tests/test_netron_export_harness.mjs) imports netron's real
``onnx.js`` ModelFactory -- no transcription -- and reports what netron
WOULD draw: matched reader, node/function populations, painted edge
labels, sidebar strings, drill-down resolution, call-graph acyclicity
(netron memo D-08; measured 0.08-0.13 s per fixture). Skips typed when
node or the netron wheel is absent; CI declares both through
the packaging-request ledger (node >= 20 ships on ubuntu-latest).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

netron_package = pytest.importorskip(
    "netron", reason="vendor-execution tier runs the exact installed pin"
)

pytestmark = pytest.mark.smoke

_HARNESS = Path(__file__).with_name("test_netron_export_harness.mjs")


def _run_harness(artifact: Path) -> dict[str, Any]:
    """Execute the node harness over one artifact and parse its JSON report."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("node >= 20 is required for the vendor-execution tier")
    netron_dir = Path(netron_package.__file__).parent
    completed = subprocess.run(
        [node, str(_HARNESS), str(netron_dir), str(artifact)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, f"harness crashed: {completed.stderr[-2000:]}"
    return json.loads(completed.stdout)


class _Inner(nn.Module):
    """Two-op child module."""

    def __init__(self) -> None:
        """Build the child linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear then relu."""

        return torch.relu(self.fc(x))


class _Nested(nn.Module):
    """Two children plus a root op, wrapped once more for nesting."""

    def __init__(self) -> None:
        """Build children."""

        super().__init__()
        self.a = _Inner()
        self.b = _Inner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine both children."""

        return self.a(x) + self.b(x)


def test_vendor_parser_accepts_and_fully_lights_the_module_artifact(
    tmp_path: Path,
) -> None:
    """Netron's reader matches, paints every edge, resolves both functions."""

    log = tl.trace(_Nested(), torch.randn(2, 4))
    artifact = tl.export.netron(log, tmp_path / "m.json")
    report = _run_harness(artifact)
    assert report["matched"] and report["reader"] == "onnx.proto"
    assert report["format"] == "ONNX v10"
    assert report["producer"].startswith("torchlens")
    assert report["rootNodeCount"] == 3  # a, b, add
    assert report["rootInputCount"] == 1 and report["rootOutputCount"] == 1
    assert set(report["domains"]) == {"ai.torchlens.lossy", "ai.torchlens.module"}
    assert report["untypedValues"] == 0, "typed coverage: 0 untyped (memo D-04)"
    assert report["paintedEdgeLabels"] >= 3
    assert "2×4" in report["edgeLabelSamples"], "the exact expected edge string"
    assert report["functionNames"] == ["a", "b"]
    assert report["emptyFunctions"] == []
    assert report["functionCallGraphAcyclic"] is True
    metadata = report["modelMetadata"]
    assert metadata["torchlens.granularity"] == "module"
    assert metadata["torchlens.runnable"] == "false"


def test_vendor_parser_nested_drill_down_resolves(tmp_path: Path) -> None:
    """A nested function call inside a function body resolves to a function."""

    class _Wrapper(nn.Module):
        """Wraps the nested model one level down."""

        def __init__(self) -> None:
            """Build the wrapped model."""

            super().__init__()
            self.mid = _Nested()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Delegate plus an outer op."""

            return torch.sigmoid(self.mid(x))

    log = tl.trace(_Wrapper(), torch.randn(2, 4))
    artifact = tl.export.netron(log, tmp_path / "n.json")
    report = _run_harness(artifact)
    assert report["functionCallGraphAcyclic"] is True
    assert set(report["nestedFunctionCalls"]) == {"mid.a", "mid.b"}, (
        "one-click drill-down resolves nested functions"
    )


def test_vendor_parser_sidebar_strings_exact(tmp_path: Path) -> None:
    """EXACT sidebar strings for one op node and one model property (D-17)."""

    log = tl.trace(_Nested(), torch.randn(2, 4))
    artifact = tl.export.netron(log, tmp_path / "op.json", granularity="op")
    report = _run_harness(artifact)
    probe = report["probeNode"]
    assert probe["name"] == "linear_1_1"
    attrs = probe["attributes"]
    assert attrs["dtype"] == "float32"
    assert attrs["module_path"] == "a.fc"
    assert attrs["params"] == "20"
    assert attrs["shape"] == "2,4"
    assert attrs["flops_estimated"] == "72"
    assert "module" not in attrs, "the collision netron's own domain row owns"
    assert probe["inputPortNames"] == ["input"], "the input_names port hook fires"
    assert report["modelMetadata"]["torchlens.netron_schema"] == "2"
