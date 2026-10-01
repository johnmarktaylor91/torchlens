"""Buffer density via the tri-state policy + counter-chain rule (F14, memo D-10).

The netron exporter reuses TorchLens's existing ``show_buffers``
``never|meaningful|always`` policy (default ``meaningful``) with ONE added
rule: counter-update chains (the zero-dim ``add`` ops between
``num_batches_tracked`` reads and writes) are ops, not buffers, so the
buffer policy alone misses them. Hidden names/counts are disclosed on the
owning module and in the model metadata; ``always`` stays the forensic
escape hatch. The train-mode BatchNorm composition (compo row C6) guards
the one place a naive filter lies.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def train_bn_log() -> Any:
    """A train-mode BatchNorm capture (running stats + counter chain)."""

    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)).train()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = tl.trace(model, torch.randn(2, 4))
    try:
        yield log
    finally:
        log.cleanup()


def _export(log: Any, path: Path, **kwargs: Any) -> dict[str, Any]:
    """Export and parse one artifact."""

    return json.loads(tl.export.netron(log, path, **kwargs).read_text(encoding="utf-8"))


def test_meaningful_drops_counter_chain_and_noise_buffers(
    train_bn_log: Any, tmp_path: Path
) -> None:
    """Compo row C6: counter chains drop; no hidden op feeds visible compute."""

    payload = _export(train_bn_log, tmp_path / "m.json", granularity="op")
    names = [node["name"] for node in payload["graph"]["node"]]
    assert names == ["linear_1_1", "batchnorm_1_3"], "compute nodes only"
    produced = {out for node in payload["graph"]["node"] for out in node["output"]}
    produced.update(row["name"] for row in payload["graph"]["input"])
    for node in payload["graph"]["node"]:
        for value in node["input"]:
            assert value in produced, "no hidden op feeds visible computation"
    props = {row["key"]: row["value"] for row in payload["metadataProps"]}
    assert props["torchlens.buffer_policy"] == "meaningful"
    assert props["torchlens.hidden_buffer_count"] == "6"
    assert props["torchlens.hidden_op_count"] == "1", "the counter add is an op, not a buffer"
    assert int(props["torchlens.omitted_value_count"]) > 0, "dropped buffer reads disclosed"


def test_always_is_the_forensic_escape_hatch(train_bn_log: Any, tmp_path: Path) -> None:
    """``always`` shows every buffer node and the intact counter topology."""

    payload = _export(train_bn_log, tmp_path / "a.json", granularity="op", show_buffers="always")
    names = [node["name"] for node in payload["graph"]["node"]]
    assert "add_1_2" in names, "counter chain visible"
    assert sum(1 for name in names if name.startswith("buffer_")) == 6
    add_node = next(node for node in payload["graph"]["node"] if node["name"] == "add_1_2")
    assert add_node["input"] == ["buffer_1"], "buffer topology intact under always"
    props = {row["key"]: row["value"] for row in payload["metadataProps"]}
    assert props["torchlens.hidden_buffer_count"] == "0"
    assert props["torchlens.hidden_op_count"] == "0"


def test_never_hides_all_buffers(train_bn_log: Any, tmp_path: Path) -> None:
    """``never`` hides every buffer plus the then-dead counter chain."""

    payload = _export(train_bn_log, tmp_path / "n.json", granularity="op", show_buffers="never")
    names = [node["name"] for node in payload["graph"]["node"]]
    assert names == ["linear_1_1", "batchnorm_1_3"]
    props = {row["key"]: row["value"] for row in payload["metadataProps"]}
    assert props["torchlens.hidden_buffer_count"] == "6"


def test_module_projection_discloses_hidden_buffers_on_owner(
    train_bn_log: Any, tmp_path: Path
) -> None:
    """Hidden names ride the owning function's docString (memo D-10)."""

    class _Wrap(nn.Module):
        """Multi-op module owning a train BatchNorm."""

        def __init__(self) -> None:
            """Build the normed block."""

            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.bn = nn.BatchNorm1d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Linear, norm, then relu so the module keeps multiple ops."""

            return torch.relu(self.bn(self.fc(x)))

    class _Root(nn.Module):
        """Root holding the wrapped block plus a root op."""

        def __init__(self) -> None:
            """Build the wrapped block."""

            super().__init__()
            self.block = _Wrap()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the block and one root-level op."""

            return self.block(x) * 2.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = tl.trace(_Root().train(), torch.randn(2, 4))
    payload = _export(log, tmp_path / "own.json")
    functions = {fn["name"]: fn for fn in payload.get("functions", [])}
    assert "block" in functions
    doc = functions["block"].get("docString", "")
    assert "hidden by show_buffers='meaningful'" in doc
    assert "buffer_" in doc, "hidden buffer names disclosed on the owning call"


def test_invalid_buffer_policy_refuses_teaching(train_bn_log: Any) -> None:
    """The tri-state vocabulary rides the one existing options validator."""

    with pytest.raises(tl.errors.ConfigurationError):
        tl.export.netron(train_bn_log, "unused.json", show_buffers="minimal")
