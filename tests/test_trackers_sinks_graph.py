"""F26: the executed-DAG graph IR + the (gated) TB ingest pin (memo 3.14).

The IR layer is dependency-free and carries the join oracle; the event-file
ingest pin runs only where tensorboard is installed (the packaged extra).
Realism: the graph builds from a REAL captured trace of a real conv
architecture (never ``jit.trace``), plus the dict-input foil that kills
stock ``add_graph``.
"""

from __future__ import annotations

import importlib.util
import json

import pytest
import torch

import torchlens as tl
import torchlens.errors
import torchlens.trackers as trk


class _DictInputNet(torch.nn.Module):
    """The stock-add_graph killer: a forward taking a dict (issue #28206)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, batch: dict) -> torch.Tensor:
        """Consume a dict input -- jit.trace-based graphs die here."""

        return self.proj(batch["x"]).relu()


@pytest.fixture(scope="module")
def conv_trace():
    """One real captured conv net (module nesting exercises namespacing)."""

    # Standing rule: construct every model variant BEFORE the first trace.
    conv = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
        torch.nn.Sequential(torch.nn.Conv2d(4, 2, 1), torch.nn.ReLU()),
    )
    dict_net = _DictInputNet()
    trace = tl.trace(conv, torch.randn(1, 3, 8, 8))
    dict_trace = tl.trace(dict_net, {"x": torch.randn(2, 4)})
    try:
        yield trace, dict_trace
    finally:
        trace.cleanup()
        dict_trace.cleanup()


class TestGraphIR:
    """The dep-free layer: names, edges, truthful fields, join oracle."""

    def test_join_oracle_catches_corruption(self, conv_trace) -> None:  # noqa: ANN001
        """Corrupt one stats name -> the oracle names the unmatched node."""

        trace, _ = conv_trace
        ir = trk.build_graph_ir(trace)
        stats = ir.stats_names()
        assert ir.join_oracle(stats) == ()
        if stats:
            corrupted = ("definitely_not_a_node/x",) + stats[1:]
            assert ir.join_oracle(corrupted) == ("definitely_not_a_node/x",)

    def test_ir_serializes_to_jsonl(self, conv_trace, tmp_path) -> None:  # noqa: ANN001
        """The graph capability on the JSONL sink round-trips the IR."""

        trace, _ = conv_trace
        ir = trk.build_graph_ir(trace)
        sink = trk.JSONLSink(tmp_path / "g.jsonl")
        sink.emit_graph(ir)
        sink.close()
        rows = [json.loads(line) for line in (tmp_path / "g.jsonl").read_text().splitlines()]
        graph_row = next(r for r in rows if r.get("kind") == "graph")
        assert graph_row["graph"]["kind"] == "torchlens.trackers.graph_ir"
        assert len(graph_row["graph"]["nodes"]) == len(ir.nodes)


class TestTensorBoardIngestPin:
    """The pinned ingest leg (runs where the tensorboard extra is installed)."""

    @pytest.mark.skipif(
        importlib.util.find_spec("tensorboard") is not None,
        reason="tensorboard installed; the refusal leg is for bare envs",
    )
    def test_missing_tensorboard_refuses_typed(self, conv_trace, tmp_path) -> None:  # noqa: ANN001
        """Without the package the serializer refuses naming the extra."""

        trace, _ = conv_trace
        ir = trk.build_graph_ir(trace)
        with pytest.raises(torchlens.errors.SinkProtocolError) as info:
            trk.graph_ir_to_tensorboard(ir, str(tmp_path / "tb"))
        assert info.value.fields["code"] == "tracker_sink_unavailable"
        assert "torchlens[tensorboard]" in info.value.fields["remedy"]
