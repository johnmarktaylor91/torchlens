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
import torchlens.trackers as trk

pytestmark = pytest.mark.smoke


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

    def test_nodes_and_edges_from_executed_dag(self, conv_trace) -> None:  # noqa: ANN001
        trace, _ = conv_trace
        ir = trk.build_graph_ir(trace)
        assert len(ir.nodes) == len(trace.layer_list)
        names = {node.name for node in ir.nodes}
        assert len(names) == len(ir.nodes), "node names must be unique"
        # Every input edge resolves to a real node (the frontend join).
        for node in ir.nodes:
            for parent in node.inputs:
                assert parent in names
        # Module namespacing: the nested Sequential shows up as a slash path.
        assert any("/" in node.name for node in ir.nodes)

    def test_dict_input_model_still_graphs(self, conv_trace) -> None:  # noqa: ANN001
        """The decade-old add_graph failure class: ours always logs."""

        _, dict_trace = conv_trace
        ir = trk.build_graph_ir(dict_trace)
        assert len(ir.nodes) == len(dict_trace.layer_list) > 0

    def test_truthful_fields_disclosed(self, conv_trace) -> None:  # noqa: ANN001
        """No allocator claim anywhere; timing carries its evidence label."""

        trace, _ = conv_trace
        ir = trk.build_graph_ir(trace)
        payload = json.dumps(ir.as_dict())
        assert "allocator" not in payload.replace(
            "memory_coloring: unavailable (no allocator source)", ""
        )
        assert any("memory_coloring: unavailable" in d for d in ir.disclosures)
        if ir.timing_evidence is not None:
            assert ir.timing_evidence == trk.TIMING_EVIDENCE
            assert any("host bracket" in d for d in ir.disclosures)

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

    def test_event_file_binds_stats_to_nodes(self, conv_trace, tmp_path) -> None:  # noqa: ANN001
        """GraphDef + StepStats land and every stats row joins a node."""

        pytest.importorskip("tensorboard")
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        trace, _ = conv_trace
        ir = trk.build_graph_ir(trace)
        logdir = str(tmp_path / "tb")
        trk.graph_ir_to_tensorboard(ir, logdir)
        accumulator = EventAccumulator(logdir)
        accumulator.Reload()
        graph = accumulator.Graph()
        graph_names = {node.name for node in graph.node}
        assert {node.name for node in ir.nodes} <= graph_names
        run_metadata = accumulator.RunMetadata("step1")
        stats_names = {
            stat.node_name
            for device in run_metadata.step_stats.dev_stats
            for stat in device.node_stats
        }
        assert stats_names, "StepStats rows must land"
        unmatched = ir.join_oracle(tuple(sorted(stats_names)))
        assert unmatched == (), f"unbound stats rows: {unmatched}"

    @pytest.mark.skipif(
        importlib.util.find_spec("tensorboard") is not None,
        reason="tensorboard installed; the refusal leg is for bare envs",
    )
    def test_missing_tensorboard_refuses_typed(self, conv_trace, tmp_path) -> None:  # noqa: ANN001
        """Without the package the serializer refuses naming the extra."""

        trace, _ = conv_trace
        ir = trk.build_graph_ir(trace)
        with pytest.raises(trk.SinkProtocolError) as info:
            trk.graph_ir_to_tensorboard(ir, str(tmp_path / "tb"))
        assert info.value.fields["code"] == "tracker_sink_unavailable"
        assert "torchlens[tensorboard]" in info.value.fields["remedy"]
