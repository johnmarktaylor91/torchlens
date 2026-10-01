"""A09 listA row 24: every tabular/file export carries capture-honesty facts.

One shared source (torchlens._capture_honesty) feeds every exporter in
torchlens.export, the debug DataFrames, Trace.to_pandas, the accessor tables,
and the fastlog tables -- through the format-appropriate slot (comment
preamble, metadata block, ``DataFrame.attrs``, dedicated key).
"""

from __future__ import annotations

import json as json_module
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_honesty import DATAFRAME_ATTRS_KEY

pytestmark = pytest.mark.smoke

pd = pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def export_trace() -> object:
    """One reusable small capture for every export assertion."""

    torch.manual_seed(0)
    trace = tl.trace(nn.Sequential(nn.Linear(3, 4), nn.ReLU()), torch.randn(2, 3))
    try:
        yield trace
    finally:
        trace.cleanup()


def _facts_of(dataframe: object) -> dict:
    facts = dataframe.attrs.get(DATAFRAME_ATTRS_KEY)
    assert isinstance(facts, dict), "export DataFrame carries no honesty attrs"
    return facts


def test_to_pandas_attrs_carry_honesty(export_trace: tl.Trace) -> None:
    """Trace.to_pandas() attaches the shared fact block to DataFrame.attrs."""

    facts = _facts_of(export_trace.to_pandas())
    assert facts["capture_status"] == "complete"
    assert facts["poisoned"] is False


def test_profile_and_accessor_tables_carry_honesty(export_trace: tl.Trace) -> None:
    """TraceProfile.to_pandas and accessor to_pandas carry the facts."""

    profile_facts = export_trace.profile().to_pandas().attrs["torchlens_capture_honesty"]
    assert profile_facts["capture_status"] == "complete"
    accessor_frame = export_trace.layers.to_pandas()
    assert _facts_of(accessor_frame)["capture_status"] == "complete"


def test_debug_dataframes_carry_honesty(export_trace: tl.Trace) -> None:
    """The debug DataFrame family attaches the shared fact block."""

    from torchlens import debug

    assert _facts_of(debug.hot_path(export_trace))["capture_status"] == "complete"
    assert _facts_of(debug.recompute_candidates(export_trace))["capture_status"] == "complete"
    assert _facts_of(debug.dead_neurons(export_trace))["capture_status"] == "complete"
    compare_frame = debug.compare(export_trace, export_trace)
    both = compare_frame.attrs[DATAFRAME_ATTRS_KEY]
    assert both["trace_a"]["capture_status"] == "complete"
    assert both["trace_b"]["capture_status"] == "complete"


def test_fastlog_tables_carry_honesty() -> None:
    """Recording.to_pandas attaches the settled recording outcome facts."""

    recording = tl.record(
        nn.Sequential(nn.Linear(3, 4), nn.ReLU()), torch.randn(2, 3), save=tl.func("relu")
    )
    facts = _facts_of(recording.to_pandas())
    assert facts["capture_status"] == "complete"


def test_csv_export_preamble(export_trace: tl.Trace, tmp_path: Path) -> None:
    """CSV opens with '#' honesty comment lines; read back with comment='#'."""

    destination = tl.export.csv(export_trace, tmp_path / "t.csv")
    text = destination.read_text(encoding="utf-8")
    assert text.startswith("# torchlens capture honesty: status=complete")
    frame = pd.read_csv(destination, comment="#")
    assert len(frame) == len(export_trace.layer_list)


def test_json_export_carries_fact_block(export_trace: tl.Trace, tmp_path: Path) -> None:
    """json() wraps rows with the fact block under torchlens.table_export.v1."""

    destination = tl.export.json(export_trace, tmp_path / "t.json")
    payload = json_module.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema"] == "torchlens.table_export.v1"
    assert payload["capture_honesty"]["capture_status"] == "complete"
    assert isinstance(payload["rows"], list) and payload["rows"]


def test_parquet_export_carries_schema_metadata(export_trace: tl.Trace, tmp_path: Path) -> None:
    """parquet() embeds the fact block in file-level schema metadata."""

    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    destination = tl.export.parquet(export_trace, tmp_path / "t.parquet")
    metadata = pyarrow_parquet.read_schema(destination).metadata
    facts = json_module.loads(metadata[b"torchlens_capture_honesty"])
    assert facts["capture_status"] == "complete"
    # read_parquet consumers are unaffected.
    assert len(pd.read_parquet(destination)) == len(export_trace.layer_list)


def test_svg_and_html_exports_carry_comment(export_trace: tl.Trace, tmp_path: Path) -> None:
    """svg()/html() embed the honesty facts as an XML/HTML comment."""

    svg_text = tl.export.svg(export_trace, tmp_path / "t.svg").read_text(encoding="utf-8")
    assert "torchlens capture honesty: status=complete" in svg_text
    assert svg_text.startswith("<?xml")  # comment did not break the declaration
    html_text = tl.export.html(export_trace, tmp_path / "t.html").read_text(encoding="utf-8")
    assert "torchlens capture honesty: status=complete" in html_text


def test_json_timeline_exports_carry_fact_key(export_trace: tl.Trace, tmp_path: Path) -> None:
    """chrome_trace/speedscope/memory_timeline carry the dedicated key."""

    chrome = json_module.loads(
        tl.export.chrome_trace(export_trace, tmp_path / "c.json").read_text(encoding="utf-8")
    )
    assert chrome["metadata"]["torchlens_capture_honesty"]["capture_status"] == "complete"
    speedscope = json_module.loads(
        tl.export.speedscope(export_trace, tmp_path / "s.json").read_text(encoding="utf-8")
    )
    assert speedscope["torchlens_capture_honesty"]["capture_status"] == "complete"
    timeline = json_module.loads(
        tl.export.memory_timeline(export_trace, tmp_path / "m.json").read_text(encoding="utf-8")
    )
    assert timeline["torchlens_capture_honesty"]["capture_status"] == "complete"


def test_flamegraph_carries_zero_weight_honesty_frame(
    export_trace: tl.Trace, tmp_path: Path
) -> None:
    """The folded-stack export carries a zero-weight honesty frame line."""

    lines = (
        tl.export.flamegraph(export_trace, tmp_path / "f.folded")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    honesty_lines = [line for line in lines if line.startswith("torchlens_capture_honesty;")]
    assert len(honesty_lines) == 1
    assert honesty_lines[0].endswith(" 0")
    assert "capture_status=complete" in honesty_lines[0]
    # Every line still parses as "stack count".
    for line in lines:
        stack, _, count = line.rpartition(" ")
        assert stack and count.isdigit()


def test_vendor_graph_exports_carry_facts(export_trace: tl.Trace, tmp_path: Path) -> None:
    """model_explorer/netron embed the facts in their vendor-legal slots."""

    explorer = json_module.loads(
        tl.export.model_explorer(export_trace, tmp_path / "me.json").read_text(encoding="utf-8")
    )
    # Schema v3: the honesty facts ride the "" groupNodeAttributes row (the
    # side-panel provenance block) -- extra top-level keys fail Model
    # Explorer's strict GraphCollection parse, so the v2 top-level block is
    # deliberately gone (modelexplorer memo D7).
    assert set(explorer) >= {"label", "graphs"}  # ingest contract intact
    for graph in explorer["graphs"]:
        root_row = graph["groupNodeAttributes"][""]
        assert root_row["capture_status"] == "complete"
        assert root_row["poisoned"] == "False"
    netron = json_module.loads(
        tl.export.netron(export_trace, tmp_path / "n.json").read_text(encoding="utf-8")
    )
    honesty_props = [
        prop for prop in netron["metadataProps"] if prop["key"] == "torchlens.capture_honesty"
    ]
    assert len(honesty_props) == 1
    assert json_module.loads(honesty_props[0]["value"])["capture_status"] == "complete"


def test_tracker_exports_return_fact_block(export_trace: tl.Trace) -> None:
    """mlflow/aim prepared-metrics mappings carry the fact block (not logged)."""

    class _Client:
        def __init__(self) -> None:
            self.logged: list[tuple[str, object]] = []

        def log_metric(self, key: str, value: object) -> None:
            self.logged.append((key, value))

    client = _Client()
    mlflow_result = tl.export.mlflow(export_trace, client)
    assert mlflow_result["capture_honesty"]["capture_status"] == "complete"
    # Only numeric metrics reach the client; facts are returned, not coerced.
    assert all(isinstance(value, int | float) for _, value in client.logged)

    class _Run:
        def __init__(self) -> None:
            self.tracked: list[tuple[object, str]] = []

        def track(self, value: object, name: str) -> None:
            self.tracked.append((value, name))

    run = _Run()
    aim_result = tl.export.aim(export_trace, run)
    assert aim_result["capture_honesty"]["capture_status"] == "complete"
    assert all(isinstance(value, int | float) for value, _ in run.tracked)


def test_tensorboard_export_writes_honesty_text(export_trace: tl.Trace) -> None:
    """tensorboard() adds a capture_honesty text summary."""

    class _Writer:
        def __init__(self) -> None:
            self.scalars: list[tuple[str, object, int]] = []
            self.texts: list[tuple[str, str, int]] = []

        def add_scalar(self, tag: str, value: object, step: int) -> None:
            self.scalars.append((tag, value, step))

        def add_text(self, tag: str, text: str, step: int) -> None:
            self.texts.append((tag, text, step))

    writer = _Writer()
    tl.export.tensorboard(export_trace, writer, step=0)
    honesty_texts = [text for tag, text, _ in writer.texts if tag.endswith("capture_honesty")]
    assert len(honesty_texts) == 1
    assert "status=complete" in honesty_texts[0]


def test_poisoned_trace_export_preamble_discloses(tmp_path: Path) -> None:
    """A poisoned capture's exports carry the poison, never a clean preamble."""

    from torchlens.runnable import PathFaithfulness

    trace = tl.trace(nn.Sequential(nn.Linear(3, 4), nn.ReLU()), torch.randn(2, 3))
    trace._runnable.poisoned = True
    trace._runnable.path_faithfulness = PathFaithfulness.DIVERGED
    svg_text = tl.export.svg(trace, tmp_path / "p.svg").read_text(encoding="utf-8")
    assert "POISONED" in svg_text
