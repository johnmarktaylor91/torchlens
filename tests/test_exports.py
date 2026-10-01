"""Tests for Phase 10 export surfaces.

Governance adjudication (b10 R78 round-3): the two export goldens under
``tests/fixtures/exports/`` are ENVIRONMENT-INDEPENDENT semantic contracts
and deliberately NOT routed through the ``tests/_oracle_env.py``
env-fingerprint resolver. The payloads are normalized structural JSON
(viewer schema fields, node ids/labels from torchlens-owned label
vocabulary, edge lists) with process-global identifiers normalized before
comparison — no floats, reprs, or emitter bytes. A torch upgrade that
changed the exported graph structure would be a REAL export-contract change
this gate must surface. Registered in the environment-independent ledger
enforced by ``tests/test_golden_governance_lint.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

pd = pytest.importorskip("pandas")

from _oracle_env import (  # noqa: E402
    flag_armed,
    guard_wrap_state_for_golden_update,
    require_update_reason,
    write_provenance,
)

import torchlens as tl  # noqa: E402

EXPORT_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "exports"
_UPDATE_ENV = "TORCHLENS_REGEN_EXPORT_GOLDENS"


def _assert_model_explorer_structure(payload: dict[str, Any]) -> None:
    """Validate required Model Explorer keys and graph referential integrity.

    Parameters
    ----------
    payload:
        Parsed Model Explorer artifact.
    """

    # Schema v3 (lane F15): the payload carries ONLY the vendor
    # GraphCollection keys -- extra top-level keys (the v2 schema/disclaimer
    # block) are measured to fail Model Explorer's strict parse; TorchLens
    # provenance now rides the "" groupNodeAttributes row.
    assert set(payload) == {"label", "graphs", "graphSorting"}
    assert payload["graphSorting"] == "name_asc"
    assert isinstance(payload["label"], str) and payload["label"]
    assert isinstance(payload["graphs"], list) and payload["graphs"]
    for graph in payload["graphs"]:
        assert isinstance(graph["id"], str)
        assert isinstance(graph["nodes"], list) and graph["nodes"]
        node_ids = [node["id"] for node in graph["nodes"]]
        assert all(isinstance(node_id, str) for node_id in node_ids)
        assert len(node_ids) == len(set(node_ids))
        for node in graph["nodes"]:
            assert isinstance(node["label"], str)
            assert isinstance(node["namespace"], str)
            assert isinstance(node["attrs"], list)
            assert all(
                isinstance(attr, dict)
                and isinstance(attr.get("key"), str)
                and isinstance(attr.get("value"), str)
                for attr in node["attrs"]
            )
            assert all(
                set(edge) <= {"sourceNodeId", "sourceNodeOutputId", "targetNodeInputId"}
                and isinstance(edge["sourceNodeId"], str)
                and edge["sourceNodeId"] in node_ids
                for edge in node.get("incomingEdges", [])
            )
        # The "" group row is the visible TorchLens provenance/disclosure
        # block (memo D7) and must be present on every emitted graph.
        assert graph["groupNodeAttributes"][""]["produced_by"].startswith("torchlens ")


def _assert_netron_structure(payload: dict[str, Any]) -> None:
    """Validate required ONNX-JSON keys and graph referential integrity (schema v2).

    Parameters
    ----------
    payload:
        Parsed ONNX ``ModelProto`` JSON artifact.
    """

    assert payload["irVersion"] == 10
    assert payload["producerName"] == "torchlens"
    domains = {row["domain"] for row in payload["opsetImport"]}
    assert "ai.torchlens.lossy" in domains
    assert domains <= {"ai.torchlens.lossy", "ai.torchlens.module"}
    assert "not a runnable ONNX model" in payload["docString"]
    props = {prop["key"]: prop["value"] for prop in payload["metadataProps"]}
    honesty_value = props["torchlens.capture_honesty"]
    assert json.loads(honesty_value)["schema"] == "torchlens.capture_honesty.v1"
    assert props["torchlens.lossy_export"] == "true"
    assert props["torchlens.runnable"] == "false"
    assert props["torchlens.netron_schema"] == "2"
    assert props["torchlens.granularity"] in {"op", "module", "rolled"}
    graph = payload["graph"]
    assert isinstance(graph["name"], str)
    assert "not a runnable ONNX model" in graph["docString"]
    assert isinstance(graph["node"], list) and graph["node"]
    node_names = [node["name"] for node in graph["node"]]
    outputs = [output for node in graph["node"] for output in node["output"]]
    graph_inputs = [row["name"] for row in graph.get("input", [])]
    assert len(node_names) == len(set(node_names))
    assert len(outputs) == len(set(outputs))
    assert graph.get("output"), "graph outputs must be present (netron #71 promotion)"
    known_values = set(outputs) | set(graph_inputs)
    assert all(row["name"] in known_values for row in graph["output"])
    for node in graph["node"]:
        assert isinstance(node["opType"], str)
        assert node["domain"] in {"ai.torchlens.lossy", "ai.torchlens.module"}
        assert isinstance(node["input"], list)
        assert all(
            isinstance(input_id, str) and input_id in known_values for input_id in node["input"]
        )
        assert isinstance(node["output"], list) and node["output"]
        assert all(
            isinstance(attribute, dict)
            and isinstance(attribute.get("name"), str)
            and isinstance(attribute.get("type"), str)
            for attribute in node.get("attribute", [])
        )


def _assert_or_regenerate_export_golden(name: str, payload: dict[str, Any]) -> bool:
    """Compare an export payload with its golden, with explicit opt-in regeneration.

    Set ``TORCHLENS_REGEN_EXPORT_GOLDENS=1`` and run the export test to regenerate
    fixtures after an intentional contract change. A regen run WRITES and never
    compares: the historical behavior compared the payload to the golden it had
    just written and reported green, blurring "verified" with "just rebaselined"
    (b7 R53-1, the last unconverted site of the 5-site fix; same doctrine as
    ``test_selector_semantics_matrix._golden``). The caller must ``pytest.skip``
    when this returns ``True`` so a regen run never reports a verifying pass.

    Parameters
    ----------
    name:
        Golden fixture filename.
    payload:
        Normalized parsed export payload.

    Returns
    -------
    bool
        ``True`` when the golden was regenerated (no comparison happened).
    """

    fixture_path = EXPORT_FIXTURE_DIR / name
    if flag_armed(os.environ, _UPDATE_ENV):
        reason = require_update_reason(_UPDATE_ENV)
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        write_provenance(EXPORT_FIXTURE_DIR, f"tests/test_exports.py ({name})", _UPDATE_ENV, reason)
        return True
    if not fixture_path.exists():
        pytest.fail(
            f"Missing golden {fixture_path}; regenerate with TORCHLENS_REGEN_EXPORT_GOLDENS=1."
        )
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert payload == expected
    return False


#: Netron schema-v2 node attributes that report per-run measurements
#: (wall-clock durations); structurally real but never byte-stable, so the
#: environment-independent golden replaces their values with a marker.
_RUN_VARYING_ATTRS = frozenset({"observed_duration_us", "observed_duration_inclusive_us"})


def _normalize_export_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize process-global identifiers and per-run measurements.

    Parameters
    ----------
    payload:
        Parsed export payload.

    Returns
    -------
    dict[str, Any]
        A detached payload with its generated graph identifier and any
        run-varying measurement values normalized.
    """

    normalized = json.loads(json.dumps(payload))
    if "graphs" in normalized:
        normalized["graphs"][0]["id"] = "<trace-id>"
        if "label" in normalized:
            normalized["label"] = "<trace-id>"
        # Measured wall-clock rows are real per-run values, not structural
        # contract: strip them so the golden stays environment-independent
        # (the same doctrine that keeps floats out of these fixtures).
        for graph in normalized["graphs"]:
            for node in graph.get("nodes", []):
                node["attrs"] = [
                    attr for attr in node.get("attrs", []) if attr.get("key") != "time"
                ]
            for row in (graph.get("groupNodeAttributes") or {}).values():
                row.pop("time", None)
        return normalized
    normalized["graph"]["name"] = "<trace-id>"
    node_lists = [normalized["graph"].get("node", [])]
    node_lists.extend(fn.get("node", []) for fn in normalized.get("functions", []))
    for nodes in node_lists:
        for node in nodes:
            for attribute in node.get("attribute", []):
                if attribute.get("name") in _RUN_VARYING_ATTRS:
                    attribute["i"] = "<measured>"
    props = normalized.get("metadataProps", [])
    for row in props:
        if row.get("key") == "torchlens.capture_honesty":
            row["value"] = "<capture-honesty-json>"
    return normalized


def test_hash_namespace_is_available_through_all_public_import_patterns() -> None:
    """The structural-hash namespace should preserve lazy facade import patterns."""

    import torchlens.hash as hash_module
    from torchlens import hash as imported_namespace
    from torchlens.hash import model as model_hash

    assert tl.hash is hash_module
    assert imported_namespace is hash_module
    assert model_hash is hash_module.model
    assert tl.assert_unchanged is hash_module.assert_unchanged


class _Tracker:
    """Small object with tracker-like custom_methods for export tests."""

    def __init__(self) -> None:
        """Initialize recorded calls."""

        self.metrics: list[tuple[str, int]] = []

    def log_metric(self, name: str, value: int) -> None:
        """Record an MLflow-like metric call.

        Parameters
        ----------
        name:
            Metric name.
        value:
            Metric value.
        """

        self.metrics.append((name, value))

    def track(self, value: int, name: str) -> None:
        """Record an Aim-like track call.

        Parameters
        ----------
        value:
            Metric value.
        name:
            Metric name.
        """

        self.metrics.append((name, value))


class _FakeHubApi:
    """Small Hugging Face API double."""

    def __init__(self) -> None:
        """Initialize recorded calls."""

        self.created: list[dict[str, Any]] = []
        self.uploaded: list[dict[str, Any]] = []
        self.uploaded_bytes: list[bytes] = []

    def create_repo(self, **kwargs: Any) -> None:
        """Record repository creation.

        Parameters
        ----------
        **kwargs:
            Repository creation arguments.
        """

        self.created.append(kwargs)

    def upload_file(self, **kwargs: Any) -> str:
        """Record file upload.

        Parameters
        ----------
        **kwargs:
            Upload arguments.

        Returns
        -------
        str
            Fake upload URL.
        """

        self.uploaded.append(kwargs)
        # Read the file's bytes immediately: the caller's temp directory is
        # cleaned up as soon as this call returns, so content must be
        # captured now rather than by re-reading the path later.
        self.uploaded_bytes.append(Path(kwargs["path_or_fileobj"]).read_bytes())
        return "https://huggingface.co/example/repo/blob/main/torchlens_artifact.pkl"


@pytest.fixture
def export_log() -> Any:
    """Build a small Trace for export tests.

    Returns
    -------
    Any
        Logged model.
    """

    if flag_armed(os.environ, _UPDATE_ENV):
        # Golden regeneration derives from this in-process capture: refuse
        # to generate on a torch earlier tests already wrapped (SF-53).
        guard_wrap_state_for_golden_update(_UPDATE_ENV)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    return tl.trace(model, torch.randn(2, 3), capture=tl.options.CaptureOptions())


def test_trace_timeline_exports_are_parseable(export_log: Any, tmp_path: Path) -> None:
    """Trace/timeline exports should write viewer-conformant payloads."""

    chrome_path = tl.export.chrome_trace(export_log, tmp_path / "trace.json")
    chrome_payload = json.loads(chrome_path.read_text(encoding="utf-8"))
    assert "traceEvents" in chrome_payload
    assert any(event.get("ph") == "X" for event in chrome_payload["traceEvents"])

    speedscope_path = tl.export.speedscope(export_log, tmp_path / "profile.json")
    speedscope_payload = json.loads(speedscope_path.read_text(encoding="utf-8"))
    assert speedscope_payload["$schema"].endswith("file-format-schema.json")
    assert speedscope_payload["profiles"][0]["type"] == "evented"

    flamegraph_path = tl.export.flamegraph(export_log, tmp_path / "profile.folded")
    assert ";" in flamegraph_path.read_text(encoding="utf-8")

    memory_path = tl.export.memory_timeline(export_log, tmp_path / "memory.json")
    memory_payload = json.loads(memory_path.read_text(encoding="utf-8"))
    assert memory_payload["scope"] == "tensor"
    assert "not an allocator trace" in memory_payload["disclaimer"]


def test_xarray_export_has_neuroidassembly_shape(export_log: Any) -> None:
    """xarray export should expose presentation and neuroid dimensions."""
    pytest.importorskip("xarray")

    assembly = tl.export.xarray(export_log)

    assert assembly.dims == ("presentation", "neuroid")
    assert "layer" in assembly.coords
    assert assembly.attrs["assembly"] == "NeuroidAssembly"
    assert assembly.sizes["presentation"] == 2
    assert assembly.sizes["neuroid"] > 0


def test_xarray_export_names_mismatched_presentation_layer() -> None:
    """Mismatched presentation counts should identify the offending layer."""
    pytest.importorskip("xarray")

    fake_log = SimpleNamespace(
        layer_list=[
            SimpleNamespace(layer_label="first", out=torch.randn(2, 3)),
            SimpleNamespace(layer_label="bad_layer", out=torch.randn(1, 3)),
        ]
    )

    with pytest.raises(ValueError, match="bad_layer.*1.*expected 2"):
        tl.export.xarray(fake_log)


def test_tracker_exports_accept_existing_objects(export_log: Any, tmp_path: Path) -> None:
    """Tracker helpers should work with caller-owned writer/run objects."""

    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=tmp_path / "tb")
    returned_writer = tl.export.tensorboard(export_log, writer, step=3, prefix="tl")
    writer.close()

    assert returned_writer is writer
    accumulator = EventAccumulator(str(tmp_path / "tb"))
    accumulator.Reload()
    assert "tl/num_layers" in accumulator.Tags()["scalars"]

    tracker = _Tracker()
    assert tl.export.mlflow(export_log, client=tracker, prefix="tl")["num_layers"] > 0
    assert tracker.metrics

    aim_tracker = _Tracker()
    assert tl.export.aim(export_log, run=aim_tracker, prefix="tl")["num_layers"] > 0
    assert aim_tracker.metrics

    pytest.importorskip("wandb")
    wandb_result = tl.export.wandb(export_log)
    assert "table" in wandb_result


def test_tracker_exports_reject_paths_with_clear_type_errors(
    export_log: Any, tmp_path: Path
) -> None:
    """Tracker helpers need live tracker objects, not filesystem paths."""

    with pytest.raises(TypeError, match="tensorboard expects an existing tracker object"):
        tl.export.tensorboard(export_log, str(tmp_path / "tb"), step=0)
    with pytest.raises(TypeError, match="mlflow expects an existing tracker object"):
        tl.export.mlflow(export_log, client=tmp_path / "mlruns")
    with pytest.raises(TypeError, match="aim expects an existing tracker object"):
        tl.export.aim(export_log, run=tmp_path / "aim")


def test_emit_nvtx_capture_option_does_not_change_capture() -> None:
    """emit_nvtx should be accepted and should not alter normal logging output."""

    log = tl.trace(
        nn.Linear(2, 2),
        torch.randn(1, 2),
        capture=tl.options.CaptureOptions(emit_nvtx=True),
    )

    assert log.emit_nvtx is True
    assert len(log.layer_list) > 0


def test_tabular_exports_round_trip(export_log: Any, tmp_path: Path) -> None:
    """Canonical tabular exports should round-trip."""

    expected = export_log.to_pandas()
    assert "func_config" in expected.columns
    assert "conditional_then_children" in expected.columns

    csv_path = tl.export.csv(export_log, tmp_path / "model.csv")
    # The capture-honesty preamble rides '#' comment lines (WT1 A-V row 24).
    assert csv_path.read_text(encoding="utf-8").startswith("# torchlens capture honesty:")
    csv_df = pd.read_csv(csv_path, comment="#")
    assert list(csv_df.columns) == list(expected.columns)
    assert len(csv_df) == len(expected)

    json_path = tl.export.json(export_log, tmp_path / "model.json")
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_payload["schema"] == "torchlens.table_export.v1"
    assert json_payload["capture_honesty"]["capture_status"] == "complete"
    json_df = pd.DataFrame(json_payload["rows"])
    assert list(json_df.columns) == list(expected.columns)
    assert len(json_df) == len(expected)

    parquet_path = tmp_path / "model.parquet"
    if importlib.util.find_spec("pyarrow") is None:
        with pytest.raises(ImportError, match=r"torchlens\[tabular\]"):
            tl.export.parquet(export_log, parquet_path)
    else:
        tl.export.parquet(export_log, parquet_path)
        parquet_df = pd.read_parquet(parquet_path)
        assert list(parquet_df.columns) == list(expected.columns)
        assert len(parquet_df) == len(expected)


def test_static_graph_adapters_and_hub_dry_run(export_log: Any, tmp_path: Path) -> None:
    """Static graph adapters and Hub publisher should write planned payloads."""

    explorer_path = tl.export.model_explorer(export_log, tmp_path / "explorer.json")
    explorer_payload = json.loads(explorer_path.read_text(encoding="utf-8"))
    _assert_model_explorer_structure(explorer_payload)
    regenerated = _assert_or_regenerate_export_golden(
        "model_explorer.json", _normalize_export_payload(explorer_payload)
    )

    netron_path = tl.export.netron(export_log, tmp_path / "netron.json")
    netron_payload = json.loads(netron_path.read_text(encoding="utf-8"))
    _assert_netron_structure(netron_payload)
    regenerated |= _assert_or_regenerate_export_golden(
        "netron.json", _normalize_export_payload(netron_payload)
    )

    result = tl.bridge.huggingface.push_to_hub(
        export_log,
        "example/repo",
        dry_run=True,
    )
    assert result["repo_id"] == "example/repo"
    assert result["dry_run"] is True

    api = _FakeHubApi()
    uploaded = tl.bridge.huggingface.push_to_hub(export_log, "example/repo", api=api)
    assert uploaded["upload_result"].startswith("https://huggingface.co/")
    assert api.created
    assert api.uploaded

    if regenerated:
        pytest.skip(
            "regenerated export goldens; re-run without TORCHLENS_REGEN_EXPORT_GOLDENS to verify"
        )


def test_recurrent_static_graph_exports_use_unique_pass_qualified_ids(tmp_path: Path) -> None:
    """Recurrent exports preserve every execution pass and every connecting edge."""

    class _LoopModel(nn.Module):
        """Three-step recurrent linear/ReLU chain."""

        def __init__(self) -> None:
            """Initialize the reused linear layer."""

            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the same two operations for three iterations."""

            for _ in range(3):
                x = torch.relu(self.linear(x))
            return x

    log = tl.trace(_LoopModel(), torch.randn(1, 2))
    explorer_path = tl.export.model_explorer(log, tmp_path / "recurrent-explorer.json")
    netron_path = tl.export.netron(log, tmp_path / "recurrent-netron.json")

    explorer_graph = json.loads(explorer_path.read_text(encoding="utf-8"))["graphs"][0]
    explorer_ids = [node["id"] for node in explorer_graph["nodes"]]
    assert len(explorer_ids) == len(set(explorer_ids)) == 8
    assert sum(len(node.get("incomingEdges", [])) for node in explorer_graph["nodes"]) == 7

    _assert_model_explorer_structure(json.loads(explorer_path.read_text(encoding="utf-8")))

    netron_payload = json.loads(netron_path.read_text(encoding="utf-8"))
    _assert_netron_structure(netron_payload)
    netron_nodes = netron_payload["graph"]["node"]
    netron_outputs = {output for node in netron_nodes for output in node["output"]}
    # Schema v2: input/output layers promote to graph I/O (memo D-04), so the
    # six op passes remain as nodes, all pass-qualified and unique.
    assert len(netron_nodes) == len(netron_outputs) == 6
    assert {node["name"] for node in netron_nodes} == {
        f"{base}:{index}" for base in ("linear_1_1", "relu_1_2") for index in (1, 2, 3)
    }
    known = netron_outputs | {row["name"] for row in netron_payload["graph"]["input"]}
    assert all(input_id in known for node in netron_nodes for input_id in node["input"])


def test_model_explorer_package_accepts_graph_schema(export_log: Any, tmp_path: Path) -> None:
    """The optional Model Explorer graph dataclass loader should accept the artifact."""

    model_explorer = pytest.importorskip("model_explorer")
    dacite = pytest.importorskip("dacite")
    explorer_path = tl.export.model_explorer(export_log, tmp_path / "explorer.json")
    payload = json.loads(explorer_path.read_text(encoding="utf-8"))

    parsed = [
        dacite.from_dict(data_class=model_explorer.graph_builder.Graph, data=graph)
        for graph in payload["graphs"]
    ]

    assert parsed and parsed[0].nodes


def test_hub_push_uploads_real_bundle_not_metadata_stub(export_log: Any) -> None:
    """push_to_hub must upload the real scrubbed artifact, never a JSON stub.

    ``push_to_hub`` previously fell back to a ~240-byte JSON manifest for
    backward-eligible captures while still reporting ``dry_run: False`` success.
    This asserts the uploaded payload is the real, larger, non-JSON artifact.

    R62-1 rebaseline (privacy): a ``Trace`` is now ALWAYS serialized through the
    privacy-scrubbed portable ``.tlspec`` bundle (a gzipped tar), never raw
    ``pickle.dumps`` -- raw pickle embeds ``$HOME``, the username, and absolute
    source paths, leaking them to a PUBLIC hub. This test previously asserted the
    raw-pickle opcode ``0x80``; it now asserts the scrubbed tar.gz artifact.
    """

    api = _FakeHubApi()
    uploaded = tl.bridge.huggingface.push_to_hub(export_log, "example/repo", api=api)
    assert uploaded["dry_run"] is False
    assert api.uploaded, "expected an upload_file call"

    payload = api.uploaded_bytes[-1]

    # A ~240-byte JSON manifest stub was the old broken fallback. The real
    # artifact must be large and must not be a bare JSON stub.
    assert len(payload) > 1000
    assert uploaded["size_bytes"] > 1000
    assert not payload.lstrip().startswith(b"{"), "expected a real artifact, not a JSON stub"

    # The scrubbed portable bundle is uploaded as a gzip tar (magic 0x1f 0x8b),
    # never a raw pickle (0x80) that would leak local paths.
    assert uploaded["format"] == "tar.gz"
    assert payload[:2] == b"\x1f\x8b", "expected a scrubbed gzip-tar bundle, not raw pickle"
    assert payload[:1] != b"\x80", "raw pickle would leak $HOME / username / source paths"


def test_hub_push_scrubs_local_paths_from_uploaded_trace(export_log: Any) -> None:
    """R62-1: the uploaded artifact must not contain $HOME / username / abs paths.

    Raw ``pickle.dumps`` of a Trace embeds the local home directory, the OS
    username, and absolute source-file paths. ``push_to_hub`` routes through the
    privacy scrub so none of those local identifiers reach the public hub.
    """

    api = _FakeHubApi()
    tl.bridge.huggingface.push_to_hub(export_log, "example/repo", api=api)
    payload = api.uploaded_bytes[-1]

    # Decompress the gzip-tar and scan the actual member bytes: a compressed
    # payload would hide a plaintext leak, so the raw archive bytes are not a
    # sufficient check.
    import io as _io
    import tarfile as _tarfile

    contents = b""
    with _tarfile.open(fileobj=_io.BytesIO(payload), mode="r:gz") as tar:
        for member in tar.getmembers():
            contents += member.name.encode("utf-8", "replace")
            if member.isfile():
                extracted = tar.extractfile(member)
                if extracted is not None:
                    contents += extracted.read()

    home = os.path.expanduser("~")
    assert home.encode() not in contents, "uploaded artifact leaked $HOME"
    user = os.environ.get("USER")
    if user and len(user) >= 3:
        assert user.encode() not in contents, "uploaded artifact leaked the username"


def test_depyf_bridge_fails_soft_when_extra_missing() -> None:
    """depyf bridge should explain the missing optional dependency."""

    if importlib.util.find_spec("depyf") is not None:
        pytest.skip("Installed depyf API varies; smoke coverage is in the extras matrix.")
    with pytest.raises(ImportError, match=r"torchlens\[depyf\]"):
        tl.bridge.depyf.dump(nn.Linear(1, 1), torch.randn(1, 1))


def test_netron_export_is_valid_onnx_modelproto_json(export_log: Any, tmp_path: Path) -> None:
    """The Netron artifact parses into ``onnx.ModelProto`` via strict protobuf JSON.

    Netron's ONNX JSON reader decodes the file with the protobuf JSON mapping
    (``onnx.ProtoReader`` ``encoding='json'``), so a strict
    ``google.protobuf.json_format.Parse`` into ``onnx.ModelProto`` — which
    refuses unknown fields — is the real external acceptance contract.
    """

    onnx = pytest.importorskip("onnx")
    json_format = pytest.importorskip("google.protobuf.json_format")

    netron_path = tl.export.netron(export_log, tmp_path / "netron.json")
    model = json_format.Parse(netron_path.read_text(encoding="utf-8"), onnx.ModelProto())
    assert model.ir_version == 10
    assert model.producer_name == "torchlens"
    assert len(model.graph.node) > 0
    assert all(
        node.domain in ("ai.torchlens.lossy", "ai.torchlens.module") for node in model.graph.node
    )
    # The official structural validator joins the gate (memo D-07): it caught
    # SSA violations, unsorted bodies, and the killer function cycle that
    # netron's permissive reader swallowed or died on.
    onnx.checker.check_model(model, full_check=True)


def test_netron_export_passes_netron_onnx_json_sniffer(export_log: Any, tmp_path: Path) -> None:
    """The artifact satisfies Netron's ONNX-JSON acceptance predicate.

    The predicate is transcribed from netron 9.2.2 ``onnx.js`` (ProtoReader
    ``open``): the object must carry NO snake_case ONNX markers and at least
    one camelCase ``ModelProto`` marker. The historical snake_case export
    failed the first conjunct and Netron refused to open it.
    """

    netron_path = tl.export.netron(export_log, tmp_path / "netron.json")
    obj = json.loads(netron_path.read_text(encoding="utf-8"))
    no_snake_markers = (
        obj.get("ir_version") is None
        and obj.get("producer_name") is None
        and not isinstance(obj.get("opset_import"), list)
        and not isinstance(obj.get("metadata_props"), list)
    )
    camel_markers = (
        obj.get("irVersion") is not None
        or obj.get("producerName") is not None
        or isinstance(obj.get("opsetImport"), list)
        or isinstance(obj.get("metadataProps"), list)
        # dict guard: a list-valued graph crashed the transcription on .get()
        # (netron memo B0's sniffer list-crash; the vendor checks objectness).
        or (isinstance(obj.get("graph"), dict) and isinstance(obj["graph"].get("node"), list))
    )
    assert no_snake_markers and camel_markers


def test_model_explorer_export_passes_ingest_normalizer(export_log: Any, tmp_path: Path) -> None:
    """The artifact satisfies Model Explorer's JSON file-ingest contract.

    The normalizer is transcribed from the ai-edge-model-explorer 0.1.32 web
    app: a file is a graph collection iff top-level ``label`` AND ``graphs``
    are both non-null (extra keys are ignored); anything else refuses with
    "Unsupported JSON format". The historical export omitted ``label`` and
    was refused. Node shapes are checked against the vendor's
    ``graph_builder`` dataclass fields (same release).
    """

    explorer_path = tl.export.model_explorer(export_log, tmp_path / "explorer.json")
    obj = json.loads(explorer_path.read_text(encoding="utf-8"))
    assert obj.get("label") is not None and obj.get("graphs") is not None

    graph_fields = {
        "id",
        "nodes",
        "groupNodeAttributes",
        "groupNodeConfigs",
        "nodeLabelsToHide",
        "tasksData",
        "layoutConfigs",
    }
    node_fields = {
        "id",
        "label",
        "namespace",
        "subgraphIds",
        "attrs",
        "incomingEdges",
        "outputsMetadata",
        "inputsMetadata",
        "style",
        "config",
    }
    for graph in obj["graphs"]:
        assert set(graph) <= graph_fields
        for node in graph["nodes"]:
            assert set(node) <= node_fields
            assert all(set(attr) <= {"key", "value"} for attr in node.get("attrs", []))
            assert all(
                set(edge) <= {"sourceNodeId", "sourceNodeOutputId", "targetNodeInputId"}
                for edge in node.get("incomingEdges", [])
            )
