"""W051-IO: load-time integrity anchors the loader never checked (AUD-CODE 3.0).

(a) forged ``Op.edge_substitutions`` carriers, (b) manifest <-> metadata
anchors and op-graph coherence, (d) TorchLens-owned annotation families,
(e) an absent ``root_entry_point`` on a tlspec v9+ artifact. Tampering is
done on the bytes this test itself wrote (plain ``pickle`` on trusted local
bytes), then ``tl.load`` must refuse typed -- never load a wrong graph.
"""

from __future__ import annotations

import json
import pickle
import shutil
import tarfile
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._io import TorchLensIOError

pytestmark = pytest.mark.smoke

CORPUS = Path(__file__).parent / "release_goldens" / "genuine_release_artifacts.tar.gz"


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))) + 1.0)


@pytest.fixture(scope="module")
def source_bundle(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    torch.manual_seed(0)
    trace = tl.trace(_Model(), torch.randn(1, 1, 12, 12))
    try:
        path = tmp_path_factory.mktemp("w051_anchor") / "source.tlspec"
        tl.save(trace, path)
        yield path
    finally:
        trace.cleanup()


def _tampered(
    source: Path,
    target: Path,
    mutate_state: Callable[[dict[str, Any]], None] | None = None,
    mutate_manifest: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    shutil.copytree(source, target)
    if mutate_state is not None:
        metadata_path = target / "metadata.pkl"
        state = pickle.loads(metadata_path.read_bytes())
        mutate_state(state)
        metadata_path.write_bytes(pickle.dumps(state))
    if mutate_manifest is not None:
        manifest_path = target / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        mutate_manifest(manifest)
        manifest_path.write_text(json.dumps(manifest))
    return target


def _assert_refuses(path: Path, code: str, **fields: Any) -> TorchLensIOError:
    with pytest.raises(TorchLensIOError) as excinfo:
        tl.load(path)
    assert excinfo.value.fields["code"] == code, excinfo.value.fields
    for key, expected in fields.items():
        assert excinfo.value.fields.get(key) == expected, excinfo.value.fields
    return excinfo.value


# --- (b) manifest <-> metadata anchors ---------------------------------------


def test_untampered_source_loads(source_bundle: Path) -> None:
    assert len(tl.load(source_bundle).layer_list) > 0


def test_dropped_op_row_refuses(source_bundle: Path, tmp_path: Path) -> None:
    # A popped output row leaves its parent's ``children`` dangling, so the
    # graph-structure validator (inside rehydrate) fires first; the row-count
    # anchor is the second net (pinned separately below).
    path = _tampered(source_bundle, tmp_path / "drop.tlspec", lambda st: st["layer_list"].pop())
    _assert_refuses(path, "artifact_graph_structure_invalid", reason="dangling_relation")


def test_manifest_row_count_disagreeing_with_metadata_refuses(
    source_bundle: Path, tmp_path: Path
) -> None:
    path = _tampered(
        source_bundle,
        tmp_path / "rows.tlspec",
        mutate_manifest=lambda manifest: manifest.__setitem__("n_layers", 999),
    )
    _assert_refuses(path, "bundle_manifest_metadata_mismatch", field="n_layers")


def test_root_stamp_disagreeing_with_manifest_refuses(source_bundle: Path, tmp_path: Path) -> None:
    def older_root(state: dict[str, Any]) -> None:
        state["tlspec_version"] = state["tlspec_version"] - 2

    path = _tampered(source_bundle, tmp_path / "stamp.tlspec", older_root)
    _assert_refuses(path, "bundle_manifest_metadata_mismatch", field="tlspec_version")


def test_migrated_artifact_keeps_its_witnessed_stamp_lineage(tmp_path: Path) -> None:
    from torchlens.ecosystem.migrate import migrate

    with tarfile.open(CORPUS, "r:gz") as tar:
        tar.extractall(tmp_path)
    artifact = tmp_path / "art_v2.31.0_portable"
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["tlspec_version"] < tl._io.TLSPEC_VERSION
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        migrate(artifact)
    migrated = json.loads((artifact / "manifest.json").read_text())
    assert migrated["tlspec_version"] == tl._io.TLSPEC_VERSION
    # Root state still carries the source stamp; the witness explains it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = tl.load(artifact)
    assert len(loaded.layer_list) == 151
    # Delete the witness: the same bytes now refuse (the lineage is unproven).
    (artifact / "tl_migration_provenance.json").unlink()
    with pytest.raises(TorchLensIOError) as excinfo:
        tl.load(artifact)
    assert excinfo.value.fields["code"] in {
        "bundle_manifest_metadata_mismatch",
        "artifact_producer_pair_ungoverned",
    }


# --- (b) op-graph coherence ---------------------------------------------------


def test_duplicate_label_refuses(source_bundle: Path, tmp_path: Path) -> None:
    def duplicate(state: dict[str, Any]) -> None:
        ops = state["layer_list"]
        ops[2].label = ops[1].label

    path = _tampered(source_bundle, tmp_path / "dup.tlspec", duplicate)
    _assert_refuses(path, "artifact_graph_structure_invalid", reason="duplicate_label")


def test_dangling_parent_refuses(source_bundle: Path, tmp_path: Path) -> None:
    def dangle(state: dict[str, Any]) -> None:
        state["layer_list"][2].parents = ["nonexistent_op_99"]

    path = _tampered(source_bundle, tmp_path / "dangle.tlspec", dangle)
    _assert_refuses(path, "artifact_graph_structure_invalid", reason="dangling_relation")


@pytest.mark.parametrize(
    ("attr", "value"),
    [("pass_index", -1), ("pass_index", 0), ("num_passes", 0), ("pass_index", "1")],
)
def test_pass_stamp_out_of_range_refuses(
    source_bundle: Path, tmp_path: Path, attr: str, value: Any
) -> None:
    def stamp(state: dict[str, Any]) -> None:
        setattr(state["layer_list"][2], attr, value)

    path = _tampered(source_bundle, tmp_path / f"{attr}_{value}.tlspec", stamp)
    _assert_refuses(path, "artifact_graph_structure_invalid", reason="pass_stamp_range")


# --- (a) tier-(ii) edge carriers ---------------------------------------------


def test_forged_edge_substitutions_refuse(source_bundle: Path, tmp_path: Path) -> None:
    def forge(state: dict[str, Any]) -> None:
        op = state["layer_list"][2]
        op.edge_substitutions = {("positional", (0,)): {"forged": True}}
        op.edge_replacement_stamps = {("positional", (0,)): {"forged": True}}

    path = _tampered(source_bundle, tmp_path / "edge.tlspec", forge)
    _assert_refuses(path, "artifact_edge_substitutions_invalid", field="Op.edge_substitutions")


def test_edge_carrier_without_matching_stamps_refuses(source_bundle: Path, tmp_path: Path) -> None:
    def forge(state: dict[str, Any]) -> None:
        state["layer_list"][2].edge_substitutions = {("positional", (0,)): {}}

    path = _tampered(source_bundle, tmp_path / "edge2.tlspec", forge)
    _assert_refuses(path, "artifact_edge_substitutions_invalid")


def test_genuine_edge_intervention_round_trips(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trace = tl.trace(
        _Model(),
        torch.randn(1, 1, 12, 12),
        capture=tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True),
    )
    edge = next(e for e in trace.edges if e.parent_label == "relu_1_2")
    fork = trace.fork()
    fork.do(edge.__selection__(), trace["relu_1_2"].out.clone() * 0.5)
    path = tmp_path / "genuine_edge.tlspec"
    tl.save(fork, path)
    loaded = tl.load(path)
    carriers = [op for op in loaded.layer_list if getattr(op, "edge_substitutions", None)]
    assert carriers, "the genuine tier-(ii) carrier must persist and load"


# --- (d) TorchLens-owned annotation families -----------------------------------


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (
            lambda st: st["annotations"].__setitem__("logged_values", 5),
            'Trace.annotations["logged_values"]',
        ),
        (
            lambda st: st["annotations"].__setitem__("distributed", "junk"),
            'Trace.annotations["distributed"]',
        ),
        (
            lambda st: st["layer_list"][2].annotations.__setitem__("collective", "junk"),
            'Op.annotations["collective"]',
        ),
        (
            lambda st: st["layer_list"][2].annotations.__setitem__("save_mode", "bogus"),
            'Op.annotations["save_mode"]',
        ),
        (
            lambda st: st["layer_list"][2].annotations.__setitem__("saved_out_version", "x"),
            'Op.annotations["saved_out_version"]',
        ),
        (
            lambda st: st["layer_list"][2].annotations.__setitem__("varying_across_passes", 12),
            'Op.annotations["varying_across_passes"]',
        ),
        (
            lambda st: st["layer_list"][2].annotations.__setitem__("dedup_source_id", "abc"),
            'Op.annotations["dedup_source_id"]',
        ),
    ],
    ids=[
        "logged_values",
        "distributed",
        "collective",
        "save_mode",
        "saved_out_version",
        "varying",
        "dedup",
    ],
)
def test_owned_annotation_families_refuse_off_shape(
    source_bundle: Path, tmp_path: Path, mutate: Callable[[dict[str, Any]], None], field: str
) -> None:
    def apply(state: dict[str, Any]) -> None:
        state.setdefault("annotations", {})
        ops = state["layer_list"]
        if getattr(ops[2], "annotations", None) is None:
            ops[2].annotations = {}
        mutate(state)

    path = _tampered(source_bundle, tmp_path / "ann.tlspec", apply)
    _assert_refuses(path, "artifact_annotations_invalid", field=field)


def test_user_annotation_keys_stay_open(source_bundle: Path, tmp_path: Path) -> None:
    def user_key(state: dict[str, Any]) -> None:
        state.setdefault("annotations", {})["my_experiment"] = {"seed": 7, "tags": ["a", "b"]}

    path = _tampered(source_bundle, tmp_path / "user.tlspec", user_key)
    assert tl.load(path).annotations["my_experiment"] == {"seed": 7, "tags": ["a", "b"]}


# --- (e) root_entry_point on tlspec v9+ ---------------------------------------


def test_absent_root_entry_point_within_v9_is_entry_dark(
    source_bundle: Path, tmp_path: Path
) -> None:
    """tlspec 9 carries pre-C07X writers without the fact, so absence LOADS.

    Evidence: ``tests/agent_surface_goldens/clean.tlspec`` (tlspec 9, torchlens
    2.34.1) has no ``root_entry_point``; the fail-closed rule is armed from the
    next coordinated bump (``_ROOT_ENTRY_POINT_REQUIRED_FROM_TLSPEC``).
    """

    from torchlens._io._forgery_identity_facts import _ROOT_ENTRY_POINT_REQUIRED_FROM_TLSPEC

    assert _ROOT_ENTRY_POINT_REQUIRED_FROM_TLSPEC == tl._io.TLSPEC_VERSION + 1
    path = _tampered(
        source_bundle, tmp_path / "rep.tlspec", lambda st: st.__setitem__("root_entry_point", None)
    )
    assert tl.load(path).root_entry_point is None


def test_absent_root_entry_point_refuses_from_the_armed_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The armed rule fires inside the governed window once the stamp reaches the floor."""

    from torchlens._io import _forgery_identity_facts as facts
    from torchlens._io.state_contract import governed_artifact_load

    monkeypatch.setattr(facts, "_ROOT_ENTRY_POINT_REQUIRED_FROM_TLSPEC", tl._io.TLSPEC_VERSION)
    torch.manual_seed(0)
    trace = tl.trace(_Model(), torch.randn(1, 1, 12, 12))
    trace.root_entry_point = None
    with governed_artifact_load(), pytest.raises(TorchLensIOError) as excinfo:
        facts._validate_root_entry_point(trace)
    assert excinfo.value.fields["code"] == "artifact_root_entry_point_invalid"
    assert excinfo.value.fields["reason"] == "absent_on_current_schema"
    # Outside the governed window (plain session state) the same trace passes.
    facts._validate_root_entry_point(trace)


def test_pre_v9_artifact_without_root_entry_point_still_loads(tmp_path: Path) -> None:
    with tarfile.open(CORPUS, "r:gz") as tar:
        tar.extractall(tmp_path)
    with pytest.warns(Warning):
        loaded = tl.load(tmp_path / "art_v2.34.1_portable")
    assert loaded.root_entry_point is None
    assert loaded.tlspec_version < 9


def test_plain_session_pickle_tolerates_absent_root_entry_point() -> None:
    torch.manual_seed(0)
    trace = tl.trace(_Model(), torch.randn(1, 1, 12, 12))
    trace.root_entry_point = None
    restored = pickle.loads(pickle.dumps(trace))
    assert restored.root_entry_point is None
