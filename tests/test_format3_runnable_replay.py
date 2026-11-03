"""The 3.0 runnable component replays exactly as the live trace reruns.

``torchlens/_io/format3/runnable.py`` carries the runnable component as typed
JSON plus safetensors (``runnable/descriptor.json``, ``state_dict-NNNN.safetensors``,
``buffers.safetensors`` and, when archived, ``activations-NNNN.safetensors``), listed by
the manifest's ``runnable`` object and ``files`` rows. These tests prove the carrier change keeps
the replay check bit for bit:

* the descriptor written is the descriptor the tlspec 11 save stamped, and the
  state families hold the same tensors;
* a trace saved at the runnable level, loaded and replayed returns the live
  rerun's output element for element, with the same attestation and path
  faithfulness verdicts, on every classics smoke-subset entry and on toy
  models with weights, non-persistent buffers and archived activations;
* a capture the runnable preflight refuses refuses identically in both formats;
* an altered, truncated or symlinked component refuses with the contract's
  typed codes.

Until the 3.0 trace writer and reader doors are built in this checkout, the
trace records travel through the tlspec 11 door and only the runnable component
travels through the 3.0 door (``_save_and_load_runnable``); once ``Trace.save``
writes the 3.0 envelope the same tests run end to end through it. That bridge is
test-only: no product code reads a mixed artifact.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.classics_corpus._loader import CorpusEntry, build_entry, corpus_entries
from torchlens._io.format3 import (
    runnable as runnable3,
    runnable_families,
)
from torchlens._io.format_errors import ArtifactError
from torchlens.errors import RunnablePreflightError, TorchLensError
from torchlens.intervention.errors import EngineDispatchError
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(replayable=True, capture_container_structure=True, cache=False)


class _StateModel(nn.Module):
    """Linear map, a persistent and a non-persistent buffer, and a relu."""

    def __init__(self) -> None:
        """Build the layer and both buffers."""

        super().__init__()
        self.fc = nn.Linear(3, 4)
        self.register_buffer("shift", torch.arange(4, dtype=torch.float32))
        self.register_buffer("scale", torch.full((4,), 0.5), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Shift, scale and rectify the linear map."""

        return torch.relu(self.fc(value) * self.scale + self.shift)


class _ReluModel(nn.Module):
    """Two linear maps around a relu an intervention can replace."""

    def __init__(self) -> None:
        """Build both maps."""

        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run linear, relu, linear."""

        return self.fc2(torch.relu(self.fc1(value)))


class _TiedModel(nn.Module):
    """An embedding whose weight the output projection shares (tied storage)."""

    def __init__(self) -> None:
        """Build the embedding and tie the projection to it."""

        super().__init__()
        self.embed = nn.Embedding(5, 3)
        self.head = nn.Linear(3, 5, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Embed the ids and project back onto the vocabulary."""

        return self.head(self.embed(value))


def _quiet_save(trace: Any, path: Path, **kwargs: Any) -> None:
    """Save while tolerating the one-time runnable disclosures."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trace.save(path, **kwargs)


def _save_and_load_runnable(
    trace: Any, root: Path, *, include_activations: bool = False
) -> tuple[Any, Path]:
    """Save and load through the real 3.0 doors, with the ruled runnable component."""

    artifact = root / "trace3"
    _quiet_save(
        trace,
        artifact,
        level="runnable",
        include_weights=True,
        include_activations=include_activations,
    )
    manifest = json.loads((artifact / "manifest.json").read_text("utf-8"))
    assert manifest["runnable"]["descriptor"] == runnable3.DESCRIPTOR_PATH
    assert "run" not in manifest and "tensors" not in manifest
    return tl.load(artifact), artifact


def _leaves(value: Any) -> Iterator[Any]:
    """Yield the leaves of a nested output in a fixed order."""

    if isinstance(value, dict):
        for key in sorted(value, key=repr):
            yield from _leaves(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _assert_identical(replayed: Any, live: Any, where: str) -> None:
    """Assert two outputs are equal element for element (NaN equal to NaN)."""

    replayed_leaves, live_leaves = list(_leaves(replayed)), list(_leaves(live))
    assert len(replayed_leaves) == len(live_leaves), where
    for index, (got, want) in enumerate(zip(replayed_leaves, live_leaves, strict=True)):
        if isinstance(want, torch.Tensor):
            assert isinstance(got, torch.Tensor), f"{where}: leaf {index} is not a tensor"
            assert got.dtype == want.dtype and got.shape == want.shape, f"{where}: leaf {index}"
            torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)
        else:
            assert got == want, f"{where}: leaf {index}"


def _attempt(action: Any) -> tuple[Any, TorchLensError | None]:
    """Run one step; return its value, or the typed TorchLens refusal it raised."""

    try:
        return action(), None
    except TorchLensError as refusal:
        return None, refusal


def _assert_same_refusal(new: BaseException | None, old: BaseException | None, where: str) -> None:
    """Assert the 3.0 path and the tlspec 11 path refused with the same type and code."""

    assert new is not None and old is not None, f"{where}: 3.0 refused {new!r}, tlspec 11 {old!r}"
    assert type(new) is type(old), f"{where}: {type(new).__name__} != {type(old).__name__}"
    new_code = getattr(new, "fields", {}).get("code")
    assert new_code == getattr(old, "fields", {}).get("code"), where


def _assert_replays_like_live(model: nn.Module, inputs: Any, tmp_path: Path, where: str) -> None:
    """Capture, save at the runnable level, load, replay; compare with the live rerun.

    The tlspec 11 runnable save of the same capture is the reference: where it
    refuses at save, load or replay, the 3.0 component must refuse with the same
    exception type and code; where it replays, the 3.0 replay must equal both it
    and the live rerun element for element.
    """

    trace = tl.trace(model, inputs, capture=_CAP)
    try:
        live = trace.run(inputs=inputs)
    except RunnablePreflightError as live_refusal:
        # The live provider refuses this capture; the saved one must refuse the same way.
        with pytest.raises(RunnablePreflightError) as saved_refusal:
            _save_and_load_runnable(trace, tmp_path)
        assert saved_refusal.value.fields.get("code") == live_refusal.fields.get("code"), where
        return
    old_path = tmp_path / "old.tlspec"

    def _old_load() -> Any:
        _quiet_save(trace, old_path, level="runnable", include_weights=True)
        return tl.load(old_path)

    old_loaded, old_refusal = _attempt(_old_load)
    loaded, refusal = _attempt(lambda: _save_and_load_runnable(trace, tmp_path)[0])
    if old_refusal is not None or refusal is not None:
        _assert_same_refusal(refusal, old_refusal, f"{where} (save and load)")
        return
    old_replay, old_refusal = _attempt(lambda: old_loaded.run(inputs=inputs))
    replayed, refusal = _attempt(lambda: loaded.run(inputs=inputs))
    if old_refusal is not None or refusal is not None:
        _assert_same_refusal(refusal, old_refusal, f"{where} (replay)")
        return
    _assert_identical(replayed.output, live.output, where)
    assert replayed.report.path_faithfulness == live.report.path_faithfulness, where
    _assert_identical(replayed.output, old_replay.output, where)
    assert replayed.report.numeric_attestation == old_replay.report.numeric_attestation, where
    assert replayed.report.path_faithfulness == old_replay.report.path_faithfulness, where


def test_descriptor_and_families_preserve_the_frozen_run_contract(tmp_path: Path) -> None:
    """The carrier preserves the frozen descriptor and the full captured state exactly."""

    from safetensors.torch import load_file

    from torchlens._io.runnable import (
        require_sparse_run_descriptor,
        sparse_descriptor_to_json,
        with_weight_payload,
    )

    model = _StateModel().eval()
    trace = tl.trace(model, torch.ones(2, 3), capture=_CAP)
    old = tmp_path / "old.tlspec"
    _quiet_save(trace, old, level="runnable", include_weights=True)
    artifact = tmp_path / "new"
    artifact.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        written = runnable3.write_runnable(trace, artifact, include_weights=True)

    manifest = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
    descriptor = json.loads((artifact / "runnable" / "descriptor.json").read_text("utf-8"))
    expected = sparse_descriptor_to_json(with_weight_payload(require_sparse_run_descriptor(trace)))
    assert descriptor == expected
    assert descriptor == json.loads((old / "runnable" / "descriptor.json").read_text("utf-8"))
    assert "run" not in manifest and "tensors" not in manifest

    references = written.manifest.pop("state_dict_tensors")
    buffer_references = written.manifest.pop("buffers_tensors")
    assert written.manifest == {
        "descriptor": "runnable/descriptor.json",
        **runnable3.manifest_versions(),
        "state_dict": ["runnable/state_dict-0001.safetensors"],
        "state_dict_family": "state_dict_v1",
        "buffers": "runnable/buffers.safetensors",
        "buffers_family": "runnable_nonpersistent_buffer_v1",
    }
    assert set(references) == set(model.state_dict())
    assert buffer_references["scale"] == {
        "$tl": "tensor",
        "blob": "runnable/buffers.safetensors",
        "key": buffer_references["scale"]["key"],
        "dtype": "float32",
        "shape": [4],
        "device": "cpu",
        "requires_grad": False,
        "codec": "torch_safetensors_v1",
    }
    assert set(written.files) == {
        "runnable/descriptor.json",
        "runnable/state_dict-0001.safetensors",
        "runnable/buffers.safetensors",
    }
    assert sorted(p.name for p in (artifact / "runnable").iterdir()) == [
        "buffers.safetensors",
        "descriptor.json",
        "state_dict-0001.safetensors",
    ]
    state = load_file(str(artifact / "runnable" / "state_dict-0001.safetensors"))
    assert set(state) == {reference["key"] for reference in references.values()}
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(state[references[name]["key"]], tensor, rtol=0, atol=0)
    buffers = load_file(str(artifact / "runnable" / "buffers.safetensors"))
    scale_key = buffer_references["scale"]["key"]
    torch.testing.assert_close(buffers[scale_key], model.scale, rtol=0, atol=0)


def test_toy_state_model_replays_like_live(tmp_path: Path) -> None:
    """Weights, a persistent and a non-persistent buffer replay element for element."""

    _assert_replays_like_live(_StateModel().eval(), torch.randn(2, 3), tmp_path, "state model")


def test_tied_weights_replay_like_live(tmp_path: Path) -> None:
    """Weights sharing one storage write as separate safetensors members and replay."""

    model = _TiedModel().eval()
    _assert_replays_like_live(model, torch.tensor([[0, 3, 4]]), tmp_path, "tied model")


def test_archived_activations_round_trip_and_attest(tmp_path: Path) -> None:
    """The activation family loads, keeps its digests and the replay attests as before."""

    model = _StateModel().eval()
    inputs = torch.randn(2, 3)
    trace = tl.trace(model, inputs, capture=_CAP)
    loaded, _artifact = _save_and_load_runnable(trace, tmp_path, include_activations=True)
    archived = loaded._runnable.archived_activations
    assert archived, "the activation family did not bind"
    replayed = loaded.run(inputs=inputs)
    old_path = tmp_path / "old.tlspec"
    _quiet_save(trace, old_path, level="runnable", include_weights=True, include_activations=True)
    old_loaded = tl.load(old_path)
    assert set(old_loaded._runnable.archived_activations) == set(archived)
    for key, record in archived.items():
        assert record.byte_digest == old_loaded._runnable.archived_activations[key].byte_digest
    old_replay = old_loaded.run(inputs=inputs)
    _assert_identical(replayed.output, old_replay.output, "activations")
    assert replayed.report.numeric_attestation == old_replay.report.numeric_attestation


@pytest.mark.heavy
@pytest.mark.parametrize(
    "entry",
    [pytest.param(entry, id=entry.id) for entry in corpus_entries("smoke")],
)
def test_classics_smoke_entry_replays_like_live(entry: CorpusEntry, tmp_path: Path) -> None:
    """Every classics smoke-subset entry replays from the 3.0 component like the live rerun."""

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        model, inputs = build_entry(entry)
        _assert_replays_like_live(model, inputs, tmp_path, entry.id)


def test_intervened_capture_refuses_the_runnable_component(tmp_path: Path) -> None:
    """An intervention-replaced capture refuses the runnable component typed, as before."""

    trace = tl.trace(
        _ReluModel(),
        torch.randn(2, 4),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
        capture=_CAP,
    )
    with pytest.raises(RunnablePreflightError) as info:
        runnable3.write_runnable(trace, tmp_path, include_weights=True)
    assert info.value.fields.get("code") == "sparse_preflight_failed"
    assert not (tmp_path / "runnable").exists()


def _intervened_rerun_case(tmp_path: Path, edit: Any) -> tuple[Any, Any, torch.Tensor]:
    """Save an intervened trace through the 3.0 writer and load it back."""

    model = _ReluModel()
    value = torch.randn(2, 4)
    trace = tl.trace(model, value, intervene=tl.when(tl.func("relu"), edit))
    path = tmp_path / "intervened3"
    _quiet_save(trace, path, level="portable")
    return tl.load(path), model, value


def test_loaded_builtin_edit_reruns_like_live(tmp_path: Path) -> None:
    """io.1: a loaded trace with a built-in edit reruns to the live intervened output."""

    loaded, model, value = _intervened_rerun_case(tmp_path, tl.zero_ablate())
    live = tl.trace(model, value, intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
    rerun = loaded.run(model, value)
    _assert_identical(
        [op.out for op in rerun.output_ops],
        [op.out for op in live.output_ops],
        "built-in edit rerun",
    )


def test_loaded_callable_placeholder_rerun_refuses_typed(tmp_path: Path) -> None:
    """io.1: a loaded recipe whose callable became a placeholder refuses the rerun typed."""

    def halve(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        """Halve the activation."""

        del ctx
        return out * 0.5

    loaded, model, value = _intervened_rerun_case(tmp_path, halve)
    with pytest.raises(EngineDispatchError) as info:
        loaded.run(model, value)
    assert info.value.fields.get("code") == "run_intervention_spec_not_persisted"


def _component(tmp_path: Path) -> tuple[Path, runnable3.RunnableWrite]:
    """Write one runnable component with weights and buffers; return the root and the write."""

    trace = tl.trace(_StateModel().eval(), torch.ones(2, 3), capture=_CAP)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        written = runnable3.write_runnable(trace, artifact, include_weights=True)
    return artifact, written


def _refusal_code(artifact: Path, manifest: Any, files: Any) -> str:
    """Load the component and return the typed refusal code."""

    with pytest.raises(ArtifactError) as info:
        runnable3.load_runnable(artifact, manifest, files)
    return str(info.value.fields["code"])


def test_component_absent_loads_as_none(tmp_path: Path) -> None:
    """An artifact without a runnable object or directory has no component."""

    assert runnable3.load_runnable(tmp_path, None, {}) is None


def test_component_round_trips_its_families(tmp_path: Path) -> None:
    """The loaded families hold the written tensors under the bound family names."""

    artifact, written = _component(tmp_path)
    component = runnable3.load_runnable(artifact, written.manifest, written.files)
    assert component is not None
    assert set(component.families) == {"state_dict", "nonpersistent_buffers"}
    assert set(component.families["state_dict"]) == {"fc.weight", "fc.bias", "shift"}


def test_state_dict_chunks_at_the_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A state dict larger than the chunk limit spans numbered files and replays."""

    monkeypatch.setattr(runnable_families, "CHUNK_BYTES", 16)
    artifact, written = _component(tmp_path)
    assert len(written.manifest["state_dict"]) == 3
    assert written.manifest["state_dict"][-1] == "runnable/state_dict-0003.safetensors"
    component = runnable3.load_runnable(artifact, written.manifest, written.files)
    assert component is not None
    assert set(component.families["state_dict"]) == {"fc.weight", "fc.bias", "shift"}
    _assert_replays_like_live(_StateModel().eval(), torch.randn(2, 3), tmp_path, "chunked")


def test_unlisted_component_directory_refuses(tmp_path: Path) -> None:
    """A runnable directory the manifest does not list refuses."""

    artifact, _written = _component(tmp_path)
    assert _refusal_code(artifact, None, {}) == "manifest_schema_invalid"


def test_altered_family_file_refuses(tmp_path: Path) -> None:
    """A family file whose bytes changed refuses as an integrity failure."""

    artifact, written = _component(tmp_path)
    target = artifact / "runnable" / "state_dict-0001.safetensors"
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(bytes(data))
    assert _refusal_code(artifact, written.manifest, written.files) == (
        "bundle_metadata_integrity_refused"
    )


def test_altered_descriptor_refuses(tmp_path: Path) -> None:
    """A descriptor edited after the write refuses as an integrity failure."""

    artifact, written = _component(tmp_path)
    target = artifact / "runnable" / "descriptor.json"
    target.write_text(target.read_text("utf-8").replace('"cpu"', '"cuda"', 1), "utf-8")
    assert _refusal_code(artifact, written.manifest, written.files) == (
        "bundle_metadata_integrity_refused"
    )


def test_missing_family_file_refuses(tmp_path: Path) -> None:
    """A family the manifest lists but the directory lacks refuses as missing."""

    artifact, written = _component(tmp_path)
    (artifact / "runnable" / "buffers.safetensors").unlink()
    assert _refusal_code(artifact, written.manifest, written.files) == "bundle_load_failed"


def _edit_reference(manifest: dict[str, Any], key: str, value: Any) -> None:
    """Set one field of the ``fc.weight`` tensor reference in a copied manifest."""

    references = {name: dict(ref) for name, ref in manifest["state_dict_tensors"].items()}
    references["fc.weight"][key] = value
    manifest["state_dict_tensors"] = references


@pytest.mark.parametrize(
    ("edit", "where"),
    [
        (lambda m: _edit_reference(m, "dtype", "float64"), "dtype"),
        (lambda m: _edit_reference(m, "shape", [3, 4]), "shape"),
        (lambda m: _edit_reference(m, "key", "t99"), "absent key"),
        (
            lambda m: m.update(
                state_dict_tensors={
                    k: v for k, v in m["state_dict_tensors"].items() if k != "fc.bias"
                }
            ),
            "unreferenced tensor",
        ),
    ],
)
def test_reference_disagreeing_with_its_file_refuses(tmp_path: Path, edit: Any, where: str) -> None:
    """A tensor reference must name a present key with the recorded dtype and shape."""

    artifact, written = _component(tmp_path)
    manifest = dict(written.manifest)
    edit(manifest)
    assert _refusal_code(artifact, manifest, written.files) == (
        "bundle_metadata_integrity_refused"
    ), where


@pytest.mark.parametrize(
    ("edit", "where"),
    [
        (lambda m, f: m.update(callables={"fc": "pickled"}), "unknown key"),
        (lambda m, f: m.pop("state_dict_family"), "family without schema"),
        (lambda m, f: m.update(call_recipe_version="v0"), "stale version"),
        (lambda m, f: m.update(descriptor="runnable/other.json"), "descriptor path"),
        (lambda m, f: m.update(state_dict=["../../outside.safetensors"]), "outside path"),
        (lambda m, f: m.update(state_dict=["runnable/state_dict-0002.safetensors"]), "numbering"),
        (lambda m, f: m.update(state_dict=[]), "empty family"),
        (lambda m, f: f.pop("runnable/buffers.safetensors"), "no files row"),
        (lambda m, f: m.pop("buffers_tensors"), "family without references"),
        (lambda m, f: m.update(buffers_tensors={}), "empty references"),
        (lambda m, f: _edit_reference(m, "blob", "runnable/buffers.safetensors"), "other file"),
        (lambda m, f: _edit_reference(m, "codec", "pickle"), "codec"),
        (lambda m, f: _edit_reference(m, "loader", "x"), "unknown reference key"),
        (lambda m, f: _edit_reference(m, "complex", False), "complex false"),
    ],
)
def test_malformed_runnable_object_refuses(tmp_path: Path, edit: Any, where: str) -> None:
    """The manifest's runnable object is closed, versioned and names fixed files."""

    artifact, written = _component(tmp_path)
    manifest, files = dict(written.manifest), dict(written.files)
    edit(manifest, files)
    assert _refusal_code(artifact, manifest, files) == "manifest_schema_invalid", where


def test_missing_state_key_refuses_at_bind(tmp_path: Path) -> None:
    """A state family consistently stripped of one tensor refuses at the strict bind."""

    from safetensors.torch import load_file, save_file

    from torchlens._utils.hashing import sha256_file
    from torchlens.errors import StateBindingError

    trace = tl.trace(_StateModel().eval(), torch.ones(2, 3), capture=_CAP)
    records = tmp_path / "records.tlspec"
    _quiet_save(trace, records, level="runnable")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        written = runnable3.write_runnable(trace, artifact, include_weights=True)
    relative = "runnable/state_dict-0001.safetensors"
    references = dict(written.manifest["state_dict_tensors"])
    dropped = references.pop("fc.bias")
    state = load_file(str(artifact / relative))
    state.pop(dropped["key"])
    save_file(state, str(artifact / relative))
    files = dict(written.files)
    files[relative] = {
        "sha256": sha256_file(artifact / relative),
        "bytes": (artifact / relative).stat().st_size,
    }
    manifest = {**written.manifest, "state_dict_tensors": references}
    with pytest.raises(StateBindingError):
        runnable3.attach_runnable(tl.load(records), artifact, manifest, files)


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlinked_component_refuses(tmp_path: Path) -> None:
    """A symlinked component file refuses before it is read."""

    artifact, written = _component(tmp_path)
    target = artifact / "runnable" / "descriptor.json"
    moved = tmp_path / "elsewhere.json"
    target.rename(moved)
    target.symlink_to(moved)
    assert _refusal_code(artifact, written.manifest, written.files) == (
        "load_path_symlink_rejected"
    )


def test_family_present_against_descriptor_refuses(tmp_path: Path) -> None:
    """A state family bound onto a descriptor that declares no weights refuses typed."""

    trace = tl.trace(_StateModel().eval(), torch.ones(2, 3), capture=_CAP)
    records = tmp_path / "records.tlspec"
    _quiet_save(trace, records, level="runnable")
    loaded = tl.load(records)
    stray = {"fc.weight": torch.zeros(4, 3)}
    buffers = {"scale": torch.full((4,), 0.5)}
    with pytest.raises(ArtifactError) as info:
        runnable3.bind_runnable_families(
            loaded, {"nonpersistent_buffers": buffers, "state_dict": stray}
        )
    assert info.value.fields["code"] == "bundle_metadata_integrity_refused"


def test_component_directory_is_written_once(tmp_path: Path) -> None:
    """Writing into an artifact that already holds a component refuses typed."""

    artifact, _written = _component(tmp_path)
    trace = tl.trace(_StateModel().eval(), torch.ones(2, 3), capture=_CAP)
    with pytest.raises(ArtifactError) as info:
        runnable3.write_runnable(trace, artifact)
    assert info.value.fields["code"] == "bundle_save_failed"


def test_complex_tensor_round_trips_as_its_real_view(tmp_path: Path) -> None:
    """A complex tensor is stored as its real view and decoded back exactly."""

    value = torch.randn(3, dtype=torch.complex64)
    files: dict[str, Any] = {}
    (tmp_path / "runnable").mkdir()
    sink = runnable_families.FamilySink(tmp_path, files, [0])
    paths, references = runnable_families.write_family(
        sink, "state_dict", [("w", value)], chunked=True
    )
    assert references["w"]["complex"] is True and references["w"]["dtype"] == "complex64"
    loaded = runnable_families.read_family(tmp_path, paths, references, files, "cpu")
    torch.testing.assert_close(loaded["w"], value, rtol=0, atol=0)


def test_runnable_and_record_payloads_share_one_tensor_key_counter(tmp_path: Path) -> None:
    """All safetensors keys are unique across ordinary and runnable component files."""

    from safetensors import safe_open

    from torchlens._io.format3.payloads import PayloadStore, host_tensor

    counter = [0]
    store = PayloadStore(tmp_path, counter=counter)
    store.submit(host_tensor(torch.ones(2)))
    store.close()
    trace = tl.trace(_StateModel().eval(), torch.ones(2, 3), capture=_CAP)
    try:
        written = runnable3.write_runnable(trace, tmp_path, counter=counter)
        assert written.manifest["descriptor"] == "runnable/descriptor.json"
        all_keys = []
        for path in sorted(tmp_path.rglob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                all_keys.extend(handle.keys())
        assert len(all_keys) > 1
        assert len(all_keys) == len(set(all_keys))
        assert sorted(int(key[1:]) for key in all_keys) == list(range(len(all_keys)))
    finally:
        trace.cleanup()
