"""Extraction-engine v2 integration tests (lane C04's row gate lives here).

The RESUME-IDENTITY rows: T-MODELSWAP (a pretrained prefix resumed with a
random-init model REFUSES typed — until this lane it completed
``status=complete``), the identity level matrix on real resumes, the strict
opaque-resume rule, completed-v1 migration through the public entry, and the
transform door's engine seams (P2 declared ctx dispatch incl. the
``torch.abs`` counterexample, T-C2 row guard, T-C6 frozen-plan refusal,
per-site Mapping).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_substrate import ExtractionArtifactError
from torchlens._errors import InvalidArgumentError
from torchlens.dataset_extraction import (
    MANIFEST_SCHEMA_V2,
    DatasetExtractionResumeError,
    load_extraction,
)
from torchlens.transforms import (
    DEFAULT,
    PlannedStep,
    TransformContext,
    TransformContractError,
    TransformDefinition,
    chain,
    register_transform,
    registered_transform_names,
    with_context,
)

pytestmark = pytest.mark.smoke

_LAYERS = {"relu": "relu", "logits": "output_1"}


class _Model(nn.Module):
    """Tiny deterministic model; distinct seeds give distinct identities."""

    def __init__(self, seed: int = 0) -> None:
        """Build the affine stack from ``seed``.

        Parameters
        ----------
        seed:
            Torch manual seed for the parameter draw.
        """

        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.head = nn.Linear(4, 2)
        self.n_forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the stack and count the call.

        Parameters
        ----------
        inputs:
            Batched stimulus rows.

        Returns
        -------
        torch.Tensor
            Model logits.
        """

        self.n_forward_calls += 1
        return self.head(self.relu(self.fc(inputs)))


def _stimuli(n: int = 10) -> torch.Tensor:
    """Return ``n`` distinctive stimulus rows.

    Parameters
    ----------
    n:
        Row count.

    Returns
    -------
    torch.Tensor
        ``(n, 3)`` float rows identifying their index.
    """

    return torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)


def _interrupt_after(tmp_path: Path, n_shards: int) -> None:
    """Reconstruct a mid-run artifact: ``n_shards`` committed, in progress.

    Parameters
    ----------
    tmp_path:
        Artifact directory.
    n_shards:
        Committed-shard count to keep.
    """

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "in_progress"
    manifest["totals"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ledger_path = tmp_path / "ledger.jsonl"
    rows = ledger_path.read_text(encoding="utf-8").splitlines()
    ledger_path.write_text("".join(line + "\n" for line in rows[:n_shards]), encoding="utf-8")
    for line in rows[n_shards:]:
        (tmp_path / json.loads(line)["file"]).unlink()


# --- RESUME-IDENTITY ROWS (the C04 row gate) ----------------------------------


def test_resume_identity_random_init_refuses_t_modelswap(tmp_path: Path) -> None:
    """T-MODELSWAP: a pretrained prefix resumed with a random-init model REFUSES.

    Before this lane the resume completed ``status=complete``, silently mixing
    checkpoint activations with random ones — the headline fails-open defect
    of the extraction slot (extract memo D6).
    """

    # Both variants constructed before the first capture in this process.
    original = _Model(seed=0).eval()
    random_init = _Model(seed=999).eval()
    tl.extract_dataset(
        original, _stimuli(), _LAYERS, batch_size=4, output_dir=tmp_path, progress=False
    )
    _interrupt_after(tmp_path, 1)
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        tl.extract_dataset(
            random_init,
            _stimuli(),
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path,
            progress=False,
            resume=True,
        )
    assert excinfo.value.fields["code"] == "extraction_resume_model_identity_mismatch"
    assert excinfo.value.fields["mismatched_fields"] == ["model_identity"]
    assert random_init.n_forward_calls == 0, "refused BEFORE any forward"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "in_progress", "never completes on a swapped model"


def test_resume_identity_same_state_resumes_and_completes(tmp_path: Path) -> None:
    """The matching model resumes the prefix and lands terminal totals."""

    model = _Model(seed=0).eval()
    tl.extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=4, output_dir=tmp_path, progress=False
    )
    _interrupt_after(tmp_path, 1)
    paths = tl.extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=4, output_dir=tmp_path, progress=False, resume=True
    )
    assert [path.name for path in paths] == [f"batch_0000{i}.safetensors" for i in range(3)]
    loaded = load_extraction(tmp_path)
    assert loaded.manifest["status"] == "complete"
    clean = tl.extract_dataset(model, _stimuli(), _LAYERS, batch_size=4, progress=False)
    for key, tensor in clean.items():
        assert torch.equal(loaded.activations[key], tensor)


def test_resume_identity_in_place_weight_edit_refuses(tmp_path: Path) -> None:
    """The LoRA-merge/lesion class: an in-place weight edit changes identity."""

    model = _Model(seed=0).eval()
    tl.extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=4, output_dir=tmp_path, progress=False
    )
    _interrupt_after(tmp_path, 1)
    with torch.no_grad():
        model.fc.weight[0, 0] += 1.0
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path,
            progress=False,
            resume=True,
        )
    assert excinfo.value.fields["code"] == "extraction_resume_model_identity_mismatch"


def test_resume_identity_cross_level_refuses(tmp_path: Path) -> None:
    """An artifact recorded at 'measured' never compares against 'none'."""

    model = _Model(seed=0).eval()
    tl.extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=4, output_dir=tmp_path, progress=False
    )
    _interrupt_after(tmp_path, 1)
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path,
            progress=False,
            resume=True,
            model_identity="none",
        )
    assert excinfo.value.fields["code"] == "extraction_resume_model_identity_mismatch"


def test_resume_identity_none_optout_is_recorded_and_resumable(tmp_path: Path) -> None:
    """'none' is an explicit recorded opt-out: resume proceeds, no claim made."""

    model = _Model(seed=0).eval()
    tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        model_identity="none",
    )
    _interrupt_after(tmp_path, 1)
    paths = tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        resume=True,
        model_identity="none",
    )
    assert len(paths) == 3
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["signature"]["model_identity"]["level"] == "none"


def test_resume_identity_assertion_matrix(tmp_path: Path) -> None:
    """Asserted identities compare by their claim: equal resumes, unequal refuses."""

    claim = {"checkpoint": "org/model", "revision": "abc123"}
    model = _Model(seed=0).eval()
    tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        model_identity=claim,
    )
    _interrupt_after(tmp_path, 1)
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path,
            progress=False,
            resume=True,
            model_identity={"checkpoint": "org/model", "revision": "OTHER"},
        )
    assert excinfo.value.fields["code"] == "extraction_resume_model_identity_mismatch"
    paths = tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        resume=True,
        model_identity=dict(claim),
    )
    assert len(paths) == 3


def test_model_identity_kwarg_refuses_outside_the_vocabulary(tmp_path: Path) -> None:
    """'sampled' is not offered; arbitrary strings refuse typed."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.extract_dataset(
            _Model(seed=0).eval(),
            _stimuli(),
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path,
            progress=False,
            model_identity="sampled",
        )
    assert excinfo.value.fields["code"] == "extraction_model_identity_invalid"


# --- completed-v1 migration through the public entry ---------------------------


def _write_v1_artifact(tmp_path: Path, model: nn.Module, status: str) -> None:
    """Write a v1 artifact whose shards hold this model's true activations.

    Parameters
    ----------
    tmp_path:
        Artifact directory.
    model:
        Model producing the shard payloads.
    status:
        v1 manifest status.
    """

    from torchlens.dataset_extraction import _shard_filename, _stimuli_signature

    reference = tl.extract_dataset(model, _stimuli(), _LAYERS, batch_size=4, progress=False)
    batches = []
    for index, start in enumerate(range(0, 10, 4)):
        payload = {key: value[start : start + 4] for key, value in reference.items()}
        torch.save(payload, tmp_path / _shard_filename(index))
        batches.append(
            {"index": index, "file": _shard_filename(index), "n_stimuli": min(4, 10 - start)}
        )
    manifest = {
        "schema": "tl_extract_manifest_v1",
        "torchlens_version": "2.99",
        "status": status,
        "signature": {
            "layer_plan": dict(_LAYERS),
            "layers_kind": "mapping",
            "batch_size": 4,
            "transform": None,
            "stimuli": _stimuli_signature(_stimuli()),
        },
        "stimulus_provenance": {"order": "iteration order", "n_stimuli": 10},
        "storage": {},
        "layers": {"relu": {}, "logits": {}},
        "batches": batches if status == "complete" else batches[:1],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_completed_v1_resume_migrates_without_a_forward(tmp_path: Path) -> None:
    """resume=True on a completed v1 artifact migrates and returns the paths."""

    model = _Model(seed=0).eval()
    _write_v1_artifact(tmp_path, model, "complete")
    calls_before = model.n_forward_calls
    paths = tl.extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=4, output_dir=tmp_path, progress=False, resume=True
    )
    assert model.n_forward_calls == calls_before, "migration never runs a forward"
    assert len(paths) == 3
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == MANIFEST_SCHEMA_V2
    assert manifest["migration"]["migrated_from"] == "tl_extract_manifest_v1"
    loaded = load_extraction(tmp_path)
    clean = tl.extract_dataset(model, _stimuli(), _LAYERS, batch_size=4, progress=False)
    for key, tensor in clean.items():
        assert torch.equal(loaded.activations[key], tensor)


def test_in_progress_v1_resume_refuses_naming_unprovable_fields(tmp_path: Path) -> None:
    """In-progress v1 recorded no identity/mode facts: v2 resume refuses typed."""

    model = _Model(seed=0).eval()
    _write_v1_artifact(tmp_path, model, "in_progress")
    with pytest.raises(ExtractionArtifactError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path,
            progress=False,
            resume=True,
        )
    assert excinfo.value.fields["code"] == "extraction_resume_v1_in_progress"
    assert "model_identity" in excinfo.value.fields["unprovable_fields"]


# --- the strict opaque-resume rule ---------------------------------------------


def test_opaque_complete_class_continuation_resumes_measured(tmp_path: Path) -> None:
    """Extract D8: a COMPLETE-class digest match resumes without ceremony.

    ``torch.abs`` is an opaque step to the static chain, but the callable
    identity classifier measures it (allowlisted torch namespace), so the
    historical blanket opaque refusal is superseded by a measured resume.
    """

    model = _Model(seed=0).eval()
    tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        transform=torch.abs,
    )
    _interrupt_after(tmp_path, 1)
    paths = tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        resume=True,
        transform=torch.abs,
    )
    assert len(paths) == 3
    loaded = load_extraction(tmp_path)
    assert loaded.manifest["status"] == "complete"
    clean = tl.extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=4, progress=False, transform=torch.abs
    )
    for key, tensor in clean.items():
        assert torch.equal(loaded.activations[key], tensor)


def test_opaque_partial_continuation_refuses_typed(tmp_path: Path) -> None:
    """Decision 14 + D8: a PARTIAL callable's continuation refuses typed.

    A closure over a set is blind territory for the classifier (unordered),
    so the callable classifies partial and resume refuses naming the
    opaque reference and both remedies (register_transform / pipeline_id).
    """

    blind = {"a", "b"}

    def opaque_step(tensor: torch.Tensor) -> torch.Tensor:
        """Scale by the closure set's size (an unmeasurable dependence)."""

        return tensor * float(len(blind))

    model = _Model(seed=0).eval()
    tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        transform=opaque_step,
    )
    _interrupt_after(tmp_path, 1)
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path,
            progress=False,
            resume=True,
            transform=opaque_step,
        )
    assert excinfo.value.fields["code"] == "extraction_resume_opaque_transform"
    assert any("set_unordered" in ref for ref in excinfo.value.fields["opaque_references"])
    message = str(excinfo.value)
    assert "register_transform" in message, "the one-line remedy is named"
    assert "pipeline_id" in message, "the assertion door is named"


def test_partial_continuation_with_recorded_pipeline_id_is_asserted(tmp_path: Path) -> None:
    """D8 strict continuity: the RECORDED pipeline_id attests a partial slot.

    An id invented at resume time refuses; the id set from the FIRST run
    resumes with the override recorded as asserted, not measured.
    """

    blind = {"a", "b"}

    def opaque_step(tensor: torch.Tensor) -> torch.Tensor:
        """Scale by the closure set's size (an unmeasurable dependence)."""

        return tensor * float(len(blind))

    model = _Model(seed=0).eval()
    tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        transform=opaque_step,
        pipeline_id="exp-7",
    )
    _interrupt_after(tmp_path, 1)
    paths = tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        resume=True,
        transform=opaque_step,
        pipeline_id="exp-7",
    )
    assert len(paths) == 3
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    audit = manifest["run"]["resume_audit"]
    assert any(entry["kind"] == "callable_identity_asserted" for entry in audit)


def test_resume_time_invented_pipeline_id_refuses(tmp_path: Path) -> None:
    """T-CALLABLE-IDENTITY: a resume-time-invented pipeline id must refuse."""

    blind = {"a", "b"}

    def opaque_step(tensor: torch.Tensor) -> torch.Tensor:
        """Scale by the closure set's size (an unmeasurable dependence)."""

        return tensor * float(len(blind))

    model = _Model(seed=0).eval()
    tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        transform=opaque_step,
    )
    _interrupt_after(tmp_path, 1)
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path,
            progress=False,
            resume=True,
            transform=opaque_step,
            pipeline_id="invented-later",
        )
    assert excinfo.value.fields["code"] == "extraction_resume_pipeline_id_invalid"


def test_registered_spec_chain_continuation_resumes(tmp_path: Path) -> None:
    """The remedy works: a spec chain is resume-verifiable and continues."""

    model = _Model(seed=0).eval()
    spec_chain = chain(tl.transforms.cast(torch.float16))
    tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        transform=spec_chain,
    )
    _interrupt_after(tmp_path, 1)
    paths = tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        resume=True,
        transform=spec_chain,
    )
    assert len(paths) == 3
    assert load_extraction(tmp_path).activations["relu"].dtype == torch.float16


def test_changed_spec_chain_mismatches_the_signature(tmp_path: Path) -> None:
    """A numerics-visible chain change mismatches transform_pipeline (D16)."""

    model = _Model(seed=0).eval()
    tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        transform=chain(tl.transforms.cast(torch.float16)),
    )
    _interrupt_after(tmp_path, 1)
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path,
            progress=False,
            resume=True,
            transform=chain(tl.transforms.cast(torch.float32)),
        )
    assert excinfo.value.fields["code"] == "extraction_resume_signature_mismatch"
    assert "transform_pipeline" in excinfo.value.fields["mismatched_fields"]


# --- transform door engine seams -------------------------------------------------


def test_c_implemented_transform_torch_abs_dispatches_unary() -> None:
    """The P2 counterexample: transform=torch.abs (C-op) must be called unary.

    ``inspect.signature`` reports ``(*args, **kwargs)`` for C-implemented
    ops, so arity sniffing would pass ctx and TypeError at batch 1.
    """

    model = _Model(seed=0).eval()
    out = tl.extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=4, progress=False, transform=torch.abs
    )
    reference = tl.extract_dataset(model, _stimuli(), _LAYERS, batch_size=4, progress=False)
    assert torch.equal(out["logits"], reference["logits"].abs())


def test_declared_ctx_transform_receives_site_label() -> None:
    """A declared ContextTransform receives the per-site context (T-C1)."""

    seen: list[str | None] = []

    def probe(tensor: torch.Tensor, ctx: TransformContext | None) -> torch.Tensor:
        seen.append(None if ctx is None else ctx.site_label)
        return tensor

    model = _Model(seed=0).eval()
    tl.extract_dataset(
        model, _stimuli(), _LAYERS, batch_size=4, progress=False, transform=with_context(probe)
    )
    assert set(seen) == {"relu", "logits"}


def test_t_c2_row_axis_guard_refuses_launderers() -> None:
    """A per-batch reduction over the stimulus axis refuses typed (T-C2)."""

    model = _Model(seed=0).eval()
    with pytest.raises(TransformContractError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=4,
            progress=False,
            transform=lambda tensor: tensor.mean(dim=0),
        )
    assert excinfo.value.fields["code"] == "transform_row_axis_violated"
    with pytest.raises(TransformContractError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=4,
            progress=False,
            transform=lambda tensor: "not a tensor",
        )
    assert excinfo.value.fields["code"] == "transform_output_invalid"


def test_t_c6_frozen_plan_refuses_lying_transform_before_publication(tmp_path: Path) -> None:
    """A shard contradicting the batch-zero frozen plan refuses BEFORE commit."""

    name = "test_c04_liar"
    if name not in registered_transform_names():
        register_transform(
            TransformDefinition(
                name=name,
                version=1,
                normalize_params=dict,
                plan_fn=lambda spec, input_spec, ctx: PlannedStep(
                    name=spec.name,
                    version=spec.version,
                    # The lie: promises float64 output but applies identity.
                    output=type(input_spec)(shape=input_spec.shape, dtype="torch.float64"),
                    stream_safe=True,
                    may_alias=False,
                    context_capable=False,
                ),
                apply_fn=lambda spec, tensor, ctx: tensor,
            )
        )
    from torchlens.transforms import TransformSpec

    model = _Model(seed=0).eval()
    with pytest.raises(TransformContractError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path,
            progress=False,
            transform=TransformSpec(name=name, version=1),
        )
    assert excinfo.value.fields["code"] == "transform_plan_violated"
    assert not list(tmp_path.glob("batch_*.pt")), "refused BEFORE publication"
    ledger = tmp_path / "ledger.jsonl"
    assert not ledger.exists() or ledger.read_text(encoding="utf-8") == ""


def test_per_site_mapping_with_default_and_unknown_key(tmp_path: Path) -> None:
    """Decision 13: heterogeneous per-site chains, engine-resolved per label."""

    model = _Model(seed=0).eval()
    out = tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        progress=False,
        transform={"relu": "magnitude", DEFAULT: chain(tl.transforms.cast(torch.float16))},
    )
    assert out["relu"].dtype == torch.float32 and torch.all(out["relu"] >= 0)
    assert out["logits"].dtype == torch.float16
    with pytest.raises(TransformContractError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=4,
            progress=False,
            transform={"typo": "magnitude"},
        )
    assert excinfo.value.fields["code"] == "transform_mapping_key_unknown"
    # Disk mode records the per-site records in the signature.
    tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        transform={"relu": "magnitude", DEFAULT: None},
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    per_site: dict[str, Any] = manifest["signature"]["transform_pipeline"]["per_site"]
    assert per_site["relu"]["steps"][0]["name"] == "magnitude"
    assert per_site["logits"] is None


# --- id facts at the engine boundary ---------------------------------------------


def test_stimulus_ids_cardinality_refusals(tmp_path: Path) -> None:
    """Sized stimuli validate upfront; iterables refuse before the short commit."""

    model = _Model(seed=0).eval()
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path / "a",
            progress=False,
            stimulus_ids=["only", "two"],
        )
    assert excinfo.value.fields["code"] == "extraction_stimulus_ids_cardinality"
    assert not (tmp_path / "a" / "manifest.json").exists(), "refused before any write"

    rows = list(_stimuli(6))
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.extract_dataset(
            model,
            rows,
            _LAYERS,
            batch_size=4,
            output_dir=tmp_path / "b",
            progress=False,
            stimulus_ids=[f"s{i}" for i in range(5)],
        )
    assert excinfo.value.fields["code"] == "extraction_stimulus_ids_cardinality"


def test_ledger_rows_carry_id_range_digests(tmp_path: Path) -> None:
    """Each committed shard ledgers the digest of its ordered id slice."""

    from torchlens._data_substrate import read_trusted_rows, stimulus_ids_digest

    model = _Model(seed=0).eval()
    ids = [f"s{i:02d}" for i in range(10)]
    tl.extract_dataset(
        model,
        _stimuli(),
        _LAYERS,
        batch_size=4,
        output_dir=tmp_path,
        progress=False,
        stimulus_ids=ids,
    )
    rows = read_trusted_rows(tmp_path)
    assert rows[0]["ids_range_digest"] == stimulus_ids_digest(ids[0:4])
    assert rows[2]["ids_range_digest"] == stimulus_ids_digest(ids[8:10])
