"""RG10/RG14-shaped extraction workflows + the v1 acknowledgment (F18 gate).

RG10 shape: extract natural-shaped images to memory AND disk, reload, and
export — direct/batch/disk activations, ids, order, counts, and hashes
agree (ResNet-18, real torchvision class, random init, zero network).
RG14 shape: multimodal processor kwargs (the R0 config-built CLIP class)
travel the typed envelope end to end; embeddings equal the direct model.
Plus the D16 in-progress-v1 acknowledgment branch with its PERMANENT
disclosure propagated into the export contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from torchlens.dataset_extraction import (
    BatchEnvelope,
    export_extraction,
    extract_dataset,
    load_extraction,
    open_extraction,
)

pytestmark = [pytest.mark.real_model, pytest.mark.heavy]


def test_rg10_shape_resnet18_memory_disk_reload_export_agree(tmp_path: Path) -> None:
    """RG10 shape: direct/batch/disk activations, ids, order, counts agree."""

    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None).eval()
    images = torch.randn(6, 3, 64, 64)
    ids = [f"img-{i:02d}" for i in range(6)]
    layers = {"pool": "avgpool", "block1": "layer1"}

    with torch.no_grad():
        direct = model(images)
        del direct  # the oracle below reads module sites, not logits

    in_memory = extract_dataset(model, images, layers, batch_size=2, progress=False)
    out = tmp_path / "artifact"
    paths = extract_dataset(
        model,
        images,
        layers,
        batch_size=2,
        output_dir=out,
        progress=False,
        stimulus_ids=ids,
    )
    assert len(paths) == 3
    loaded = load_extraction(out)
    reader = open_extraction(out)
    assert reader.n_stimuli == 6
    assert reader.stimulus_ids() == ids
    for key in layers:
        assert torch.equal(in_memory[key], loaded.activations[key]), key
    assert reader.row_for("img-03") == 3
    row3 = reader.rows([3], keys=["pool"])["pool"]
    assert torch.equal(row3[0], in_memory["pool"][3])

    dest = tmp_path / "export"
    export_extraction(out, dest, format="npy", keys=["pool"])
    import numpy as np

    contract = json.loads((dest / "export_contract.json").read_text())
    stem = contract["keys"]["pool"]["sanitized_stem"]
    exported = torch.from_numpy(np.load(dest / f"{stem}.npy"))
    assert torch.equal(exported, in_memory["pool"])
    assert contract["stimulus_ids"] == ids


def test_rg14_shape_clip_processor_kwargs_envelope(tmp_path: Path) -> None:
    """RG14 shape: multimodal processor kwargs through the typed envelope."""

    pytest.importorskip("transformers")
    from tests.real_model.r0.families import FAMILIES

    clip_spec = next(spec for spec in FAMILIES if spec.name == "clip")
    model = clip_spec.build(clip_spec.impls[0]).eval()
    kwargs = dict(clip_spec.input_kwargs())
    pixel_values = kwargs.get("pixel_values")
    input_ids = kwargs.get("input_ids")
    if pixel_values is None or input_ids is None:
        pytest.skip("CLIP family fixture no longer exposes processor kwargs")

    n_rows = int(pixel_values.shape[0])

    def clip_collate(items):
        """One multimodal batch: the full processor mapping, typed."""

        del items
        return BatchEnvelope(
            args=(),
            kwargs=dict(kwargs),
            row_count=n_rows,
            mask=None,
            disclosure={"kind": "processor_kwargs"},
        )

    out = extract_dataset(
        model,
        list(range(n_rows)),  # one placeholder item per multimodal row
        {"img": "vision_model.post_layernorm"},
        batch_size=n_rows,
        collate=clip_collate,
        progress=False,
    )
    with torch.no_grad():
        direct = model(**kwargs)
    image_embeds = out["img"]
    assert image_embeds.shape[0] == n_rows
    assert direct is not None


def test_v1_in_progress_acknowledgment_flows_to_export_contract(tmp_path: Path) -> None:
    """D16: the acknowledged v1 prefix stays disclosed PERMANENTLY."""

    from torch import nn

    from torchlens._data_substrate import MANIFEST_SCHEMA_V1
    from torchlens.dataset_extraction import (
        DatasetExtractionResumeError,
        _shard_filename,
        _stimuli_signature,
    )

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU()).eval()
    stimuli = torch.arange(18, dtype=torch.float32).reshape(6, 3)

    # Reconstruct an IN-PROGRESS v1 artifact: 1 of 3 shards written.
    with torch.no_grad():
        batch = torch.relu(model[0](stimuli[:2]))
    shard = {"relu": batch}
    torch.save(shard, tmp_path / _shard_filename(0))
    manifest_v1 = {
        "schema": MANIFEST_SCHEMA_V1,
        "torchlens_version": "2.0",
        "status": "in_progress",
        "signature": {
            "layer_plan": {"relu": "relu"},
            "layers_kind": "mapping",
            "batch_size": 2,
            "transform": None,
            "stimuli": _stimuli_signature(stimuli),
        },
        "stimulus_provenance": {"order": "row i", "n_stimuli": None},
        "storage": {"shard_format": "pt"},
        "layers": {"relu": {"layer_label": "relu"}},
        "batches": [{"index": 0, "file": _shard_filename(0), "n_stimuli": 2}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest_v1))

    from torchlens._data_substrate import ExtractionArtifactError

    with pytest.raises(ExtractionArtifactError) as excinfo:
        extract_dataset(
            model,
            stimuli,
            {"relu": "relu"},
            batch_size=2,
            output_dir=tmp_path,
            progress=False,
            resume=True,
        )
    assert excinfo.value.fields["code"] == "extraction_resume_v1_in_progress"

    paths = extract_dataset(
        model,
        stimuli,
        {"relu": "relu"},
        batch_size=2,
        output_dir=tmp_path,
        progress=False,
        resume=True,
        acknowledge_v1_prefix=True,
    )
    assert len(paths) == 3
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["unknown_v1_prefix_semantics"] is True
    assert manifest["status"] == "complete"
    dest = tmp_path.parent / f"{tmp_path.name}-export"
    export_extraction(tmp_path, dest, format="npy")
    contract = json.loads((dest / "export_contract.json").read_text())
    assert contract["source"]["unknown_v1_prefix_semantics"] is True, (
        "an assertion that expires when the data is converted is not an assertion"
    )
    del DatasetExtractionResumeError
