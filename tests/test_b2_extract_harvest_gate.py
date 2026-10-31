"""The counterfactual-harvest execution gate (R2-08, lane EXTRACT).

``harvest_schema.EXECUTABLE_ADDRESS_CLASSES`` admits a class only beside its
measured gate evidence: the receipt lane, the save path, and the real
64-image resnet18 workload green. This module is that evidence for the two
structural classes (``module``, ``type``): a torchvision resnet18 over 64
images, extracted clean and intervened to disk in four shards, with every
shard's fire receipt attested, the counterfactual confined to the
intervention's forward cone, and an interrupted intervened run resumed to
the same bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import torchlens as tl
from torchlens import extract_dataset
from torchlens._data_substrate import read_trusted_rows
from torchlens._extraction.harvest_schema import (
    EXECUTABLE_ADDRESS_CLASSES,
    HARVEST_BLOCK_KEY,
    validate_harvest_block,
)
from torchlens._extraction.resume import MANIFEST_FILENAME
from torchlens.dataset_extraction import open_extraction

pytestmark = pytest.mark.slow

_LAYERS = {"stem": "maxpool", "block1": "layer1", "block2": "layer2", "pool": "avgpool"}


@pytest.fixture(scope="module")
def resnet18_and_images():
    """A seeded torchvision resnet18 and 64 seeded 224x224 images."""

    models = pytest.importorskip("torchvision.models")
    torch.manual_seed(0)
    model = models.resnet18(weights=None).eval()
    images = torch.randn(64, 3, 224, 224, generator=torch.Generator().manual_seed(1))
    return model, images


def _interrupt_after(path: Path, n_shards: int) -> None:
    """Reconstruct the on-disk state of a kill after ``n_shards`` commits."""

    manifest_path = path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(status="in_progress", totals=None, ledger_digest=None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ledger = path / "ledger.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    ledger.write_text("".join(line + "\n" for line in lines[:n_shards]), encoding="utf-8")
    for line in lines[n_shards:]:
        (path / json.loads(line)["file"]).unlink()


@pytest.mark.parametrize(
    ("address_class", "spec_factory", "zeroed"),
    [
        ("module", lambda: tl.when(tl.module("layer1"), tl.zero_ablate()), "block1"),
        ("type", lambda: tl.when(tl.func("adaptive_avg_pool2d"), tl.zero_ablate()), "pool"),
    ],
)
def test_resnet18_64_image_harvest_gate(
    resnet18_and_images, tmp_path: Path, address_class: str, spec_factory, zeroed: str
) -> None:
    """Receipts, save path, cone confinement, and resume on the real workload."""

    assert address_class in EXECUTABLE_ADDRESS_CLASSES
    model, images = resnet18_and_images
    clean_dir, edited_dir = tmp_path / "clean", tmp_path / "edited"
    kwargs = {"batch_size": 16, "progress": False}
    extract_dataset(model, images, _LAYERS, path=clean_dir, **kwargs)
    extract_dataset(model, images, _LAYERS, path=edited_dir, intervene=spec_factory(), **kwargs)

    manifest = json.loads((edited_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    harvest = validate_harvest_block(manifest[HARVEST_BLOCK_KEY])
    assert (harvest.policy, harvest.address_class) == ("intervened", address_class)
    rows = read_trusted_rows(edited_dir)
    assert [row["n_rows"] for row in rows] == [16, 16, 16, 16]
    assert all(row["intervention"]["fire_count"] >= 1 for row in rows)
    assert all(row["intervention"]["site_keys"] == list(harvest.target_site_keys) for row in rows)

    clean = open_extraction(clean_dir).materialize()
    edited = open_extraction(edited_dir).materialize()
    order = list(_LAYERS)
    cut = order.index(zeroed)
    for key in order[:cut]:
        assert torch.equal(edited[key], clean[key]), f"{key} is upstream of the edit"
    assert torch.count_nonzero(edited[zeroed]) == 0
    for key in order[cut:]:
        assert not torch.equal(edited[key], clean[key]), f"{key} is in the edit's cone"

    _interrupt_after(edited_dir, 2)
    extract_dataset(
        model,
        images,
        _LAYERS,
        path=edited_dir,
        intervene=spec_factory(),
        resume=True,
        **kwargs,
    )
    resumed = open_extraction(edited_dir).materialize()
    assert all(torch.equal(resumed[key], edited[key]) for key in order)
    final = json.loads((edited_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert final[HARVEST_BLOCK_KEY] == manifest[HARVEST_BLOCK_KEY]
