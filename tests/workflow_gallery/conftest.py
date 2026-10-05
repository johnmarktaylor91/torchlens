"""Workflow-gallery venue gate + pinned-artifact loaders (testing memo D2/E2).

VENUE CONTRACT (testing memo 5.1/5.3): gallery scenarios run ONLY inside the
offline preflighted venue (``HF_HUB_OFFLINE=1`` + ``TRANSFORMERS_OFFLINE=1``,
warmed caches). IN VENUE nothing here may skip and nothing may xfail -- a
missing artifact is a loud load failure, and the exact-passed-ID manifest
(``rg_passed_ids.txt``) is the floor. OUT of venue the whole directory
skips with the gate id below.

GATE-ID: RG_OFFLINE_VENUE
  kind: env gate (offline-venue signature; same shape as R1_OFFLINE_VENUE)
  executing legs: the R1 train gate (megaplan s6: ``pytest -q
  tests/real_model/r1 tests/workflow_gallery``); the nightly full-gallery
  leg (packaging_requests.tsv row, F36). Manifested in
  tests/support/proofnet/gate_manifest.tsv.

Every checkpoint loads THROUGH the artifact registry (pinned ``revision=``,
never a bare model id), via ``checkpoint_evidence`` so an R0 fixture can
never satisfy a gallery obligation.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from support.r1_venue import IN_OFFLINE_VENUE

from tests.real_model.registry import NATURAL_INPUTS_DIR, Registry


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gallery: workflow-gallery acceptance scenario (RG id; zero skip/xfail in venue)",
    )
    # real_model/real_checkpoint are registered under tests/real_model/'s
    # conftest, which is not a parent of this tree; mirror them so gallery
    # collection is warning-clean in any invocation shape.
    config.addinivalue_line(
        "markers",
        "real_checkpoint: R1/R2 bands -- pinned released weights, offline verified cache",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Read the pre-import snapshot: some test modules setdefault the offline
    # flags at import, which would make a plain box look like the venue.
    if IN_OFFLINE_VENUE:
        return
    marker = pytest.mark.skip(
        reason="GATE RG_OFFLINE_VENUE: not in the offline preflighted venue;"
        " export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 over warmed caches"
        " and rerun. In venue this suite can NEVER skip."
    )
    this_dir = os.path.dirname(__file__)
    for item in items:
        # The membership-closure module reads SOURCE, not models: it runs in
        # every venue (a renamed scenario must be red on a cold box too).
        if item.path.name == "test_proofnet_rg_manifest.py":
            continue
        if str(item.path).startswith(this_dir):
            item.add_marker(marker)


@pytest.fixture(scope="session")
def gallery_registry() -> Registry:
    from tests.real_model.registry import load_registry

    return load_registry()


@pytest.fixture(scope="session")
def rg_loader(gallery_registry: Registry) -> Any:
    """Load a pinned checkpoint through its registry row (session cache)."""

    cache: dict[str, Any] = {}

    def _load(artifact_id: str) -> dict[str, Any]:
        if artifact_id in cache:
            return cache[artifact_id]
        row = gallery_registry.checkpoint_evidence(artifact_id)
        if row.kind == "hf_hub":
            from transformers import (
                AutoModel,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )

            if "t5" in row.model_id:
                loader = AutoModelForSeq2SeqLM
            elif "gpt" in row.model_id or "SmolLM" in row.model_id:
                loader = AutoModelForCausalLM
            else:
                loader = AutoModel
            model = loader.from_pretrained(row.model_id, revision=row.revision).eval()
            tokenizer = AutoTokenizer.from_pretrained(row.model_id, revision=row.revision)
            result = {"row": row, "model": model, "tokenizer": tokenizer}
        elif row.kind == "torchvision_weights":
            import torchvision.models as tv_models

            enum_path = row.model_id.split(":", 1)[1]
            enum_name, member = enum_path.split(".", 1)
            family = enum_name.removesuffix("_Weights").lower()
            if "fasterrcnn" in family:
                import torchvision.models.detection as tv_detection

                weights = getattr(
                    tv_detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, member
                )
                model = tv_detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).eval()
            else:
                weights = getattr(tv_models.get_model_weights(family), member)
                model = tv_models.get_model(family, weights=weights).eval()
            result = {"row": row, "model": model, "weights": weights}
        else:  # pragma: no cover - registry drift guard
            raise ValueError(f"{artifact_id}: kind {row.kind} has no gallery loader")
        cache[artifact_id] = result
        return result

    return _load


@pytest.fixture(scope="session")
def natural_image_batch() -> Any:
    """The committed license-documented natural image, model-ready."""

    import torch
    from torchvision.io import read_image
    from torchvision.transforms.functional import center_crop, resize

    image = read_image(str(NATURAL_INPUTS_DIR / "pd_astronaut_256.jpg"))
    batch = center_crop(resize(image, 256, antialias=True), 224).unsqueeze(0).float() / 255.0
    return batch.clone().requires_grad_(False), torch.flip(batch, dims=[3]).contiguous()


def prompt(prompt_id: str) -> dict[str, Any]:
    """Read one committed natural prompt by id (data file, never a literal)."""

    for line in (NATURAL_INPUTS_DIR / "prompts.jsonl").read_text().splitlines():
        if line.strip() and json.loads(line)["id"] == prompt_id:
            return json.loads(line)
    raise AssertionError(f"prompt {prompt_id!r} not in the committed corpus")
