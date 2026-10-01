"""R1 venue gate + session-scoped real-checkpoint fixtures (memo A1/A2).

VENUE CONTRACT (memo 4.3): R1 tests run ONLY behind a green preflight
(``scripts/preflight_fetch_artifacts.py``) under the offline env it prints.
IN VENUE (offline env set) nothing here can skip -- a missing artifact fails
loudly at load. OUT of venue (a developer box or the P gate without the
cache) the whole directory skips with the gate id below.

GATE-ID: R1_OFFLINE_VENUE
  kind: env gate (offline-venue signature)
  executing legs: latest-canary R1-core job (A6); the R1 train gate
  (megaplan s6); the PR telemetry leg (B1, lands with the fix bundle).
  This gate enters the C1 gate-witness manifest when that lands.

Every model here is loaded THROUGH the registry (checkpoint_evidence -- the
never-count hook -- plus pinned revision=), never by a bare model id.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from tests.real_model.registry import Registry


def _in_offline_venue() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _in_offline_venue():
        return
    marker = pytest.mark.skip(
        reason="GATE R1_OFFLINE_VENUE: not in the offline preflighted venue; run"
        " scripts/preflight_fetch_artifacts.py fetch, export its print-env, then"
        " rerun. In venue this suite can NEVER skip (missing artifact = loud"
        " preflight/load failure)."
    )
    this_dir = os.path.dirname(__file__)
    for item in items:
        if str(item.path).startswith(this_dir):
            item.add_marker(marker)


@pytest.fixture(scope="session")
def r1_loader(artifact_registry: Registry):
    """Factory: load a pinned R1 checkpoint through its registry row.

    Session-scoped cache: each checkpoint loads once per session. The row is
    served through ``checkpoint_evidence`` so an R0 row can never launder in.
    """

    cache: dict[str, Any] = {}

    def _load(artifact_id: str) -> Any:
        if artifact_id in cache:
            return cache[artifact_id]
        row = artifact_registry.checkpoint_evidence(artifact_id)
        if row.kind == "hf_hub":
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

            loader = (
                AutoModelForCausalLM
                if any("lm" in g.lower() or g in ("RG04", "RG05") for g in row.gallery_ids)
                or "gpt" in row.model_id
                else AutoModel
            )
            model = loader.from_pretrained(row.model_id, revision=row.revision).eval()
            tokenizer = AutoTokenizer.from_pretrained(row.model_id, revision=row.revision)
            result = {"row": row, "model": model, "tokenizer": tokenizer}
        elif row.kind == "torchvision_weights":
            import torchvision.models as tv_models

            enum_path = row.model_id.split(":", 1)[1]  # e.g. ResNet18_Weights.IMAGENET1K_V1
            enum_name, member = enum_path.split(".", 1)
            weights = getattr(
                tv_models.get_model_weights(enum_name.removesuffix("_Weights").lower()), member
            )
            model = tv_models.get_model(
                enum_name.removesuffix("_Weights").lower(), weights=weights
            ).eval()
            result = {"row": row, "model": model, "weights": weights}
        else:
            raise ValueError(f"{artifact_id}: kind {row.kind} has no loader")
        cache[artifact_id] = result
        return result

    return _load
