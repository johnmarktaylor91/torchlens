"""Regression tests for bucket-E metadata and anti-slop proposal depth."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.metadata import (
    MANDATORY_EXTERNAL_FIELDS,
    MetadataValidationError,
    TORCHLENS_DERIVABLE_FIELDS,
    validate_external_metadata_for_write,
)
from menagerie.crawler.proposal import ProposalValidationError, validate_author_proposal
from menagerie.crawler.tests.conftest import make_author_proposal, make_model

NEWLY_GATED_EXTERNAL_FIELDS = (
    "field",
    "subfield",
    "predecessors",
    "family",
    "era",
    "original_framework",
    "run_framework",
    "modes",
)


def _accurate_gate_item() -> dict[str, Any]:
    """Return exhaustive accurate leaf checks for canonical metadata write.

    Returns
    -------
    dict[str, Any]
        Independently accurate metadata gate item.
    """

    return {
        "verdict": "accurate",
        "integrity": {"verdict": "accurate"},
        "field_checks": [
            {"field": f"external_metadata.{field}", "verdict": "accurate"}
            for field in MANDATORY_EXTERNAL_FIELDS
        ],
    }


def _accepted_metadata() -> dict[str, Any]:
    """Return complete accepted external metadata.

    Returns
    -------
    dict[str, Any]
        Schema-valid authored metadata fixture.
    """

    return deepcopy(make_model(accepted=True)["external_metadata"])


@pytest.mark.parametrize("field", NEWLY_GATED_EXTERNAL_FIELDS)
def test_new_external_field_missing_blocks_canonical_write(field: str) -> None:
    """Every newly covered authored field is mandatory on the write path.

    Parameters
    ----------
    field:
        Required external metadata field removed from the proposal.
    """

    metadata = _accepted_metadata()
    del metadata[field]
    with pytest.raises(MetadataValidationError, match="missing mandatory"):
        validate_external_metadata_for_write(metadata, _accurate_gate_item())


@pytest.mark.parametrize("field", NEWLY_GATED_EXTERNAL_FIELDS)
def test_new_external_field_ungated_blocks_canonical_write(field: str) -> None:
    """Every newly covered authored field needs its own accurate leaf check.

    Parameters
    ----------
    field:
        Required external metadata field whose independent check is removed.
    """

    gate_item = _accurate_gate_item()
    gate_item["field_checks"] = [
        check
        for check in gate_item["field_checks"]
        if check["field"] != f"external_metadata.{field}"
    ]
    with pytest.raises(MetadataValidationError, match="ungated mandatory"):
        validate_external_metadata_for_write(_accepted_metadata(), gate_item)


def test_duplicate_external_leaf_check_blocks_canonical_write() -> None:
    """A duplicated leaf cannot hide behind an otherwise exhaustive accurate gate."""

    gate_item = _accurate_gate_item()
    gate_item["field_checks"].append(deepcopy(gate_item["field_checks"][0]))
    with pytest.raises(MetadataValidationError, match="duplicate metadata field check"):
        validate_external_metadata_for_write(_accepted_metadata(), gate_item)


def test_torchlens_derivable_fields_remain_optional_at_write() -> None:
    """Missing TorchLens structural observations do not block authored metadata."""

    metadata = _accepted_metadata()
    assert TORCHLENS_DERIVABLE_FIELDS.isdisjoint(metadata)
    report = validate_external_metadata_for_write(metadata, _accurate_gate_item())
    assert report.derivable_fields_present == frozenset()


def _proposal_source(
    tmp_path: Path, text: str, *, extra_supports: tuple[str, ...] = ()
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an exact-source R1 proposal for focused proposal-gate tests.

    Parameters
    ----------
    tmp_path:
        Isolated source and staged-code directory.
    text:
        Exact controlled source bytes and verbatim excerpt.
    extra_supports:
        Additional self-declared labels attached to the excerpt.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Schema-valid proposal and exact controlled-fetch manifest.
    """

    proposal = make_author_proposal()
    source_path = tmp_path / "source.txt"
    source_path.write_text(text)
    source_hash = hash_bytes(text.encode())
    excerpt = proposal["proposed_facts"]["evidence"]["excerpts"][0]
    excerpt.update(
        {
            "locator": f"bytes:0-{len(text.encode())}",
            "text": text,
            "text_sha256": source_hash,
            "supports": [
                "external_metadata.citation",
                "external_metadata.family",
                *extra_supports,
            ],
            "family_level": True,
        }
    )
    proposal["proposed_facts"]["evidence"]["coverage"].update(
        {
            "all_agent_fields_have_support": True,
            "missing_support": [],
            "family_grounding_complete": True,
        }
    )
    manifest = {
        "sources": [
            {
                "source_id": "source-1",
                "url": "https://example.com/paper",
                "revision": "v1",
                "content_sha256": source_hash,
                "cas_path": str(source_path),
                "retrieval_status": "fetched",
            }
        ]
    }
    return proposal, manifest


def _validate_focused(proposal: dict[str, Any], manifest: dict[str, Any], tmp_path: Path) -> None:
    """Validate a proposal while focusing mandatory support on its family claim.

    Parameters
    ----------
    proposal:
        Complete author proposal.
    manifest:
        Exact controlled-fetch source manifest.
    tmp_path:
        Allowed staged-code directory.
    """

    validate_author_proposal(
        proposal,
        allowed_model_dir=tmp_path,
        source_manifest=manifest,
        required_claims={"external_metadata.family"},
    )


def test_self_declared_support_without_value_binding_is_refused(tmp_path: Path) -> None:
    """An unrelated verbatim excerpt cannot support a family merely by its label."""

    proposal, manifest = _proposal_source(
        tmp_path, "Example Model was published at TestConf in 2020 about weather forecasting."
    )
    with pytest.raises(ProposalValidationError, match="substantively support"):
        _validate_focused(proposal, manifest, tmp_path)


def test_excerpt_genuinely_supporting_proposed_value_passes(tmp_path: Path) -> None:
    """Literal text naming the proposed family satisfies deterministic binding."""

    proposal, manifest = _proposal_source(
        tmp_path,
        "Example Model introduced the ExampleNet family at TestConf in 2020.",
    )
    _validate_focused(proposal, manifest, tmp_path)


def _stage_typed_adapter(proposal: dict[str, Any], tmp_path: Path, code: str, *, rung: str) -> None:
    """Convert an R1 fixture into a schema-valid typed staged-code rung.

    Parameters
    ----------
    proposal:
        Proposal modified in place.
    tmp_path:
        Adapter staging directory.
    code:
        Typed adapter source.
    rung:
        Selected canonical source rung.
    """

    code_path = tmp_path / "adapter.py"
    code_path.write_text(code)
    facts = proposal["proposed_facts"]
    resolution = facts["source_resolution"]
    resolution["rung"] = rung
    selected_index = ("R1_LIBRARY", "R2_VENDOR", "R3_PORT", "R4_REIMPLEMENT").index(rung)
    resolution["attempted_rungs"] = [
        {
            "rung": attempted,
            "result": "selected" if attempted == rung else "unavailable",
            "reason_code": "documented-search",
            "evidence_ids": ["evidence-1"],
        }
        for attempted in ("R1_LIBRARY", "R2_VENDOR", "R3_PORT", "R4_REIMPLEMENT")[
            : selected_index + 1
        ]
    ]
    implementation = facts["implementation"]
    implementation.update(
        {
            "recipe_type": "typed-adapter" if rung == "R2_VENDOR" else "reimplementation",
            "code_path": "adapter.py",
            "code_sha256": hash_bytes(code.encode()),
            "builder_symbol": "build_model",
            "dummy_call_symbol": "make_dummy_call",
            "library_recipe": None,
        }
    )
    proposal["verified_hashes"]["code"] = hash_bytes(code.encode())


def _source_map(source_locator: str = "bytes:0-82") -> list[dict[str, Any]]:
    """Return one exact source-to-adapter mapping.

    Parameters
    ----------
    source_locator:
        Exact verified excerpt locator in the controlled source bytes.

    Returns
    -------
    list[dict[str, Any]]
        Complete source-map row.
    """

    return [
        {
            "material_item": "upstream forward implementation",
            "source_id": "source-1",
            "source_locator": source_locator,
            "evidence_ids": ["evidence-1"],
            "code_path": "adapter.py",
            "code_locator": "lines 1-5",
            "disposition": "wrapped-exactly",
        }
    ]


def test_generic_sequential_mlp_as_exotic_family_trips_tripwire(tmp_path: Path) -> None:
    """A plain Sequential MLP cannot masquerade as the named ExampleNet family."""

    proposal, manifest = _proposal_source(
        tmp_path,
        "Example Model introduced the complete ExampleNet architecture at TestConf in 2020.",
        extra_supports=("implementation.architecture",),
    )
    code = (
        "import torch.nn as nn\n\n"
        "class ExampleNet(nn.Module):\n"
        "    def __init__(self) -> None:\n"
        "        super().__init__()\n"
        "        self.layers = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))\n\n"
        "    def forward(self, value: object) -> object:\n"
        "        return self.layers(value)\n\n"
        "def build_model() -> object:\n"
        "    return ExampleNet()\n\n"
        "def make_dummy_call(seed: int, device: str) -> tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )
    _stage_typed_adapter(proposal, tmp_path, code, rung="R4_REIMPLEMENT")
    facts = proposal["proposed_facts"]
    facts["source_resolution"]["sources"][0].update({"role": "introducing-paper", "kind": "paper"})
    facts["implementation"]["source_to_code_map"] = _source_map()
    facts["fidelity"].update(
        {"required": True, "reason": "source-faithful architecture", "current": False}
    )
    with pytest.raises(ProposalValidationError, match="structural slop tripwire"):
        _validate_focused(proposal, manifest, tmp_path)


def _r2_proposal(tmp_path: Path, *, bound: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an R2 proposal with optional exact byte/map binding.

    Parameters
    ----------
    tmp_path:
        Isolated source and adapter directory.
    bound:
        Whether exact upstream bytes and source-map rows are supplied.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        R2 proposal and exact source manifest.
    """

    proposal, manifest = _proposal_source(
        tmp_path,
        "Example Model introduced the official upstream ExampleNet architecture in 2020.",
        extra_supports=("implementation.architecture",),
    )
    code = (
        "def build_model() -> object:\n"
        "    return object()\n\n"
        "def make_dummy_call(seed: int, device: str) -> tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )
    _stage_typed_adapter(proposal, tmp_path, code, rung="R2_VENDOR")
    implementation = proposal["proposed_facts"]["implementation"]
    if bound:
        proposal["proposed_facts"]["source_resolution"]["sources"][0]["content_sha256"] = manifest[
            "sources"
        ][0]["content_sha256"]
        implementation["upstream_files"] = [
            {
                "source_id": "source-1",
                "path": "upstream/model.py",
                "sha256": manifest["sources"][0]["content_sha256"],
                "use": "exact upstream model source",
            }
        ]
        excerpt = proposal["proposed_facts"]["evidence"]["excerpts"][0]
        implementation["source_to_code_map"] = _source_map(excerpt["locator"])
    return proposal, manifest


def test_r2_without_exact_source_binding_is_refused(tmp_path: Path) -> None:
    """A named vendor adapter alone cannot earn the R2 rung."""

    proposal, manifest = _r2_proposal(tmp_path, bound=False)
    with pytest.raises(ProposalValidationError, match="exact mirrored upstream files"):
        _validate_focused(proposal, manifest, tmp_path)


def test_r2_with_exact_source_bytes_and_map_passes(tmp_path: Path) -> None:
    """R2 passes when adapter code maps to hash-matched controlled source bytes."""

    proposal, manifest = _r2_proposal(tmp_path, bound=True)
    _validate_focused(proposal, manifest, tmp_path)


@pytest.mark.parametrize(
    "phrase",
    [
        "an approximate implementation",
        "a toy model substitute",
        "a knowingly simplified version",
        "a placeholder for the real architecture",
        "a stand-in for the source model",
    ],
)
def test_broadened_approximation_language_is_refused(tmp_path: Path, phrase: str) -> None:
    """Expanded explicit approximation vocabulary blocks acceptance.

    Parameters
    ----------
    tmp_path:
        Isolated source directory.
    phrase:
        Explicit approximation admission inserted into authored prose.
    """

    proposal, manifest = _proposal_source(
        tmp_path, "Example Model introduced the ExampleNet family at TestConf in 2020."
    )
    proposal["proposed_facts"]["external_metadata"]["description"] = f"ExampleNet is {phrase}."
    with pytest.raises(ProposalValidationError, match="forbidden approximation language"):
        _validate_focused(proposal, manifest, tmp_path)


def test_divergent_top_level_citation_is_refused(tmp_path: Path) -> None:
    """An unvetted public citation cannot differ from checked external metadata."""

    proposal, manifest = _proposal_source(
        tmp_path, "Example Model introduced the ExampleNet family at TestConf in 2020."
    )
    proposal["proposed_facts"]["citation"]["title"] = "Different Unvetted Work"
    with pytest.raises(ProposalValidationError, match="top-level citation differs"):
        _validate_focused(proposal, manifest, tmp_path)
