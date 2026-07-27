"""Anti-slop proposal and one-model author-dispatch tests for Slice D."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorDispatchError,
    BlockedRecommendation,
    DeferRecommendation,
    ProposedAuthorResult,
    SkipRecommendation,
    build_author_envelope,
    serialize_author_result_cache,
    validate_author_result,
    validate_author_result_cache,
)
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.constants import (
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    AUTHOR_RESULT_SCHEMA_VERSION,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.proposal import (
    DEFAULT_GATED_CLAIMS,
    ProposalValidationError,
    model_code_manifest,
    validate_author_proposal,
)
from menagerie.crawler.tests.conftest import bind_handoff_execution, make_author_proposal

import shutil
import sys
import zipfile
from menagerie.crawler.driver import (
    EnvironmentBinding,
    _attempt_policy_satisfied,
    _attempts_from_supervised,
)
from menagerie.crawler.identity import compute_recipe_revision
from menagerie.crawler.policy import detect_os_sandbox
from menagerie.crawler.tests.conftest import HASH, make_proposed_artifact
from menagerie.crawler.worker_supervisor import supervise_worker


def _ground_proposal(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a schema-valid R1 proposal with exact fetched evidence.

    Parameters
    ----------
    tmp_path:
        Isolated model/CAS directory.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Proposal and controlled source manifest.
    """

    proposal = make_author_proposal()
    text = (
        "Example Model introduced ExampleNet in TestConf 2020 by A. Author at Example Lab in "
        "the US. ExampleNet is an official PyTorch library CNN architecture for supervised computer vision "
        "classification in machine learning. This modern ExampleNet family uses vision modality "
        "and has the example and cnn keywords. It is a small source-grounded example network "
        "whose grounded contribution uses the Apache-2.0 license. It runs in PyTorch eval mode "
        "with no train eval divergence. The input contract is one small RGB image and the output "
        "is class scores."
    )
    source_path = tmp_path / "source.txt"
    source_path.write_text(text)
    source_hash = hash_bytes(text.encode())
    claims = [*sorted(DEFAULT_GATED_CLAIMS), "implementation.architecture"]
    excerpt = proposal["proposed_facts"]["evidence"]["excerpts"][0]
    excerpt.update(
        {
            "locator": f"bytes:0-{len(text.encode())}",
            "text": text,
            "text_sha256": hash_bytes(text.encode()),
            "supports": claims,
            "family_level": True,
        }
    )
    coverage = proposal["proposed_facts"]["evidence"]["coverage"]
    coverage.update(
        {
            "all_agent_fields_have_support": True,
            "missing_support": [],
            "family_grounding_complete": True,
        }
    )
    manifest: dict[str, Any] = {
        "sources": [
            {
                "source_id": "source-1",
                "url": "https://example.com/model",
                "revision": "v1",
                "content_sha256": source_hash,
                "cas_path": str(source_path),
                "retrieval_status": "fetched",
            }
        ]
    }
    manifest["manifest_sha256"] = stable_hash(manifest["sources"])
    proposal["verified_hashes"]["source_manifest"] = manifest["manifest_sha256"]
    return proposal, manifest


def _make_r4(
    proposal: dict[str, Any], manifest: dict[str, Any], model_dir: Path, code: str
) -> None:
    """Convert a grounded R1 fixture into a typed R4 proposal in place.

    Parameters
    ----------
    proposal:
        Proposal fixture.
    manifest:
        Source manifest fixture.
    model_dir:
        Allowed staged-code directory.
    code:
        Typed adapter source.
    """

    code_path = model_dir / "adapter.py"
    code_path.write_text(code)
    facts = proposal["proposed_facts"]
    resolution = facts["source_resolution"]
    resolution["rung"] = "R4_REIMPLEMENT"
    resolution["attempted_rungs"] = [
        {
            "rung": rung,
            "result": "unavailable" if rung != "R4_REIMPLEMENT" else "selected",
            "reason_code": "documented-search",
            "evidence_ids": ["evidence-1"],
        }
        for rung in ("R1_LIBRARY", "R2_VENDOR", "R3_PORT", "R4_REIMPLEMENT")
    ]
    resolution["sources"][0]["role"] = "introducing-paper"
    resolution["sources"][0]["kind"] = "paper"
    implementation = facts["implementation"]
    implementation.update(
        {
            "recipe_type": "reimplementation",
            "code_path": "adapter.py",
            "code_sha256": hash_bytes(code.encode()),
            "builder_symbol": "build_model",
            "dummy_call_symbol": "make_dummy_call",
            "library_recipe": None,
            "source_to_code_map": [
                {
                    "material_item": "complete forward architecture",
                    "source_id": "source-1",
                    "source_locator": "bytes:0-73",
                    "evidence_ids": ["evidence-1"],
                    "code_path": "adapter.py",
                    "code_locator": "lines 1-5",
                    "disposition": "transcribed",
                }
            ],
        }
    )
    code_manifest = [dict(row) for row in model_code_manifest(code_path, model_dir)]
    implementation["code_manifest"] = code_manifest
    facts["fidelity"].update({"required": True, "reason": "R4 reimplementation", "current": False})
    manifest["sources"][0].pop("role", None)
    proposal["verified_hashes"]["code"] = hash_bytes(code.encode())
    proposal["verified_hashes"]["code_manifest"] = stable_hash(code_manifest)


def test_valid_typed_r1_proposal_passes(tmp_path: Path) -> None:
    """A complete grounded declarative library proposal is accepted."""

    proposal, manifest = _ground_proposal(tmp_path)
    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )
    assert report.rung.value == "R1_LIBRARY"


def test_valid_r4_with_cited_descriptive_text_passes(tmp_path: Path) -> None:
    """A typed R4 with no implementation source and a literal source map passes."""

    proposal, manifest = _ground_proposal(tmp_path)
    code = (
        "def build_model() -> object:\n"
        "    return object()\n\n"
        "def make_dummy_call(seed: int, device: str) -> tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )
    _make_r4(proposal, manifest, tmp_path, code)
    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )
    assert report.rung.value == "R4_REIMPLEMENT"


def test_r4_checked_candidate_withheld_from_fetch_is_rejected(tmp_path: Path) -> None:
    """A checked repository omitted from the CAS is a detectable coverage gap."""

    proposal, manifest = _ground_proposal(tmp_path)
    code = (
        "def build_model() -> object:\n"
        "    return object()\n\n"
        "def make_dummy_call(seed: int, device: str) -> tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )
    _make_r4(proposal, manifest, tmp_path, code)
    proposal["proposed_facts"]["source_resolution"]["search_report"]["links_checked"].append(
        "https://code.example.org/example-net"
    )

    with pytest.raises(ProposalValidationError, match="checked-link coverage gap"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )


def test_recursive_helper_structural_slop_is_rejected(tmp_path: Path) -> None:
    """A generic stand-in hidden in an imported helper remains statically visible."""

    proposal, manifest = _ground_proposal(tmp_path)
    code = (
        "from helper import build_architecture\n\n"
        "def build_model() -> object:\n"
        "    return build_architecture()\n\n"
        "def make_dummy_call(seed: int, device: str) -> "
        "tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )
    (tmp_path / "helper.py").write_text(
        "import torch.nn as nn\n\n"
        "def build_architecture() -> object:\n"
        "    return nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 2))\n",
        encoding="utf-8",
    )
    _make_r4(proposal, manifest, tmp_path, code)
    with pytest.raises(ProposalValidationError, match="structural slop"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )


@pytest.mark.parametrize("forbidden", ["eval", "exec", "compile"])
def test_dynamic_execution_code_is_rejected(tmp_path: Path, forbidden: str) -> None:
    """Every dynamic execution primitive is rejected.

    Parameters
    ----------
    forbidden:
        Forbidden builtin called by staged code.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    code = (
        "def build_model() -> object:\n"
        f"    return {forbidden}('1 + 1')\n\n"
        "def make_dummy_call(seed: int, device: str) -> tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )
    _make_r4(proposal, manifest, tmp_path, code)
    with pytest.raises(ProposalValidationError, match="forbidden dynamic execution"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_out_of_path_code_and_write_are_rejected(tmp_path: Path) -> None:
    """Both staged-code path escape and a literal outside write fail."""

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    proposal, manifest = _ground_proposal(tmp_path)
    code = (
        "def build_model() -> object:\n"
        "    with open('/tmp/forbidden', 'w') as handle:\n"
        "        handle.write('x')\n"
        "    return object()\n\n"
        "def make_dummy_call(seed: int, device: str) -> tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )
    _make_r4(proposal, manifest, model_dir, code)
    with pytest.raises(ProposalValidationError, match="writes outside"):
        validate_author_proposal(proposal, allowed_model_dir=model_dir, source_manifest=manifest)
    proposal["proposed_facts"]["implementation"]["code_path"] = str(
        (tmp_path / "outside.py").resolve()
    )
    (tmp_path / "outside.py").write_text(code)
    with pytest.raises(ProposalValidationError, match="escapes"):
        validate_author_proposal(proposal, allowed_model_dir=model_dir, source_manifest=manifest)


def test_absolute_patch_path_is_rejected_before_proposal_identity(tmp_path: Path) -> None:
    """Accepted patch locators must be repository-relative just like adapter paths."""

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(
        proposal,
        manifest,
        model_dir,
        "def build_model() -> object:\n"
        "    return object()\n\n"
        "def make_dummy_call(seed: int, device: str) -> "
        "tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n",
    )
    patch = tmp_path / "outside.patch"
    patch.write_text("diff --git a/a b/a\n", encoding="utf-8")
    proposal["proposed_facts"]["implementation"]["patches"] = [
        {
            "path": str(patch.resolve()),
            "sha256": hash_bytes(patch.read_bytes()),
            "classification": "adapter-fix",
            "semantic": False,
            "rationale": "test path validation",
            "evidence_ids": ["evidence-1"],
        }
    ]
    with pytest.raises(ProposalValidationError, match="repository-relative"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=model_dir,
            source_manifest=manifest,
        )


def test_r4_source_classification_does_not_trust_author_role(tmp_path: Path) -> None:
    """An implementation role without code bytes cannot fabricate a higher rung."""

    proposal, manifest = _ground_proposal(tmp_path)
    code = (
        "def build_model() -> object:\n"
        "    return object()\n\n"
        "def make_dummy_call(seed: int, device: str) -> tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )
    _make_r4(proposal, manifest, tmp_path, code)
    manifest["sources"][0]["role"] = "implementation"
    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )
    assert report.rung.value == "R4_REIMPLEMENT"


def test_fabricated_citation_and_empty_description_are_rejected(tmp_path: Path) -> None:
    """Citation evidence and authored prose are both mandatory and literal."""

    proposal, manifest = _ground_proposal(tmp_path)
    proposal["proposed_facts"]["citation"]["source_evidence_ids"] = ["invented"]
    with pytest.raises(ProposalValidationError, match="fabricated evidence"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)
    proposal, manifest = _ground_proposal(tmp_path)
    proposal["proposed_facts"]["external_metadata"]["description"] = "   "
    with pytest.raises(ProposalValidationError, match="must be non-empty"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_author_envelope_round_trip_and_result_binding(tmp_path: Path) -> None:
    """A complete hash-matched result validates against its one-model packet."""

    proposal, manifest = _ground_proposal(tmp_path)
    prompt_hash = hash_bytes(
        (Path(__file__).parents[1] / "prompts" / "claude_crawler_author_v2.txt").read_bytes()
    )
    proposal["author"]["prompt_sha256"] = prompt_hash
    proposal.update(
        {
            "schema_version": AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
            "campaign_id": "campaign-1",
            "intake_snapshot_id": "intake-1",
            "intake_snapshot_sha256": "sha256:" + "1" * 64,
            "intake_item_sha256": stable_hash(
                {"stable_id": proposal["stable_id"], "variant": "base"}
            ),
            "source_manifest_identity": str(manifest["manifest_sha256"]),
            "dispatcher_identity": "sha256:" + "2" * 64,
        }
    )
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    context = _author_context(proposal, prompt_hash)
    result_path = tmp_path / "output" / "result.json"
    envelope = build_author_envelope(
        context=context,
        work_id=proposal["work_id"],
        stable_id=proposal["stable_id"],
        campaign_id="campaign-1",
        created_at="2026-07-16T00:00:00Z",
        untrusted_hints={"legacy": "hint"},
        source_manifest=manifest,
        allowed_model_dir=tmp_path,
        output_path=result_path,
    )
    result = _author_result(envelope, "PROPOSED", {"arm": "PROPOSED", "proposal": proposal})
    result_path.parent.mkdir()
    result_path.write_text(json.dumps(result))
    validated = validate_author_result(result_path, envelope)
    assert isinstance(validated, ProposedAuthorResult)
    assert validated.binding.stable_id == validated.validation_report.stable_id
    cache = serialize_author_result_cache(validated, source_manifest=manifest, model_dir=tmp_path)
    assert isinstance(validate_author_result_cache(cache, envelope), ProposedAuthorResult)


@pytest.mark.parametrize("value", [None, "adapter.py"])
def test_embedded_author_result_rejects_input_contract_code_path(
    tmp_path: Path, value: object
) -> None:
    """The result callback rejects both forms of the deleted embedded v3 leaf.

    Parameters
    ----------
    tmp_path:
        Isolated author result directory.
    value:
        Legacy null or string value whose presence must reject.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    prompt_hash = hash_bytes(
        (Path(__file__).parents[1] / "prompts" / "claude_crawler_author_v2.txt").read_bytes()
    )
    proposal["author"]["prompt_sha256"] = prompt_hash
    proposal.update(
        {
            "campaign_id": "campaign-1",
            "intake_snapshot_id": "intake-1",
            "intake_snapshot_sha256": "sha256:" + "1" * 64,
            "intake_item_sha256": stable_hash(
                {"stable_id": proposal["stable_id"], "variant": "base"}
            ),
            "source_manifest_identity": str(manifest["manifest_sha256"]),
            "dispatcher_identity": "sha256:" + "2" * 64,
        }
    )
    proposal["proposed_facts"]["input_contract"]["code_path"] = value
    proposal["proposal_sha256"] = stable_hash(
        {key: item for key, item in proposal.items() if key != "proposal_sha256"}
    )
    result_path = tmp_path / "result.json"
    envelope = build_author_envelope(
        context=_author_context(proposal, prompt_hash),
        work_id=proposal["work_id"],
        stable_id=proposal["stable_id"],
        campaign_id="campaign-1",
        created_at="2026-07-16T00:00:00Z",
        untrusted_hints={},
        source_manifest=manifest,
        allowed_model_dir=tmp_path,
        output_path=result_path,
    )
    result_path.write_text(
        json.dumps(_author_result(envelope, "PROPOSED", {"arm": "PROPOSED", "proposal": proposal}))
    )
    with pytest.raises(AuthorDispatchError):
        validate_author_result(result_path, envelope)


@pytest.mark.parametrize("corruption", ["mismatched", "partial"])
def test_author_result_rejects_mismatch_or_partial(tmp_path: Path, corruption: str) -> None:
    """Mismatched identity and partial JSON never enter the proposal lane.

    Parameters
    ----------
    corruption:
        Result corruption applied.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    prompt_hash = hash_bytes(
        (Path(__file__).parents[1] / "prompts" / "claude_crawler_author_v2.txt").read_bytes()
    )
    proposal["author"]["prompt_sha256"] = prompt_hash
    context = _author_context(proposal, prompt_hash)
    result_path = tmp_path / "result.json"
    envelope = build_author_envelope(
        context=context,
        work_id=proposal["work_id"],
        stable_id=proposal["stable_id"],
        campaign_id="campaign-1",
        created_at="2026-07-16T00:00:00Z",
        untrusted_hints={},
        source_manifest=manifest,
        allowed_model_dir=tmp_path,
        output_path=result_path,
    )
    if corruption == "partial":
        result_path.write_text('{"schema_version":')
    else:
        payload = {
            "arm": "DEFER_RECOMMENDATION",
            "platform": "cuda",
            "source_ids": ["source-1"],
            "evidence_ids": ["evidence-1"],
            "evidence_identity": "sha256:" + "3" * 64,
            "license_identity": "sha256:" + "4" * 64,
        }
        payload["recommendation_sha256"] = stable_hash(payload)
        result = _author_result(envelope, "DEFER_RECOMMENDATION", payload)
        result["stable_id"] = "m_other"
        result["result_sha256"] = stable_hash(
            {key: value for key, value in result.items() if key != "result_sha256"}
        )
        result_path.write_text(json.dumps(result))
    with pytest.raises(AuthorDispatchError):
        validate_author_result(result_path, envelope)


@pytest.mark.parametrize(
    ("kind", "payload", "expected_type"),
    [
        (
            "DEFER_RECOMMENDATION",
            {
                "arm": "DEFER_RECOMMENDATION",
                "platform": "cuda",
                "source_ids": ["source-1"],
                "evidence_ids": ["evidence-1"],
                "evidence_identity": "sha256:" + "3" * 64,
                "license_identity": "sha256:" + "4" * 64,
            },
            DeferRecommendation,
        ),
        (
            "SKIP_RECOMMENDATION",
            {
                "arm": "SKIP_RECOMMENDATION",
                "status_code": "skipped:no-description",
                "source_ids": ["source-1"],
                "evidence_ids": ["evidence-1"],
                "evidence_identity": "sha256:" + "3" * 64,
                "search_report_identity": "sha256:" + "5" * 64,
                "license_identity": "sha256:" + "4" * 64,
            },
            SkipRecommendation,
        ),
        (
            "BLOCKED",
            {
                "arm": "BLOCKED",
                "stage": "source",
                "reason_code": "missing-prerequisite",
                "prerequisite_ids": ["prerequisite-1"],
                "evidence_ids": ["evidence-1"],
                "evidence_identity": "sha256:" + "3" * 64,
                "license_identity": "sha256:" + "4" * 64,
            },
            BlockedRecommendation,
        ),
    ],
)
def test_advisory_author_result_arms_are_production_parsed(
    tmp_path: Path,
    kind: str,
    payload: dict[str, Any],
    expected_type: type[object],
) -> None:
    """Every non-proposal arm reaches the same production parser and cache.

    Parameters
    ----------
    tmp_path:
        Isolated author result directory.
    kind, payload, expected_type:
        Closed union fixture and its arm-specific dataclass type.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    prompt_hash = hash_bytes(
        (Path(__file__).parents[1] / "prompts" / "claude_crawler_author_v2.txt").read_bytes()
    )
    proposal["author"]["prompt_sha256"] = prompt_hash
    context = _author_context(proposal, prompt_hash)
    envelope = build_author_envelope(
        context=context,
        work_id=proposal["work_id"],
        stable_id=proposal["stable_id"],
        campaign_id="campaign-1",
        created_at="2026-07-16T00:00:00Z",
        untrusted_hints={},
        source_manifest=manifest,
        allowed_model_dir=tmp_path,
        output_path=tmp_path / "result.json",
    )
    if kind == "DEFER_RECOMMENDATION":
        payload["handoff_execution"] = bind_handoff_execution(
            proposal,
            context=context,
            work_id=str(proposal["work_id"]),
            campaign_id="campaign-1",
            source_manifest_identity=str(manifest["manifest_sha256"]),
        )
    payload["recommendation_sha256"] = stable_hash(payload)
    raw = _author_result(envelope, kind, payload)
    (tmp_path / "result.json").write_text(json.dumps(raw))
    parsed = validate_author_result(tmp_path / "result.json", envelope)
    assert isinstance(parsed, expected_type)
    cache = serialize_author_result_cache(parsed, source_manifest=manifest, model_dir=tmp_path)
    assert isinstance(validate_author_result_cache(cache, envelope), expected_type)


def _author_context(proposal: dict[str, Any], prompt_hash: str) -> AuthorityContext:
    """Return the mandatory frozen context for one author-dispatch fixture."""

    intake = {"stable_id": proposal["stable_id"], "variant": "base"}
    proposal["intake_item_sha256"] = stable_hash(intake)
    return AuthorityContext(
        active_intake_snapshot_id="intake-1",
        active_intake_snapshot_sha256="sha256:" + "1" * 64,
        intake_by_stable_id={proposal["stable_id"]: intake},
        family_bindings={},
        author_prompt_identity=prompt_hash,
        author_model_identity=stable_hash(proposal["author"]),
        author_schema_identity="sha256:" + "6" * 64,
        author_dispatcher_identity="sha256:" + "2" * 64,
        checker_prompt_identity="sha256:" + "7" * 64,
        checker_model_identity="sha256:" + "8" * 64,
        checker_schema_identity="sha256:" + "9" * 64,
        environment_generations={},
        reducer_policy_identity="sha256:" + "a" * 64,
        runner_policy_identity="sha256:" + "b" * 64,
        terminal_policy_identity="sha256:" + "c" * 64,
        publication_policy_identity="sha256:" + "d" * 64,
    )


def _author_result(envelope: dict[str, Any], kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Build one exact self-hashed author-result.v4 fixture."""

    result = {
        **envelope["expected_result"],
        "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
        "result_id": f"result-{kind.lower()}",
        "result_sha256": "sha256:" + "0" * 64,
        "kind": kind,
        "created_at": "2026-07-16T00:01:00Z",
        "payload": payload,
    }
    result["result_sha256"] = stable_hash(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    return result


def _typed_adapter(outside_path: Path) -> str:
    """Return a model adapter that catches a native denied write and returns a tensor.

    Parameters
    ----------
    outside_path:
        Read-only path targeted from the model's forward method.

    Returns
    -------
    str
        Complete typed adapter source.
    """

    return f"""from __future__ import annotations
import ctypes
import os
import torch

class CaughtDenial(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        libc = ctypes.CDLL(None, use_errno=True)
        descriptor = libc.open({str(outside_path)!r}.encode(), os.O_WRONLY | os.O_CREAT, 0o600)
        if descriptor >= 0:
            libc.close(descriptor)
        return value + 1

def build_model() -> object:
    return CaughtDenial()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 3, 8, 8, device=device),), {{}})
"""


@pytest.mark.skipif(sys.platform != "linux", reason="Linux denial-audit regression")
def test_caught_os_sandbox_denial_in_flat_v1_cannot_satisfy_run_award(
    tmp_path: Path,
) -> None:
    """A caught denial under legacy flat-v1 execution remains non-awarding."""

    if detect_os_sandbox("Linux") is None or shutil.which("strace") is None:
        pytest.skip("working Linux sandbox denial broker is unavailable")
    outside_path = tmp_path.parent / f"{tmp_path.name}-forbidden.bin"
    outside_path.unlink(missing_ok=True)
    adapter = tmp_path / "adapter.py"
    adapter.write_text(_typed_adapter(outside_path), encoding="utf-8")
    proposal = make_author_proposal("m_caught_denial")
    scratch = tmp_path / "scratch"
    receipt_path = scratch / "result" / "receipt.json"
    request_path = tmp_path / "request.json"
    expected_revision = compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": adapter.name},
        proposal["source_identity"],
        adapter_bytes=adapter.read_bytes(),
    )
    proposal["recipe_revision"] = expected_revision
    proposal["proposed_facts"]["implementation"]["recipe_revision"] = expected_revision
    request_path.write_text(
        json.dumps(
            {
                "stable_id": proposal["stable_id"],
                "recipe": {
                    "kind": "typed-adapter",
                    "path": str(adapter),
                    "adapter_sha256": hash_bytes(adapter.read_bytes()),
                },
                "modality": "vision",
                "input_spec": {"shape": [1, 3, 8, 8], "dtype": "float32"},
                "scratch_root": str(scratch),
                "receipt_path": str(receipt_path),
                "meaningful_modes": ["eval"],
                "source_identity": proposal["source_identity"],
                "recipe_revision": expected_revision,
                "execution_identity": HASH,
            }
        ),
        encoding="utf-8",
    )

    result = supervise_worker(
        request_path,
        receipt_path,
        scratch / "supervisor",
        timeout_seconds=20,
        rss_limit_bytes=12 * 1024**3,
    )

    assert result.observation.exit_code == 0
    assert result.worker_receipt is None
    assert result.receipt_error == "invalid-receipt:worker-result-envelope"
    environment = EnvironmentBinding(
        prefix=tmp_path / "env",
        python_executable=Path(sys.executable),
        family="core",
        target="linux-64",
        env_generation=HASH,
        lock_sha256=HASH,
        resolved_export_sha256=HASH,
        packages_manifest_sha256=HASH,
        python_version="3.11",
        compiler_identity="test-compiler",
        sdk_identity="test-sdk",
    )
    artifact = make_proposed_artifact(proposal, {"sources": []}, tmp_path)
    attempts = _attempts_from_supervised(
        artifact,
        result,
        environment,
        HASH,
        0,
        20,
        12 * 1024**3,
        diagnostics_root=tmp_path / ".crawl-local" / "diagnostics",
    )

    assert len(attempts) == 1
    assert attempts[0]["result"] == "failed"
    assert attempts[0]["stage"] == "runner"
    assert attempts[0]["error"]["reason_code"] == "protocol-violation"
    assert _attempt_policy_satisfied(attempts, proposal, 1) is False
    assert not outside_path.exists()


def _add_archive_source(
    manifest: dict[str, Any], archive_path: Path, members: dict[str, str]
) -> None:
    """Append one deliberately mislabeled fetched archive to a source manifest.

    Parameters
    ----------
    manifest:
        Controlled-fetch manifest fixture.
    archive_path:
        CAS object path to create.
    members:
        Archive member names and text bytes.
    """

    with zipfile.ZipFile(archive_path, mode="w") as archive:
        for name, member_text in members.items():
            archive.writestr(name, member_text)
    archive_bytes = archive_path.read_bytes()
    manifest["sources"].append(
        {
            "source_id": "archive-source",
            "url": "https://example.com/supplement.zip",
            "revision": "v1",
            "content_sha256": hash_bytes(archive_bytes),
            "cas_path": str(archive_path),
            "retrieval_status": "fetched",
            "role": "introducing-paper",
            "content_kind": "paper-supplement",
        }
    )


def test_r4_inventory_uses_fetched_archive_bytes_not_author_labels(tmp_path: Path) -> None:
    """Code-bearing CAS bytes refuse R4 while a genuine no-code archive still permits it."""

    adapter_code = (
        "def build_model() -> object:\n"
        "    return object()\n\n"
        "def make_dummy_call(seed: int, device: str) -> tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )
    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, adapter_code)
    _add_archive_source(
        manifest,
        tmp_path / "source-code.zip",
        {
            "upstream/src/example_net.py": (
                "import torch\n\n"
                "class ExampleNet(torch.nn.Module):\n"
                "    def __init__(self) -> None:\n"
                "        super().__init__()\n"
                "        self.conv = torch.nn.Conv2d(3, 4, 3)\n\n"
                "    def forward(self, value: torch.Tensor) -> torch.Tensor:\n"
                "        return self.conv(value)\n"
            )
        },
    )
    proposal["proposed_facts"]["source_resolution"]["search_report"]["links_checked"].append(
        "https://example.com/supplement.zip"
    )

    with pytest.raises(ProposalValidationError, match="source code is available"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )

    no_code_root = tmp_path / "no-code"
    no_code_root.mkdir()
    no_code_proposal, no_code_manifest = _ground_proposal(no_code_root)
    _make_r4(no_code_proposal, no_code_manifest, no_code_root, adapter_code)
    _add_archive_source(
        no_code_manifest,
        no_code_root / "paper-materials.zip",
        {
            "README.md": "Architecture equations and prose only.\n",
            "supplement/metrics.py": "def accuracy(expected, observed):\n    return 1.0\n",
            "supplement/plotting.c": "void plot_metrics(void) { return; }\n",
        },
    )
    no_code_proposal["proposed_facts"]["source_resolution"]["search_report"][
        "links_checked"
    ].append("https://example.com/supplement.zip")
    report = validate_author_proposal(
        no_code_proposal,
        allowed_model_dir=no_code_root,
        source_manifest=no_code_manifest,
    )

    assert report.rung.value == "R4_REIMPLEMENT"


def _adapter_code() -> str:
    """Return a minimal typed R4 adapter used by proposal fixtures.

    Returns
    -------
    str
        Complete staged adapter source.
    """

    return (
        "def build_model() -> object:\n"
        "    return object()\n\n"
        "def make_dummy_call(seed: int, device: str) -> "
        "tuple[tuple[()], dict[str, object]]:\n"
        "    return (), {}\n"
    )


@pytest.mark.parametrize(
    "source_code",
    [
        (
            "import flax.linen as nn\n"
            "import jax\n"
            "import jax.numpy as jnp\n\n"
            "class ExampleNetArchitecture(nn.Module):\n"
            "    @nn.compact\n"
            "    def __call__(self, value):\n"
            "        scanned, _ = jax.lax.scan(custom_step, value, value)\n"
            "        return jnp.einsum('...d,df->...f', scanned, custom_weights())\n"
        ),
        (
            "import paddle\n\n"
            "class ExampleNetArchitecture(CustomPaddleBase):\n"
            "    def forward(self, value):\n"
            "        mixed = custom_paddle_stage(value)\n"
            "        return paddle.add(mixed, value)\n"
        ),
        (
            "import torch\n\n"
            "def example_net_architecture(value, weights):\n"
            "    mixed = custom_channel_mix(value, weights)\n"
            "    return torch.einsum('bcd,ce->bed', mixed, weights)\n"
        ),
    ],
    ids=("jax-flax", "paddle", "custom-functional-pytorch"),
)
def test_framework_neutral_implementation_bytes_refuse_r4(tmp_path: Path, source_code: str) -> None:
    """JAX/Flax, Paddle, and custom-functional model sources all block R4.

    Parameters
    ----------
    tmp_path:
        Isolated model and CAS directory.
    source_code:
        Exact framework-specific upstream implementation bytes.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, _adapter_code())
    _add_archive_source(
        manifest,
        tmp_path / "implementation.zip",
        {"upstream/src/example_net.py": source_code},
    )

    with pytest.raises(ProposalValidationError, match="source code is available"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )


def test_irrelevant_code_archive_still_permits_r4(tmp_path: Path) -> None:
    """Metrics and plotting files do not masquerade as model implementations."""

    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, _adapter_code())
    _add_archive_source(
        manifest,
        tmp_path / "paper-materials.zip",
        {
            "supplement/metrics.py": (
                "def example_net_accuracy(expected, observed):\n"
                "    return (expected == observed).mean()\n"
            ),
            "supplement/plotting.c": "void plot_example_net_metrics(void) { return; }\n",
        },
    )

    report = validate_author_proposal(
        proposal,
        allowed_model_dir=tmp_path,
        source_manifest=manifest,
    )

    assert report.rung.value == "R4_REIMPLEMENT"


def test_split_registry_to_model_symbol_refuses_r4(tmp_path: Path) -> None:
    """Identity-bearing config linked to a separate executable model blocks R4."""

    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, _adapter_code())
    _add_archive_source(
        manifest,
        tmp_path / "split-implementation.zip",
        {
            "configs/model.py": (
                "from src.net import Net\n\nMODEL_REGISTRY = {'ExampleNet': Net}\n"
            ),
            "src/net.py": (
                "class Net:\n"
                "    def forward(self, value):\n"
                "        hidden = self.encoder(value)\n"
                "        return self.decoder(hidden)\n"
            ),
        },
    )

    with pytest.raises(ProposalValidationError, match="source code is available"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )


def test_unrelated_linked_generic_helper_does_not_block_r4(tmp_path: Path) -> None:
    """A generic forward helper with only a prose identity mention is not an implementation."""

    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, _adapter_code())
    _add_archive_source(
        manifest,
        tmp_path / "generic-helper.zip",
        {
            "utils/helper.py": (
                "# Used by the ExampleNet documentation build.\n"
                "class Helper:\n"
                "    def forward(self, value):\n"
                "        return normalize(value)\n"
            )
        },
    )

    report = validate_author_proposal(
        proposal,
        allowed_model_dir=tmp_path,
        source_manifest=manifest,
    )
    assert report.rung.value == "R4_REIMPLEMENT"


def test_large_notebook_is_streamed_and_structurally_inspected(tmp_path: Path) -> None:
    """A model notebook above the former 8 MiB cap still blocks source-free R4."""

    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, _adapter_code())
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["x" * (8 * 1024**2 + 1)]},
            {
                "cell_type": "code",
                "source": [
                    "class ExampleNet:\n",
                    "    def forward(self, value):\n",
                    "        hidden = self.encoder(value)\n",
                    "        return self.decoder(hidden)\n",
                ],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    _add_archive_source(
        manifest,
        tmp_path / "large-notebook.zip",
        {"notebooks/example_net.ipynb": json.dumps(notebook)},
    )

    with pytest.raises(ProposalValidationError, match="source code is available"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )


@pytest.mark.parametrize("missing_proof", ["negative-attempt", "bounded-report"])
def test_r4_requires_explicit_bounded_negative_proof(tmp_path: Path, missing_proof: str) -> None:
    """R4 fails unless higher-rung absence and a bounded search are explicit.

    Parameters
    ----------
    tmp_path:
        Isolated model and CAS directory.
    missing_proof:
        Negative-proof component removed from the otherwise valid fixture.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _make_r4(proposal, manifest, tmp_path, _adapter_code())
    resolution = proposal["proposed_facts"]["source_resolution"]
    if missing_proof == "negative-attempt":
        resolution["attempted_rungs"][1]["result"] = "not-reached"
    else:
        resolution["search_report"]["queries"] = []

    with pytest.raises(ProposalValidationError, match="explicit negative proof|bounded search"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )
