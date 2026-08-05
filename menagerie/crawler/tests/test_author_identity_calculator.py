"""The two identity walls, and the OPPOSITE fixes they take.

Ten genuine ``c1-mech`` models published clean author results and every one refused at
identity binding. Two independent walls, and fixing only the first leaves all ten failing:

**Wall 1 -- ``proposal.author``.** ``{provider, model, version, prompt_sha256}`` is a
function of nothing the author knows: four machine-held facts about its own dispatch.
Recomputing it CATCHES NOTHING, because a "lifted" value here would be the correct value.
So the machine supplies it. These tests prove the exact block the ten authors published
now binds, and that the check which refused it is intact and still refuses it when the
stamp is not applied.

**Wall 2 -- the five derived identities.** These ARE pure functions of the author's own
drafted facts, so the driver's recompute is a genuine tripwire: it catches a proposal
that lifted an identity from somewhere its facts do not produce. They are NOT stamped and
the recompute is NOT loosened. What is handed over is the arithmetic, as a calculator.
These tests prove a calculator-computed identity satisfies the driver, and -- the half
that matters -- that a FABRICATED fact still fails it.

Every test drives the REAL production functions. A reimplementation of a derivation
proves only that the reimplementation agrees with itself.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorEngineFaultError,
    AuthorResultBinding,
    AuthorResultMalformedError,
    _validate_proposal_binding,
    author_identity_binding,
    build_author_envelope,
    checker_identity_binding,
)
from menagerie.crawler.author_executor import (
    AuthorExecutorError,
    _stamp_machine_owned_proposal_fields,
    identity_tool_command,
)
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.metadata import recompute_accepted_identities
from menagerie.crawler.schema import MODEL_SCHEMA_VERSION_V3
from menagerie.crawler.tests.conftest import attach_paper_evidence, make_author_proposal
from menagerie.crawler.tools.author_identities import (
    DERIVED_IDENTITY_FIELDS,
    IdentityToolError,
    derive_identities,
    main,
)

#: The exact author block the ten refused c1-mech proposals published, and the exact one
#: the driver's authority context hashed. They differ in ONE named field -- ``version`` --
#: and asserting that explicitly is deliberate: an all-placeholder pair whose two halves
#: happen to be equal would let a stamp that does nothing pass vacuously.
_AUTHORED_AUTHOR_BLOCK: dict[str, Any] = {
    "provider": "anthropic",
    "model": "claude-sonnet",
    "version": "claude_crawler_author_v2",
    "prompt_sha256": "sha256:" + "d3" * 32,
}
_MACHINE_AUTHOR_BLOCK: dict[str, Any] = {
    **_AUTHORED_AUTHOR_BLOCK,
    "version": "current",
}

_CHECKER_MODEL = "gpt-5.6-terra"
_CHECKER_VERSION = "current"


def test_the_two_author_blocks_differ_in_exactly_one_named_field() -> None:
    """Pin the fixture's own discriminating power before anything relies on it."""

    differing = {
        key
        for key in {*_AUTHORED_AUTHOR_BLOCK, *_MACHINE_AUTHOR_BLOCK}
        if _AUTHORED_AUTHOR_BLOCK.get(key) != _MACHINE_AUTHOR_BLOCK.get(key)
    }
    assert differing == {"version"}
    assert stable_hash(_AUTHORED_AUTHOR_BLOCK) != stable_hash(_MACHINE_AUTHOR_BLOCK)


# -- Wall 1: the machine supplies what only the machine knows -----------------


def _binding() -> AuthorResultBinding:
    """Return a result binding carrying the DRIVER's author identity."""

    return AuthorResultBinding(
        result_id="result-1",
        result_sha256="sha256:" + "0" * 64,
        stable_id="m3671",
        work_id="work-m3671",
        campaign_id="c1-mech",
        author_identity=stable_hash(_MACHINE_AUTHOR_BLOCK),
        prompt_identity=_MACHINE_AUTHOR_BLOCK["prompt_sha256"],
        dispatcher_identity="sha256:" + "2" * 64,
        source_manifest_identity="sha256:" + "3" * 64,
        intake_snapshot_id="snapshot-1",
        intake_snapshot_sha256="sha256:" + "4" * 64,
        intake_item_sha256="sha256:" + "5" * 64,
        created_at="2026-07-31T00:00:00Z",
        raw_result={},
    )


def _bound_proposal(binding: AuthorResultBinding, author: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal proposal repeating every common binding, self-hashed."""

    proposal: dict[str, Any] = {
        "stable_id": binding.stable_id,
        "work_id": binding.work_id,
        "campaign_id": binding.campaign_id,
        "intake_snapshot_id": binding.intake_snapshot_id,
        "intake_snapshot_sha256": binding.intake_snapshot_sha256,
        "intake_item_sha256": binding.intake_item_sha256,
        "source_manifest_identity": binding.source_manifest_identity,
        "dispatcher_identity": binding.dispatcher_identity,
        "author": deepcopy(author),
    }
    proposal["proposal_sha256"] = stable_hash(proposal)
    return proposal


def test_the_authored_block_is_still_refused_when_it_is_not_stamped() -> None:
    """The check that refused ten models is intact and still refuses that value.

    This is the failing direction. Nothing about Wall 1's fix loosens
    ``_validate_proposal_binding``: hand it the author's own guess and it raises the
    same exact message it raised on all ten.
    """

    binding = _binding()
    with pytest.raises(AuthorResultMalformedError) as excinfo:
        _validate_proposal_binding(_bound_proposal(binding, _AUTHORED_AUTHOR_BLOCK), binding)
    assert str(excinfo.value) == "proposal author identity does not match its result binding"


def test_the_stamped_proposal_carrying_the_observed_value_now_binds(tmp_path: Path) -> None:
    """A proposal whose author wrote the OBSERVED wrong value now passes the real check.

    The proposal that goes in carries ``version: "claude_crawler_author_v2"`` -- the exact
    string every one of the ten published. The stamp replaces it with the machine's own,
    and the unchanged ``_validate_proposal_binding`` then accepts it.
    """

    binding = _binding()
    authored = _bound_proposal(binding, _AUTHORED_AUTHOR_BLOCK)
    expected = {
        "campaign_id": binding.campaign_id,
        "stable_id": binding.stable_id,
        "work_id": binding.work_id,
        "intake_snapshot_id": binding.intake_snapshot_id,
        "intake_snapshot_sha256": binding.intake_snapshot_sha256,
        "intake_item_sha256": binding.intake_item_sha256,
        "source_manifest_identity": binding.source_manifest_identity,
        "dispatcher_identity": binding.dispatcher_identity,
    }
    stamped = _stamp_machine_owned_proposal_fields(
        authored, expected, author_binding=_MACHINE_AUTHOR_BLOCK
    )
    assert stamped["author"] == _MACHINE_AUTHOR_BLOCK
    assert stamped["author"] != authored["author"]
    # The real check, unmodified, on the stamped object.
    _validate_proposal_binding(stamped, binding)
    assert stable_hash(stamped["author"]) == binding.author_identity


def test_the_stamp_overwrites_a_conflict_rather_than_refusing_it() -> None:
    """An authored ``author`` is silently replaced, not treated as a contradiction.

    The eight request bindings refuse a conflicting authored value, because an author
    that disagrees there is contradicting a fact it was handed. ``author`` is the
    opposite: the author was never told the value, so refusing its guess would only
    rebuild the same wall one layer up.
    """

    binding = _binding()
    for guess in ("claude_crawler_author_v2", "claude-opus-5", "v2", ""):
        authored = _bound_proposal(binding, {**_AUTHORED_AUTHOR_BLOCK, "version": guess})
        stamped = _stamp_machine_owned_proposal_fields(
            authored, {}, author_binding=_MACHINE_AUTHOR_BLOCK
        )
        assert stamped["author"] == _MACHINE_AUTHOR_BLOCK


def test_an_undisclosed_author_binding_fails_closed() -> None:
    """A missing or open binding is an engine fault, never a licence to keep the guess."""

    binding = _binding()
    authored = _bound_proposal(binding, _AUTHORED_AUTHOR_BLOCK)
    for bad in ({}, {"provider": "anthropic"}, {**_MACHINE_AUTHOR_BLOCK, "actor": "x"}):
        with pytest.raises(AuthorExecutorError, match="identity_inputs.author"):
            _stamp_machine_owned_proposal_fields(authored, {}, author_binding=bad)


# -- the envelope disclosure ---------------------------------------------------


def _context(prompt_hash: str, stable_id: str) -> AuthorityContext:
    """Return a frozen authority context whose author prompt matches shipped bytes."""

    author_fields = {**_MACHINE_AUTHOR_BLOCK, "prompt_sha256": prompt_hash}
    return AuthorityContext(
        active_intake_snapshot_id="intake-1",
        active_intake_snapshot_sha256="sha256:" + "1" * 64,
        intake_by_stable_id={stable_id: {"stable_id": stable_id, "variant": "base"}},
        family_bindings={},
        author_prompt_identity=prompt_hash,
        author_model_identity=stable_hash(author_fields),
        author_schema_identity="sha256:" + "6" * 64,
        author_dispatcher_identity="sha256:" + "2" * 64,
        author_model_fields=author_fields,
        checker_prompt_identity=_checker_prompt_hash(),
        checker_model_identity="sha256:" + "8" * 64,
        checker_schema_identity="sha256:" + "9" * 64,
        checker_model_fields={
            "provider": "openai",
            "model": _CHECKER_MODEL,
            "version": _CHECKER_VERSION,
            "prompt_sha256": _checker_prompt_hash(),
        },
        environment_generations={},
        reducer_policy_identity="sha256:" + "a" * 64,
        runner_policy_identity="sha256:" + "b" * 64,
        terminal_policy_identity="sha256:" + "c" * 64,
        publication_policy_identity="sha256:" + "d" * 64,
    )


def _checker_prompt_hash() -> str:
    """Hash the exact shipped checker prompt bytes."""

    from menagerie.crawler.constants import CHECKER_PROMPT_NAME

    path = Path(__file__).resolve().parents[1] / "prompts" / f"{CHECKER_PROMPT_NAME}.txt"
    return hash_bytes(path.read_bytes())


def _author_prompt_hash() -> str:
    """Hash the exact shipped author prompt bytes."""

    from menagerie.crawler.constants import AUTHOR_PROMPT_NAME

    path = Path(__file__).resolve().parents[1] / "prompts" / f"{AUTHOR_PROMPT_NAME}.txt"
    return hash_bytes(path.read_bytes())


def _envelope(tmp_path: Path, stable_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Build one real author envelope through the production builder."""

    return build_author_envelope(
        context=_context(_author_prompt_hash(), stable_id),
        work_id=f"work-{stable_id}",
        stable_id=stable_id,
        campaign_id="c1-mech",
        created_at="2026-07-31T00:00:00Z",
        untrusted_hints={},
        source_manifest=manifest,
        allowed_model_dir=tmp_path,
        output_path=tmp_path / "out" / "result.json",
    )


def test_the_envelope_discloses_the_preimage_of_the_identity_it_already_published(
    tmp_path: Path,
) -> None:
    """Disclosure is strictly weaker than the digest the envelope has always shipped."""

    envelope = _envelope(tmp_path, "m_example", {"sources": []})
    author = author_identity_binding(envelope)
    assert stable_hash(author) == envelope["expected_result"]["author_identity"]
    assert set(author) == {"provider", "model", "version", "prompt_sha256"}
    # The preimage does NOT ride in expected_result: those keys are splatted verbatim
    # into an author-result.v4 body whose additionalProperties is false.
    assert "author" not in envelope["expected_result"]


def test_a_malformed_disclosure_is_an_engine_fault(tmp_path: Path) -> None:
    """Both accessors fail closed rather than defaulting a binding."""

    envelope = _envelope(tmp_path, "m_example", {"sources": []})
    for mutate in (
        lambda e: e.pop("identity_inputs"),
        lambda e: e["identity_inputs"].pop("author"),
        lambda e: e["identity_inputs"]["author"].pop("version"),
        lambda e: e["identity_inputs"]["author"].update({"version": ""}),
    ):
        broken = deepcopy(envelope)
        mutate(broken)
        with pytest.raises(AuthorEngineFaultError, match="identity_inputs"):
            author_identity_binding(broken)
    assert checker_identity_binding(envelope)["model"] == _CHECKER_MODEL


# -- Wall 2: the calculator ----------------------------------------------------


def _grounded(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a MULTI-SOURCE grounded proposal and its controlled manifest.

    Multi-source on purpose: a fixture freezing exactly one source hid a
    width-dependent bug here for a whole sprint. Two sources with distinct bytes
    exercise the ordering and per-source projection the source identity depends on.
    """

    proposal = make_author_proposal()
    text = (
        "Example Model introduced ExampleNet in TestConf 2020 by A. Author at Example Lab "
        "in the US. ExampleNet is an official PyTorch library CNN architecture for "
        "supervised computer vision classification in machine learning. This modern "
        "ExampleNet family uses vision modality and has the example and cnn keywords. It "
        "is a small source-grounded example network whose grounded contribution uses the "
        "Apache-2.0 license. It runs in PyTorch eval mode with no train eval divergence. "
        "The input contract is one small RGB image and the output is class scores."
    )
    source_path = tmp_path / "source.txt"
    source_path.write_text(text)
    excerpt = proposal["proposed_facts"]["evidence"]["excerpts"][0]
    excerpt.update(
        {
            "locator": f"bytes:0-{len(text.encode())}",
            "text": text,
            "text_sha256": hash_bytes(text.encode()),
            "family_level": True,
        }
    )
    manifest: dict[str, Any] = {
        "sources": [
            {
                "source_id": "source-1",
                "url": "https://example.com/model",
                "revision": "v1",
                "content_sha256": hash_bytes(text.encode()),
                "cas_path": str(source_path),
                "retrieval_status": "fetched",
            }
        ]
    }
    manifest["manifest_sha256"] = stable_hash(manifest["sources"])
    # Adds the second, independently-fetched introducing-paper source and its excerpt.
    attach_paper_evidence(proposal, manifest, tmp_path)
    assert len(proposal["proposed_facts"]["source_resolution"]["sources"]) >= 2
    assert len(proposal["proposed_facts"]["evidence"]["excerpts"]) >= 2
    return proposal, manifest


def _driver_recompute(facts: dict[str, Any]) -> dict[str, Any]:
    """Run the exact derivation ``_validate_artifact_identities`` runs."""

    identities = recompute_accepted_identities(
        facts,
        checker_prompt_hash=_checker_prompt_hash(),
        checker_model=_CHECKER_MODEL,
        checker_version=_CHECKER_VERSION,
        schema_version=MODEL_SCHEMA_VERSION_V3,
    )
    return {
        "source_identity": identities.source,
        "evidence_identity": identities.evidence,
        "recipe_revision": identities.recipe,
        "vet_identity": identities.vet,
        "fidelity_identity": identities.fidelity,
    }


def test_calculator_output_satisfies_the_driver_recompute(tmp_path: Path) -> None:
    """The tool's five identities are exactly what the driver derives. All five."""

    proposal, manifest = _grounded(tmp_path)
    envelope = _envelope(tmp_path, proposal["stable_id"], manifest)
    computed = derive_identities(envelope=envelope, facts_document=proposal)
    assert set(computed) == set(DERIVED_IDENTITY_FIELDS)
    # Compared against the recompute over the PUBLISHED fact block -- the one carrying
    # the two embedded copies -- because that is the object the driver actually sees.
    assert computed == _driver_recompute(_settled(proposal["proposed_facts"], computed))


def test_a_fabricated_fact_still_fails_the_recompute(tmp_path: Path) -> None:
    """THE proof that arithmetic was removed and the check was not.

    Compute the identities honestly from facts F. Then change EXACTLY ONE named fact
    and publish the identities computed from F beside the changed facts -- the shape a
    proposal takes when it lifted an identity from somewhere its own facts do not
    produce. The driver's recompute disagrees, on the identity that fact feeds.
    """

    proposal, manifest = _grounded(tmp_path)
    envelope = _envelope(tmp_path, proposal["stable_id"], manifest)
    honest = derive_identities(envelope=envelope, facts_document=proposal)

    fabricated = deepcopy(proposal["proposed_facts"])
    original = fabricated["evidence"]["excerpts"][0]["text"]
    forged = original.replace("TestConf 2020", "PrestigeConf 2019")
    assert forged != original
    fabricated["evidence"]["excerpts"][0]["text"] = forged
    fabricated["evidence"]["excerpts"][0]["text_sha256"] = hash_bytes(forged.encode())

    differing = {
        key
        for key in ("text", "text_sha256")
        if fabricated["evidence"]["excerpts"][0][key]
        != proposal["proposed_facts"]["evidence"]["excerpts"][0][key]
    }
    assert differing == {"text", "text_sha256"}

    # Publish the fabricated facts carrying the HONEST identities -- the exact shape a
    # proposal takes when it lifted an identity from somewhere its own facts do not
    # produce -- and the driver's recompute disagrees.
    published = _settled(fabricated, honest)
    recomputed = _driver_recompute(published)
    assert recomputed["evidence_identity"] != honest["evidence_identity"]
    assert recomputed["vet_identity"] != honest["vet_identity"]
    # And the tool is not a laundry: asked about the fabricated facts it returns the
    # fabrication's own identity, which is exactly what the driver refuses to match
    # against the real artifacts.
    fabricated_ids = derive_identities(envelope=envelope, facts_document=fabricated)
    assert fabricated_ids != honest
    assert fabricated_ids == _driver_recompute(_settled(fabricated, fabricated_ids))


def _settled(facts: dict[str, Any], identities: dict[str, Any]) -> dict[str, Any]:
    """Return the fact block with both embedded identity copies written."""

    published = deepcopy(facts)
    published["implementation"]["recipe_revision"] = identities["recipe_revision"]
    published["evidence"]["evidence_identity"] = identities["evidence_identity"]
    return published


def test_the_calculator_settles_the_embedded_copies_before_reporting(
    tmp_path: Path,
) -> None:
    """The reported ``vet_identity`` is the one the PUBLISHED proposal will produce.

    ``_validate_artifact_identities`` requires ``implementation.recipe_revision`` and
    ``evidence.evidence_identity`` to equal the identities it recomputes, and both are
    authored leaves that ``vet_identity`` projects. A calculator that derived once from
    a draft lacking them would hand back a ``vet_identity`` the driver rejects the
    moment the author writes them in. This asserts the reported numbers survive the
    round trip through the complete published fact block.
    """

    proposal, manifest = _grounded(tmp_path)
    envelope = _envelope(tmp_path, proposal["stable_id"], manifest)

    draft = deepcopy(proposal["proposed_facts"])
    draft["implementation"].pop("recipe_revision", None)
    draft["evidence"].pop("evidence_identity", None)

    reported = derive_identities(envelope=envelope, facts_document=draft)
    published = _settled(draft, reported)
    assert _driver_recompute(published) == reported


def test_deriving_without_settling_would_have_produced_a_refused_vet_identity(
    tmp_path: Path,
) -> None:
    """Pin the trap the settling pass exists for, in the failing direction.

    Derive naively from the copy-less draft, then write the copies as an author must.
    ``vet_identity`` moves -- and only ``vet_identity`` -- so a calculator without the
    settling pass would be confidently wrong on exactly one of the five.
    """

    proposal, manifest = _grounded(tmp_path)
    envelope = _envelope(tmp_path, proposal["stable_id"], manifest)
    draft = deepcopy(proposal["proposed_facts"])
    draft["implementation"].pop("recipe_revision", None)
    draft["evidence"].pop("evidence_identity", None)

    naive = _driver_recompute(draft)
    published = _settled(draft, naive)
    after = _driver_recompute(published)

    moved = {field for field in DERIVED_IDENTITY_FIELDS if naive[field] != after[field]}
    assert moved == {"vet_identity"}
    # And the settled calculator is right where the naive derivation was wrong.
    assert derive_identities(envelope=envelope, facts_document=draft) == after


def test_the_calculator_refuses_to_echo_an_identity_it_was_handed(tmp_path: Path) -> None:
    """A guessed identity supplied on the way IN is stripped, never returned."""

    proposal, manifest = _grounded(tmp_path)
    envelope = _envelope(tmp_path, proposal["stable_id"], manifest)
    honest = derive_identities(envelope=envelope, facts_document=proposal)

    poisoned = deepcopy(proposal)
    lie = "sha256:" + "e" * 64
    for field in DERIVED_IDENTITY_FIELDS:
        poisoned["proposed_facts"][field] = lie
    computed = derive_identities(envelope=envelope, facts_document=poisoned)
    assert computed == honest
    assert lie not in computed.values()


def test_the_calculator_never_dereferences_a_source(tmp_path: Path) -> None:
    """It computes from the facts supplied, not from anything they point at.

    Deleting every fetched byte on disk changes nothing: a calculator that reached for
    a source would break here, and a calculator that reached for a source would be a
    second author.
    """

    proposal, manifest = _grounded(tmp_path)
    envelope = _envelope(tmp_path, proposal["stable_id"], manifest)
    before = derive_identities(envelope=envelope, facts_document=proposal)
    deleted = 0
    for source in manifest["sources"]:
        path = Path(str(source.get("cas_path", "")))
        if path.is_file():
            path.unlink()
            deleted += 1
    # Without this the test would pass vacuously on a fixture that froze no bytes.
    assert deleted >= 2, f"multi-source fixture must have had bytes to delete, got {deleted}"
    assert derive_identities(envelope=envelope, facts_document=proposal) == before


def test_the_calculator_refuses_a_doctored_machine_binding(tmp_path: Path) -> None:
    """A hand-edited ``identity_inputs`` fails at the calculator, not silently later."""

    proposal, manifest = _grounded(tmp_path)
    envelope = _envelope(tmp_path, proposal["stable_id"], manifest)
    doctored = deepcopy(envelope)
    doctored["identity_inputs"]["checker"]["model"] = "some-other-checker"

    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(proposal), encoding="utf-8")
    request_path = tmp_path / "doctored-request.json"
    request_path.write_text(json.dumps(doctored), encoding="utf-8")
    assert main(["--request", str(request_path), "--facts", str(facts_path)]) == 2


def test_the_calculator_refuses_incomplete_facts_rather_than_guessing(
    tmp_path: Path,
) -> None:
    """Missing facts produce a typed refusal, never a filled-in default."""

    proposal, manifest = _grounded(tmp_path)
    envelope = _envelope(tmp_path, proposal["stable_id"], manifest)
    stripped = deepcopy(proposal)
    stripped["proposed_facts"].pop("evidence")
    with pytest.raises(IdentityToolError, match="proposed_facts"):
        derive_identities(envelope=envelope, facts_document=stripped)

    hollow = deepcopy(proposal)
    hollow["proposed_facts"]["source_resolution"].pop("sources")
    with pytest.raises(IdentityToolError, match="incomplete"):
        derive_identities(envelope=envelope, facts_document=hollow)


def test_the_command_the_grant_names_actually_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end through the CLI the permission specifier pins.

    The author reaches this only as a command line, so the argv path is the one that
    has to work -- an importable function nobody can invoke would be no capability.
    """

    proposal, manifest = _grounded(tmp_path)
    envelope = _envelope(tmp_path, proposal["stable_id"], manifest)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(envelope), encoding="utf-8")
    facts_path = tmp_path / "facts.json"
    # The author may hand over the whole draft proposal or just its fact block.
    facts_path.write_text(json.dumps(proposal["proposed_facts"]), encoding="utf-8")

    assert main(["--request", str(request_path), "--facts", str(facts_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == _driver_recompute(_settled(proposal["proposed_facts"], printed))


def test_the_grant_command_runs_as_a_real_subprocess_from_the_attempt_directory(
    tmp_path: Path,
) -> None:
    """The exact granted command, in a fresh process, cwd'd where the author is.

    In-process tests cannot see this failure mode and did not: every one of them
    runs from a repository root, where ``menagerie`` resolves through the current
    directory. ``menagerie`` is not an installed distribution, and the author's cwd
    is its own attempt directory, so a live probe under the real permission recipe
    got ``No module named 'menagerie'`` AFTER the harness had already granted the
    command -- a capability that was reachable but not runnable. Only a real
    subprocess started somewhere else reproduces it, so that is what this does.
    """

    proposal, manifest = _grounded(tmp_path)
    envelope = _envelope(tmp_path, proposal["stable_id"], manifest)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(envelope), encoding="utf-8")
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(proposal["proposed_facts"]), encoding="utf-8")

    attempt = tmp_path / "attempts" / "attempt-001-probe"
    attempt.mkdir(parents=True)
    completed = subprocess.run(
        [*shlex.split(identity_tool_command()), "--request", str(request_path),
         "--facts", str(facts_path)],
        cwd=str(attempt),
        capture_output=True,
        text=True,
        timeout=120,
        # A cleared PYTHONPATH is the honest reproduction: the grant carries no
        # environment, so the script has to bootstrap its own imports.
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    assert completed.returncode == 0, completed.stderr
    printed = json.loads(completed.stdout)
    assert printed == _driver_recompute(_settled(proposal["proposed_facts"], printed))


# -- the two arithmetic modes rung 8 proved missing --------------------------
#
# The stage-2 allowlist grants exactly ONE command: this calculator. Rung 8
# measured what each missing arithmetic cost. No clock: eight sessions guessed
# at time and self-blocked with 63-88% of their grant unused. No hash: the
# brief instructed ``sha256sum``, the allowlist denied it, and m8189 published
# eight sequential PLACEHOLDER digests over byte-perfect excerpts. Both
# arithmetics now live behind the same single granted prefix.


def test_the_clock_mode_reports_now_and_the_exact_remaining_seconds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--clock --deadline`` is the granted time observation, end to end."""

    from datetime import datetime, timedelta, timezone

    deadline = datetime.now(timezone.utc) + timedelta(seconds=600)
    deadline_iso = deadline.isoformat().replace("+00:00", "Z")
    assert main(["--clock", "--deadline", deadline_iso]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"now", "deadline", "remaining_seconds"}
    assert report["deadline"] == deadline_iso
    assert 0 < report["remaining_seconds"] <= 600
    assert report["now"].endswith("Z")

    # Without a deadline the observation is just "now" -- still an observation.
    assert main(["--clock"]) == 0
    bare = json.loads(capsys.readouterr().out)
    assert set(bare) == {"now"}

    # A passed deadline reads NEGATIVE, never wraps or clamps: the session must
    # be able to see that it is already in the external-kill window.
    stale = datetime.now(timezone.utc) - timedelta(seconds=120)
    assert main(["--clock", "--deadline", stale.isoformat()]) == 0
    late = json.loads(capsys.readouterr().out)
    assert late["remaining_seconds"] < 0


def test_a_malformed_deadline_is_a_typed_refusal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Garbage in is a refusal naming the fix, never a guessed instant."""

    assert main(["--clock", "--deadline", "half past nine"]) == 2
    assert "ISO-8601" in capsys.readouterr().err


def test_the_hash_modes_produce_the_digest_the_evidence_validator_recomputes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--hash-string``/``--hash-file`` are the real excerpt-digest arithmetic.

    The digest must equal ``identity.hash_bytes`` over the exact UTF-8 bytes,
    because that is precisely what the evidence validator recomputes from the
    ``text`` field.
    """

    text = "PoolFormer is instantiated from metaformer configs"
    assert main(["--hash-string", text]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["sha256"] == hash_bytes(text.encode("utf-8"))
    assert printed["bytes_len"] == len(text.encode("utf-8"))
    assert "sha256_without_trailing_newline" not in printed

    target = tmp_path / "excerpt.txt"
    target.write_bytes(text.encode("utf-8"))
    assert main(["--hash-file", str(target)]) == 0
    from_file = json.loads(capsys.readouterr().out)
    assert from_file["sha256"] == printed["sha256"]


def test_the_trailing_newline_trap_is_reported_beside_the_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file ending in a newline reports BOTH digests, labelled.

    The one observed real mismatch class: a file-writing tool appends a final
    newline the quoted string never had, and the excerpt digest is wrong by
    exactly one byte. The report names the alternative instead of leaving the
    session to rediscover the trap one dead proposal at a time.
    """

    text = "exact quoted line"
    target = tmp_path / "excerpt.txt"
    target.write_bytes(text.encode("utf-8") + b"\n")
    assert main(["--hash-file", str(target)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["trailing_newline"] is True
    assert report["sha256"] == hash_bytes(text.encode("utf-8") + b"\n")
    assert report["sha256_without_trailing_newline"] == hash_bytes(
        text.encode("utf-8")
    )


def test_an_unreadable_hash_target_is_a_typed_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing file refuses with the path named, never an empty-bytes digest."""

    assert main(["--hash-file", str(tmp_path / "absent.txt")]) == 2
    assert "unreadable" in capsys.readouterr().err


def test_exactly_one_mode_per_invocation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Modes are exclusive, and the identity mode still demands both inputs."""

    assert main(["--clock", "--hash-string", "x"]) == 2
    assert "exactly one mode" in capsys.readouterr().err
    assert main([]) == 2
    capsys.readouterr()
    request_only = tmp_path / "request.json"
    request_only.write_text("{}", encoding="utf-8")
    assert main(["--request", str(request_only)]) == 2
    assert "BOTH --request and --facts" in capsys.readouterr().err


def test_the_granted_command_carries_the_new_modes_as_a_real_subprocess(
    tmp_path: Path,
) -> None:
    """The exact granted prefix runs ``--clock`` and ``--hash-file`` from an
    attempt directory -- reachable AND runnable, like the identity mode."""

    attempt = tmp_path / "attempts" / "attempt-001-modes"
    attempt.mkdir(parents=True)
    target = attempt / "excerpt.txt"
    target.write_bytes(b"granted bytes")
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    clock = subprocess.run(
        [*shlex.split(identity_tool_command()), "--clock"],
        cwd=str(attempt),
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    assert clock.returncode == 0, clock.stderr
    assert "now" in json.loads(clock.stdout)

    digest = subprocess.run(
        [*shlex.split(identity_tool_command()), "--hash-file", str(target)],
        cwd=str(attempt),
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    assert digest.returncode == 0, digest.stderr
    assert json.loads(digest.stdout)["sha256"] == hash_bytes(b"granted bytes")
