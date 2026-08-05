"""Anti-slop proposal and one-model author-dispatch tests for Slice D."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorDispatchError,
    AuthorEffortExhaustionClaim,
    BlockedRecommendation,
    DeferRecommendation,
    ProposedAuthorResult,
    SkipRecommendation,
    _validate_blocked_reason,
    build_author_envelope,
    serialize_author_result_cache,
    validate_author_result,
    validate_author_result_cache,
)
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.constants import (
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    AUTHOR_RESULT_SCHEMA_VERSION,
    ACCESS_BARRIER_REJECTION_CLASS,
    ACCESS_BLOCKED_REASON_CODE,
    EFFORT_EXHAUSTION_REASON_CODES,
    EXHAUSTION_TERMINAL_REASON_BY_STAGE,
    FAILURE_REASON_CODES,
    TERMINAL_STATUS_CODES,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.proposal import (
    CHECKER_EVALUATED_CLAIMS,
    DEFAULT_GATED_CLAIMS,
    VALUE_MATCHED_CLAIMS,
    ProposalValidationError,
    _identifier_grounded,
    _normalize_support_text,
    model_code_manifest,
    validate_author_proposal,
)
from menagerie.crawler.tests.conftest import (
    attach_paper_evidence,
    bind_handoff_execution,
    make_author_proposal,
)

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
    attach_paper_evidence(proposal, manifest, tmp_path)
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


def _strip_paper_evidence(proposal: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Reduce a grounded fixture to the historical code-only evidence set.

    This is the exact shape every R1/R2 model had before source triage pinned the
    paper: implementation bytes in the manifest, and the citation grounded on the
    code's own docstring-style mention.

    Parameters
    ----------
    proposal, manifest:
        Grounded proposal and controlled-fetch manifest mutated in place.
    """

    facts = proposal["proposed_facts"]
    facts["source_resolution"]["sources"] = [
        source
        for source in facts["source_resolution"]["sources"]
        if source["source_id"] != "source-paper"
    ]
    facts["evidence"]["excerpts"] = [
        excerpt
        for excerpt in facts["evidence"]["excerpts"]
        if excerpt["source_id"] != "source-paper"
    ]
    facts["citation"]["source_evidence_ids"] = ["evidence-1"]
    facts["external_metadata"]["citation"]["source_evidence_ids"] = ["evidence-1"]
    manifest["sources"] = [
        source for source in manifest["sources"] if source["source_id"] != "source-paper"
    ]
    manifest["manifest_sha256"] = stable_hash(manifest["sources"])
    proposal["verified_hashes"]["source_manifest"] = manifest["manifest_sha256"]


def test_citation_without_a_fetched_paper_source_is_refused(tmp_path: Path) -> None:
    """Paper metadata cannot be grounded in an implementation-only evidence set.

    Twenty-seven claims are gated on the frozen manifest, and the provenance half of
    them lives in the paper, not the code. An author whose manifest holds only code is
    structurally unable to ground them, so the gate demands the paper itself.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _strip_paper_evidence(proposal, manifest)
    with pytest.raises(ProposalValidationError, match="introducing paper"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_citation_grounded_only_in_implementation_code_is_refused(tmp_path: Path) -> None:
    """Fetching the paper is not enough; the citation must be grounded on its bytes."""

    proposal, manifest = _ground_proposal(tmp_path)
    facts = proposal["proposed_facts"]
    paper_excerpt = next(
        excerpt
        for excerpt in facts["evidence"]["excerpts"]
        if excerpt["source_id"] == "source-paper"
    )
    paper_excerpt["supports"] = ["external_metadata.venue"]
    with pytest.raises(ProposalValidationError, match="controlled-fetched paper source"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_fabricated_citation_title_is_still_refused(tmp_path: Path) -> None:
    """Adding the paper to the evidence set does not soften the accuracy gate.

    The paper is now present and fetched, and the citation is bound to its excerpt --
    every structural requirement is satisfied. An invented title must still fail,
    because the excerpt does not contain it.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["title"] = "Imaginary Hypernetwork Transformer"
    with pytest.raises(ProposalValidationError, match="not grounded verbatim.*title"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_fabricated_citation_year_is_still_refused(tmp_path: Path) -> None:
    """A year absent from the fetched paper text is not grounded."""

    proposal, manifest = _ground_proposal(tmp_path)
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["year"] = 1997
    with pytest.raises(ProposalValidationError, match="not grounded verbatim.*year"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_declared_arxiv_identifier_must_occur_in_the_cited_text(tmp_path: Path) -> None:
    """A resolvable identifier is an exact anchor and is required when declared.

    ``1905.09791`` is strictly more checkable than title-token overlap, so declaring
    one and failing to show it in the fetched paper text is a grounding gap.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["arxiv_id"] = "1905.09791"
    with pytest.raises(ProposalValidationError, match="not grounded verbatim.*arxiv_id"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_arxiv_identifier_present_in_the_fetched_paper_grounds_the_citation(
    tmp_path: Path,
) -> None:
    """The pykeen shape: a citation keyed on an arXiv ID the paper page carries."""

    proposal, manifest = _ground_proposal(tmp_path)
    text = (
        "arXiv:1905.09791. Example Model. A. Author, Example Lab, US. "
        "Published at TestConf in 2020."
    )
    _strip_paper_evidence(proposal, manifest)
    attach_paper_evidence(proposal, manifest, tmp_path, text=text, source_id="source-arxiv")
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["arxiv_id"] = "1905.09791"
    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )
    assert report.rung.value == "R1_LIBRARY"


def _r1_recipe(proposal: dict[str, Any]) -> dict[str, Any]:
    """Return the mutable R1 library recipe of a grounded proposal fixture."""

    recipe = proposal["proposed_facts"]["implementation"]["library_recipe"]
    assert isinstance(recipe, dict)
    return recipe


@pytest.mark.smoke
def test_r1_with_no_pretrained_capable_constructor_passes_on_assertion(
    tmp_path: Path,
) -> None:
    """The pilot's most common R1 shape is satisfiable again, honestly.

    ``MiniMaxForCausalLM(config)``, ``TAGConv``, and ``DiehlAndCook2015v2`` all
    died on ``R1_LIBRARY must explicitly disable pretrained fields`` because
    their constructors expose nothing to disable. The checked positive
    assertion replaces that wall: an empty disable list plus
    ``pretrained_fields_absent: true`` passes the proposal gate, and the
    assertion is re-verified against the real constructor signature at load.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    recipe = _r1_recipe(proposal)
    recipe["kwargs"] = {"hidden_size": 8}
    recipe["pretrained_disable_fields"] = []
    recipe["pretrained_fields_absent"] = True
    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )
    assert report.rung.value == "R1_LIBRARY"


@pytest.mark.smoke
def test_r1_silence_about_pretrained_fields_is_still_refused(tmp_path: Path) -> None:
    """An empty disable list with no assertion remains the undeclared state.

    The wall came from conflating "nothing to disable" with "did not think
    about it"; only the first is now expressible, and silence still refuses.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    recipe = _r1_recipe(proposal)
    recipe["kwargs"] = {"hidden_size": 8}
    recipe["pretrained_disable_fields"] = []
    with pytest.raises(ProposalValidationError, match="must declare its pretrained disposition"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


@pytest.mark.smoke
def test_r1_enabled_pretrained_flag_is_still_refused(tmp_path: Path) -> None:
    """The protection the rule exists for fires unchanged: enabling refuses."""

    proposal, manifest = _ground_proposal(tmp_path)
    recipe = _r1_recipe(proposal)
    recipe["kwargs"] = {"weights": "IMAGENET1K_V1"}
    recipe["pretrained_disable_fields"] = ["weights"]
    with pytest.raises(ProposalValidationError, match="does not carry a disabling value"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


@pytest.mark.smoke
def test_r1_unlisted_pretrained_capable_kwarg_is_refused(tmp_path: Path) -> None:
    """A pretrained-capable kwargs key cannot ride through unlisted.

    Listing a harmless disabled field beside an enabling ``weights`` value used
    to satisfy the non-empty rule; the kwargs scan now refuses the dodge.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    recipe = _r1_recipe(proposal)
    recipe["kwargs"] = {"weights": "IMAGENET1K_V1", "progress": False}
    recipe["pretrained_disable_fields"] = ["progress"]
    with pytest.raises(ProposalValidationError, match="leave known pretrained keywords enabled"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)
    # Even a DISABLING value on a known key must be declared where readers look.
    recipe["kwargs"] = {"weights": None, "progress": False}
    with pytest.raises(ProposalValidationError, match="pretrained-capable keys"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


@pytest.mark.smoke
def test_r1_assertion_beside_disable_fields_is_refused_as_contradiction(
    tmp_path: Path,
) -> None:
    """Both declarations at once answer the same question twice and refuse."""

    proposal, manifest = _ground_proposal(tmp_path)
    recipe = _r1_recipe(proposal)
    recipe["pretrained_fields_absent"] = True
    with pytest.raises(ProposalValidationError, match="contradicts"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def _blank_search_report(proposal: dict[str, Any]) -> None:
    """Remove the bounded search that would justify an unanswered field."""

    proposal["proposed_facts"]["source_resolution"]["search_report"]["queries"] = []


def test_fabricated_citation_authors_venue_and_bibtex_are_refused_per_leaf(
    tmp_path: Path,
) -> None:
    """A citation with fabricated authors, venue, and BibTeX no longer passes.

    The old matcher checked title+year only, so every other leaf was free to invent.
    Every positive leaf is now grounded against the fetched paper bytes, and BibTeX
    must be exactly consistent with the grounded title/year/authors.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation.update(
            {
                "authors": ["J. Fabricated", "N. Invented"],
                "venue": "NeurIPS",
                "bibtex": (
                    "@inproceedings{fabricated2020, title={A Different Paper Entirely}, "
                    "author={Fabricated, J.}, booktitle={NeurIPS}, year={2019}}"
                ),
            }
        )
    with pytest.raises(
        ProposalValidationError,
        match="not grounded verbatim.*authors.*venue.*do not agree.*bibtex",
    ):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_fabricated_bibtex_alone_is_refused_by_consistency(tmp_path: Path) -> None:
    """A BibTeX entry for a different work fails against the grounded leaves.

    The refusal must name the check BibTeX actually failed. A constructed record is
    never quoted by the paper it cites, so reporting it as "not grounded verbatim in
    the fetched paper text" would send the author after an excerpt that cannot exist.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["bibtex"] = (
            "@article{other2019, title={A Different Paper Entirely}, "
            "author={Somebody, Else}, year={2019}}"
        )
    with pytest.raises(
        ProposalValidationError, match="do not agree with the grounded title.*bibtex"
    ) as refusal:
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)
    assert "not grounded verbatim" not in str(refusal.value)


def test_honest_bibtex_consistent_with_grounded_leaves_passes(tmp_path: Path) -> None:
    """An honest BibTeX carrying the grounded title, year, and authors is accepted."""

    proposal, manifest = _ground_proposal(tmp_path)
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["bibtex"] = (
            "@inproceedings{author2020example, title={Example Model}, "
            "author={Author, A.}, booktitle={TestConf}, year={2020}}"
        )
    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )
    assert report.rung.value == "R1_LIBRARY"


ACCENTED_PAPER_TEXT = (
    "Example Model. Lélio Renard Lavaud, Théophile Gervet, Example Lab, US. "
    "Published at TestConf in 2020."
)


def _accented_citation(
    proposal: dict[str, Any], bibtex: str, authors: list[str] | None = None
) -> None:
    """Point both citation copies at the accented-author fixture.

    Parameters
    ----------
    proposal:
        Author proposal mutated in place.
    bibtex:
        Constructed BibTeX record under test.
    authors:
        Claimed author list; defaults to the two accented names in the paper text.
    """

    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["authors"] = list(
            authors if authors is not None else ["Lélio Renard Lavaud", "Théophile Gervet"]
        )
        citation["bibtex"] = bibtex


def test_bibtex_spelling_accents_as_tex_escapes_still_grounds_the_same_authors(
    tmp_path: Path,
) -> None:
    """The m5915 shape: an honest BibTeX whose accented names use TeX escapes.

    ``L\\'elio`` is the only spelling BibTeX has for ``Lélio``, and the author is
    obliged to write it that way. Before the canonicalizer decoded TeX escapes the
    backslash and quote were dropped as punctuation, the name tokenized to ``l`` plus
    ``elio`` instead of ``lelio``, and a correct entry naming exactly the grounded
    authors was refused as if it cited a different work. Refusing this is refusing a
    requirement no honest author can satisfy.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _strip_paper_evidence(proposal, manifest)
    attach_paper_evidence(
        proposal, manifest, tmp_path, text=ACCENTED_PAPER_TEXT, source_id="source-arxiv"
    )
    _accented_citation(
        proposal,
        "@inproceedings{lavaud2020example, title={Example Model}, "
        "author={Renard Lavaud, L\\'elio and Gervet, Th\\'eophile}, "
        "booktitle={TestConf}, year={2020}}",
    )
    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )
    assert report.rung.value == "R1_LIBRARY"


def test_tex_escapes_cannot_launder_a_bibtex_naming_different_authors(
    tmp_path: Path,
) -> None:
    """Decoding TeX escapes must not become a hole a fabricated entry fits through.

    The escape decoder only rejoins a token the TeX syntax split. An entry whose author
    list, fully decoded, names people the grounded citation does not claim is still an
    entry for a different work, and is still refused.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _strip_paper_evidence(proposal, manifest)
    attach_paper_evidence(
        proposal, manifest, tmp_path, text=ACCENTED_PAPER_TEXT, source_id="source-arxiv"
    )
    _accented_citation(
        proposal,
        "@inproceedings{other2020, title={Example Model}, "
        "author={Renard Lavaud, S\\'ebastien and Gervet, Ana\\\"is}, "
        "booktitle={TestConf}, year={2020}}",
    )
    with pytest.raises(
        ProposalValidationError, match="do not agree with the grounded title.*bibtex"
    ):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_tex_escaped_bibtex_still_fails_against_a_fabricated_author_claim(
    tmp_path: Path,
) -> None:
    """An author the paper never names is refused before BibTeX is even consulted."""

    proposal, manifest = _ground_proposal(tmp_path)
    _strip_paper_evidence(proposal, manifest)
    attach_paper_evidence(
        proposal, manifest, tmp_path, text=ACCENTED_PAPER_TEXT, source_id="source-arxiv"
    )
    _accented_citation(
        proposal,
        "@inproceedings{ghost2020, title={Example Model}, "
        "author={Ghostwriter, S\\'ebastien}, booktitle={TestConf}, year={2020}}",
        authors=["Sébastien Ghostwriter"],
    )
    with pytest.raises(ProposalValidationError, match="not grounded verbatim.*authors"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_tex_escape_decoding_is_symmetric_with_the_unicode_spelling() -> None:
    """Both spellings of one name must canonicalize to the same tokens.

    Symmetry is the whole property: the decoder emits the precomposed character the
    escape denotes, so the existing NFKD fold reduces it exactly as it reduces the
    character typed directly. ``\\l`` and ``ł`` matter separately because NFKD does not
    decompose a barred l, so decoding it to a bare ``l`` would have been the asymmetry
    in the other direction.
    """

    for escaped, unicode_spelling in (
        ("L\\'elio", "Lélio"),
        ("Th\\'eophile", "Théophile"),
        ("Timoth\\'ee", "Timothée"),
        ('Ana\\"is', "Anaïs"),
        ("Fran\\c{c}ois", "François"),
        ("Erd\\H{o}s", "Erdős"),
        ("{\\L}ukasz", "Łukasz"),
        ("Wei\\ss{}", "Weiß"),
        ("Sm\\o{}rrebr\\o{}d", "Smørrebrød"),
    ):
        assert _normalize_support_text(escaped) == _normalize_support_text(unicode_spelling)


def test_tex_escape_decoding_leaves_unrelated_backslash_commands_alone() -> None:
    """Only accent and special-letter commands decode; nothing else is interpreted."""

    assert _normalize_support_text("\\varphi \\ref{fig:1} \\dots") == "varphi ref fig 1 dots"


def test_declared_arxiv_id_present_on_the_page_but_absent_from_the_excerpt_is_refused(
    tmp_path: Path,
) -> None:
    """The m8245 shape: the grounding text was available and was not excerpted.

    The MetaFormer author declared ``arxiv_id`` and bound citation-header excerpts
    (``citation_title``, ``citation_author``) that carry the title and authors but not
    the identifier -- which occurs twenty-eight times elsewhere on the same fetched
    page. The requirement is satisfiable from the bytes the author already held, so the
    refusal is correct and the remedy is one more excerpt, never a looser check. Both
    halves are asserted here so the pair cannot drift apart.
    """

    page = (
        '<meta name="citation_title" content="Example Model" />'
        '<meta name="citation_author" content="Author, A." />'
        "Published at TestConf in 2020. [Submitted 2020] arXiv:1905.09791"
    )
    header_only, with_identifier = page.split("[Submitted 2020] ")

    proposal, manifest = _ground_proposal(tmp_path)
    _strip_paper_evidence(proposal, manifest)
    attach_paper_evidence(
        proposal, manifest, tmp_path, text=header_only, source_id="source-arxiv"
    )
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["arxiv_id"] = "1905.09791"
    with pytest.raises(ProposalValidationError, match="not grounded verbatim.*arxiv_id"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)

    grounded, grounded_manifest = _ground_proposal(tmp_path)
    _strip_paper_evidence(grounded, grounded_manifest)
    attach_paper_evidence(
        grounded,
        grounded_manifest,
        tmp_path,
        text=header_only + with_identifier,
        source_id="source-arxiv",
    )
    for citation in (
        grounded["proposed_facts"]["citation"],
        grounded["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["arxiv_id"] = "1905.09791"
    report = validate_author_proposal(
        grounded, allowed_model_dir=tmp_path, source_manifest=grounded_manifest
    )
    assert report.rung.value == "R1_LIBRARY"


#: The fixture header that grounds every citation leaf except the identifier, so the
#: identifier is the only variable in the tests below.
_CITATION_HEADER_TEXT = (
    '<meta name="citation_title" content="Example Model" />'
    '<meta name="citation_author" content="Author, A." />'
    "Published at TestConf in 2020."
)
#: Line 148 of the arXiv abstract page the pilot crawler actually froze for model m8245
#: (source ``paper-arxiv-abs``, content sha256
#: ``18e4beca562216cce440a60aab0d736f4fb0b75606631d54564d8cea807f7ac0``), quoted byte for
#: byte. It is the real shape the bug is about: the identifier appears on this line only
#: with its revision selector attached.
M8245_VERSIONED_ONLY_LINE = (
    '              <a href="https://arxiv.org/abs/2111.11418v3">arXiv:2111.11418v3</a>'
    " [cs.CV]</span> for this version)\n"
)
#: The two excerpts model m8245's recorded proposal actually bound to its citation claim
#: (``ev-arxiv-citation`` and ``ev-arxiv-venue``), quoted byte for byte from the archived
#: author result. Both are honest quotes of the frozen page and neither names the
#: identifier, which occurs twenty-eight times elsewhere in the same fetched bytes.
M8245_RECORDED_CITATION_EXCERPTS = (
    '<meta name="citation_title" content="MetaFormer Is Actually What You Need for'
    ' Vision" /><meta name="citation_author" content="Yu, Weihao" />'
    '<meta name="citation_author" content="Luo, Mi" />'
    '<meta name="citation_author" content="Zhou, Pan" />'
    '<meta name="citation_author" content="Si, Chenyang" />'
    '<meta name="citation_author" content="Zhou, Yichen" />'
    '<meta name="citation_author" content="Wang, Xinchao" />'
    '<meta name="citation_author" content="Feng, Jiashi" />'
    '<meta name="citation_author" content="Yan, Shuicheng" />'
    '<meta name="citation_date" content="2021/11/22" />'
    '<meta name="citation_online_date" content="2022/07/04" />',
    '          <td class="tablecell comments mathjax">CVPR 2022 (Oral). Code: '
    '<a href="https://github.com/sail-sg/poolformer" rel="external noopener nofollow"'
    ' class="link-external link-https">this https URL</a></td>\n',
)


def _citation_against(
    tmp_path: Path, text: str, arxiv_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a grounded proposal whose paper evidence is exactly ``text``.

    Parameters
    ----------
    tmp_path:
        Per-test temporary directory.
    text:
        Verbatim paper-excerpt text to bind as the sole citation evidence.
    arxiv_id:
        Identifier both citation copies declare.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Proposal and source manifest ready for validation.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _strip_paper_evidence(proposal, manifest)
    attach_paper_evidence(proposal, manifest, tmp_path, text=text, source_id="source-arxiv")
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["arxiv_id"] = arxiv_id
    return proposal, manifest


def test_versioned_only_arxiv_mention_grounds_the_unversioned_declared_id(
    tmp_path: Path,
) -> None:
    """A real page line naming ``2111.11418v3`` grounds a declared ``2111.11418``.

    The canonicalizer reduces ``2111.11418`` to the token pair ``2111 11418``, but the
    revision selector fuses into the trailing token, so the phrase was no longer
    contiguous and a true claim was refused. The authoring stage runs once per model, so
    that refusal is a permanent dead record. The bytes here are the arXiv abstract line
    the pilot crawler actually froze for m8245, not a fixture invented to pass.
    """

    assert "2111.11418v3" in M8245_VERSIONED_ONLY_LINE
    assert "2111.11418<" not in M8245_VERSIONED_ONLY_LINE
    proposal, manifest = _citation_against(
        tmp_path, _CITATION_HEADER_TEXT + M8245_VERSIONED_ONLY_LINE, "2111.11418"
    )
    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )
    assert report.rung.value == "R1_LIBRARY"


def test_m8245_recorded_citation_excerpts_still_do_not_ground_its_arxiv_id(
    tmp_path: Path,
) -> None:
    """The archived m8245 proposal stays refused; the version fix must not reach it.

    Its two bound excerpts are honest quotes that simply do not contain the identifier,
    and the remedy was always one more excerpt. If the version-suffix tolerance ever
    accepts this pair it has stopped being an identifier rule.
    """

    combined = _normalize_support_text("\n".join(M8245_RECORDED_CITATION_EXCERPTS))
    assert not _identifier_grounded("2111.11418", combined)

    proposal, manifest = _citation_against(
        tmp_path,
        _CITATION_HEADER_TEXT + "".join(M8245_RECORDED_CITATION_EXCERPTS),
        "2111.11418",
    )
    with pytest.raises(ProposalValidationError, match="not grounded verbatim.*arxiv_id"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_a_bare_page_mention_does_not_ground_a_declared_arxiv_version(
    tmp_path: Path,
) -> None:
    """The weaker direction is deliberately not granted.

    An arXiv id names a work and ``vN`` names one revision of it, so a versioned mention
    has necessarily named the work. A bare mention has not established that a third
    revision exists, so declaring ``v3`` asserts strictly more than the excerpt shows.
    """

    proposal, manifest = _citation_against(
        tmp_path,
        _CITATION_HEADER_TEXT + " Cited as arXiv:2111.11418 [cs.CV]",
        "2111.11418v3",
    )
    with pytest.raises(ProposalValidationError, match="not grounded verbatim.*arxiv_id"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_a_numerically_extended_arxiv_id_cannot_launder_either_way(
    tmp_path: Path,
) -> None:
    """The tolerance consumes ``v`` plus digits or nothing -- never loose trailing text.

    A general "ignore what follows" rule would make ``2111.114189`` ground
    ``2111.11418``, which is a different identifier entirely. Both directions are
    asserted so neither can regress into a laundering path.
    """

    extended = _normalize_support_text("Cited as arXiv:2111.114189 [cs.CV]")
    assert not _identifier_grounded("2111.11418", extended)
    base = _normalize_support_text("Cited as arXiv:2111.11418 [cs.CV]")
    assert not _identifier_grounded("2111.114189", base)

    proposal, manifest = _citation_against(
        tmp_path,
        _CITATION_HEADER_TEXT + " Cited as arXiv:2111.114189 [cs.CV]",
        "2111.11418",
    )
    with pytest.raises(ProposalValidationError, match="not grounded verbatim.*arxiv_id"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_a_fabricated_arxiv_id_is_refused_against_the_real_page_line(
    tmp_path: Path,
) -> None:
    """An identifier appearing nowhere in the bound text is still refused."""

    proposal, manifest = _citation_against(
        tmp_path, _CITATION_HEADER_TEXT + M8245_VERSIONED_ONLY_LINE, "2199.99999"
    )
    with pytest.raises(ProposalValidationError, match="not grounded verbatim.*arxiv_id"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_doi_and_openreview_identifiers_get_no_version_tolerance() -> None:
    """Only arXiv's grammar licenses the widening; the opaque identifiers keep exact match.

    A DOI suffix and an OpenReview id are registrant-chosen opaque strings, so
    ``10.1234/xyzv2`` may be a separately registered work rather than a revision of
    ``10.1234/xyz``. There is no grammar that could tell the two apart, so no near
    variant may ground either. Exact occurrence still grounds them, including inside the
    URL and query-string forms real pages print.
    """

    assert not _identifier_grounded(
        "10.1234/xyz", _normalize_support_text("https://doi.org/10.1234/xyzv2")
    )
    assert _identifier_grounded(
        "10.1234/xyz", _normalize_support_text("https://doi.org/10.1234/xyz")
    )
    assert not _identifier_grounded("SygXPaEYv", _normalize_support_text("?id=SygXPaEYvH"))
    assert _identifier_grounded(
        "SygXPaEYvH", _normalize_support_text("https://openreview.net/forum?id=SygXPaEYvH&noteId=x")
    )


def test_the_old_style_arxiv_identifier_gets_the_same_one_way_tolerance() -> None:
    """Pre-2007 ``archive[.SS]/YYMMNNN`` ids carry the same revision selector."""

    assert _identifier_grounded(
        "math.GT/0309136", _normalize_support_text("arXiv:math.GT/0309136v2")
    )
    assert not _identifier_grounded(
        "math.GT/0309136v2", _normalize_support_text("arXiv:math.GT/0309136")
    )


def test_omitting_the_citation_while_the_paper_is_bound_is_refused(tmp_path: Path) -> None:
    """With the introducing paper fetched, "no citation" is a checkable false claim.

    The citation is gated whenever a paper source is bound, not only when the author
    volunteers one -- supplying a true fact must never be what triggers the check.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation.update(
            {
                "status": "not-found-after-search",
                "title": None,
                "authors": [],
                "year": None,
                "venue": None,
                "url": None,
            }
        )
    with pytest.raises(ProposalValidationError, match="controlled-fetched source"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_taxonomy_leaves_are_gated_individually_never_as_an_aggregate(
    tmp_path: Path,
) -> None:
    """A taxonomy with unsupported leaves no longer rides through on its family name.

    The old aggregate claim passed when the family plus ANY one scalar matched. Every
    taxonomy leaf now requires its own excerpt binding; an aggregate ``taxonomy``
    support string covers nothing.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    for excerpt in proposal["proposed_facts"]["evidence"]["excerpts"]:
        excerpt["supports"] = [
            support for support in excerpt["supports"] if not support.startswith("taxonomy.")
        ] + ["taxonomy"]
    with pytest.raises(ProposalValidationError, match="ungrounded claim categories.*taxonomy"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_partially_supported_taxonomy_is_refused(tmp_path: Path) -> None:
    """Binding only the family leaf leaves every other taxonomy leaf ungrounded."""

    proposal, manifest = _ground_proposal(tmp_path)
    for excerpt in proposal["proposed_facts"]["evidence"]["excerpts"]:
        excerpt["supports"] = [
            support
            for support in excerpt["supports"]
            if not support.startswith("taxonomy.") or support == "taxonomy.family"
        ]
    with pytest.raises(
        ProposalValidationError, match="ungrounded claim categories.*taxonomy.domains"
    ):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_judgment_claims_move_to_the_checker_and_the_token_oracle_is_deleted() -> None:
    """The token-entailment matcher is deleted, not tuned.

    Measured against real prose, ``country = "US"`` passed by matching the English
    pronoun "us", while ``GB``, ``CN``, and ``DE`` could never pass however correct the
    evidence was, and a fabricated citation passed on title+year alone. Judgment claims
    keep their provenance requirement and gain per-leaf checker verdicts; the only
    deterministically value-matched claim left is the citation, per-leaf against the
    fetched paper bytes.
    """

    from menagerie.crawler import proposal as proposal_module

    assert not hasattr(proposal_module, "_text_supports_claim")
    assert not hasattr(proposal_module, "_scalar_matches")
    assert not hasattr(proposal_module, "_significant_tokens")
    assert CHECKER_EVALUATED_CLAIMS == DEFAULT_GATED_CLAIMS - {"external_metadata.citation"}
    assert VALUE_MATCHED_CLAIMS == {"external_metadata.citation"}
    assert "external_metadata.country" in CHECKER_EVALUATED_CLAIMS
    assert "external_metadata.era" in CHECKER_EVALUATED_CLAIMS


def test_country_iso_codes_pass_with_real_supporting_evidence(tmp_path: Path) -> None:
    """``GB``/``CN``/``DE``/``UK`` are no longer structurally impossible.

    Under token overlap these codes could never occur in honest prose, so a correct
    claim was refused however good the evidence; the deterministic layer now checks
    provenance and structure, and the checker judges the value.
    """

    for code in ("GB", "CN", "DE", "UK"):
        proposal, manifest = _ground_proposal(tmp_path)
        proposal["proposed_facts"]["external_metadata"]["country"] = code
        report = validate_author_proposal(
            proposal, allowed_model_dir=tmp_path, source_manifest=manifest
        )
        assert report.rung.value == "R1_LIBRARY"


@pytest.mark.parametrize("field", ["country", "era"])
def test_checker_evaluated_fields_are_still_mandatory_at_the_accuracy_gate(field: str) -> None:
    """The claim the excerpt matcher stopped judging is still gated, by the checker."""

    from menagerie.crawler.metadata import (
        MANDATORY_EXTERNAL_FIELDS,
        MetadataValidationError,
        _validate_external_field_checks,
    )

    checks = [
        {"field": f"external_metadata.{name}", "verdict": "accurate"}
        for name in MANDATORY_EXTERNAL_FIELDS
        if name != field
    ]
    with pytest.raises(MetadataValidationError, match="ungated mandatory external metadata"):
        _validate_external_field_checks(checks)


def test_checker_evaluated_claims_still_require_evidence_provenance(tmp_path: Path) -> None:
    """Moving the value verdict does not remove the requirement to cite a source."""

    proposal, manifest = _ground_proposal(tmp_path)
    excerpt = proposal["proposed_facts"]["evidence"]["excerpts"][0]
    excerpt["supports"] = [
        support for support in excerpt["supports"] if support != "external_metadata.country"
    ]
    with pytest.raises(ProposalValidationError, match="ungrounded claim categories"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_diacritics_no_longer_break_a_correct_author_claim(tmp_path: Path) -> None:
    """The same author spelled two real ways must ground the same claim.

    ``Balazevic`` and ``Balazevic`` with diacritics previously normalized to different
    token sets, so a correct claim failed exactly as if it had been fabricated.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    text = "Example Model. Ivana Balažević, Example Lab, US. Published at TestConf in 2020."
    attach_paper_evidence(proposal, manifest, tmp_path, text=text, source_id="source-authors")
    facts = proposal["proposed_facts"]
    excerpt = next(
        item for item in facts["evidence"]["excerpts"] if item["source_id"] == "source-authors"
    )
    excerpt["supports"] = ["external_metadata.authors", "external_metadata.citation"]
    facts["external_metadata"]["authors"] = ["Ivana Balazevic"]

    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )

    assert report.rung.value == "R1_LIBRARY"


def test_omitting_the_citation_without_a_recorded_search_is_refused(tmp_path: Path) -> None:
    """Declaring "no paper" must cost the search that establishes it.

    The gate returned early on any non-present citation, so omission was the cheapest
    way past every check. A campaign that completes with its citations silently blank
    is worse than one that stops, because nothing surfaces it.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _strip_paper_evidence(proposal, manifest)
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation.update(
            {
                "status": "not-found-after-search",
                "title": None,
                "authors": [],
                "year": None,
                "venue": None,
                "url": None,
            }
        )
    _blank_search_report(proposal)
    with pytest.raises(ProposalValidationError, match="bounded search"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def _declare_absent(
    proposal: dict[str, Any],
    field: str,
    *,
    status: str = "not-found-after-search",
    basis: str = "search-exhausted",
) -> None:
    """Empty one gated field and declare its typed availability state."""

    metadata = proposal["proposed_facts"]["external_metadata"]
    metadata[field] = [] if isinstance(metadata[field], list) else None
    metadata.setdefault("availability", {})[field] = {
        "status": status,
        "values": [],
        "basis": basis,
        "evidence": [],
    }


@pytest.mark.parametrize(
    ("field", "empty_value"),
    [("venue", None), ("authors", []), ("institution", []), ("country", None)],
)
def test_bare_null_or_empty_gated_claim_is_refused(
    tmp_path: Path, field: str, empty_value: object
) -> None:
    """A bare null or empty list never passes a gated claim again.

    The old ``value is None -> passes`` arm made omission the cheapest route past the
    gate, silently emptying exactly the fields the catalog exists to collect.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    proposal["proposed_facts"]["external_metadata"][field] = empty_value
    with pytest.raises(ProposalValidationError, match="bare null/empty"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_honestly_unknown_authors_and_institution_are_representable(tmp_path: Path) -> None:
    """An honest unknown is an explicit, searched, typed state -- and it passes.

    The schema previously forbade empty ``authors``/``institution`` while the gate
    rewarded nulls elsewhere, leaving fabrication as the only path for a fact that
    genuinely is not findable. The availability state fixes both halves.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _strip_paper_evidence(proposal, manifest)
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation.update(
            {
                "status": "not-found-after-search",
                "title": None,
                "authors": [],
                "year": None,
                "venue": None,
                "url": None,
            }
        )
    for field in ("authors", "institution", "venue", "year"):
        _declare_absent(proposal, field)

    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )

    assert report.rung.value == "R1_LIBRARY"


def test_not_found_availability_without_a_recorded_search_is_refused(tmp_path: Path) -> None:
    """Claiming not-found still costs the bounded search that establishes it."""

    proposal, manifest = _ground_proposal(tmp_path)
    _declare_absent(proposal, "venue")
    _blank_search_report(proposal)
    with pytest.raises(ProposalValidationError, match="recorded bounded search"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_availability_state_contradicting_a_carried_value_is_refused(tmp_path: Path) -> None:
    """A declared absence over a present value is a false claim, not bookkeeping."""

    proposal, manifest = _ground_proposal(tmp_path)
    metadata = proposal["proposed_facts"]["external_metadata"]
    metadata.setdefault("availability", {})["venue"] = {
        "status": "not-found-after-search",
        "values": [],
        "basis": "search-exhausted",
        "evidence": [],
    }
    with pytest.raises(ProposalValidationError, match="carries a value"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_availability_state_with_noncanonical_vocabulary_is_refused(tmp_path: Path) -> None:
    """Status and basis are closed vocabularies, not free text.

    The refusal moved EARLIER, not away: ``basis`` is now a declared ``enum`` in the
    registered schema rather than a ``nonempty_string`` whose description merely
    promised a closed vocabulary, so payload validation rejects a coined value before
    ``_validate_availability_record`` is reached. Both layers still refuse -- the
    deterministic guard is exercised directly in
    ``test_claim_vocabulary_and_absence`` -- and the message must now ENUMERATE the
    members, which is the whole point of the move. m8245 and m9617 each lost a model to
    a coined basis that no surface the author reads ever listed.
    """

    from menagerie.crawler.metadata import AVAILABILITY_BASES

    proposal, manifest = _ground_proposal(tmp_path)
    _declare_absent(proposal, "venue", basis="vibes")
    with pytest.raises(ProposalValidationError) as excinfo:
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)
    message = str(excinfo.value)
    assert "'vibes'" in message, "the rejected spelling must still be named"
    assert "availability.venue.basis" in message, "the refusal must locate the exact leaf"
    for basis in AVAILABILITY_BASES:
        assert basis in message, f"the refusal must enumerate the member {basis!r}"


def test_availability_state_citing_fabricated_evidence_is_refused(tmp_path: Path) -> None:
    """Availability evidence IDs must name real hash-verified excerpts."""

    proposal, manifest = _ground_proposal(tmp_path)
    _declare_absent(proposal, "venue")
    proposal["proposed_facts"]["external_metadata"]["availability"]["venue"]["evidence"] = [
        "invented"
    ]
    with pytest.raises(ProposalValidationError, match="missing or fabricated evidence"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_genuine_model_named_with_antislop_vocabulary_passes(tmp_path: Path) -> None:
    """A model whose NAME is the banned vocabulary can be honestly described.

    Roughly 32 real roster models -- surrogate-gradient SNNs, Neural Mesh
    Simplification, approximate message passing -- are named with words the anti-slop
    list bans, so their honest descriptions could never pass. The pattern list is
    scoped to implementation-fidelity surfaces where the words mean what the tripwire
    thinks they mean.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    metadata = proposal["proposed_facts"]["external_metadata"]
    website = proposal["proposed_facts"]["website"]
    honest = (
        "Neural Mesh Simplification is a surrogate-gradient spiking network for "
        "approximate message passing over simplified meshes; it is a small "
        "source-grounded example network."
    )
    metadata["description"] = honest
    website["description"] = honest
    website["tagline"] = "A surrogate-gradient mesh simplification proxy model"

    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )

    assert report.rung.value == "R1_LIBRARY"


def test_antislop_vocabulary_on_fidelity_surfaces_is_still_refused(tmp_path: Path) -> None:
    """The tripwire the rescope preserves: approximation admissions about the CODE."""

    proposal, manifest = _ground_proposal(tmp_path)
    proposal["proposed_facts"]["fidelity"]["reason"] = (
        "the staged code is a simplified approximation of the published architecture"
    )
    with pytest.raises(ProposalValidationError, match="forbidden approximation language"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_diacritic_citation_authors_ground_in_both_directions(tmp_path: Path) -> None:
    """ASCII and diacritic spellings of one author name ground each other."""

    for claimed, published in (
        ("Ivana Balazevic", "Ivana Balažević"),
        ("Ivana Balažević", "Ivana Balazevic"),
    ):
        proposal, manifest = _ground_proposal(tmp_path)
        text = f"Example Model. {published}, Example Lab, US. Published at TestConf in 2020."
        _strip_paper_evidence(proposal, manifest)
        source_id = f"source-{abs(hash((claimed, published)))}"
        attach_paper_evidence(proposal, manifest, tmp_path, text=text, source_id=source_id)
        for citation in (
            proposal["proposed_facts"]["citation"],
            proposal["proposed_facts"]["external_metadata"]["citation"],
        ):
            citation["authors"] = [claimed]
        report = validate_author_proposal(
            proposal, allowed_model_dir=tmp_path, source_manifest=manifest
        )
        assert report.rung.value == "R1_LIBRARY"


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
                "reason_code": "missing-mandatory-link",
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


def _parse_blocked_result(
    tmp_path: Path, *, stage: str, reason_code: str
) -> BlockedRecommendation:
    """Drive one complete BLOCKED author result through the production parser.

    Everything except ``stage``/``reason_code`` is exact, so nothing ahead of the reason
    vocabulary can short-circuit the parse: the arm reaches the blocked branch and that
    branch is what decides.

    Parameters
    ----------
    tmp_path:
        Isolated author result directory.
    stage, reason_code:
        Blocking claim under test.

    Returns
    -------
    BlockedRecommendation
        Parsed arm when the claim is admissible.
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
    payload: dict[str, Any] = {
        "arm": "BLOCKED",
        "stage": stage,
        "reason_code": reason_code,
        "prerequisite_ids": ["prerequisite-1"],
        "evidence_ids": ["evidence-1"],
        "evidence_identity": "sha256:" + "3" * 64,
        "license_identity": "sha256:" + "4" * 64,
    }
    if reason_code in {"needs-higher-tier", ACCESS_BLOCKED_REASON_CODE}:
        # The two deferrable arms carry a mandatory stage-1 research summary, enforced by
        # an earlier clause. Without it those probes would be refused before the reason
        # vocabulary ever decided, and would prove nothing about the vocabulary.
        payload["research_summary"] = {
            "queries": ["ExampleNet architecture implementation"],
            "places": ["upstream repositories", "introducing paper"],
            "candidate_links": [
                {
                    "url": "https://example.com/model.txt",
                    "why_rejected": "The material could not be adjudicated at this tier.",
                    "rejection_class": (
                        ACCESS_BARRIER_REJECTION_CLASS
                        if reason_code == ACCESS_BLOCKED_REASON_CODE
                        else "no-material-detail"
                    ),
                }
            ],
            "languages": ["English"],
            "conclusion": "The source is real; this campaign cannot author it faithfully.",
        }
    payload["recommendation_sha256"] = stable_hash(payload)
    raw = _author_result(envelope, "BLOCKED", payload)
    (tmp_path / "result.json").write_text(json.dumps(raw))
    parsed = validate_author_result(tmp_path / "result.json", envelope)
    assert isinstance(parsed, BlockedRecommendation)
    return parsed


#: Blocking stages read from the shipped schema rather than mirrored in Python, so a stage
#: added to the enum is covered here without a second list to keep in step.
_BLOCKED_SCHEMA_STAGES: tuple[str, ...] = tuple(
    json.loads(
        (Path(__file__).parents[1] / "schemas" / "author-result-v3.schema.json").read_text(
            encoding="utf-8"
        )
    )["$defs"]["blocked_payload"]["properties"]["stage"]["enum"]
)


def test_blocked_schema_stages_are_read_not_guessed() -> None:
    """The parametrized stage list really is the shipped schema enum, and is non-trivial."""

    assert "author" in _BLOCKED_SCHEMA_STAGES
    assert "source" in _BLOCKED_SCHEMA_STAGES
    assert len(_BLOCKED_SCHEMA_STAGES) >= 6


def test_effort_exhaustion_set_covers_every_exhaustion_code_a_blocked_arm_can_name() -> None:
    """No budget-exhaustion code reachable from a BLOCKED arm escapes the refused class.

    The sweep is scoped to the stages a BLOCKED arm may actually name. The remaining
    ``*-cap-exhausted`` codes are VERDICT caps, not budget outcomes: they say the checker
    rejected the model until its repair rounds ran out, which is a quality finding and a
    truthful thing to record. They live only on the ``accuracy-gate`` and ``fidelity``
    stages, which a BLOCKED arm cannot name, and they are deliberately NOT refused.
    """

    reachable = {
        code
        for stage in _BLOCKED_SCHEMA_STAGES
        for code in FAILURE_REASON_CODES.get(stage, frozenset())
        if code.startswith("effort-exhausted:") or "exhaust" in code or code == "wall-exceeded"
    }
    assert reachable, "the sweep found nothing, so it proves nothing"
    assert reachable <= EFFORT_EXHAUSTION_REASON_CODES

    verdict_caps = {
        "slop-cap-exhausted",
        "major-drift-cap-exhausted",
        "cannot-verify-cap-exhausted",
        "inaccurate-cap-exhausted",
    }
    assert not verdict_caps & reachable
    for stage in _BLOCKED_SCHEMA_STAGES:
        assert not verdict_caps & FAILURE_REASON_CODES.get(stage, frozenset())

    # The pilot's free-form spelling is not a vocabulary member at all, and is still refused.
    assert "budget-exhausted" in EFFORT_EXHAUSTION_REASON_CODES
    assert "budget-exhausted" not in reachable


def test_every_exhausting_stage_has_a_stage_valid_exhaustion_terminal() -> None:
    """The refused-claim terminal must be a reason its own stage actually admits.

    ``author`` carries the ``effort-exhausted:`` family rather than
    ``effort-cap-exhausted``, so a single hardcoded reason would record an invalid pair.
    """

    assert EXHAUSTION_TERMINAL_REASON_BY_STAGE["author"] == "effort-exhausted:wall-seconds"
    assert EXHAUSTION_TERMINAL_REASON_BY_STAGE["source"] == "effort-cap-exhausted"
    for stage, reason_code in EXHAUSTION_TERMINAL_REASON_BY_STAGE.items():
        assert reason_code in FAILURE_REASON_CODES[stage]
        assert reason_code in EFFORT_EXHAUSTION_REASON_CODES
        assert f"failed:{stage}" in TERMINAL_STATUS_CODES


@pytest.mark.parametrize("stage", sorted(_BLOCKED_SCHEMA_STAGES))
def test_blocked_arm_still_accepts_every_honest_prerequisite_reason(
    tmp_path: Path, stage: str
) -> None:
    """The exhaustion refusal rejects nothing an honest prerequisite block can say.

    Includes both deferrable capability arms. ``needs-source-access`` routes to a
    ``deferred:`` terminal and is deliberately absent from ``FAILURE_REASON_CODES``, so a
    guard that closed to the failure set alone would have silently rejected every
    access-barrier deferral.

    Parameters
    ----------
    tmp_path:
        Isolated author result directory.
    stage:
        Blocking stage under test.
    """

    honest = (FAILURE_REASON_CODES.get(stage, frozenset()) - EFFORT_EXHAUSTION_REASON_CODES) | (
        {ACCESS_BLOCKED_REASON_CODE} if stage == "author" else set()
    )
    assert honest, f"{stage} has no honest prerequisite reason to test"
    for reason_code in sorted(honest):
        parsed = _parse_blocked_result(tmp_path, stage=stage, reason_code=reason_code)
        assert parsed.stage == stage
        assert parsed.reason_code == reason_code


@pytest.mark.smoke
@pytest.mark.parametrize("reason_code", sorted(EFFORT_EXHAUSTION_REASON_CODES))
def test_blocked_arm_refuses_effort_exhaustion_dressed_as_a_prerequisite(
    tmp_path: Path, reason_code: str
) -> None:
    """An exhausted session cannot claim the model is unresolvable.

    ``effort-cap-exhausted`` is a member of most stages' own vocabularies, so nothing but
    this explicit refusal stops it; ``budget-exhausted`` is in no vocabulary at all and is
    refused by the same clause rather than degrading to ``malformed-result``.

    Parameters
    ----------
    tmp_path:
        Isolated author result directory.
    reason_code:
        Exhaustion spelling under test.
    """

    with pytest.raises(AuthorEffortExhaustionClaim) as caught:
        _parse_blocked_result(tmp_path, stage="source", reason_code=reason_code)
    message = str(caught.value)
    assert "unfinished, not" in message
    assert "failed:<stage>" in message


@pytest.mark.smoke
@pytest.mark.parametrize("stage", sorted(_BLOCKED_SCHEMA_STAGES))
def test_refused_exhaustion_claim_carries_a_stage_valid_terminal(stage: str) -> None:
    """Every stage's refusal names a terminal that stage's own vocabulary admits.

    Parameters
    ----------
    stage:
        Blocking stage under test.
    """

    with pytest.raises(AuthorEffortExhaustionClaim) as caught:
        _validate_blocked_reason(stage, "budget-exhausted")
    claim = caught.value
    assert claim.reason_code in FAILURE_REASON_CODES[claim.stage]
    assert f"failed:{claim.stage}" in TERMINAL_STATUS_CODES


def test_an_unrecognized_reason_is_left_to_the_total_terminal_mapping() -> None:
    """An odd reason string is NOT refused here; ``_blocked_terminal`` records it honestly.

    ``driver._blocked_terminal`` is total and maps an unrecordable reason to
    ``failed:author``/``malformed-result``. Refusing it at the parse boundary as well would
    reintroduce the failure mode that mapping exists to prevent: one odd reason string on
    one model taking the whole campaign down.
    """

    for reason_code in ("some-reason-the-record-cannot-express", "missing-runtime-dependency"):
        assert reason_code not in EFFORT_EXHAUSTION_REASON_CODES
        # Returns rather than raising: the guard is scoped to the exhaustion class alone.
        assert _validate_blocked_reason("source", reason_code) is None


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
        author_model_fields=dict(proposal["author"]),
        checker_prompt_identity="sha256:" + "7" * 64,
        checker_model_identity="sha256:" + "8" * 64,
        checker_schema_identity="sha256:" + "9" * 64,
        checker_model_fields={
            "provider": "openai",
            "model": "gpt-fixture",
            "version": "current",
            "prompt_sha256": "sha256:" + "7" * 64,
        },
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


def _retext_paper(
    proposal: dict[str, Any], manifest: dict[str, Any], tmp_path: Path, body: str
) -> None:
    """Replace the fixture's paper bytes in place, rebinding every digest it feeds."""

    digest = hash_bytes(body.encode())
    path = tmp_path / "source-paper.txt"
    path.write_text(body)
    facts = proposal["proposed_facts"]
    for row in (*facts["source_resolution"]["sources"], *manifest["sources"]):
        if row.get("source_id") == "source-paper":
            row.update(
                {
                    "content_sha256": digest,
                    "mirror_digest": digest,
                    "byte_count": len(body.encode()),
                }
            )
    for excerpt in facts["evidence"]["excerpts"]:
        if excerpt["source_id"] == "source-paper":
            excerpt.update(
                {
                    "locator": f"bytes:0-{len(body.encode())}",
                    "text": body,
                    "text_sha256": digest,
                }
            )
    manifest["manifest_sha256"] = stable_hash(manifest["sources"])
    proposal["verified_hashes"]["source_manifest"] = manifest["manifest_sha256"]


def _paper_without_year(
    tmp_path: Path, *, arxiv_id: str = "2007.04044"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an honest proposal whose paper source never prints the work's own year.

    This is the measured shape of a body rendering. Across every ar5iv page in the
    ``pilot`` campaign's archived rungs the declared year occurred outside the
    bibliography zero times, so the fixture's paper text carries the title, authors,
    and arXiv identifier but no date at all.
    """

    proposal, manifest = _ground_proposal(tmp_path)
    _retext_paper(
        proposal,
        manifest,
        tmp_path,
        f"Example Model. A. Author, Example Lab, US. arXiv:{arxiv_id}. "
        "Abstract: ExampleNet is a small convolutional network.",
    )
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation.update({"arxiv_id": arxiv_id, "bibtex": None, "venue": None})
    return proposal, manifest


def test_year_is_grounded_by_the_arxiv_identifier_when_the_body_omits_it(
    tmp_path: Path,
) -> None:
    """A body rendering prints no date, so the grounded identifier must carry the year.

    ``2007.04044`` announces in 2020-07. Requiring the token anyway left exactly one
    satisfying witness on such a page -- some other work's year in the reference list,
    bound as evidence for THIS paper's citation -- which is the laundering the check
    exists to refuse. The identifier is machine-derived from an already-grounded leaf.
    """

    proposal, manifest = _paper_without_year(tmp_path)
    assert "2020" not in _paper_text(proposal)
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["year"] = 2020
    validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_venue_publication_year_after_the_announcement_is_entailed(tmp_path: Path) -> None:
    """A preprint announced late in one year is routinely published the next.

    PoolFormer is announced as ``2111.11418`` and published at CVPR 2022, so refusing
    the venue year would refuse a true claim.
    """

    proposal, manifest = _paper_without_year(tmp_path)
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["year"] = 2021
    validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


@pytest.mark.parametrize("year", [2018, 2019, 2022, 2023, 2030, 1997])
def test_year_outside_the_entailed_window_is_still_refused(tmp_path: Path, year: int) -> None:
    """The entailment runs one way and one year only.

    A year EARLIER than announcement is impossible for the work the identifier names,
    and two or more years later is not entailed by it, so both stay refused.
    """

    proposal, manifest = _paper_without_year(tmp_path)
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["year"] = year
    with pytest.raises(ProposalValidationError, match="not grounded verbatim.*year"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_year_without_an_arxiv_identifier_still_needs_the_paper_text(tmp_path: Path) -> None:
    """Nothing is entailed when no identifier is declared; the excerpt must carry it."""

    proposal, manifest = _paper_without_year(tmp_path)
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["arxiv_id"] = None
        citation["year"] = 2020
    with pytest.raises(ProposalValidationError, match="not grounded verbatim.*year"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def test_old_style_arxiv_identifier_entails_no_year(tmp_path: Path) -> None:
    """``archive/YYMMNNN`` is not decoded, so the text check governs unchanged."""

    proposal, manifest = _paper_without_year(tmp_path, arxiv_id="cs.CV/0309136")
    for citation in (
        proposal["proposed_facts"]["citation"],
        proposal["proposed_facts"]["external_metadata"]["citation"],
    ):
        citation["year"] = 2003
    with pytest.raises(ProposalValidationError, match="not grounded verbatim.*year"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


def _paper_text(proposal: dict[str, Any]) -> str:
    """Return the concatenated excerpt text bound to the paper source."""

    return " ".join(
        excerpt["text"]
        for excerpt in proposal["proposed_facts"]["evidence"]["excerpts"]
        if excerpt["source_id"] == "source-paper"
    )
