"""Rung-7 checker-calibration pins: prompt clauses and their machine mirrors.

The rung-7 adjudication (dual blind cross-lab grading of the 8 frozen
``terminal_disposition`` gate verdicts) found every disposition correct on the
merits and exactly three prompt-clarity gaps, closed by three additive clauses
in ``codex_accuracy_checker_v2.txt``:

1. the rejected/cannot-verify boundary on a terminal item (a resolved pack
   whose excerpts do not entail the arm's own predicate is an OVERSTATED-support
   rejection; cannot-verify is reserved for evidence that could not be resolved
   or bound) -- the frozen rung-7 ledger split 7 rejected / 1 cannot-verify on
   materially identical wall-exhaustion claims because this boundary was
   unstated;
2. the typed inapplicable ``rung_check`` form on a terminal item -- the eight
   frozen sessions improvised four different shapes for a block nothing
   terminal-side consumes;
3. the exact R4 ``search-attested:`` / ``search-cannot-verify:`` correlation,
   which ``metadata._validate_rung_and_search_attestation`` already enforces
   mechanically but the prompt only half-stated.

This module pins each clause into the shipped prompt bytes (so the calibration
cannot silently drift back out) and pins the two machine facts the clauses
mirror: terminal routing is ``rung_check``-inert, and the R4 correlation stated
in prose is byte-for-byte the rule the metadata validator enforces. None of
these pins loosens anything: every probe that failed before still fails.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from menagerie.crawler.gates import validate_terminal_disposition_gate
from menagerie.crawler.metadata import (
    MetadataValidationError,
    _validate_rung_and_search_attestation,
)
from menagerie.crawler.schema import validate_payload
from menagerie.crawler.tests.conftest import HASH
from menagerie.crawler.tests.test_slice_d_checker_gates import _blocked_terminal_fixture

_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "codex_accuracy_checker_v2.txt"
)

#: The exact typed inapplicable ``rung_check`` the prompt prescribes for a
#: ``terminal_disposition`` item. A terminal advisory arm selects no
#: implementation rung, so this block is canonical filler; the tests below
#: prove the deterministic terminal route never reads it.
TYPED_INAPPLICABLE_RUNG_CHECK = {
    "selected_rung": "R5_SKIP",
    "highest_applicable": "R5_SKIP",
    "verdict": "accurate",
    "findings": ["not-applicable: terminal advisory arm; no implementation rung selected"],
}

#: rung_check shapes the eight frozen rung-7 checker sessions actually emitted
#: on terminal items (improvised, because no canonical form was stated).
_RUNG7_IMPROVISED_RUNG_CHECKS = (
    {
        "selected_rung": "R1_LIBRARY",
        "highest_applicable": "R1_LIBRARY",
        "verdict": "accurate",
        "findings": ["an available library implementation route exists"],
    },
    {
        "selected_rung": "R5_SKIP",
        "highest_applicable": "R5_SKIP",
        "verdict": "cannot-verify",
        "findings": ["no independently evidenced skip rationale was supplied"],
    },
)


def _prompt_text() -> str:
    """Return the shipped checker prompt bytes as text."""

    return _PROMPT_PATH.read_text(encoding="utf-8")


@pytest.mark.smoke
def test_prompt_states_the_terminal_rejected_cannot_verify_boundary() -> None:
    """The rejected/cannot-verify boundary clause ships in the prompt bytes.

    Rung 7 split 7 rejected / 1 cannot-verify (m5445) over materially identical
    author-process wall claims; the boundary was checker-session jitter until
    the prompt stated it. These literals are the load-bearing phrases.
    """

    text = _prompt_text()
    assert "OVERSTATES its support and the disposition is rejected" in text
    assert "never source-establishable" in text
    assert "Reserve cannot-verify for evidence you could not" in text


@pytest.mark.smoke
def test_prompt_states_the_typed_inapplicable_terminal_rung_check() -> None:
    """The canonical terminal rung_check form ships verbatim in the prompt."""

    text = _prompt_text()
    assert "selected_rung and highest_applicable both R5_SKIP, verdict accurate" in text
    assert TYPED_INAPPLICABLE_RUNG_CHECK["findings"][0] in text


@pytest.mark.smoke
def test_prompt_states_the_exact_r4_search_marker_correlation() -> None:
    """The R4 marker/verdict correlation clause ships in the prompt bytes."""

    text = _prompt_text()
    assert "search_report.links_checked" in text
    assert "each link spelled byte-identically, the set complete" in text
    assert "carries at least one `search-cannot-verify:<reason>`" in text
    assert "Both markers are R4-only" in text


@pytest.mark.smoke
def test_terminal_routing_is_rung_check_inert() -> None:
    """One terminal gate routes identically under every rung_check shape.

    ``validate_terminal_disposition_gate`` keys on the terminal_disposition
    block, the exact identities, and item integrity -- never on ``rung_check``.
    Standardizing the typed inapplicable form therefore cannot drift any
    terminal verdict: the decision under the canonical form equals the decision
    under both improvised rung-7 shapes, field for field.
    """

    gate, result, source_manifest, evidence_pack = _blocked_terminal_fixture(
        ("source-1", "source-2", "source-3"), ("evidence-1",)
    )

    decisions = []
    for rung_check in (TYPED_INAPPLICABLE_RUNG_CHECK, *_RUNG7_IMPROVISED_RUNG_CHECKS):
        probe = deepcopy(gate)
        probe["items"][0]["rung_check"] = deepcopy(rung_check)
        validate_payload(probe)
        decisions.append(
            validate_terminal_disposition_gate(
                probe,
                result,
                source_manifest=source_manifest,
                evidence_pack=evidence_pack,
                license_identity=HASH,
            )
        )

    first = decisions[0]
    assert first.accepted is True
    assert first.predicate == "blocked-prerequisite"
    for other in decisions[1:]:
        assert other == first


def _r4_probe(
    *,
    verdict: str,
    findings: list[str],
    links: list[str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build minimal facts and gate-item mappings for the R4 correlation rule."""

    facts = {
        "source_resolution": {
            "rung": "R4_REIMPLEMENT",
            "search_report": {"links_checked": links},
        }
    }
    gate_item = {
        "rung_check": {
            "selected_rung": "R4_REIMPLEMENT",
            "highest_applicable": "R4_REIMPLEMENT",
            "verdict": verdict,
            "findings": findings,
        }
    }
    return facts, gate_item


@pytest.mark.smoke
def test_the_prompt_r4_correlation_is_the_machine_rule() -> None:
    """Every branch of the stated R4 correlation is what the validator enforces.

    The prompt clause and ``_validate_rung_and_search_attestation`` must agree
    or checker sessions get mechanically discarded for obeying the prose. Each
    probe below is one clause of the stated correlation.
    """

    links = ["https://example.org/search-a", "https://example.org/search-b"]

    # accurate + complete byte-identical attestation + no cannot-verify: passes.
    facts, item = _r4_probe(
        verdict="accurate",
        findings=[f"search-attested:{link}" for link in links],
        links=links,
    )
    _validate_rung_and_search_attestation(facts, item)

    # accurate with one link unattested: refused, set-complete is mandatory.
    facts, item = _r4_probe(
        verdict="accurate",
        findings=[f"search-attested:{links[0]}"],
        links=links,
    )
    with pytest.raises(MetadataValidationError, match="lacks re-executed search attestations"):
        _validate_rung_and_search_attestation(facts, item)

    # accurate alongside any search-cannot-verify marker: refused.
    facts, item = _r4_probe(
        verdict="accurate",
        findings=[f"search-attested:{link}" for link in links]
        + ["search-cannot-verify:rate-limited"],
        links=links,
    )
    with pytest.raises(MetadataValidationError, match="cannot carry search-cannot-verify"):
        _validate_rung_and_search_attestation(facts, item)

    # every non-accurate R4 rung verdict must carry the typed marker -- both
    # cannot-verify and inaccurate (including inaccurate-for-found-code).
    for verdict in ("cannot-verify", "inaccurate"):
        facts, item = _r4_probe(verdict=verdict, findings=[], links=links)
        with pytest.raises(MetadataValidationError, match="requires typed search-cannot-verify"):
            _validate_rung_and_search_attestation(facts, item)

        facts, item = _r4_probe(
            verdict=verdict,
            findings=["search-cannot-verify:usable code found at the first link"],
            links=links,
        )
        _validate_rung_and_search_attestation(facts, item)
