"""Lock public documentation names, paths, links, and dated tier claims to code."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

import torchlens as tl

# DECLARED-surface constant (cited by the reachable-surface gate in
# tests/oracles/test_oracle_w0_surface.py and the denominator 3-root lint in
# tests/oracles/test_oracle_w0_lints.py -- one cites the other, never forks).
PUBLIC_SURFACE_SIZE = 115
#: Docs carrying a parseable ``__all__`` count claim. AGENTS.md and torchlens/AGENTS.md
#: no longer carry this prose directly (2026-10-01 docs move: "docs: move agent
#: reference material out of startup instructions" / "docs: reserve nested instruction
#: budget for module rules") -- the claim now lives in their docs/agent-reference/
#: pointer targets below.
PUBLIC_SURFACE_DOCS = (
    "docs/agent-reference/current-2-x-surface.md",
    "docs/agent-reference/package/public-surface.md",
    "docs/for-ai-agents.md",
    "docs/migration/v2.0_api_changes.md",
)
CANONICAL_SYMBOLS = {
    "torchlens.backends.torch.wrappers": ("unwrap_torch",),
    "torchlens": ("Bundle", "merge_ranks", "merge_report"),
    "torchlens.distributed": ("arm",),
}
MARKDOWN_LINK_RE = re.compile(r"\[[^]]*\]\((?P<target>[^)]+)\)")


def _repo_root() -> Path:
    """Return the repository root.

    Returns
    -------
    Path
        Absolute repository root.
    """

    return Path(__file__).resolve().parents[1]


def _public_markdown_files() -> tuple[Path, ...]:
    """Return public Markdown files covered by link lockstep.

    Returns
    -------
    tuple[Path, ...]
        README, root guides, and every documentation page.
    """

    root = _repo_root()
    return (
        root / "README.md",
        root / "CLAUDE.md",
        root / "AGENTS.md",
        *sorted((root / "docs").rglob("*.md")),
    )


def test_public_surface_count_claims_match_runtime() -> None:
    """Keep every numeric ``__all__`` claim synchronized with the runtime."""

    root = _repo_root()
    assert len(tl.__all__) == PUBLIC_SURFACE_SIZE
    count_claim = re.compile(
        r"(?:(?P<before>\d+)[ -]?(?:name|top-level)|"
        r"(?:has|exposes|reserves(?: exactly)?)[^\d\n]*(?P<after>\d+))"
    )
    for relative_path in PUBLIC_SURFACE_DOCS:
        text = (root / relative_path).read_text(encoding="utf-8")
        claims = [
            int(match.group("before") or match.group("after"))
            for line in text.splitlines()
            if "__all__" in line
            for match in count_claim.finditer(line)
        ]
        assert claims, f"Expected an __all__ count claim in {relative_path}"
        assert set(claims) == {PUBLIC_SURFACE_SIZE}, (relative_path, claims)


def test_canonical_documented_symbols_resolve() -> None:
    """Resolve public symbol paths whose drift previously broke the guides."""

    for module_name, attributes in CANONICAL_SYMBOLS.items():
        module = importlib.import_module(module_name)
        for attribute in attributes:
            assert hasattr(module, attribute), f"{module_name}.{attribute} does not resolve"


def test_agent_docs_use_current_internal_paths() -> None:
    """Pin corrected backward, schema, utility, and Bundle documentation paths."""

    root = _repo_root()
    expected_paths = (
        "torchlens/backends/torch/backward.py",
        "torchlens/_source_links.py",
        "torchlens/schemas/tlspec_manifest_v1.json",
        "torchlens/schemas/tlspec_manifest_v2.json",
    )
    for relative_path in expected_paths:
        assert (root / relative_path).is_file(), relative_path

    validation_docs = "\n".join(
        (root / path).read_text(encoding="utf-8") for path in ("torchlens/validation/AGENTS.md",)
    )
    assert "tlspec_manifest_v{schema_version}.json" in validation_docs
    assert "capture/backward.py" not in validation_docs

    intervention_doc = (root / "torchlens/intervention/AGENTS.md").read_text(encoding="utf-8")
    assert "tl.Bundle" in intervention_doc
    assert "torchlens.bundle.Bundle" not in intervention_doc


#: Structured smoke-count claim in AGENTS.md's tier section. The lockstep test
#: below requires the claim to PARSE (dated, with raw collect-only numbers);
#: the drift tripwire compares the parsed numbers against a live collection so
#: the gate detects staleness instead of freezing it (the pre-r3 version
#: hard-asserted the literal count, ENFORCING the stale doc; R41/R81/R88).
TIER_CLAIM_RE = re.compile(
    r"~\d+(?:\.\d+)?k tests \((?P<smoke>\d[\d,]*)/(?P<total>\d[\d,]*) "
    r"collect-only, measured (?P<date>20\d{2}-\d{2}-\d{2})\)"
)


def test_dated_test_tier_claim_is_present_and_selector_is_additive() -> None:
    """Require a parseable dated tier record and default rare-test exclusion."""

    root = _repo_root()
    guide = (root / "AGENTS.md").read_text(encoding="utf-8")
    assert TIER_CLAIM_RE.search(guide), (
        "AGENTS.md's Testing Tiers section lost its structured smoke-count "
        "claim ('~Nk tests (S/T collect-only, measured YYYY-MM-DD)'); the "
        "drift tripwire needs it parseable"
    )
    assert re.search(r"measured 20\d{2}-\d{2}-\d{2}.{0,200}took \d+s \(~\d+ min\)", guide, re.S), (
        "AGENTS.md lost its dated smoke wall-clock measurement record"
    )
    assert 'pytest tests/ -m "not rare and not slow"' in guide

    test_guide = (root / "tests/AGENTS.md").read_text(encoding="utf-8")
    assert 'pytest tests/ -m "not rare and not slow"' in test_guide


#: Structured duration-budget claim, required VERBATIM-parseable in both agent
#: docs. The r3 re-tier shipped a 5s load-scaled budget while CLAUDE.md said
#: "15s" three lines from its own "<5s" partition table and tests/AGENTS.md
#: repeated the 15s — no gate parsed either sentence (R41-1/R81-2/R88-3,
#: cross-filed by all three labs). The numbers are compared against the
#: shipped conftest constants below, so budget and doc can only move together.
BUDGET_CLAIM_RE = re.compile(
    r"budget (?P<smoke>\d+(?:\.\d+)?)s and heavy (?P<heavy>\d+(?:\.\d+)?)s "
    r"\(load-scaled 1x-4x plus a (?P<grace>\d+(?:\.\d+)?)s\s+boundary-noise grace, "
    r"charged\s+on min\(wall, cpu\)\)"
)


def test_documented_duration_budgets_match_the_shipped_constants() -> None:
    """AGENTS.md and tests/AGENTS.md budget sentences track conftest reality."""

    root = _repo_root()
    conftest_text = (root / "tests" / "conftest.py").read_text(encoding="utf-8")
    shipped = {
        name: float(match)
        for name, pattern in (
            ("smoke", r"^SMOKE_DURATION_BUDGET_SECONDS\s*=\s*([\d.]+)"),
            ("heavy", r"^HEAVY_DURATION_BUDGET_SECONDS\s*=\s*([\d.]+)"),
            ("grace", r"^DURATION_BUDGET_GRACE_SECONDS\s*=\s*([\d.]+)"),
        )
        for match in re.findall(pattern, conftest_text, flags=re.MULTILINE)
    }
    assert set(shipped) == {"smoke", "heavy", "grace"}, "conftest budget constants moved or renamed"
    for doc in ("AGENTS.md", "tests/AGENTS.md"):
        text = (root / doc).read_text(encoding="utf-8")
        claim = BUDGET_CLAIM_RE.search(text)
        assert claim is not None, (
            f"{doc} lost its structured duration-budget sentence ('budget Ns and "
            "heavy Ms (load-scaled 1x-4x, charged on min(wall, cpu))'); the doc "
            "and the shipped budget must move together (R41-1)"
        )
        for tier in ("smoke", "heavy", "grace"):
            assert float(claim.group(tier)) == shipped[tier], (
                f"{doc} documents a {tier} budget of {claim.group(tier)}s but "
                f"tests/conftest.py ships {shipped[tier]}s — update both together"
            )


def test_public_relative_markdown_links_resolve() -> None:
    """Ensure every local Markdown link names an existing repository path."""

    missing: list[tuple[str, str]] = []
    root = _repo_root()
    for page in _public_markdown_files():
        for match in MARKDOWN_LINK_RE.finditer(page.read_text(encoding="utf-8")):
            target = match.group("target").split("#", maxsplit=1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (page.parent / target).resolve().exists():
                missing.append((str(page.relative_to(root)), target))
    assert not missing


def test_capture_outcomes_doc_vocabulary_locksteps_with_the_enums() -> None:
    """Gate the capture-outcome doc of record against the settlement enums.

    ``docs/reference/capture_outcomes.md`` is the doc of record for the
    outcome unification (CLAUDE.md), yet it was the one enum-vocabulary
    contract doc with ZERO automated lockstep (grind b7 R53-F1): a
    ``CaptureStatus``/``CapturePhase``/``FailureOrigin`` or N-gate change
    would have shipped with a silently stale doc. The status table is
    compared bidirectionally (a phantom or missing row fails either way);
    the FAILED-only prose vocabularies and every chokepoint gate code must
    appear in their sections.
    """

    from torchlens.capture.outcome import (
        _REFUSAL_HINTS,
        CapturePhase,
        CaptureStatus,
        FailureOrigin,
    )

    text = (_repo_root() / "docs" / "reference" / "capture_outcomes.md").read_text(encoding="utf-8")

    status_section = text.split("## The status vocabulary", 1)[1].split("\n## ", 1)[0]
    documented_statuses = re.findall(r"^\|\s*`([A-Z_]+)`\s*\|", status_section, flags=re.M)
    assert len(documented_statuses) == len(set(documented_statuses)), (
        f"duplicated status rows in capture_outcomes.md: {documented_statuses}"
    )
    assert set(documented_statuses) == {member.name for member in CaptureStatus}, (
        "the capture_outcomes.md status table and tl.types.CaptureStatus have "
        f"drifted apart: doc rows {sorted(documented_statuses)} vs enum "
        f"{sorted(member.name for member in CaptureStatus)} — update both in "
        "the same change"
    )

    for member in CapturePhase:
        assert f"`{member.name}`" in text, (
            f"CapturePhase.{member.name} is not documented in capture_outcomes.md"
        )
    for member in FailureOrigin:
        assert f"`{member.value}`" in text, (
            f"FailureOrigin.{member.value} is not documented in capture_outcomes.md"
        )

    capability_section = text.split("## The capability table", 1)[1].split("\n## ", 1)[0]
    for gate_code in _REFUSAL_HINTS:
        assert gate_code in capability_section, (
            f"chokepoint gate code {gate_code} is missing from the "
            "capture_outcomes.md capability table"
        )


def test_shipped_glossary_covers_the_public_surface() -> None:
    """Every `tl.__all__` name appears in the shipped user glossary.

    docs/reference/glossary.md was complete (97/97) and gated by NOTHING —
    the state was held purely by hand against the repo's own LOCKED
    "a rename is not done until the docs match" rule (b2 rounds 4+5,
    B2R5-6). Word-boundary matching: the glossary is prose-styled
    (bold terms + `tl.name(...)` spellings), not one-heading-per-name.
    """

    import torchlens as tl

    glossary = (_repo_root() / "docs" / "reference" / "glossary.md").read_text(encoding="utf-8")
    missing = [name for name in tl.__all__ if not re.search(rf"\b{re.escape(name)}\b", glossary)]
    assert not missing, (
        f"public names absent from docs/reference/glossary.md: {missing} — the "
        "shipped glossary is release surface; update it in the same change as "
        "the rename/addition (LOCKED lockstep rule)"
    )


#: Deliberate doc spellings that do not resolve at runtime yet (each entry
#: needs a one-line reason). Empty today; a glossary entry naming a future
#: surface must be listed here explicitly instead of silently passing.
GLOSSARY_FUTURE_SPELLINGS: dict[str, str] = {}

DOTTED_SPELLING_RE = re.compile(r"`(?:tl|torchlens)((?:\.[A-Za-z_][A-Za-z0-9_]*)+)")


@pytest.mark.smoke
def test_documented_dotted_spellings_resolve() -> None:
    """Every backticked `tl.*`/`torchlens.*` glossary spelling resolves in code.

    The reverse half of the LOCKED docs-lockstep rule ("a rename is not done
    until the docs match"): a doc entry naming something that no longer
    exists must FAIL. The wave-3 audit found the forward gaps by hand; this
    keeps renames from stranding stale glossary spellings. Attribute chains
    resolve via getattr with a submodule-import fallback (lazy namespaces).
    """

    import warnings

    text = (_repo_root() / "docs" / "reference" / "glossary.md").read_text(encoding="utf-8")
    unresolved: list[str] = []
    for token in sorted({match.group(1) for match in DOTTED_SPELLING_RE.finditer(text)}):
        if token in GLOSSARY_FUTURE_SPELLINGS:
            continue
        parts = token.strip(".").split(".")
        obj: object = tl
        for index, part in enumerate(parts):
            try:
                with warnings.catch_warnings():
                    # The glossary documents deprecated aliases deliberately;
                    # resolving them must not fail a warnings-as-errors run.
                    warnings.simplefilter("ignore", DeprecationWarning)
                    obj = getattr(obj, part)
            except AttributeError:
                try:
                    obj = importlib.import_module("torchlens." + ".".join(parts[: index + 1]))
                except Exception:
                    unresolved.append(token)
                    break
    assert not unresolved, (
        f"glossary spellings that no longer resolve at runtime: {unresolved} — "
        "either the code renamed without updating docs/reference/glossary.md "
        "(fix the docs in the same change, LOCKED lockstep rule) or the entry "
        "deliberately names future surface (list it in GLOSSARY_FUTURE_SPELLINGS "
        "with a reason)"
    )


#: Curated user-facing analysis namespaces and each one's doc of record.
#: Every public CALLABLE in these modules' ``__all__`` must be named in the
#: glossary, an agent guide, or the module's doc of record — this is the
#: check that would have caught `logit_lens` / `bisect_precision` /
#: `compare_params` / `audit_trace` / `dtype_range_audit` shipping with zero
#: documentation (wave-3 audit residue). Classes/constants are exempt: the
#: docs cover them at concept level, name-by-name coverage would be noise.
CURATED_NAMESPACE_DOCS = {
    "torchlens.attribution": ("docs/reference/attribution.md",),
    "torchlens.semantic": ("docs/facets.md",),
    "torchlens.debug": ("docs/reference/debug.md",),
}
#: ``docs/agent-reference/current-2-x-surface.md`` carries the "Current 2.x Surface"
#: prose that used to live directly in AGENTS.md (2026-10-01 docs move: "docs: move
#: agent reference material out of startup instructions"), e.g. the
#: ``tl.debug.graph_breaks()`` mention -- its doc-of-record row stays
#: docs/reference/debug.md, but this is the common venue a bare mention counts from.
CURATED_COMMON_VENUES = (
    "docs/reference/glossary.md",
    "AGENTS.md",
    "docs/agent-reference/current-2-x-surface.md",
)


def test_curated_namespace_callables_are_documented() -> None:
    """Every public callable in the curated analysis namespaces has a doc row."""

    import inspect

    root = _repo_root()
    common = "\n".join(
        (root / venue).read_text(encoding="utf-8") for venue in CURATED_COMMON_VENUES
    )
    problems: list[str] = []
    for module_name, doc_paths in CURATED_NAMESPACE_DOCS.items():
        module = importlib.import_module(module_name)
        text = common + "\n".join((root / path).read_text(encoding="utf-8") for path in doc_paths)
        for name in module.__all__:
            member = getattr(module, name)
            if not (inspect.isfunction(member) or inspect.isbuiltin(member)):
                continue
            if not re.search(rf"\b{re.escape(name)}\b", text):
                problems.append(f"{module_name}.{name} (doc of record: {doc_paths[0]})")
    assert not problems, (
        f"public callables with no documentation row anywhere: {problems} — "
        "register each in its doc of record (or the glossary) in the same "
        "change that ships it (LOCKED lockstep rule)"
    )


def test_tier_census_figures_agree_across_docs() -> None:
    """r7 R41 (sol MED): the two documented tier censuses must be ONE census.

    CLAUDE.md and tests/AGENTS.md carried contradictory totals (12,430 vs
    12,651) with the SAME claimed measurement date, and the only live check
    was slow-marked and CLAUDE-only. This half is a pure textual
    consistency gate (no collection; lives here in the smoke-marked docs
    lockstep module -- its old home tests/test_docs_tier_drift.py is
    module-slow): the total in AGENTS.md's
    smoke sentence must equal the total in tests/AGENTS.md's unmarked row.
    The live-count validation above stays slow (it pays a full collection).
    """

    root = Path(__file__).resolve().parents[1]
    claude = (root / "AGENTS.md").read_text(encoding="utf-8")
    agents = (root / "tests" / "AGENTS.md").read_text(encoding="utf-8")
    claude_match = re.search(r"\((\d[\d,]*)/(\d[\d,]*) collect-only, measured ([0-9-]+)", claude)
    assert claude_match, "AGENTS.md lost its smoke census sentence"
    agents_match = re.search(r"\((\d[\d,]*)/(\d[\d,]*), \d+%, measured ([0-9-]+)", agents)
    assert agents_match, "tests/AGENTS.md lost its unmarked census row"
    claude_total = int(claude_match.group(2).replace(",", ""))
    agents_total = int(agents_match.group(2).replace(",", ""))
    assert claude_total == agents_total, (
        f"tier census totals contradict: AGENTS.md says {claude_total} "
        f"(measured {claude_match.group(3)}) but tests/AGENTS.md says "
        f"{agents_total} (measured {agents_match.group(3)}) — remeasure ONCE "
        "and update both docs in the same change"
    )
