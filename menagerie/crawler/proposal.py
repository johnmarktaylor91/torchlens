"""Staged author-proposal validation and deterministic anti-slop gates."""

from __future__ import annotations

import ast
import hashlib
import re
import tarfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Any, Iterable, Mapping, Optional, Sequence, Union

from menagerie.crawler.constants import AUTHOR_PROPOSAL_SCHEMA_VERSION_V3, SourceRung
from menagerie.crawler.evidence import (
    EvidenceValidationError,
    evidence_ids,
    fetched_sources_for_checked_links,
    validate_evidence,
)
from menagerie.crawler.fetcher import cas_path as source_cas_path
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.metadata import (
    AVAILABILITY_BASES,
    AVAILABILITY_FIELDS,
    AVAILABILITY_RECORD_KEYS,
    AVAILABILITY_STATUSES,
    FOREIGN_AVAILABILITY_KEYS,
    MANDATORY_EXTERNAL_FIELDS,
)
from menagerie.crawler.recipe import RecipeError, validate_pretrained_disposition
from menagerie.crawler.schema import (
    PayloadValidationError,
    RequiredFieldProjection,
    SchemaOwner,
    required_field_projection_spec,
    validate_payload,
)

#: Per-leaf taxonomy claims. The old aggregate ``taxonomy`` claim passed whenever ONE
#: child matched -- a taxonomy with unsupported leaves rode through on its family name
#: alone. Every leaf now needs its own excerpt binding, and the Codex checker returns a
#: per-leaf verdict. No aggregate claim ever passes on a child's support.
TAXONOMY_LEAF_CLAIMS = frozenset(
    {
        "taxonomy.family",
        "taxonomy.domains",
        "taxonomy.tasks",
        "taxonomy.modalities",
        "taxonomy.era",
        "taxonomy.architecture_tags",
        "taxonomy.novel_ops",
    }
)
DEFAULT_GATED_CLAIMS = frozenset(
    {f"external_metadata.{field}" for field in MANDATORY_EXTERNAL_FIELDS if field != "keywords"}
    | {
        "external_metadata.description",
        "source_resolution.rung",
        "input_contract",
        # The website block is the PUBLIC-FACING prose of the permanent record, and it
        # was the one authored block no lane verified structurally: the checker prompt
        # reviews website text under its holistic item verdict, but nothing forced a
        # per-item verdict or an evidence binding, so an inattentive checker could pass
        # invented catalog-page prose silently. One block-level claim closes that: the
        # prose restates facts the author already grounds (family, task, contribution,
        # description), so tagging those same excerpts with ``website`` costs one
        # supports entry and is always satisfiable -- the schema requires non-empty
        # website prose for every proposal, so no availability route is needed.
        #
        # "Always satisfiable" was an ARGUMENT when this claim landed; it is now a
        # MEASUREMENT. Replaying this gate over all 39 proposals archived from the
        # 2026-08-05 twenty-model rung: 37 carried a ``website`` supports tag and 39 of
        # 39 dereferenced ``website.family_grounding_id`` to a real family-level
        # excerpt, so the harder half of the claim was satisfied by literally every
        # proposal an author produced. The two misses are one model's third attempt
        # (m7362), which dropped ``website``, ``input_contract`` AND
        # ``external_metadata.country`` together in a whole-proposal rewrite after its
        # own attempts 1 and 2 had grounded all three. That is a regression, not a
        # wall: no proposal ever failed this claim for want of groundable evidence.
        # Do not re-litigate it from theory -- replay the archive.
        "website",
    }
    | TAXONOMY_LEAF_CLAIMS
)
#: Judgment claims: every gated claim whose VALUE is an inference over the source rather
#: than a string the source contains. The deterministic layer can decide structure and
#: provenance for them -- the claim is bound to hash-verified excerpts, empties carry a
#: typed availability state -- but it cannot decide entailment. The deleted token-overlap
#: oracle was simultaneously too weak (``country = "US"`` passed on the English pronoun
#: "us"; a fabricated citation passed on title+year alone) and too strong (``GB``/``CN``/
#: ``DE`` and era buckets could never pass honest text), so semantic entailment belongs
#: to the Codex accuracy checker, which must still return ``accurate`` for every
#: mandatory external field before any canonical write
#: (:func:`menagerie.crawler.metadata._validate_external_field_checks`).
CHECKER_EVALUATED_CLAIMS = DEFAULT_GATED_CLAIMS - {"external_metadata.citation"}
#: Gated claims whose value is checked deterministically, per-leaf, against evidence
#: bytes or machine records. Citation leaves must occur (Unicode-canonicalized) verbatim
#: in the controlled-fetched paper text; see `_validate_citation_leaves`.
VALUE_MATCHED_CLAIMS = frozenset({"external_metadata.citation"})
#: Keys inside a structured claim that record provenance or disposition rather than the
#: fact itself, and so cannot make an otherwise-empty block look answered.
_HOLLOW_BOOKKEEPING_KEYS = frozenset(
    {"status", "source_evidence_ids", "evidence_ids", "basis", "confidence", "note"}
)
#: Gated claims for which an empty collection can be a true, ordinary fact rather than
#: an unfilled field (a model with no recorded predecessors, lineage, or novel ops is
#: the common honest case). Emptiness here is still an ASSERTION, not an exemption: it
#: must be declared as a typed ``none-exist`` (or ``not-found-after-search``)
#: availability state carrying its bounded search, exactly like every other unknown.
#:
#: These claims previously had no honest exit at all. They were required to carry an
#: excerpt tag by the coverage gate, yet forbidden from declaring an availability state
#: by :func:`_validate_claim_state`, so the only way past the gate was to tag an
#: arbitrary excerpt that says nothing about lineage. Because coverage is nominal, that
#: worked -- the gate rewarded fabricated attribution for exactly the facts it defined
#: as having no attribution. A recorded absence replaces a fabricated presence.
EMPTIABLE_CLAIMS = frozenset(
    {
        "external_metadata.lineage",
        "external_metadata.predecessors",
        "taxonomy.novel_ops",
    }
)
#: Availability statuses that assert a claim carries no value. Every one of them is a
#: typed, recorded, queryable declaration, so every one of them discharges its own
#: claim's evidence coverage -- there is no excerpt that says a fact is not there. The
#: distinction that matters for coverage is declared-versus-bare, not which absence.
ABSENCE_AVAILABILITY_STATUSES = frozenset(
    {"none-exist", "not-found-after-search", "not-applicable"}
)
#: The absence statuses that additionally assert a search happened. ``none-exist`` says
#: the fact does not exist and ``not-found-after-search`` says it could not be
#: established; both are findings and both must carry the bounded search that produced
#: them. ``not-applicable`` asserts the field does not pertain, which no search informs.
SEARCH_BACKED_AVAILABILITY_STATUSES = frozenset({"none-exist", "not-found-after-search"})
#: Claims that may declare a typed availability state instead of a value. These are the
#: judgment facts that can be honestly unknowable for a real model, plus the collection
#: facts whose honest answer is often "there are none". See
#: :data:`menagerie.crawler.metadata.AVAILABILITY_FIELDS` (single source of truth).
AVAILABILITY_CLAIMS = (
    frozenset(f"external_metadata.{field}" for field in AVAILABILITY_FIELDS) | EMPTIABLE_CLAIMS
)
#: Source roles whose bytes are the *paper*, not the implementation. Paper metadata
#: (`authors`, `institution`, `country`, `venue`, `year`, `era`, `citation`) essentially
#: never appears verbatim in implementation code, so a proposal that asserts a citation
#: must bring the paper itself through the controlled fetcher and ground the citation
#: there. See `_validate_paper_evidence_source`.
PAPER_EVIDENCE_ROLES = frozenset({"introducing-paper", "supplement", "project-page"})
#: Citation fields that carry a resolvable, machine-checkable identifier. When the author
#: declares one it must occur in the excerpt text: an exact identifier is a strictly
#: stronger anchor than title-token overlap, so requiring it tightens the gate.
CITATION_IDENTIFIER_FIELDS = ("arxiv_id", "doi", "openreview_id")
#: The arXiv identifier grammar, with the optional ``vN`` revision selector captured
#: separately. Both eras are covered: the post-2007 ``YYMM.NNNN(N)`` form and the old
#: ``archive[.SS]/YYMMNNN`` form. The base identifier can never itself end in ``v`` plus
#: digits -- the new form ends in a fixed-width digit run and the old form ends in seven
#: digits -- so a trailing ``vN`` is unambiguously the version selector and never part of
#: the identifier being named. That unambiguity is what makes
#: :func:`_identifier_grounded` safe for arXiv and is exactly what DOI and OpenReview
#: identifiers lack; see that function for why neither gets the same treatment.
_ARXIV_IDENTIFIER_PATTERN = re.compile(
    r"\A(?:arxiv:)?"
    r"(?:\d{4}\.\d{4,5}|[a-z][a-z-]*(?:\.[a-z]{2})?/\d{7})"
    r"(?P<version>v\d+)?\Z",
    re.IGNORECASE,
)
#: TeX accent commands mapped to the combining mark they place on their argument. Used
#: only by :func:`_decode_tex_escapes` to make a BibTeX spelling of an accented name
#: canonicalize identically to the same name spelled in Unicode.
_TEX_ACCENT_COMBINING = {
    "'": "\u0301",  # acute
    "`": "\u0300",  # grave
    "^": "\u0302",  # circumflex
    '"': "\u0308",  # diaeresis
    "~": "\u0303",  # tilde
    "=": "\u0304",  # macron
    ".": "\u0307",  # dot above
    "u": "\u0306",  # breve
    "v": "\u030c",  # caron
    "H": "\u030b",  # double acute
    "c": "\u0327",  # cedilla
    "k": "\u0328",  # ogonek
    "r": "\u030a",  # ring above
    "d": "\u0323",  # dot below
    "b": "\u0331",  # macron below
}
#: TeX commands for letters that carry no separable combining mark. They must decode to
#: the real character rather than to a bare ASCII stand-in, so that ``\l`` and ``ł`` --
#: which NFKD does not decompose -- reduce to the same token.
_TEX_LETTERS = {
    "ss": "\u00df",
    "AE": "\u00c6",
    "ae": "\u00e6",
    "OE": "\u0152",
    "oe": "\u0153",
    "AA": "\u00c5",
    "aa": "\u00e5",
    "O": "\u00d8",
    "o": "\u00f8",
    "L": "\u0141",
    "l": "\u0142",
    "i": "\u0131",
    "j": "\u0237",
}
#: ``\'e``, ``\'{e}``, ``{\'e}``, and ``\c{c}`` are all the same accent applied to one
#: letter; the outer braces are punctuation the tokenizer already drops.
_TEX_ACCENT_PATTERN = re.compile(
    r"\\(?P<accent>['`^\"~=.]|[uvHckrdb](?=\s*\{|\s))\s*\{?\s*(?P<letter>[A-Za-z])\s*\}?"
)
#: Longest command name first so ``\oe`` never matches as ``\o`` followed by ``e``.
_TEX_LETTER_PATTERN = re.compile(
    r"\\(?P<letter>" + "|".join(sorted(_TEX_LETTERS, key=len, reverse=True)) + r")(\{\}|\b)"
)
#: Gated claim that is required only when a paper source is bound or a citation is
#: volunteered; every other member of :data:`DEFAULT_GATED_CLAIMS` is always required.
CONDITIONAL_GATED_CLAIMS = frozenset({"external_metadata.citation"})
#: The audited coverage map for every author-gated leaf that is NOT under a gated
#: claim: which lane actually verifies it, or the recorded reason it is deliberately
#: unverified. This is the durable answer to "the claim vocabulary covers 30-odd
#: claims but the schema declares ~290 author-gated paths -- who checks the rest?",
#: audited leaf-by-leaf on 2026-08-04 by reading each lane's enforcing code. It is
#: exercised by the test suite: every author-gated ``model.v3`` schema path must fall
#: under a gated claim or under exactly this table, so a new authored leaf cannot
#: land silently uncovered -- adding one forces its author to declare, here, which
#: lane verifies it.
#:
#: Lane vocabulary (each names its enforcing code, not a hope):
#:
#: ``integrity-machine``
#:     Deterministically verified against frozen bytes. Excerpts re-hash and
#:     byte-locate against CAS sources (:func:`menagerie.crawler.evidence.validate_evidence`);
#:     declared source rows must mirror the frozen manifest
#:     (``artifact_transactions``: "proposal and source manifest source sets differ");
#:     coverage assertions are recomputed, not trusted.
#: ``identity-recompute``
#:     Recomputed from fact bytes by
#:     :func:`menagerie.crawler.metadata.recompute_accepted_identities`; a divergent
#:     declared value refuses at admission.
#: ``citation-grounding``
#:     The top-level citation must equal the claim-gated
#:     ``external_metadata.citation`` (`_validate_citation_consistency`), its leaves
#:     are value-checked verbatim against controlled-fetched paper bytes
#:     (`_validate_citation_leaves`), and its absence must sit behind a bounded
#:     search (`_validate_citation`).
#: ``rung-lane``
#:     Deterministic source-ladder validation (`_validate_source_ladder`,
#:     `_validate_r4_negative_proof`, checked-link fetch coverage) plus the checker's
#:     ``rung_check`` verdict, which acceptance requires to be accurate
#:     (``gates.route_metadata_gate`` / ``route_fidelity_gate``) and which the write
#:     gate cross-validates (``metadata._validate_rung_and_search_attestation``).
#:     The identity naming leaves ride this lane: rung_check's contract is "this
#:     source IS this architecture and variant".
#: ``execution-lane``
#:     Consumed and verified by real isolated execution: the recipe is built and run,
#:     the input contract must byte-match the worker receipt
#:     (``metadata.input_signature_matches_contract``), declared meaningful modes must
#:     equal the claim-gated external copy at admission ("proposal meaningful-mode
#:     declarations disagree") and each declared mode produces a receipt.
#: ``fidelity-lane``
#:     ``fidelity.required`` has a machine floor (R3/R4 always require it,
#:     ``driver_models._fidelity_required``); gate id and fidelity identity must match
#:     the actual bound fidelity gate at admission; the five-way verdict routes
#:     through ``gates.route_fidelity_gate``.
#: ``license-lane``
#:     Redistribution effects are recomputed fail-closed from hash-verified excerpts
#:     (``licenses.recompute_license_decision``): fabricated evidence ids yield no
#:     findings and therefore no public redistribution. Semantic license claims are
#:     additionally under the ``external_metadata.license`` gated claim.
#: ``availability-machine``
#:     Typed availability records validate structurally per-record
#:     (`_validate_availability_record`) and are spot-verified by the checker against
#:     the evidence pack (checker prompt step 8).
#: ``checker-holistic``
#:     No per-claim field check. Verified only by the checker's item-wide verdict,
#:     which the write gate requires to be ``accurate``
#:     (``metadata._validate_gate_header``) and whose prompt enumerates these blocks
#:     as explicit ordered steps; the exact leaf bytes are bound into the vet
#:     identity, so the verdict cannot survive any later edit. DELIBERATE non-claims,
#:     re-litigated 2026-08-04: ``people_and_origin`` and ``dates`` restate facts
#:     whose canonical copies (authors, institution, country, year, venue, era) are
#:     already per-claim gated with availability routes; a block claim here would
#:     need excerpt coverage even for models whose people/dates are honestly unknown
#:     -- external_metadata declares those absences through the availability
#:     register, but these blocks have no such route, so the claim would be a wall
#:     (an author could not ground it), and their extra columns (labs, bases,
#:     confidence, note) are provenance judgments about already-gated facts.
#:     Deterministic equality is equally unavailable: the fixture-visible semantics
#:     legitimately diverge (institutions vs labs vs institution; year bases differ
#:     from citation year), so equating them would invent semantics.
#: ``unverified-bookkeeping``
#:     ``evidence.family_grounding_path`` is a nullable operator-side note with no
#:     consumer and no defined ground truth; nothing reads it, nothing derives from
#:     it, and no checker step names it. Recorded here as deliberately unverified so
#:     the next audit does not re-litigate it. (The family grounding FACT is
#:     verified: at least one ``family_level`` excerpt is mandatory, and
#:     ``website.family_grounding_id`` must reference one -- see
#:     `_validate_website_grounding`.)
AUTHORED_LEAF_COVERAGE_LANES: tuple[tuple[str, str], ...] = (
    ("citation", "citation-grounding"),
    ("dates", "checker-holistic"),
    ("evidence.coverage", "integrity-machine"),
    ("evidence.evidence_identity", "identity-recompute"),
    ("evidence.excerpts[]", "integrity-machine"),
    ("evidence.family_grounding_path", "unverified-bookkeeping"),
    ("external_metadata.availability", "availability-machine"),
    ("fidelity", "fidelity-lane"),
    ("identity", "rung-lane"),
    ("implementation", "execution-lane"),
    ("licenses", "license-lane"),
    ("modes.meaningful_modes[]", "execution-lane"),
    ("people_and_origin", "checker-holistic"),
    ("source_resolution", "rung-lane"),
    ("source_resolution.sources[]", "integrity-machine"),
)
#: Delimiters of the generated claim-vocabulary region in the author prompt. The region
#: is rendered from :data:`DEFAULT_GATED_CLAIMS` by
#: ``menagerie.crawler.tools.render_claim_vocabulary`` and re-derived on every test run,
#: so the prompt cannot silently drift out of step with the set the gate enforces. A
#: hand-copied list is what produced the closed-vocabulary wall in the first place: the
#: required strings existed only in Python and were invisible to the author.
CLAIM_VOCABULARY_BEGIN = "<<<BEGIN GENERATED CLAIM VOCABULARY -- DO NOT HAND-EDIT>>>"
CLAIM_VOCABULARY_END = "<<<END GENERATED CLAIM VOCABULARY>>>"


def gated_claim_vocabulary_block() -> str:
    """Render the closed gated-claim vocabulary exactly as the prompt must carry it.

    Returns
    -------
    str
        Delimited block listing every claim-category string a ``supports`` entry may be
        matched against, derived from :data:`DEFAULT_GATED_CLAIMS`.
    """

    always = sorted(DEFAULT_GATED_CLAIMS - CONDITIONAL_GATED_CLAIMS)
    conditional = sorted(CONDITIONAL_GATED_CLAIMS)
    lines = [CLAIM_VOCABULARY_BEGIN]
    lines.append(
        f"These {len(always)} strings are ALWAYS required. A supports entry is matched by"
    )
    lines.append("EXACT STRING EQUALITY against this list. Nothing else counts.")
    lines.extend(f"  {claim}" for claim in always)
    lines.append("")
    lines.append(
        f"These {len(conditional)} are required whenever the introducing paper is a fetched"
    )
    lines.append("source or you volunteer a citation:")
    lines.extend(f"  {claim}" for claim in conditional)
    lines.append(CLAIM_VOCABULARY_END)
    return "\n".join(lines)


#: The one gated claim that is a relevance judgment rather than an excerpt-tagged
#: fact. It is excluded from :data:`DEFAULT_GATED_CLAIMS` because author-side
#: evidence coverage cannot demand an excerpt for it, but the accuracy gate still
#: requires the checker's independent verdict on it, so the required-check
#: derivation adds it back explicitly.
KEYWORD_CLAIM = "external_metadata.keywords"


def required_metadata_field_checks(facts: Mapping[str, Any]) -> tuple[str, ...]:
    """Derive the closed, ordered field-check set one metadata gate item must cover.

    This is the checker-facing half of the same closed gated-claim vocabulary the
    author's evidence coverage is validated against (:data:`DEFAULT_GATED_CLAIMS`),
    so the two gates verify one contract: every claim the author had to ground with
    excerpt tags or a typed availability state is exactly the claim set the checker
    must return one independent verdict for. The derivation is machine-owned,
    deterministic, and stably ordered; it is computed from the proposed facts alone
    so the checker-item builder, the envelope boundary, and the canonical-write
    validator all derive byte-identical sets from the same bytes.

    The citation claim follows the author-side conditionality: it is required
    whenever the proposal asserts a present citation or binds a controlled-fetched
    paper-role source. The declared source rows mirror the frozen manifest -- a
    divergent set is refused upstream as ``proposal and source manifest source sets
    differ`` -- so deriving from the declared rows needs no manifest.

    Parameters
    ----------
    facts:
        Complete ``proposed_facts`` tree, or the corresponding canonical fact
        mapping recovered at write time.

    Returns
    -------
    tuple[str, ...]
        Sorted claim paths; exactly one ``field_check`` per member is required.
    """

    claims = set(DEFAULT_GATED_CLAIMS - CONDITIONAL_GATED_CLAIMS)
    claims.add(KEYWORD_CLAIM)
    if _citation_is_present(facts) or _declared_fetched_paper_sources(facts):
        claims.add("external_metadata.citation")
    return tuple(sorted(claims))


def _declared_fetched_paper_sources(facts: Mapping[str, Any]) -> bool:
    """Return whether the facts declare a hash-bound paper-role source.

    Facts-only mirror of :func:`_fetched_paper_source_ids`: the declared rows
    carry the same ``content_sha256`` binding as their frozen manifest rows, and a
    proposal whose declared set diverges from its manifest never reaches a gate.

    Parameters
    ----------
    facts:
        Complete proposed fact tree, possibly partial in non-proposal contexts.

    Returns
    -------
    bool
        True when a declared source row has a paper role and a content digest.
    """

    resolution = facts.get("source_resolution")
    declared = resolution.get("sources") if isinstance(resolution, Mapping) else None
    if not isinstance(declared, list):
        return False
    return any(
        isinstance(source, Mapping)
        and source.get("role") in PAPER_EVIDENCE_ROLES
        and isinstance(source.get("content_sha256"), str)
        and bool(source.get("content_sha256"))
        for source in declared
    )
VERIFIED_HASH_CODE_MANIFEST_KEY = "code_manifest"
_AUTHOR_VERIFIED_HASH_SPEC = required_field_projection_spec(
    RequiredFieldProjection.AUTHOR_PROPOSAL_VERIFIED_HASH
)
_AUTHOR_VERIFIED_HASH_KEYS = _AUTHOR_VERIFIED_HASH_SPEC.names_for(SchemaOwner.REDUCER_DERIVED)
VERIFIED_HASH_COMMON_KEYS = frozenset(
    key for key in _AUTHOR_VERIFIED_HASH_KEYS if key != VERIFIED_HASH_CODE_MANIFEST_KEY
)
_GATE_VERIFIED_HASH_SPEC = required_field_projection_spec(
    RequiredFieldProjection.GATE_VERIFIED_HASH
)
VERIFIED_HASH_PROPOSAL_KEY = _GATE_VERIFIED_HASH_SPEC.field_order[-1]
_FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile"})
_BUILTIN_NAMESPACE_ROOTS = frozenset({"builtins", "__builtins__"})
_FORBIDDEN_DOTTED_CALLS = frozenset(
    {
        "torch.compile",
        # Known library entry points that evaluate strings as code. These are
        # dynamic execution with a module prefix, refused by EXACT dotted name
        # (canonical module plus its ubiquitous import alias), never by pattern:
        # pattern-matching on the last segment is precisely the rule that made
        # ``model.eval()`` unsatisfiable.
        "pandas.eval",
        "pd.eval",
        "numexpr.evaluate",
        "numexpr.re_evaluate",
        "ne.evaluate",
    }
)
"""Dotted calls refused by exact name rather than by their last segment.

The last-segment rule that used to stand in for this refused every method whose
name happens to collide with a builtin, and the collisions are not exotic:
``model.eval()`` is how a staged adapter enters eval mode, which the author
contract REQUIRES it to do, and ``re.compile()`` is ordinary library use. Pilot
model ``m8189``'s adapter was refused for ``model.eval`` -- an unsatisfiable
instruction pair, since the prompt mandates the call the validator rejects.

Reaching the real builtin through an attribute needs the builtin namespace, so
that is matched by ROOT instead; ``torch.compile`` stays refused by name because
logging a compiled artifact is a separate locked anti-pattern; and the named
string-evaluation entry points of pandas and numexpr are refused so that
dropping the suffix rule does not silently legalise ``pd.eval("...")``.

What this denylist explicitly does NOT cover, and cannot: dynamic-evaluation
METHODS on arbitrary receivers (``df.eval("...")`` is statically
indistinguishable from ``model.eval()``), non-standard import aliases
(``import pandas as q; q.eval(...)``), and any module we have never heard of
that evaluates strings. A static name list bounds known entry points; it is not,
and must never be presented as, a sandbox. The runtime sandbox the staged code
executes under is the actual containment boundary.
"""
_SLOP_PATTERNS = (
    r"\bcompact\s+(?:stand[- ]?in|substitute|approximation|version)\b",
    r"\bgeneric\s+(?:stand[- ]?in|substitute|implementation|version)\b",
    r"\b(?:knowingly\s+)?simplif(?:ied|ication)\b",
    r"\b(?:rough(?:ly)?\s+)?approximat(?:e|ed|ion)\b",
    r"\btoy\s+(?:model|implementation|version|replica|example)\b",
    r"\b(?:stand[- ]?in|placeholder|mock|surrogate|proxy)\b",
    r"\b(?:minimal|lightweight|reduced)\s+(?:facsimile|imitation|replica|substitute)\b",
    r"\brepresentative\s+(?:approximation|substitute|implementation)\b",
)
_SUPPORT_ALIASES = {
    "citation": "external_metadata.citation",
    "country": "external_metadata.country",
    "license": "external_metadata.license",
    "year": "external_metadata.year",
}
_GENERIC_MODEL_CALLS = frozenset(
    {
        "AdaptiveAvgPool1d",
        "AdaptiveAvgPool2d",
        "AvgPool1d",
        "AvgPool2d",
        "BatchNorm1d",
        "BatchNorm2d",
        "Conv1d",
        "Conv2d",
        "Dropout",
        "Flatten",
        "GELU",
        "LayerNorm",
        "Linear",
        "MaxPool1d",
        "MaxPool2d",
        "ReLU",
        "Sequential",
        "Sigmoid",
        "Softmax",
        "Tanh",
    }
)
_GENERIC_FAMILY_NAMES = frozenset(
    {"feedforward", "mlp", "multilayer perceptron", "sequential", "simple neural network"}
)
_SUPPORT_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "architecture",
        "based",
        "from",
        "into",
        "model",
        "network",
        "source",
        "that",
        "their",
        "this",
        "using",
        "with",
    }
)
_WRITE_METHODS = frozenset(
    {"write_text", "write_bytes", "touch", "mkdir", "rename", "replace", "unlink", "rmdir"}
)
_IMPLEMENTATION_SOURCE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".cxx",
        ".go",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".ipynb",
        ".java",
        ".jl",
        ".js",
        ".kt",
        ".lua",
        ".m",
        ".mm",
        ".py",
        ".pyx",
        ".r",
        ".rs",
        ".scala",
        ".swift",
        ".ts",
    }
)
_NON_IMPLEMENTATION_CODE_NAMES = frozenset(
    {
        "__init__.py",
        "conftest.py",
        "setup.py",
        "version.py",
    }
)
_NON_IMPLEMENTATION_SOURCE_STEMS = frozenset({"metric", "metrics", "plot", "plots", "plotting"})
_NON_IMPLEMENTATION_SOURCE_DIRS = frozenset({".github", "ci", "doc", "docs", "test", "tests"})
_MODEL_ENTRY_METHODS = frozenset({"__call__", "apply", "call", "forward"})
_MAX_STREAM_INVENTORY_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class _InventoryMember:
    """One implementation-role archive member and its structural links."""

    name: str
    text: str
    linked: bool
    structured: bool
    defined_symbols: frozenset[str]
    referenced_symbols: frozenset[str]
    imported_modules: frozenset[str]


class ProposalValidationError(ValueError):
    """Raised when a staged proposal fails schema, grounding, or anti-slop checks."""


@dataclass(frozen=True)
class ProposalValidationReport:
    """Summary of a fully validated staged proposal.

    Parameters
    ----------
    stable_id:
        Validated model identity.
    rung:
        Validated source-ladder rung.
    code_path:
        Validated staged code path, if applicable.
    supported_claims:
        Claim categories backed by literal excerpts.
    """

    stable_id: str
    rung: SourceRung
    code_path: Optional[Path]
    supported_claims: frozenset[str]


def validate_author_proposal(
    proposal: Mapping[str, Any],
    *,
    allowed_model_dir: Union[str, Path],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    required_claims: Optional[Iterable[str]] = None,
    cas_root: Union[str, Path, None] = None,
    expected_schema_version: str = AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
) -> ProposalValidationReport:
    """Validate one complete author proposal without modifying it.

    Parameters
    ----------
    proposal:
        Complete ``author-proposal.v3`` object.
    allowed_model_dir:
        Only directory in which staged typed code may reside or write.
    source_manifest:
        Exact controlled-fetch source manifests.
    required_claims:
        Optional gated claim categories. The default enforces the plan's core
        externally-authored categories.
    cas_root:
        Optional source CAS root.
    expected_schema_version:
        Exact current proposal discriminator required by the caller.

    Returns
    -------
    ProposalValidationReport
        Immutable validation summary.

    Raises
    ------
    ProposalValidationError
        If schema, evidence, code, rung, link, or anti-slop validation fails.
    """

    try:
        validate_payload(proposal, expected_schema_version)
    except PayloadValidationError as exc:
        raise ProposalValidationError(str(exc)) from exc
    _validate_verified_hash_keys(proposal)
    facts = _mapping(proposal.get("proposed_facts"), "proposed_facts")
    resolution = _mapping(facts.get("source_resolution"), "source_resolution")
    try:
        rung = SourceRung(str(resolution.get("rung")))
    except ValueError as exc:
        raise ProposalValidationError("source_resolution.rung is not canonical") from exc
    _validate_mandatory_source_link(resolution)
    _validate_description(facts)
    evidence = _mapping(facts.get("evidence"), "evidence")
    claims = set(required_claims if required_claims is not None else DEFAULT_GATED_CLAIMS)
    # The citation is gated whenever a paper source is bound, not only when the author
    # volunteers one: supplying a true fact must never be what triggers extra checks
    # while omitting it sails through.
    if _citation_is_present(facts) or _fetched_paper_source_ids(facts, source_manifest):
        claims.add("external_metadata.citation")
    try:
        evidence_report = validate_evidence(
            evidence,
            source_manifest,
            claims,
            cas_root=cas_root,
            require_family_grounding=True,
            declared_absences=declared_absence_coverage(facts, claims),
        )
    except EvidenceValidationError as exc:
        raise ProposalValidationError(str(exc)) from exc
    known_evidence = evidence_ids(evidence)
    _validate_claim_support(facts, evidence, claims, known_evidence)
    _validate_website_grounding(facts, evidence)
    _validate_citation(facts, known_evidence)
    _validate_citation_consistency(facts)
    _validate_paper_evidence_source(facts, evidence, source_manifest)
    implementation = _mapping(facts.get("implementation"), "implementation")
    allowed_dir = Path(allowed_model_dir).resolve()
    _validate_author_read_grants(facts, allowed_dir)
    code_path = _validate_code(implementation, rung, allowed_dir, source_manifest)
    _validate_source_ladder(
        rung,
        facts,
        resolution,
        implementation,
        evidence,
        known_evidence,
        source_manifest,
        cas_root,
    )
    code_paths = resolve_model_code_closure(code_path, allowed_dir) if code_path is not None else ()
    _validate_structural_slop(facts, code_paths)
    _validate_anti_slop(facts)
    return ProposalValidationReport(
        stable_id=str(proposal["stable_id"]),
        rung=rung,
        code_path=code_path,
        supported_claims=evidence_report.supported_claims,
    )


def required_verified_hash_keys(
    proposal: Mapping[str, Any], *, include_proposal: bool = False
) -> frozenset[str]:
    """Return the exact verified-hash keys required for a proposal type.

    Parameters
    ----------
    proposal:
        Proposal whose implementation determines whether recursive model code
        must be bound.
    include_proposal:
        Whether to include the checker-only digest of the complete proposal.

    Returns
    -------
    frozenset[str]
        Exact key set for a declarative or typed proposal binding.

    Raises
    ------
    ProposalValidationError
        If the proposal does not expose an unambiguous implementation type.
    """

    facts = proposal.get("proposed_facts")
    implementation = facts.get("implementation") if isinstance(facts, Mapping) else None
    if not isinstance(implementation, Mapping):
        raise ProposalValidationError("proposal implementation is incomplete")
    code_path = implementation.get("code_path")
    if code_path is not None and (not isinstance(code_path, str) or not code_path.strip()):
        raise ProposalValidationError("implementation.code_path must be null or non-empty")
    keys = set(VERIFIED_HASH_COMMON_KEYS)
    if isinstance(code_path, str):
        keys.add(VERIFIED_HASH_CODE_MANIFEST_KEY)
    if include_proposal:
        keys.add(VERIFIED_HASH_PROPOSAL_KEY)
    return frozenset(keys)


def _validate_verified_hash_keys(proposal: Mapping[str, Any]) -> None:
    """Validate the proposal-side exact verified-hash key binding.

    Parameters
    ----------
    proposal:
        Complete author proposal.

    Raises
    ------
    ProposalValidationError
        If typed code lacks a recursive manifest digest or declarative code
        carries one.
    """

    verified_hashes = proposal.get("verified_hashes")
    required_keys = required_verified_hash_keys(proposal)
    if not isinstance(verified_hashes, Mapping) or set(verified_hashes) != required_keys:
        raise ProposalValidationError(
            "verified_hashes must bind the exact declarative or typed proposal artifact pack"
        )


def _validate_mandatory_source_link(resolution: Mapping[str, Any]) -> None:
    """Validate the public primary-link invariant.

    Parameters
    ----------
    resolution:
        Source-resolution block.

    Raises
    ------
    ProposalValidationError
        If the primary source link is absent or inconsistent.
    """

    if resolution.get("mandatory_link_status") != "ok":
        raise ProposalValidationError("mandatory source link is not satisfied")
    primary_id = resolution.get("primary_source_id")
    sources = resolution.get("sources")
    if not isinstance(primary_id, str) or not primary_id or not isinstance(sources, list):
        raise ProposalValidationError("primary source identity is missing")
    primary = next(
        (
            source
            for source in sources
            if isinstance(source, Mapping) and source.get("source_id") == primary_id
        ),
        None,
    )
    if primary is None:
        raise ProposalValidationError("primary_source_id does not name a declared source")
    url = primary.get("url")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise ProposalValidationError("primary source must have an exact public URL")


def _validate_description(facts: Mapping[str, Any]) -> None:
    """Reject absent or whitespace-only authored descriptions.

    Parameters
    ----------
    facts:
        Proposed fact tree.

    Raises
    ------
    ProposalValidationError
        If external or website description is empty.
    """

    metadata = _mapping(facts.get("external_metadata"), "external_metadata")
    website = _mapping(facts.get("website"), "website")
    for field, value in (
        ("external_metadata.description", metadata.get("description")),
        ("website.description", website.get("description")),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ProposalValidationError(f"{field} must be non-empty")


def _validate_website_grounding(
    facts: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    """Resolve ``website.family_grounding_id`` to a real family-level excerpt.

    The leaf is a REFERENCE, and the governing ownership rule is that a reference
    the machine can dereference must be dereferenced: before this check it was
    free text that nothing read, so the one field naming which excerpt grounds
    the family prose of a permanent public record was unverifiable by anyone.
    Every proposal is already required to carry at least one ``family_level``
    excerpt (``validate_evidence(require_family_grounding=True)``), so pointing
    at it is always satisfiable and costs the author nothing new.

    Parameters
    ----------
    facts:
        Complete proposed fact tree.
    evidence:
        Literal evidence block already verified against source bytes.

    Raises
    ------
    ProposalValidationError
        If the id names no excerpt, or names one that is not family-level.
    """

    website = _mapping(facts.get("website"), "website")
    grounding_id = website.get("family_grounding_id")
    if not isinstance(grounding_id, str) or not grounding_id.strip():
        raise ProposalValidationError("website.family_grounding_id must be a non-empty string")
    excerpts = evidence.get("excerpts")
    if not isinstance(excerpts, list):
        raise ProposalValidationError("evidence.excerpts must be a list")
    for excerpt in excerpts:
        if isinstance(excerpt, Mapping) and excerpt.get("evidence_id") == grounding_id:
            if excerpt.get("family_level") is not True:
                raise ProposalValidationError(
                    "website.family_grounding_id must name a family_level excerpt: "
                    f"{grounding_id} is not family-level"
                )
            return
    raise ProposalValidationError(
        "website.family_grounding_id references missing or fabricated evidence: "
        f"{grounding_id}"
    )


def _citation_is_present(facts: Mapping[str, Any]) -> bool:
    """Return whether the proposal asserts an introducing citation.

    Parameters
    ----------
    facts:
        Proposed fact tree.

    Returns
    -------
    bool
        True for a present citation.
    """

    citation = facts.get("citation")
    return isinstance(citation, Mapping) and citation.get("status") == "present"


def _validate_citation(facts: Mapping[str, Any], known_evidence: frozenset[str]) -> None:
    """Reject fabricated or evidence-free citation claims.

    Parameters
    ----------
    facts:
        Proposed fact tree.
    known_evidence:
        Literal evidence identifiers.

    Raises
    ------
    ProposalValidationError
        If a present citation lacks a source, identity, or valid evidence.
    """

    citation = _mapping(facts.get("citation"), "citation")
    if citation.get("status") != "present":
        # Declaring no citation must cost at least the search that establishes it. The
        # gate previously returned here unconditionally, which made "there is no paper"
        # the cheapest possible answer and the one an author under budget pressure is
        # trained to give.
        _validate_absence_is_searched(facts, ["citation"])
        return
    if not all(
        isinstance(citation.get(field), str) and str(citation[field]).strip()
        for field in ("title", "url")
    ):
        raise ProposalValidationError("present citation must name a title and public URL")
    cited_ids = citation.get("source_evidence_ids")
    if not isinstance(cited_ids, list) or not cited_ids or not set(cited_ids) <= known_evidence:
        raise ProposalValidationError("citation references missing or fabricated evidence")


def _validate_citation_consistency(facts: Mapping[str, Any]) -> None:
    """Require the public citation to equal the accuracy-gated metadata citation.

    Parameters
    ----------
    facts:
        Proposed fact tree.

    Raises
    ------
    ProposalValidationError
        If the top-level and external-metadata citation blocks diverge.
    """

    metadata = _mapping(facts.get("external_metadata"), "external_metadata")
    external_citation = _mapping(metadata.get("citation"), "external_metadata.citation")
    citation = _mapping(facts.get("citation"), "citation")
    if external_citation != citation:
        raise ProposalValidationError(
            "top-level citation differs from accuracy-checked external_metadata.citation"
        )


def _validate_paper_evidence_source(
    facts: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> None:
    """Require an asserted citation to be grounded in controlled-fetched paper bytes.

    The gated claims are grounded on literal excerpts drawn from the frozen source
    manifest, and roughly half of them -- ``authors``, ``institution``, ``country``,
    ``venue``, ``year``, ``era``, ``citation`` -- are paper metadata that does not occur
    in implementation code. An author whose manifest holds only code therefore cannot
    ground them from any honest excerpt. The remedy is more evidence, never a looser
    check: when the proposal asserts an introducing work, that work's own page must be a
    controlled-fetch source in the frozen manifest, and the citation must be grounded on
    its exact bytes. This also removes a real fabrication surface, because a citation URL
    the author merely recalled can no longer support the claim.

    Parameters
    ----------
    facts:
        Complete proposed fact tree.
    evidence:
        Literal evidence block already verified against controlled source bytes.
    source_manifest:
        Exact controlled-fetch source rows.

    Raises
    ------
    ProposalValidationError
        If a present citation names no fetched paper-role source, or no excerpt from
        such a source supports the citation claim.
    """

    paper_source_ids = _fetched_paper_source_ids(facts, source_manifest)
    if not _citation_is_present(facts):
        if paper_source_ids:
            citation = _mapping(facts.get("citation"), "citation")
            raise ProposalValidationError(
                "citation availability cannot be "
                f"{citation.get('status')!r} while the introducing paper is a "
                "controlled-fetched source; the fetched paper names its own citation"
            )
        return
    if not paper_source_ids:
        raise ProposalValidationError(
            "a present citation requires the introducing paper or landing page as a "
            "controlled-fetched source (role in "
            f"{sorted(PAPER_EVIDENCE_ROLES)}); paper metadata such as authors, venue, "
            "institution, country, and year does not occur in implementation code and "
            "cannot be grounded from it"
        )
    excerpts = evidence.get("excerpts")
    if not isinstance(excerpts, list):
        raise ProposalValidationError("evidence.excerpts must be a list")
    paper_texts = [
        str(excerpt.get("text"))
        for excerpt in excerpts
        if isinstance(excerpt, Mapping)
        and str(excerpt.get("source_id")) in paper_source_ids
        and isinstance(excerpt.get("text"), str)
        and any(
            _SUPPORT_ALIASES.get(support, support) == "external_metadata.citation"
            for support in excerpt.get("supports", [])
            if isinstance(support, str)
        )
    ]
    if not paper_texts:
        raise ProposalValidationError(
            "external_metadata.citation must be supported by a literal excerpt from the "
            "controlled-fetched paper source, not only from implementation code"
        )
    _validate_citation_leaves(_mapping(facts.get("citation"), "citation"), paper_texts)


def _fetched_paper_source_ids(
    facts: Mapping[str, Any],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> frozenset[str]:
    """Return declared paper-role sources backed by controlled-fetched bytes.

    Parameters
    ----------
    facts:
        Complete proposed fact tree.
    source_manifest:
        Exact controlled-fetch source rows.

    Returns
    -------
    frozenset[str]
        Source identifiers whose declared role is a paper role and whose bytes are
        hash-bound in the frozen manifest.

    Raises
    ------
    ProposalValidationError
        If the declared source list is malformed.
    """

    resolution = _mapping(facts.get("source_resolution"), "source_resolution")
    declared = resolution.get("sources")
    if not isinstance(declared, list):
        raise ProposalValidationError("source_resolution.sources must be a list")
    fetched = _source_manifest_index(source_manifest)
    return frozenset(
        str(source["source_id"])
        for source in declared
        if isinstance(source, Mapping)
        and source.get("role") in PAPER_EVIDENCE_ROLES
        and isinstance(source.get("source_id"), str)
        and _is_controlled_fetch(fetched.get(str(source["source_id"])))
    )


def _is_controlled_fetch(source: Optional[Mapping[str, Any]]) -> bool:
    """Return whether one manifest row names hash-bound controlled-fetch bytes.

    Parameters
    ----------
    source:
        Candidate frozen manifest row, or ``None`` when the identifier is absent.

    Returns
    -------
    bool
        True when the row records a completed retrieval bound to a content digest.
    """

    if not isinstance(source, Mapping):
        return False
    digest = source.get("content_sha256")
    status = source.get("retrieval_status")
    # Presence in the frozen manifest is itself the controlled-fetch record, so the
    # binding requirement is the content digest. A declared retrieval status may
    # corroborate it but must never contradict it.
    return (
        isinstance(digest, str)
        and bool(digest)
        and (status is None or status in {"fetched", "already-present"})
    )


def _validate_claim_support(
    facts: Mapping[str, Any],
    evidence: Mapping[str, Any],
    required_claims: Iterable[str],
    known_evidence: frozenset[str],
) -> None:
    """Validate structure and provenance of every gated claim, per-leaf.

    The deterministic layer decides only what it can actually decide: every gated
    claim binds at least one hash-verified excerpt, no gated claim is bare null or
    empty, and an empty judgment claim carries a structurally valid typed availability
    state. Semantic entailment of judgment values is the Codex checker's job
    (:data:`CHECKER_EVALUATED_CLAIMS`); citation leaves are value-checked against the
    fetched paper bytes in :func:`_validate_paper_evidence_source`. The former
    token-overlap entailment oracle is deleted, not tuned: it certified fabrication
    (``country="US"`` on the pronoun "us") while refusing honest facts (diacritic
    spellings, ISO codes, era buckets), and no strictness setting makes that sound.

    Parameters
    ----------
    facts:
        Complete proposed fact tree.
    evidence:
        Literal evidence block already verified against source bytes.
    required_claims:
        Claim paths requiring deterministic support.
    known_evidence:
        Valid literal evidence identifiers.

    Raises
    ------
    ProposalValidationError
        If a claim has no bound excerpt, is bare-empty, or declares an invalid
        availability state.
    """

    excerpts = evidence.get("excerpts")
    if not isinstance(excerpts, list):
        raise ProposalValidationError("evidence.excerpts must be a list")
    support_texts: dict[str, list[str]] = {}
    for excerpt in excerpts:
        if not isinstance(excerpt, Mapping):
            continue
        text = excerpt.get("text")
        supports = excerpt.get("supports")
        if not isinstance(text, str) or not isinstance(supports, list):
            continue
        for support in supports:
            if isinstance(support, str):
                canonical = _SUPPORT_ALIASES.get(support, support)
                support_texts.setdefault(canonical, []).append(text)

    absence_covered = frozenset(declared_absence_coverage(facts, required_claims))
    unsupported: list[str] = []
    for claim in required_claims:
        canonical = _SUPPORT_ALIASES.get(claim, claim)
        if not support_texts.get(canonical):
            if canonical not in absence_covered:
                unsupported.append(canonical)
                continue
            # A typed absence state stands in for excerpt text: there is no excerpt that
            # says a fact is not there. The record itself is still validated below.
            _validate_claim_state(
                canonical, _claim_value(facts, canonical), facts, known_evidence
            )
            continue
        if canonical == "external_metadata.citation":
            # Citation leaves are value-checked against the paper source bytes in
            # `_validate_paper_evidence_source`; availability is checked in
            # `_validate_citation`.
            continue
        value = _claim_value(facts, canonical)
        _validate_claim_state(canonical, value, facts, known_evidence)
    if unsupported:
        raise ProposalValidationError(
            "evidence excerpts do not substantively support claimed values: "
            f"{sorted(set(unsupported))}"
        )


def _validate_claim_state(
    claim: str, value: object, facts: Mapping[str, Any], known_evidence: frozenset[str]
) -> None:
    """Require a gated claim to carry a value or a typed availability state.

    A null/omitted value used to satisfy the gate for free, which trained authors to
    write null: the cheapest route past the gate was to empty exactly the fields the
    catalog exists to collect, and a run-once campaign would have reported success over
    hollow records. Omission is now itself a checkable, evidence-carrying assertion.

    Parameters
    ----------
    claim:
        Canonical claim path.
    value:
        Proposed value at that path.
    facts:
        Complete proposed fact tree.
    known_evidence:
        Valid literal evidence identifiers.

    Raises
    ------
    ProposalValidationError
        If the claim is bare-empty, or its availability state is structurally invalid
        or contradicts the carried value.
    """

    record = _availability_record(facts, claim)
    empty = _claim_is_hollow(claim, value)
    if record is None:
        if empty:
            raise ProposalValidationError(
                f"gated claim {claim} is bare null/empty; a value must be present or the "
                "claim must declare a typed availability state "
                "(external_metadata.availability) of none-exist, not-found-after-search, "
                "or not-applicable with its evidence"
            )
        return
    if claim not in AVAILABILITY_CLAIMS:
        raise ProposalValidationError(
            f"availability states are not declarable for gated claim {claim}"
        )
    _validate_availability_record(claim, record, value, empty, facts, known_evidence)


def _availability_key(claim: str) -> Optional[str]:
    """Return the availability-register key for one claim, if it has one.

    ``external_metadata`` claims are keyed by their bare field name, which is how the
    register has always been spelled. Claims owned by another block are keyed by their
    FULL canonical path, so a key can never be confused with a field of a different
    block that happens to share a leaf name.

    Parameters
    ----------
    claim:
        Canonical claim path.

    Returns
    -------
    str | None
        Register key, or ``None`` when the claim has no availability route.
    """

    prefix = "external_metadata."
    if claim.startswith(prefix):
        return claim.removeprefix(prefix)
    if claim in FOREIGN_AVAILABILITY_KEYS:
        return claim
    return None


def _availability_record(facts: Mapping[str, Any], claim: str) -> Optional[Mapping[str, Any]]:
    """Return the declared availability record for one gated claim.

    Parameters
    ----------
    facts:
        Complete proposed fact tree.
    claim:
        Canonical claim path.

    Returns
    -------
    Mapping[str, Any] | None
        Declared availability record, or ``None`` when absent.

    Raises
    ------
    ProposalValidationError
        If the availability block is present but not an object of objects.
    """

    key = _availability_key(claim)
    if key is None:
        return None
    metadata = facts.get("external_metadata")
    if not isinstance(metadata, Mapping):
        return None
    availability = metadata.get("availability")
    if availability is None:
        return None
    if not isinstance(availability, Mapping):
        raise ProposalValidationError("external_metadata.availability must be an object")
    record = availability.get(key)
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise ProposalValidationError(f"availability state for {claim} must be an object")
    return record


def declared_absence_coverage(
    facts: Mapping[str, Any], required_claims: Iterable[str]
) -> dict[str, list[str]]:
    """Return the claims a typed absence state covers, with the evidence each cites.

    This is the ONLY route by which an unfilled gated claim reaches coverage, and it is
    strictly narrower than tagging an arbitrary excerpt: the claim must be permitted an
    availability state at all, the state must assert absence rather than presence, and
    the record must be structurally well-formed. Everything else the record asserts --
    the bounded search, the value/status agreement, the evidence IDs actually existing
    -- is enforced by :func:`_validate_availability_record` in the same validation pass.

    Parameters
    ----------
    facts:
        Complete proposed fact tree.
    required_claims:
        Gated claim categories under evaluation.

    Returns
    -------
    dict[str, list[str]]
        Claim category to the evidence IDs its absence state cites.

    Raises
    ------
    ProposalValidationError
        If the availability block is structurally malformed.
    """

    coverage: dict[str, list[str]] = {}
    for claim in required_claims:
        if claim not in AVAILABILITY_CLAIMS:
            continue
        record = _availability_record(facts, claim)
        if record is None or record.get("status") not in ABSENCE_AVAILABILITY_STATUSES:
            continue
        cited = record.get("evidence")
        if not isinstance(cited, list):
            # Structurally invalid; _validate_availability_record raises the precise
            # error. Granting no coverage here keeps the failure a refusal either way.
            continue
        coverage[claim] = [
            evidence_id for evidence_id in cited if isinstance(evidence_id, str)
        ]
    return coverage


def _validate_availability_record(
    claim: str,
    record: Mapping[str, Any],
    value: object,
    empty: bool,
    facts: Mapping[str, Any],
    known_evidence: frozenset[str],
) -> None:
    """Validate one typed availability state structurally, per-leaf.

    ``not-found-after-search`` is a positive, evidence-carrying claim: it must sit on
    the recorded bounded search that could have found the fact, so an honestly-unknown
    author or institution is exactly as auditable as a present one. Whether the pinned
    evidence actually *supports* a present value -- or contradicts a claimed absence --
    is the Codex checker's per-leaf judgment, not token matching.

    Parameters
    ----------
    claim:
        Canonical claim path.
    record:
        Declared availability record.
    value:
        Proposed claim value.
    empty:
        Whether the carried value is empty.
    facts:
        Complete proposed fact tree.
    known_evidence:
        Valid literal evidence identifiers.

    Raises
    ------
    ProposalValidationError
        If the record is structurally invalid or contradicts the carried value.
    """

    if set(record) != AVAILABILITY_RECORD_KEYS:
        raise ProposalValidationError(
            f"availability state for {claim} must carry exactly {sorted(AVAILABILITY_RECORD_KEYS)}"
        )
    status = record.get("status")
    if status not in AVAILABILITY_STATUSES:
        raise ProposalValidationError(
            f"availability state for {claim} has a non-canonical status: {status!r}; "
            f"status is a closed vocabulary, one of {sorted(AVAILABILITY_STATUSES)}"
        )
    basis = record.get("basis")
    if basis not in AVAILABILITY_BASES:
        # Name the vocabulary in the refusal. Two of twenty models in the 2026-08-05
        # rung died here on their FIRST and ONLY attempt, both by writing an honest
        # descriptive phrase for what they actually did ('bounded-source-read',
        # 'bounded-frozen-source-read') where the enum wanted 'search-exhausted'. The
        # schema said "closed-vocabulary basis" without declaring the vocabulary, so
        # the members existed only in Python -- the exact shape of the wall the
        # generated claim vocabulary was built to remove. The enum is now declared in
        # both schemas; this message is the second surface, for the human reading the
        # terminal record.
        raise ProposalValidationError(
            f"availability state for {claim} has a non-canonical basis: {basis!r}; "
            f"basis is a closed vocabulary, one of {sorted(AVAILABILITY_BASES)} "
            "(an absence established by reading the frozen sources is 'search-exhausted')"
        )
    values = record.get("values")
    if not isinstance(values, list):
        raise ProposalValidationError(f"availability state for {claim} values must be a list")
    cited = record.get("evidence")
    if not isinstance(cited, list) or not all(
        isinstance(evidence_id, str) and evidence_id for evidence_id in cited
    ):
        raise ProposalValidationError(
            f"availability state for {claim} evidence must be a list of evidence IDs"
        )
    if not set(cited) <= known_evidence:
        raise ProposalValidationError(
            f"availability state for {claim} references missing or fabricated evidence"
        )
    if status == "present":
        if empty or not values:
            raise ProposalValidationError(
                f"a present availability state for {claim} requires the field to carry "
                "its non-empty values"
            )
        if sorted(map(str, _positive_scalars(value))) != sorted(map(str, values)):
            raise ProposalValidationError(
                f"availability state for {claim} does not match the proposed value"
            )
        if not cited:
            raise ProposalValidationError(
                f"a present availability state for {claim} requires supporting evidence"
            )
        return
    if not empty or values:
        raise ProposalValidationError(
            f"availability state for {claim} declares {status} but the field carries a value"
        )
    if status in SEARCH_BACKED_AVAILABILITY_STATUSES:
        # Until the source broker ships probe receipts, the recorded bounded search IS
        # the evidence for an absence state; explicit excerpt IDs may corroborate it.
        # ``none-exist`` is held to the same bar as ``not-found-after-search``: asserting
        # that a model HAS no predecessors is a finding, and a finding needs the search
        # that produced it, or "there are none" becomes the cheapest thing to write.
        _validate_absence_is_searched(facts, [claim])


def _claim_is_hollow(claim: str, value: object) -> bool:
    """Return whether a gated claim was left empty rather than answered.

    An empty value used to satisfy the gate for free, which made omission the cheapest
    way past every check the campaign exists to enforce. A run that completes with its
    metadata silently blank is a worse outcome than one that stops, because nothing
    surfaces it.

    :data:`EMPTIABLE_CLAIMS` are no longer short-circuited to "not hollow" here. Empty
    IS a real fact for them, but a real fact is a claim and a claim is declared: they
    reach the gate through a typed ``none-exist`` availability state, not through an
    exemption. Reporting emptiness as non-hollow made the state unrecordable and
    unqueryable, and left tagging an unrelated excerpt as the only way to pass coverage.

    Parameters
    ----------
    claim:
        Canonical claim path.
    value:
        Proposed value at that path.

    Returns
    -------
    bool
        True when the claim carries no answer.
    """

    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        # A structured block is hollow when every leaf it is judged on is absent.
        # Bookkeeping keys are not answers: a citation carrying nothing but a status
        # and the evidence IDs that point at nothing is exactly the hollow case.
        return not _positive_scalars(
            {key: item for key, item in value.items() if key not in _HOLLOW_BOOKKEEPING_KEYS}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return not _positive_scalars(value)
    return False


def _validate_absence_is_searched(facts: Mapping[str, Any], hollow: Sequence[str]) -> None:
    """Require an empty gated claim to sit behind a bounded search that could find it.

    This does not make emptiness free and does not make it impossible. It makes it
    *accountable*: a proposal may report that a fact is not there, but only after
    recording the search that looked for it, so an unfilled field is distinguishable
    from a genuinely absent one.

    Parameters
    ----------
    facts:
        Complete proposed fact tree.
    hollow:
        Gated claims left empty.

    Raises
    ------
    ProposalValidationError
        If the bounded search report does not record a real search.
    """

    resolution = _mapping(facts.get("source_resolution"), "source_resolution")
    search_report = resolution.get("search_report")
    queries = search_report.get("queries") if isinstance(search_report, Mapping) else None
    conclusion = search_report.get("conclusion") if isinstance(search_report, Mapping) else None
    if (
        not isinstance(queries, list)
        or not any(isinstance(query, str) and query.strip() for query in queries)
        or not isinstance(conclusion, str)
        or not conclusion.strip()
    ):
        raise ProposalValidationError(
            "an absence state (none-exist or not-found-after-search) requires a recorded "
            f"bounded search that could have found the fact: {sorted(set(hollow))}"
        )


def _claim_value(facts: Mapping[str, Any], claim: str) -> object:
    """Resolve a supported claim path into the proposed fact tree.

    Parameters
    ----------
    facts:
        Proposed fact tree.
    claim:
        Dot-separated claim path or supported aggregate category.

    Returns
    -------
    object
        Proposed value for deterministic excerpt comparison.

    Raises
    ------
    ProposalValidationError
        If the claim does not name a proposed value.
    """

    value: object = facts
    for part in claim.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ProposalValidationError(f"gated evidence claim does not name a fact: {claim}")
        value = value[part]
    return value


def _validate_citation_leaves(citation: Mapping[str, Any], texts: Sequence[str]) -> None:
    """Require every positive citation leaf to occur verbatim in the paper text.

    Per-leaf, never aggregate: the old matcher checked title+year only, so a citation
    with fabricated authors, venue, URL, and BibTeX passed the gate. Each present leaf
    must now occur -- Unicode-canonicalized, so ``Balazevic`` and the diacritic
    spelling of the same author ground each other -- in the controlled-fetched paper
    excerpts bound to the citation claim. That covers ``title``, ``venue``, ``year``,
    every ``authors`` entry, and every member of :data:`CITATION_IDENTIFIER_FIELDS`. The
    identifier leaves go through :func:`_identifier_grounded`, which is the same phrase
    check plus the one arXiv entailment a bare phrase comparison gets wrong; author
    names likewise get exactly one entailment,
    :func:`_email_fused_component_grounded`, for pages that print a name fused with
    its own name-derived email address.

    Two leaves are deliberately not excerpt-checked, because neither is quoted from the
    paper, and each names what checks it instead:

    * ``bibtex`` is a *constructed* record. No paper prints its own BibTeX, so demanding
      it verbatim would be a requirement no honest author could satisfy. It is checked
      for exact consistency with the already-grounded title, year, and authors, which an
      entry for a different work cannot satisfy -- so a fabricated entry is still
      refused, just by the check that can actually decide it.
    * ``citation.url`` is machine-derived work verified by the source broker's resolver
      receipt.

    Parameters
    ----------
    citation:
        Present citation block.
    texts:
        Verbatim excerpt texts from controlled-fetched paper-role sources bound to the
        citation claim.

    Raises
    ------
    ProposalValidationError
        If any excerpt-checked citation leaf is not grounded in the paper text, or if
        ``bibtex`` disagrees with the grounded title, year, or authors.
    """

    combined = _normalize_support_text("\n".join(texts))
    combined_tokens = set(combined.split())

    def phrase_grounded(value: object) -> bool:
        """Return whether one canonicalized value occurs contiguously in the text."""

        normalized = _normalize_support_text(str(value))
        return not normalized or f" {normalized} " in f" {combined} "

    def name_grounded(value: object) -> bool:
        """Return whether every canonical component of one name occurs in the text."""

        components = _normalize_support_text(str(value)).split()
        tokens = set(components)
        if tokens <= combined_tokens:
            return True
        return all(
            token in combined_tokens
            or _email_fused_component_grounded(token, components, combined_tokens)
            for token in tokens
        )

    failures: list[str] = []
    inconsistent: list[str] = []
    for leaf in ("title", "venue"):
        value = citation.get(leaf)
        if isinstance(value, str) and value.strip() and not phrase_grounded(value):
            failures.append(leaf)
    year = citation.get("year")
    if year is not None and not _year_grounded(year, citation, combined_tokens):
        failures.append("year")
    authors = citation.get("authors")
    for index, author in enumerate(authors if isinstance(authors, list) else []):
        # Name order varies between "Ivana Balazevic" and "Balazevic, Ivana"; exact
        # per-component membership is required, never overlap thresholds.
        if isinstance(author, str) and author.strip() and not name_grounded(author):
            failures.append(f"authors[{index}]")
    for leaf in CITATION_IDENTIFIER_FIELDS:
        value = citation.get(leaf)
        if isinstance(value, str) and value.strip() and not _identifier_grounded(value, combined):
            failures.append(leaf)
    bibtex = citation.get("bibtex")
    if isinstance(bibtex, str) and bibtex.strip():
        bibtex_text = _normalize_support_text(bibtex)
        bibtex_tokens = set(bibtex_text.split())
        title = citation.get("title")
        consistent = (
            not isinstance(title, str)
            or not title.strip()
            or f" {_normalize_support_text(title)} " in f" {bibtex_text} "
        )
        if year is not None and str(year) not in bibtex_tokens:
            consistent = False
        for author in authors if isinstance(authors, list) else []:
            if isinstance(author, str) and author.strip():
                author_tokens = set(_normalize_support_text(author).split())
                if author_tokens and not author_tokens <= bibtex_tokens:
                    consistent = False
        if not consistent:
            inconsistent.append("bibtex")
    problems: list[str] = []
    if failures:
        problems.append(
            "citation leaves are not grounded verbatim in the fetched paper text: "
            f"{sorted(set(failures))}"
        )
    if inconsistent:
        # Never fold this into the verbatim clause. A BibTeX entry is a constructed
        # record that no paper prints about itself, so an author told its BibTeX was
        # "not grounded verbatim in the fetched paper text" is sent after an excerpt
        # that cannot exist. The check it actually failed is the consistency one.
        problems.append(
            "citation leaves do not agree with the grounded title, year, and authors: "
            f"{sorted(set(inconsistent))}"
        )
    if problems:
        raise ProposalValidationError("; ".join(problems))


def _year_grounded(
    year: object, citation: Mapping[str, Any], combined_tokens: AbstractSet[str]
) -> bool:
    """Return whether a declared publication year is grounded by the paper evidence.

    The plain token check is the rule; this adds exactly one entailment, for the same
    reason the ``arxiv_id`` widening exists, and it is licensed by a measured fact about
    the sources authors actually quote.

    A paper's own publication year is *bibliographic* metadata, and a rendering of the
    paper's BODY does not print it. Across every ar5iv page in the ``pilot`` campaign's
    archived rungs the declared year occurred outside the bibliography ZERO times, and on
    two of them it did not occur at all. The only text on such a page that carries the
    token is some OTHER work's year in the reference list -- so on a body rendering the
    old check had exactly one satisfying witness, and that witness was a bibliography
    entry for a different paper bound as evidence for THIS paper's citation. A rule whose
    only satisfying assignment is a dishonest one is not a tripwire; it is a wall that
    refuses honest authors and rewards laundering, and the authoring stage runs once per
    model, so each refusal is a permanent dead record (model ``m7362``, MobileOne).

    The entailment: a modern arXiv identifier ``YYMM.NNNNN`` *encodes* the announcement
    month, so an ``arxiv_id`` -- which this same function has already required to be
    grounded verbatim in the paper evidence -- establishes when the work was announced
    without anyone asserting it. This is machine-derived from an already-grounded leaf,
    never from author recollection.

    The admitted window is the announcement year or the single following year, and
    nothing else. The ``+1`` is not slack: a preprint is routinely announced late in one
    year and published at a venue in the next, which is a true claim the announcement
    year alone would refuse (model ``m8245``, PoolFormer, announced 2111 and published at
    CVPR 2022). It runs one way only -- a year EARLIER than announcement is impossible
    for the work the identifier names, and two or more years later is not entailed, so
    both stay refused. Old-style identifiers (``cs.CV/0309136``) are not decoded here and
    fall through to the text check unchanged.

    Parameters
    ----------
    year:
        Declared citation year.
    citation:
        Present citation block, read for its already-grounded ``arxiv_id``.
    combined_tokens:
        Canonicalized token set of the bound paper-role excerpt text.

    Returns
    -------
    bool
        Whether the declared year is grounded by the excerpt text or entailed by the
        grounded arXiv identifier.
    """

    if str(year) in combined_tokens:
        return True
    announced = _arxiv_announcement_year(citation.get("arxiv_id"))
    if announced is None:
        return False
    try:
        declared = int(str(year))
    except (TypeError, ValueError):
        return False
    return declared in (announced, announced + 1)


def _arxiv_announcement_year(arxiv_id: object) -> Optional[int]:
    """Return the four-digit year a modern arXiv identifier announces, if any.

    Modern identifiers are ``YYMM.NNNNN`` where ``YY`` is the two-digit year and ``MM``
    is a real month. The scheme began in April 2007 and arXiv has stated it runs to 2029
    before renumbering, so ``07``-``99`` maps into the 2000s unambiguously. Anything that
    is not exactly this shape -- an old-style ``archive/YYMMNNN`` locator, a malformed
    month, a pre-2007 stamp -- returns ``None`` so the caller falls back to the plain
    text check rather than inventing a year.

    Parameters
    ----------
    arxiv_id:
        Declared identifier, possibly carrying an ``arXiv:`` prefix or a ``vN`` suffix.

    Returns
    -------
    int | None
        Announcement year, or ``None`` when the identifier does not encode one.
    """

    if not isinstance(arxiv_id, str):
        return None
    parsed = _ARXIV_IDENTIFIER_PATTERN.match(arxiv_id.strip())
    if parsed is None:
        return None
    core = arxiv_id.strip().lower().removeprefix("arxiv:")
    if "." not in core:
        return None
    stamp = core.split(".", 1)[0]
    if len(stamp) != 4 or not stamp.isdigit():
        return None
    year_part, month_part = int(stamp[:2]), int(stamp[2:])
    if year_part < 7 or not 1 <= month_part <= 12:
        return None
    return 2000 + year_part


def _identifier_grounded(value: str, combined: str) -> bool:
    """Return whether a declared citation identifier is named by the excerpt text.

    The plain phrase check is the rule; this adds exactly one entailment that the plain
    check got wrong. :func:`_normalize_support_text` reduces ``2111.11418`` to the token
    pair ``2111 11418``, but a page that mentions the work only as ``arXiv:2111.11418v3``
    reduces to ``2111 11418v3`` -- the revision selector fuses into the trailing token,
    the phrase is no longer contiguous, and a true claim is refused. The authoring stage
    runs once per model, so that refusal is a permanent dead record, not a retry.

    The entailment runs one way only, from the *more* specific mention to the *less*
    specific claim. An arXiv identifier names a work and ``vN`` names one revision of
    that same work, so a page saying ``2111.11418v3`` has necessarily named
    ``2111.11418``: the widening is on the page side, and the claim being checked is
    unchanged. The converse is deliberately not granted -- a bare ``2111.11418`` on the
    page does not establish that a third revision exists, so a declared ``2111.11418v3``
    asserts strictly more than the excerpt shows and stays refused. Version-suffix
    tolerance therefore applies only when the *declared* identifier carries no version.

    This is not a general "ignore trailing characters" rule, which would be a laundering
    path: it consumes only a literal ``v`` followed by digits and then demands a token
    boundary, so ``2111.114189`` is still refused against ``2111.11418`` and vice versa.
    The tolerance is licensed by the arXiv grammar specifically -- a base arXiv id can
    never end in ``v`` plus digits (see :data:`_ARXIV_IDENTIFIER_PATTERN`), so a trailing
    ``vN`` cannot be anything but the revision selector.

    Neither sibling identifier has the same flaw, and neither may borrow the fix:

    * A DOI suffix is an *opaque* string chosen by the registrant. ``10.1234/xyzv2`` is
      not required to be a revision of ``10.1234/xyz``; it may be a separately
      registered DOI for an unrelated work. Versioned deposits (Zenodo, DataCite) mint
      distinct DOIs rather than suffixing one, so there is no grammar that could tell an
      entailed mention from a different identifier, and accepting one would let a real
      but wrong DOI ground the claim.
    * An OpenReview id is likewise an opaque token, and OpenReview expresses revisions as
      separate note ids rather than as a suffix on the forum id. It has no version
      surface at all, so there is nothing to entail. Its ids do sit inside query strings
      (``?id=SygXPaEYvH&noteId=...``), but ``&`` and ``=`` are punctuation the tokenizer
      already splits on, so no fusion occurs.

    Parameters
    ----------
    value:
        Declared identifier as the author wrote it.
    combined:
        Canonicalized excerpt text from the bound paper-role sources.

    Returns
    -------
    bool
        Whether the excerpt text names the declared identifier.
    """

    normalized = _normalize_support_text(value)
    if not normalized:
        return True
    if f" {normalized} " in f" {combined} ":
        return True
    parsed = _ARXIV_IDENTIFIER_PATTERN.match(value.strip())
    if parsed is None or parsed.group("version") is not None:
        return False
    # ``combined`` holds only lowercase alphanumeric tokens separated by single spaces,
    # so these lookarounds are exactly token boundaries. The suffix is consumed as a
    # whole or not at all, which is why a numeric extension cannot pass.
    versioned = re.compile(rf"(?<![0-9a-z]){re.escape(normalized)}v[0-9]+(?![0-9a-z])")
    return versioned.search(combined) is not None


def _email_fused_component_grounded(
    token: str, components: Sequence[str], text_tokens: AbstractSet[str]
) -> bool:
    """Return whether one missing name component is grounded by a name-email fusion.

    The plain component-membership check is the rule; this adds exactly one entailment
    for a rendering defect the check got wrong. ar5iv renders a failed LaTeX author
    macro (BMVC's ``\\addauthor``) by running its arguments together with NO separator,
    so the paper's own author line prints as ``Hanchao Lilihanchao@bit.edu.com1`` --
    the surname fused with a name-derived email local part. ``Li`` then tokenizes into
    ``lilihanchao`` rather than ``li``, the fetched page contains no separated spelling
    of the name anywhere, and a correct author list is refused as fabricated. The
    authoring stage runs once per model, so that refusal is a permanent dead record
    (menagerie campaign ``pilot``, model ``m9666``, Pyramid Attention Network: both
    ``Hanchao Li`` and ``Pengfei Xiong`` fused this way, and ``Xiong`` occurred nowhere
    else in any fetched byte).

    The entailment is deliberately stronger than "the component prefixes a token",
    which would let ``Li`` ground ``Liu``. A fused token grounds the missing component
    only when it is EXACTLY the component followed by the WHOLE declared name --
    every component, each exactly once, concatenated in some order -- which is the
    shape a name-derived email local part actually has (``li`` + ``lihanchao``,
    ``xiong`` + ``xiongpengfei``). The token must therefore spell out the full
    declared name contiguously inside the source bytes, with the missing component
    appearing twice; a name the page does not contain still cannot match:

    * a name differing in any visible character fails (``Pengfei Xiang`` shares no
      such token, and ``Li`` does not prefix ``liu``);
    * padding the declared name to fit the fused token fails (``Pengfei Xiongxiong``
      leaves a remainder that no longer spells the whole declared list);
    * re-segmenting the fused bytes as a different name fails (``Li Lihanchao``
      requires a thirteen-character token where the page has eleven); and
    * single-component names are excluded outright, so a doubled ordinary word
      (``murmur``) can never ground a declared mononym ``Mur``.

    A fusion with a non-name-derived email (``Wangneobull@...``) is NOT recovered --
    the remainder spells no part of the declared name, so nothing links the token to
    the claim, and the refusal stands as unsatisfiable-evidence rather than admit a
    guess. This helper never rewrites text and never relaxes the other components:
    each one still needs its own exact membership or its own fusion witness.

    Parameters
    ----------
    token:
        Canonical name component absent from the excerpt token set.
    components:
        Every canonical component of the declared name, in declared order.
    text_tokens:
        Canonical excerpt token set.

    Returns
    -------
    bool
        Whether a fused excerpt token grounds the missing component.
    """

    if len(components) < 2:
        return False
    fused_length = len(token) + sum(len(component) for component in components)
    return any(
        len(candidate) == fused_length
        and candidate.startswith(token)
        and _spells_all_components(candidate[len(token) :], list(components))
        for candidate in text_tokens
    )


def _spells_all_components(text: str, components: list[str]) -> bool:
    """Return whether ``text`` is a concatenation of ``components``, each used once.

    Parameters
    ----------
    text:
        Candidate remainder of a fused token.
    components:
        Multiset of canonical name components still to be consumed.

    Returns
    -------
    bool
        Whether some ordering of the components concatenates to exactly ``text``.
    """

    if not components:
        return not text
    return any(
        text.startswith(component)
        and _spells_all_components(
            text[len(component) :], components[:index] + components[index + 1 :]
        )
        for index, component in enumerate(components)
    )


def _positive_scalars(value: object) -> list[object]:
    """Flatten positive JSON-like leaves used for evidence comparison.

    Parameters
    ----------
    value:
        Proposed scalar, sequence, or mapping.

    Returns
    -------
    list[object]
        Non-null, non-empty string and numeric leaves.
    """

    if isinstance(value, Mapping):
        return [scalar for child in value.values() for scalar in _positive_scalars(child)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [scalar for child in value for scalar in _positive_scalars(child)]
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return [value] if str(value).strip() else []
    return []


def _normalize_support_text(value: str) -> str:
    """Normalize text for conservative deterministic token comparison.

    Parameters
    ----------
    value:
        Text to normalize.

    Returns
    -------
    str
        Lowercase alphanumeric tokens separated by single spaces.
    """

    # Compatibility-decompose and drop combining marks first. Without this the regex
    # splits on the diacritic itself, so "Balazevic" and "Balažević" -- the
    # same author, spelled the two ways real sources actually spell them -- normalize to
    # different token sets and a correct claim fails as if it were fabricated.
    decomposed = unicodedata.normalize("NFKD", _decode_tex_escapes(value))
    folded = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", folded.lower()))


def _decode_tex_escapes(value: str) -> str:
    """Rewrite TeX accent and special-letter escapes as the character they denote.

    BibTeX is a *constructed* citation record, and its accented names are written
    ``L\\'elio``, ``Th\\'eophile``, ``Timoth\\'ee`` -- the only spelling BibTeX has for
    them. :func:`_normalize_support_text` is Unicode-aware but not TeX-aware, so the
    backslash and the quote were dropped as punctuation and ``L\\'elio`` tokenized to
    ``l`` plus ``elio`` while the same author spelled ``Lélio`` in the fetched page
    tokenized to ``lelio``. The two spellings of one name did not compare equal, and an
    honest BibTeX entry naming exactly the grounded authors was refused as if it cited a
    different work (menagerie campaign ``pilot``, model ``m5915``, Mixtral of Experts).

    This decodes syntax, never semantics: each escape becomes the precomposed Unicode
    character it denotes, which the existing NFKD fold then reduces exactly as it
    reduces the same character typed directly. Symmetry with the Unicode spelling is
    therefore by construction. Nothing is loosened -- an escape can only rejoin a token
    the TeX syntax split, so a name the entry does not actually contain still cannot
    match.

    Parameters
    ----------
    value:
        Text that may carry TeX escapes.

    Returns
    -------
    str
        Text with recognized TeX escapes replaced by their Unicode characters.
    """

    def replace_accent(match: re.Match[str]) -> str:
        """Compose one accent command and its single-letter argument."""

        combining = _TEX_ACCENT_COMBINING[match.group("accent")]
        return unicodedata.normalize("NFC", match.group("letter") + combining)

    decoded = _TEX_ACCENT_PATTERN.sub(replace_accent, value)
    return _TEX_LETTER_PATTERN.sub(lambda match: _TEX_LETTERS[match.group("letter")], decoded)


def _validate_code(
    implementation: Mapping[str, Any],
    rung: SourceRung,
    allowed_dir: Path,
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> Optional[Path]:
    """Validate staged typed code, path isolation, and forbidden execution APIs.

    Parameters
    ----------
    implementation:
        Proposed implementation block.
    rung:
        Selected source rung.
    allowed_dir:
        Resolved model sandbox directory.
    source_manifest:
        Exact controlled-fetch source manifest, used to prove which closure
        members are unmodified vendored upstream bytes.

    Returns
    -------
    pathlib.Path | None
        Resolved code path for typed-code rungs.

    Raises
    ------
    ProposalValidationError
        If code is missing, outside the sandbox, untyped, or unsafe.
    """

    code_value = implementation.get("code_path")
    if rung in {SourceRung.LIBRARY, SourceRung.SKIP}:
        if rung is SourceRung.LIBRARY and code_value is not None:
            raise ProposalValidationError(
                "R1_LIBRARY must use a declarative recipe, not staged code"
            )
        return None
    if not isinstance(code_value, str) or not code_value.strip():
        raise ProposalValidationError(f"{rung.value} requires a staged code_path")
    candidate = Path(code_value)
    if candidate.is_absolute():
        raise ProposalValidationError(
            "implementation.code_path escapes repository-relative proposal identity"
        )
    resolved = (allowed_dir / candidate).resolve()
    if not resolved.is_relative_to(allowed_dir):
        raise ProposalValidationError("implementation.code_path escapes the model sandbox")
    patches = implementation.get("patches", [])
    if not isinstance(patches, list):
        raise ProposalValidationError("implementation.patches must be a list")
    for patch in patches:
        if not isinstance(patch, Mapping) or not isinstance(patch.get("path"), str):
            raise ProposalValidationError("implementation patch requires a relative path")
        patch_path = Path(str(patch["path"]))
        if patch_path.is_absolute():
            raise ProposalValidationError(
                "implementation patch paths must be repository-relative before proposal identity"
            )
        if not (allowed_dir / patch_path).resolve().is_relative_to(allowed_dir):
            raise ProposalValidationError("implementation patch path escapes the model sandbox")
    try:
        code = resolved.read_bytes()
    except OSError as exc:
        raise ProposalValidationError(f"cannot read staged code_path {resolved}: {exc}") from exc
    expected_hash = implementation.get("code_sha256")
    if expected_hash != hash_bytes(code):
        raise ProposalValidationError("implementation.code_sha256 does not match staged bytes")
    verbatim = _verbatim_upstream_members(implementation, allowed_dir, source_manifest)
    for member in resolve_model_code_closure(resolved, allowed_dir):
        try:
            tree = ast.parse(member.read_text(encoding="utf-8"), filename=str(member))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise ProposalValidationError(
                f"staged model-code member is not valid UTF-8 Python: {exc}"
            ) from exc
        if member == resolved or member not in verbatim:
            _validate_typed_functions(tree)
        # No exemption here, ever. Full annotation is a LEGIBILITY constraint on
        # what the author wrote; dynamic execution and out-of-sandbox writes are
        # a SAFETY constraint on what will run, and verbatim upstream bytes run
        # exactly like authored ones.
        _validate_calls_and_writes(tree, allowed_dir)
    return resolved


def _verbatim_upstream_members(
    implementation: Mapping[str, Any],
    allowed_dir: Path,
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> frozenset[Path]:
    """Return closure members proven to be unmodified fetched upstream bytes.

    ``_validate_typed_functions`` exists so that the code an author WRITES is
    legible: a reviewer can read a staged adapter and see the contract. Running
    it over the whole recursive closure applied that requirement to bytes the
    author did not write and is forbidden to touch. ``METHODOLOGY.md`` mandates
    vendoring real upstream source verbatim, essentially no real PyTorch repo is
    fully annotated (``def forward(self, x):``), and the two rules together made
    R2_VENDOR -- a locked rung of the source-fidelity ladder -- structurally
    unreachable: pilot model ``m8189`` vendored ``naver-ai/pit`` and was refused
    on ``pit.py``'s upstream signatures, which no correct author could have
    fixed without editing the vendored bytes.

    The exemption is keyed on PROOF, not on a rung and not on a claim. A member
    qualifies only when the proposal declares it in ``implementation.upstream_files``,
    the row's ``source_id`` names a completed controlled fetch whose frozen
    ``content_sha256`` equals the row's declared digest, AND the staged file's
    own bytes hash to that same digest. An author cannot manufacture that chain:
    it does not write the CAS, and the digest is recomputed here from the file
    on disk rather than read from the declaration. Editing one character of a
    vendored file breaks the digest and the file goes straight back under the
    annotation rule; a row pointing at a digest some OTHER source froze binds
    nothing.

    The staged ENTRY POINT is never exempt, whatever it hashes to: ``build_model``
    and ``make_dummy_call`` are the author's contract with the runner.

    Parameters
    ----------
    implementation:
        Proposal implementation block carrying ``upstream_files``.
    allowed_dir:
        Resolved model-local staging root.
    source_manifest:
        Exact controlled-fetch source manifest.

    Returns
    -------
    frozenset[pathlib.Path]
        Resolved members whose bytes are provably unmodified fetched source.
    """

    upstream_files = implementation.get("upstream_files")
    if not isinstance(upstream_files, list) or not upstream_files:
        return frozenset()
    try:
        indexed = _source_manifest_index(source_manifest)
    except ProposalValidationError:
        # A manifest we cannot read grants no exemption.
        return frozenset()
    members: set[Path] = set()
    for upstream in upstream_files:
        if not isinstance(upstream, Mapping):
            continue
        declared_path = upstream.get("path")
        declared_hash = upstream.get("sha256")
        source_id = upstream.get("source_id")
        if (
            not isinstance(declared_path, str)
            or not isinstance(declared_hash, str)
            or not isinstance(source_id, str)
        ):
            continue
        # The binding is per ROW: the named source must exist, must be an
        # actually completed controlled fetch, and must have frozen exactly the
        # digest this row declares. A digest that merely appears somewhere in
        # the manifest under a different source is not this row's provenance.
        source = indexed.get(source_id)
        if not _is_controlled_fetch(source):
            continue
        assert source is not None  # _is_controlled_fetch refused None
        if source.get("content_sha256") != declared_hash:
            continue
        candidate = Path(declared_path)
        if candidate.is_absolute():
            continue
        staged = (allowed_dir / candidate).resolve()
        if not staged.is_relative_to(allowed_dir) or not staged.is_file():
            continue
        try:
            observed = hash_bytes(staged.read_bytes())
        except OSError:
            continue
        if observed == declared_hash:
            members.add(staged)
    return frozenset(members)


def _validate_author_read_grants(facts: Mapping[str, Any], allowed_dir: Path) -> None:
    """Reject author-controlled filesystem grants before proposal acceptance.

    Parameters
    ----------
    facts:
        Complete authored fact tree.
    allowed_dir:
        Resolved model-local staging directory.

    Raises
    ------
    ProposalValidationError
        If the deleted input code-path leaf is present, or a builder symbol, scalar
        path value, or source CAS locator could grant access outside the model-local
        regular-file boundary.
    """

    input_contract = _mapping(facts.get("input_contract"), "input_contract")
    if "code_path" in input_contract:
        raise ProposalValidationError("v3 input_contract forbids code_path presence")
    builder_symbol = input_contract.get("builder_symbol")
    if not isinstance(builder_symbol, str) or not re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", builder_symbol
    ):
        raise ProposalValidationError("input_contract.builder_symbol must be a dotted symbol")
    non_tensor_values = input_contract.get("non_tensor_values")
    if not isinstance(non_tensor_values, list):
        raise ProposalValidationError("input_contract.non_tensor_values must be a list")
    for leaf in non_tensor_values:
        if not isinstance(leaf, Mapping):
            raise ProposalValidationError("input_contract non-tensor leaf must be an object")
        value = leaf.get("value")
        if isinstance(value, str):
            possible_path = Path(value)
            if possible_path.is_absolute() or ".." in possible_path.parts:
                raise ProposalValidationError(
                    "input_contract.non_tensor_values cannot carry absolute or escaping paths"
                )
            value_type = str(leaf.get("type", "")).casefold().replace("_", "-")
            if value_type in {"file", "file-path", "filepath", "path", "pathlib.path"}:
                resolved_value = (allowed_dir / possible_path).resolve()
                if not resolved_value.is_relative_to(allowed_dir) or not resolved_value.is_file():
                    raise ProposalValidationError(
                        "path-valued input_contract.non_tensor_values must name a model-local "
                        "regular file"
                    )
    resolution = _mapping(facts.get("source_resolution"), "source_resolution")
    sources = resolution.get("sources")
    if not isinstance(sources, list):
        raise ProposalValidationError("source_resolution.sources must be a list")
    if any(isinstance(source, Mapping) and "cas_path" in source for source in sources):
        raise ProposalValidationError(
            "source_resolution.sources cannot carry author-controlled CAS paths"
        )


def resolve_model_code_closure(code_path: Path, allowed_dir: Path) -> tuple[Path, ...]:
    """Resolve the closed recursive model-local Python import manifest.

    Parameters
    ----------
    code_path:
        Accepted adapter or port entry point.
    allowed_dir:
        Model-local root that every closure member must remain below.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Entry point and recursively imported local Python modules, sorted by
        repository-relative path.

    Raises
    ------
    ProposalValidationError
        If a member escapes the model root or cannot be parsed.
    """

    root = allowed_dir.resolve()
    entry = code_path.resolve()
    if not entry.is_relative_to(root) or not entry.is_file():
        raise ProposalValidationError("model-code entry point escapes the model sandbox")
    pending = [entry]
    members: set[Path] = set()
    while pending:
        member = pending.pop()
        if member in members:
            continue
        try:
            tree = ast.parse(member.read_text(encoding="utf-8"), filename=str(member))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise ProposalValidationError(f"cannot resolve model-code imports: {exc}") from exc
        members.add(member)
        for imported in _local_import_paths(tree, member, root):
            if imported not in members:
                pending.append(imported)
    return tuple(sorted(members, key=lambda path: path.relative_to(root).as_posix()))


def model_code_manifest(code_path: Path, allowed_dir: Path) -> tuple[dict[str, str], ...]:
    """Return the path-and-digest manifest for a closed model-code import graph.

    Parameters
    ----------
    code_path, allowed_dir:
        Entry point and model-local root accepted by
        :func:`resolve_model_code_closure`.

    Returns
    -------
    tuple[dict[str, str], ...]
        Canonically ordered relative paths and exact byte digests.
    """

    root = allowed_dir.resolve()
    return tuple(
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hash_bytes(path.read_bytes()),
        }
        for path in resolve_model_code_closure(code_path, root)
    )


def _local_import_paths(tree: ast.AST, member: Path, root: Path) -> tuple[Path, ...]:
    """Resolve statically named imports that refer to model-local Python files.

    Parameters
    ----------
    tree:
        Parsed closure member.
    member:
        Absolute path of that member.
    root:
        Closed model-local import root.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Existing local modules and package initializers referenced by the AST.
    """

    resolved: set[Path] = set()
    for node in ast.walk(tree):
        candidates: list[tuple[str, int]] = []
        if isinstance(node, ast.Import):
            candidates.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            candidates.append((module, node.level))
            candidates.extend(
                (
                    f"{module}.{alias.name}" if module else alias.name,
                    node.level,
                )
                for alias in node.names
                if alias.name != "*"
            )
        for module, level in candidates:
            resolved.update(_resolve_local_module(module, level, member, root))
    return tuple(sorted(resolved, key=lambda path: path.relative_to(root).as_posix()))


def _resolve_local_module(module: str, level: int, member: Path, root: Path) -> tuple[Path, ...]:
    """Resolve one import name to local module/package files when present.

    Parameters
    ----------
    module, level:
        Static import name and relative-import level from the AST.
    member:
        Importing closure member.
    root:
        Closed model-local root.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Package initializer chain plus the imported module, or an empty tuple
        for an external dependency.
    """

    if level:
        base = member.parent
        for _ in range(level - 1):
            base = base.parent
        if not base.is_relative_to(root):
            raise ProposalValidationError("relative model-code import escapes the model sandbox")
    else:
        base = root
    parts = tuple(part for part in module.split(".") if part)
    candidate = base.joinpath(*parts) if parts else base
    target: Optional[Path] = None
    if candidate.with_suffix(".py").is_file():
        target = candidate.with_suffix(".py").resolve()
    elif (candidate / "__init__.py").is_file():
        target = (candidate / "__init__.py").resolve()
    if target is None:
        return ()
    if not target.is_relative_to(root):
        raise ProposalValidationError("model-code import escapes the model sandbox")
    initializers: list[Path] = []
    current = target.parent
    while current != root and current.is_relative_to(root):
        initializer = current / "__init__.py"
        if initializer.is_file():
            initializers.append(initializer.resolve())
        current = current.parent
    return tuple(dict.fromkeys((*reversed(initializers), target)))


def _validate_typed_functions(tree: ast.AST) -> None:
    """Require annotations on every staged function and method.

    Parameters
    ----------
    tree:
        Parsed staged Python module.

    Raises
    ------
    ProposalValidationError
        If a function argument or return is untyped.
    """

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments = [argument for argument in arguments if argument.arg not in {"self", "cls"}]
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        if node.returns is None or any(argument.annotation is None for argument in arguments):
            raise ProposalValidationError(f"staged function {node.name!r} must be fully typed")


def _validate_calls_and_writes(tree: ast.AST, allowed_dir: Path) -> None:
    """Reject dynamic execution and statically unsafe writes.

    Parameters
    ----------
    tree:
        Parsed staged Python module.
    allowed_dir:
        Resolved write sandbox.

    Raises
    ------
    ProposalValidationError
        If forbidden execution or an unsafe write is found.
    """

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if _is_forbidden_call(call_name):
            raise ProposalValidationError(f"forbidden dynamic execution call: {call_name}")
        if call_name == "open" and _open_writes(node):
            _validate_literal_write_target(node, allowed_dir, call_name)
        elif call_name.rsplit(".", 1)[-1] in _WRITE_METHODS:
            _validate_literal_write_target(node, allowed_dir, call_name)


def _is_forbidden_call(call_name: str) -> bool:
    """Return whether a static call name reaches dynamic execution.

    Parameters
    ----------
    call_name:
        Dotted static call name, or an empty string when none is derivable.

    Returns
    -------
    bool
        True for a bare builtin, an explicit builtin-namespace attribute, or a
        refused-by-name dotted call.
    """

    if call_name in _FORBIDDEN_CALLS or call_name in _FORBIDDEN_DOTTED_CALLS:
        return True
    root, _, leaf = call_name.rpartition(".")
    return root in _BUILTIN_NAMESPACE_ROOTS and leaf in _FORBIDDEN_CALLS


def _call_name(function: ast.expr) -> str:
    """Return a dotted static call name when available.

    Parameters
    ----------
    function:
        Call target expression.

    Returns
    -------
    str
        Dotted call name or an empty string.
    """

    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        prefix = _call_name(function.value)
        return f"{prefix}.{function.attr}" if prefix else function.attr
    if isinstance(function, ast.Call):
        return _call_name(function.func)
    return ""


def _open_writes(node: ast.Call) -> bool:
    """Return whether a built-in ``open`` call may write.

    Parameters
    ----------
    node:
        Static ``open`` call.

    Returns
    -------
    bool
        True for write/append/create/update modes or dynamic mode values.
    """

    mode_node: Optional[ast.expr] = node.args[1] if len(node.args) > 1 else None
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
    if mode_node is None:
        return False
    if not isinstance(mode_node, ast.Constant) or not isinstance(mode_node.value, str):
        return True
    return any(character in mode_node.value for character in "wax+")


def _validate_literal_write_target(node: ast.Call, allowed_dir: Path, call_name: str) -> None:
    """Require write targets to be literal paths inside the model sandbox.

    Parameters
    ----------
    node:
        Static write call.
    allowed_dir:
        Resolved model sandbox.
    call_name:
        Call name used in error reporting.

    Raises
    ------
    ProposalValidationError
        If the target is dynamic or outside the sandbox.
    """

    target_node: Optional[ast.expr] = None
    if call_name == "open" and node.args:
        target_node = node.args[0]
    elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
        path_call = node.func.value
        if _call_name(path_call.func).endswith("Path") and path_call.args:
            target_node = path_call.args[0]
    if not isinstance(target_node, ast.Constant) or not isinstance(target_node.value, str):
        raise ProposalValidationError(f"{call_name} has a dynamic or unverifiable write target")
    candidate = Path(target_node.value)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (allowed_dir / candidate).resolve()
    )
    if not resolved.is_relative_to(allowed_dir):
        raise ProposalValidationError(f"{call_name} writes outside the model sandbox")


def _validate_source_ladder(
    rung: SourceRung,
    facts: Mapping[str, Any],
    resolution: Mapping[str, Any],
    implementation: Mapping[str, Any],
    evidence: Mapping[str, Any],
    known_evidence: frozenset[str],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    cas_root: Union[str, Path, None],
) -> None:
    """Enforce rung-specific source-ladder honesty.

    Parameters
    ----------
    rung:
        Selected rung.
    facts:
        Complete proposal facts used to bind source symbols to the claimed family.
    resolution, implementation, evidence:
        Proposal source and implementation blocks.
    known_evidence:
        Valid literal evidence identifiers.
    source_manifest:
        Exact fetched sources.
    cas_root:
        Optional source CAS root for source inventory inspection.

    Raises
    ------
    ProposalValidationError
        If the chosen rung contradicts its evidence or requirements.
    """

    attempted = resolution.get("attempted_rungs")
    if not isinstance(attempted, list) or not attempted:
        raise ProposalValidationError("source ladder must record attempted rungs")
    attempted_values = [item.get("rung") for item in attempted if isinstance(item, Mapping)]
    rung_order = [member.value for member in SourceRung]
    selected_index = rung_order.index(rung.value)
    if any(required not in attempted_values for required in rung_order[: selected_index + 1]):
        raise ProposalValidationError("selected rung does not document every higher rung")
    if rung is SourceRung.LIBRARY:
        recipe = implementation.get("library_recipe")
        # ``artifact_sha256`` is deliberately absent: it identifies the INSTALLED
        # distribution, and the author stage has no package inventory, no
        # environment identity, and no interpreter with which to derive it. The
        # driver resolves it from the routed intent's exact resolved export
        # before gating (``recipe.bind_library_artifact_digest``) and refuses a
        # conflicting supplied value, so requiring it here only forced authors to
        # fabricate a digest nothing ever verified.
        required = ("distribution", "version", "module", "symbol")
        if implementation.get("recipe_type") != "declarative-library" or not isinstance(
            recipe, Mapping
        ):
            raise ProposalValidationError("R1_LIBRARY requires a declarative library recipe")
        if any(
            not isinstance(recipe.get(field), str) or not str(recipe[field]).strip()
            for field in required
        ):
            raise ProposalValidationError("R1_LIBRARY recipe is incomplete")
        # The pretrained rule is a real safety requirement -- a weight download
        # inside a capture poisons the record with weights the catalog does not
        # describe -- but "the array must be non-empty" was the wrong shape for
        # it in both directions. It refused constructors that expose no such
        # keyword (``MiniMaxForCausalLM(config)``: no value satisfies it, so
        # every such model died a permanent dead record), and it accepted a
        # recipe that named one harmless disabled field while a second, genuinely
        # enabling keyword sat untouched in the same kwargs. What replaces it is
        # a positive assertion that silence still fails, plus an unconditional
        # refusal of any known pretrained keyword left enabled; the pinned
        # SIGNATURE is then checked against that assertion at load time in
        # ``recipe.load_declarative_recipe``, where the constructor is real.
        kwargs = recipe.get("kwargs")
        disable_fields = recipe.get("pretrained_disable_fields")
        fields_absent = recipe.get("pretrained_fields_absent", False)
        if (
            not isinstance(kwargs, Mapping)
            or not isinstance(disable_fields, list)
            or not isinstance(fields_absent, bool)
        ):
            raise ProposalValidationError("R1_LIBRARY pretrained disable declaration is malformed")
        try:
            validate_pretrained_disposition(kwargs, disable_fields, fields_absent=fields_absent)
        except RecipeError as exc:
            raise ProposalValidationError(str(exc)) from exc
    if rung is SourceRung.REIMPLEMENT:
        _validate_r4_negative_proof(resolution, known_evidence)
        _validate_checked_link_fetch_coverage(resolution, source_manifest, rung=rung)
        if _implementation_source_available(
            source_manifest,
            cas_root=cas_root,
            linkage_terms=_model_linkage_terms(facts),
        ):
            raise ProposalValidationError(
                "R4_REIMPLEMENT is forbidden when source code is available"
            )
    if rung is SourceRung.VENDOR:
        _validate_checked_link_fetch_coverage(resolution, source_manifest, rung=rung)
        _validate_r2_source_binding(implementation, resolution, evidence, source_manifest)
    if rung in {SourceRung.VENDOR, SourceRung.PORT, SourceRung.REIMPLEMENT}:
        source_map = implementation.get("source_to_code_map")
        if not isinstance(source_map, list) or not source_map:
            raise ProposalValidationError(f"{rung.value} requires a material source-to-code map")
        cited = {
            evidence_id
            for item in source_map
            if isinstance(item, Mapping)
            for evidence_id in item.get("evidence_ids", [])
            if isinstance(evidence_id, str)
        }
        if not cited or not cited <= known_evidence:
            raise ProposalValidationError(
                f"{rung.value} source map lacks literal descriptive evidence"
            )
        excerpts = evidence.get("excerpts", [])
        descriptive_ids = {
            item.get("evidence_id")
            for item in excerpts
            if isinstance(item, Mapping)
            and any(
                token in str(support).lower()
                for support in item.get("supports", [])
                for token in ("architecture", "implementation", "input_contract", "fidelity")
            )
        }
        if not cited & descriptive_ids:
            raise ProposalValidationError(f"{rung.value} did not cite transcribed descriptive text")


def _validate_r4_negative_proof(
    resolution: Mapping[str, Any], known_evidence: frozenset[str]
) -> None:
    """Require a bounded, evidence-backed negative source search for R4.

    Parameters
    ----------
    resolution:
        Complete source-resolution block.
    known_evidence:
        Validated literal evidence identifiers.

    Raises
    ------
    ProposalValidationError
        If higher-rung attempts do not explicitly establish unavailability or
        the bounded search report is empty.
    """

    attempted = resolution.get("attempted_rungs")
    if not isinstance(attempted, list):
        raise ProposalValidationError("R4 requires explicit negative proof from a bounded search")
    attempts_by_rung = {item.get("rung"): item for item in attempted if isinstance(item, Mapping)}
    for higher_rung in ("R1_LIBRARY", "R2_VENDOR", "R3_PORT"):
        attempt = attempts_by_rung.get(higher_rung)
        if not isinstance(attempt, Mapping) or attempt.get("result") != "unavailable":
            raise ProposalValidationError(
                "R4 requires explicit negative proof that every higher rung is unavailable"
            )
        reason_code = attempt.get("reason_code")
        attempt_evidence = attempt.get("evidence_ids")
        if (
            not isinstance(reason_code, str)
            or "search" not in reason_code.lower()
            or not isinstance(attempt_evidence, list)
            or not attempt_evidence
            or any(
                not isinstance(evidence_id, str) or evidence_id not in known_evidence
                for evidence_id in attempt_evidence
            )
        ):
            raise ProposalValidationError(
                "R4 higher-rung unavailability must be backed by bounded-search evidence"
            )
    selected_attempt = attempts_by_rung.get(SourceRung.REIMPLEMENT.value)
    if not isinstance(selected_attempt, Mapping) or selected_attempt.get("result") != "selected":
        raise ProposalValidationError("R4 source ladder must explicitly select R4_REIMPLEMENT")
    search_report = resolution.get("search_report")
    if not isinstance(search_report, Mapping) or any(
        not isinstance(search_report.get(field), list) or not search_report[field]
        for field in ("queries", "places_checked", "links_checked", "languages_checked")
    ):
        raise ProposalValidationError(
            "R4 requires an explicit bounded search report with no usable implementation found"
        )


def _validate_r2_source_binding(
    implementation: Mapping[str, Any],
    resolution: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> None:
    """Bind an R2 adapter to exact mirrored upstream bytes and mapped code.

    Parameters
    ----------
    implementation:
        R2 implementation block.
    resolution:
        Source-resolution block classifying authoritative implementation sources.
    evidence:
        Literal excerpt block already verified against controlled source bytes.
    source_manifest:
        Controlled-fetch manifest containing exact source hashes.

    Raises
    ------
    ProposalValidationError
        If upstream files or source-map rows do not bind to exact fetched bytes.
    """

    sources = _source_manifest_index(source_manifest)
    declared_sources = resolution.get("sources")
    if not isinstance(declared_sources, list):
        raise ProposalValidationError("R2_VENDOR has no declared implementation sources")
    declared_by_id = {
        source.get("source_id"): source
        for source in declared_sources
        if isinstance(source, Mapping) and isinstance(source.get("source_id"), str)
    }
    upstream_files = implementation.get("upstream_files")
    if not isinstance(upstream_files, list) or not upstream_files:
        raise ProposalValidationError("R2_VENDOR requires exact mirrored upstream files")
    upstream_source_ids: set[str] = set()
    for upstream in upstream_files:
        if not isinstance(upstream, Mapping):
            raise ProposalValidationError("R2_VENDOR upstream file binding must be an object")
        source_id = upstream.get("source_id")
        source = sources.get(source_id) if isinstance(source_id, str) else None
        if source is None:
            raise ProposalValidationError("R2_VENDOR upstream file references an unfetched source")
        declared = declared_by_id.get(source_id)
        if not isinstance(declared, Mapping) or declared.get("role") != "implementation":
            raise ProposalValidationError(
                "R2_VENDOR upstream bytes are not classified as implementation source"
            )
        if upstream.get("sha256") != source.get("content_sha256"):
            raise ProposalValidationError(
                "R2_VENDOR upstream file hash does not match exact source bytes"
            )
        if declared.get("content_sha256") != source.get("content_sha256"):
            raise ProposalValidationError(
                "R2_VENDOR declared source does not match controlled source bytes"
            )
        upstream_source_ids.add(str(source_id))
    source_map = implementation.get("source_to_code_map")
    if not isinstance(source_map, list) or not source_map:
        raise ProposalValidationError("R2_VENDOR requires an exact source-to-code map")
    excerpts = evidence.get("excerpts")
    if not isinstance(excerpts, list):
        raise ProposalValidationError("R2_VENDOR source map has no literal evidence")
    excerpt_bindings = {
        (excerpt.get("evidence_id"), excerpt.get("source_id"), excerpt.get("locator"))
        for excerpt in excerpts
        if isinstance(excerpt, Mapping)
    }
    for mapping in source_map:
        if not isinstance(mapping, Mapping):
            raise ProposalValidationError("R2_VENDOR source-to-code binding must be an object")
        source_id = mapping.get("source_id")
        locator = mapping.get("source_locator")
        if not isinstance(source_id, str) or source_id not in upstream_source_ids:
            raise ProposalValidationError(
                "R2_VENDOR source map does not reference bound upstream bytes"
            )
        if not isinstance(locator, str) or not locator.strip():
            raise ProposalValidationError("R2_VENDOR source map lacks an exact source locator")
        mapping_evidence = mapping.get("evidence_ids")
        if not isinstance(mapping_evidence, list) or not any(
            (evidence_id, source_id, locator) in excerpt_bindings
            for evidence_id in mapping_evidence
        ):
            raise ProposalValidationError(
                "R2_VENDOR source map locator is not bound to a verified exact excerpt"
            )


def _validate_checked_link_fetch_coverage(
    resolution: Mapping[str, Any],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    *,
    rung: SourceRung,
) -> None:
    """Require every checked R2/R4 candidate link to have fetched CAS bytes.

    Parameters
    ----------
    resolution:
        Source-resolution block containing the bounded search report.
    source_manifest:
        Exact controlled-fetch source rows.
    rung:
        Selected source rung used in fail-closed diagnostics.

    Raises
    ------
    ProposalValidationError
        If an author-reported checked link was withheld from controlled fetch.
    """

    search_report = resolution.get("search_report")
    if not isinstance(search_report, Mapping):
        raise ProposalValidationError(f"{rung.value} requires a bounded search report")
    try:
        fetched_sources_for_checked_links(search_report, source_manifest)
    except EvidenceValidationError as exc:
        raise ProposalValidationError(f"{rung.value} checked-link coverage gap: {exc}") from exc


def _source_manifest_index(
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> dict[str, Mapping[str, Any]]:
    """Index exact controlled-fetch rows by source identifier.

    Parameters
    ----------
    source_manifest:
        Manifest wrapper or direct source sequence.

    Returns
    -------
    dict[str, Mapping[str, Any]]
        Exact source rows keyed by source ID.

    Raises
    ------
    ProposalValidationError
        If the manifest is malformed or duplicates an identifier.
    """

    raw_sources: object
    if isinstance(source_manifest, Mapping):
        raw_sources = source_manifest.get("sources")
        if raw_sources is None and "source_id" in source_manifest:
            raw_sources = [source_manifest]
    else:
        raw_sources = source_manifest
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise ProposalValidationError("source manifest must contain a source list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for source in raw_sources:
        if not isinstance(source, Mapping):
            raise ProposalValidationError("every source manifest must be an object")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ProposalValidationError("source manifest row has no source_id")
        if source_id in indexed:
            raise ProposalValidationError(f"duplicate source_id: {source_id}")
        indexed[source_id] = source
    return indexed


def _implementation_source_available(
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    *,
    cas_root: Union[str, Path, None],
    linkage_terms: frozenset[str],
) -> bool:
    """Return whether exact fetched CAS bytes expose implementation source.

    Parameters
    ----------
    source_manifest:
        Controlled-fetch manifest wrapper or rows.
    cas_root:
        Optional CAS root for manifests without an explicit object path.
    linkage_terms:
        Normalized model/family symbols that source bytes must actually reference.

    Returns
    -------
    bool
        True when a fetched object inventory contains usable source code.
    """

    sources = _source_manifest_index(source_manifest)
    inventory_results = [
        _source_cas_contains_implementation(
            source,
            cas_root=cas_root,
            linkage_terms=linkage_terms,
        )
        for source in sources.values()
    ]
    return any(inventory_results)


def _source_cas_contains_implementation(
    source: Mapping[str, Any],
    *,
    cas_root: Union[str, Path, None],
    linkage_terms: frozenset[str],
) -> bool:
    """Inspect one hash-bound CAS object for usable implementation code.

    Parameters
    ----------
    source:
        Controlled-fetch manifest row.
    cas_root:
        Optional CAS root for manifests without an explicit object path.
    linkage_terms:
        Normalized model/family symbols required for relevance.

    Returns
    -------
    bool
        True when archive names, a byte manifest, or raw source bytes expose code.

    Raises
    ------
    ProposalValidationError
        If fetched bytes are absent or no longer match their declared digest.
    """

    path_value = source.get("cas_path")
    digest = source.get("content_sha256")
    if not isinstance(digest, str):
        raise ProposalValidationError("fetched source manifest has no content_sha256")
    if isinstance(path_value, str) and path_value:
        path = Path(path_value)
    elif cas_root is not None:
        path = source_cas_path(cas_root, digest)
    else:
        raise ProposalValidationError("R4 source inventory has no inspectable CAS path")
    if not _cas_object_matches_digest(path, digest):
        raise ProposalValidationError(f"R4 source inventory CAS object does not match {digest}")
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                members = [
                    _inventory_member(
                        member.filename,
                        _read_inventory_stream(archive.open(member), member.file_size),
                        linkage_terms,
                    )
                    for member in archive.infolist()
                    if not member.is_dir()
                    and member.file_size > 0
                    and _inventory_name_is_implementation(member.filename)
                ]
                return _archive_inventory_has_implementation(members, linkage_terms)
        if tarfile.is_tarfile(path):
            with tarfile.open(path, mode="r:*") as archive:
                tar_members: list[_InventoryMember] = []
                for member in archive:
                    if (
                        not member.isfile()
                        or member.size <= 0
                        or not _inventory_name_is_implementation(member.name)
                    ):
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        tar_members.append(
                            _inventory_member(
                                member.name,
                                _read_inventory_stream(extracted, member.size),
                                linkage_terms,
                            )
                        )
                return _archive_inventory_has_implementation(tar_members, linkage_terms)
        with path.open("rb") as handle:
            content = _read_inventory_stream(handle, path.stat().st_size)
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ProposalValidationError(
            f"cannot inventory fetched source CAS object {path}: {exc}"
        ) from exc
    return _code_bytes_are_relevant_implementation(
        str(source.get("url") or path.name),
        content,
        source,
        linkage_terms,
    )


def _read_inventory_stream(handle: Any, size: int) -> bytes:
    """Read one bounded archive member incrementally for structural inspection.

    Parameters
    ----------
    handle:
        Binary member stream.
    size:
        Declared uncompressed byte count.

    Returns
    -------
    bytes
        Complete bounded member bytes, including members above the former 8 MiB cap.
    """

    if size > _MAX_STREAM_INVENTORY_BYTES:
        raise ProposalValidationError(
            f"source inventory member exceeds {_MAX_STREAM_INVENTORY_BYTES} byte safety bound"
        )
    chunks: list[bytes] = []
    observed = 0
    while True:
        chunk = handle.read(1024**2)
        if not chunk:
            break
        observed += len(chunk)
        if observed > _MAX_STREAM_INVENTORY_BYTES:
            raise ProposalValidationError("source inventory member exceeds safety bound")
        chunks.append(chunk)
    return b"".join(chunks)


def _inventory_member(name: str, content: bytes, linkage_terms: frozenset[str]) -> _InventoryMember:
    """Classify one implementation-role member and extract its explicit links.

    Parameters
    ----------
    name, content, linkage_terms:
        Archive locator, exact bytes, and model/family linkage terms.

    Returns
    -------
    _InventoryMember
        Whole-archive graph facts for one member.
    """

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return _InventoryMember(name, "", False, False, frozenset(), frozenset(), frozenset())
    context = re.sub(r"[^a-z0-9]+", "", f"{name} {text}".lower())
    linked = any(term in context for term in linkage_terms)
    defined: set[str] = set()
    referenced: set[str] = set()
    imported: set[str] = set()
    structured = False
    if Path(name).suffix.lower() in {".py", ".pyx", ""}:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            defined = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            }
            referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            structured = _python_has_model_structure(text, linkage_terms)
    else:
        definitions = re.findall(
            r"\b(?:class|def|function|struct)\s+([A-Za-z_][A-Za-z0-9_]*)", text
        )
        defined.update(definitions)
        referenced.update(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text))
        structured = bool(
            definitions
            and re.search(
                r"\b(?:attention|conv(?:olution)?|embedding|forward|layer|lstm|module)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
    if Path(name).suffix.lower() == ".ipynb":
        notebook_text = text.replace("\\n", "\n").replace('\\"', '"')
        defined.update(
            re.findall(r"\b(?:class|def|function)\s+([A-Za-z_][A-Za-z0-9_]*)", notebook_text)
        )
        referenced.update(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", notebook_text))
        structured = bool(
            re.search(r"\b(?:class|def)\s+[A-Za-z_]", notebook_text)
            and re.search(r"\b(?:forward|call|apply)\s*\(", notebook_text)
            and re.search(r"\breturn\b", notebook_text)
        )
    return _InventoryMember(
        name,
        text,
        linked,
        structured,
        frozenset(defined),
        frozenset(referenced),
        frozenset(imported),
    )


def _archive_inventory_has_implementation(
    members: Sequence[_InventoryMember], linkage_terms: frozenset[str]
) -> bool:
    """Follow bounded symbol/import links from identity files to model structure.

    Parameters
    ----------
    members:
        Every implementation-role file in the archive.
    linkage_terms:
        Specific normalized family/model terms.

    Returns
    -------
    bool
        True only when identity is structurally bound within the archive graph.
    """

    if not linkage_terms:
        return False
    for member in members:
        linked_definitions = {
            symbol
            for symbol in member.defined_symbols
            if any(
                term in re.sub(r"[^a-z0-9]+", "", symbol.lower())
                or re.sub(r"[^a-z0-9]+", "", symbol.lower()) in term
                for term in linkage_terms
            )
        }
        if member.linked and member.structured and linked_definitions:
            return True

    adjacency: dict[int, set[int]] = {index: set() for index in range(len(members))}
    for left_index, left in enumerate(members):
        for right_index, right in enumerate(members):
            if left_index == right_index or not right.defined_symbols:
                continue
            module_name = Path(right.name).with_suffix("").as_posix().replace("/", ".")
            symbol_link = bool(left.referenced_symbols & right.defined_symbols)
            import_link = any(
                imported == module_name
                or imported.endswith(f".{module_name}")
                or module_name.endswith(f".{imported}")
                for imported in left.imported_modules
            )
            if symbol_link or import_link:
                adjacency[left_index].add(right_index)

    frontier = {index for index, member in enumerate(members) if member.linked}
    visited = set(frontier)
    for _depth in range(4):
        frontier = {
            neighbor
            for index in frontier
            for neighbor in adjacency[index]
            if neighbor not in visited
        }
        visited.update(frontier)
        if not frontier:
            return False
        if any(members[index].structured for index in frontier):
            return True
    return False


def _cas_object_matches_digest(path: Path, digest: str) -> bool:
    """Return whether a CAS file exists and matches its prefixed SHA-256 digest.

    Parameters
    ----------
    path:
        Candidate exact CAS object.
    digest:
        Declared prefixed SHA-256 digest.

    Returns
    -------
    bool
        True only for a byte-identical regular file.
    """

    if not path.is_file() or not digest.startswith("sha256:"):
        return False
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024**2), b""):
                hasher.update(chunk)
    except OSError:
        return False
    return f"sha256:{hasher.hexdigest()}" == digest


def _inventory_name_is_implementation(value: str) -> bool:
    """Return whether an archive-internal path denotes usable source code.

    Parameters
    ----------
    value:
        Archive member or byte-manifest path.

    Returns
    -------
    bool
        True for a non-packaging code file outside documentation/test trees.
    """

    normalized = value.replace("\\", "/").strip("/")
    path = Path(normalized)
    lowered_parts = {part.lower() for part in path.parts[:-1]}
    name = path.name.lower()
    return (
        bool(name)
        and name not in _NON_IMPLEMENTATION_CODE_NAMES
        and path.stem.lower() not in _NON_IMPLEMENTATION_SOURCE_STEMS
        and not lowered_parts & _NON_IMPLEMENTATION_SOURCE_DIRS
        and path.suffix.lower() in _IMPLEMENTATION_SOURCE_SUFFIXES
    )


def _model_linkage_terms(facts: Mapping[str, Any]) -> frozenset[str]:
    """Return normalized identity/family tokens for source relevance checks.

    Parameters
    ----------
    facts:
        Complete proposed facts with identity and taxonomy blocks.

    Returns
    -------
    frozenset[str]
        Specific lowercase alphanumeric model/family tokens.
    """

    values: list[str] = []
    identity = facts.get("identity")
    if isinstance(identity, Mapping):
        for field in ("canonical_name", "acronym"):
            value = identity.get(field)
            if isinstance(value, str):
                values.append(value)
        aliases = identity.get("aliases")
        if isinstance(aliases, list):
            values.extend(value for value in aliases if isinstance(value, str))
    taxonomy = facts.get("taxonomy")
    if isinstance(taxonomy, Mapping) and isinstance(taxonomy.get("family"), str):
        values.append(str(taxonomy["family"]))
    terms: set[str] = set()
    for value in values:
        split_camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
        normalized_full = re.sub(r"[^a-z0-9]+", "", value.lower())
        if len(normalized_full) >= 4:
            terms.add(normalized_full)
        for token in re.findall(r"[a-z0-9]+", split_camel.lower()):
            if len(token) >= 4 and token not in _SUPPORT_STOPWORDS:
                terms.add(token)
    return frozenset(terms)


def _python_has_model_structure(text: str, linkage_terms: frozenset[str]) -> bool:
    """Return whether Python source contains executable architecture structure.

    Parameters
    ----------
    text:
        Decoded candidate source.
    linkage_terms:
        Specific normalized model/family symbols.

    Returns
    -------
    bool
        True for a linked callable or a class/function model entry point with
        internal computation.
    """

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    function_types = (ast.AsyncFunctionDef, ast.FunctionDef)
    for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        if any(
            isinstance(member, function_types)
            and member.name in _MODEL_ENTRY_METHODS
            and _function_has_internal_computation(member)
            for member in class_node.body
        ):
            return True
    for node in ast.walk(tree):
        if not isinstance(node, function_types) or not _function_has_internal_computation(node):
            continue
        normalized_name = re.sub(r"[^a-z0-9]+", "", node.name.lower())
        if node.name in _MODEL_ENTRY_METHODS or any(
            term in normalized_name or normalized_name in term for term in linkage_terms
        ):
            return True
    return False


def _function_has_internal_computation(
    node: Union[ast.AsyncFunctionDef, ast.FunctionDef],
) -> bool:
    """Return whether a callable computes and returns an internal value.

    Parameters
    ----------
    node:
        Parsed Python callable definition.

    Returns
    -------
    bool
        True when the body contains a return/yield and framework-neutral
        computation or control-flow syntax.
    """

    body_nodes = [descendant for statement in node.body for descendant in ast.walk(statement)]
    has_result = any(
        isinstance(descendant, (ast.Return, ast.Yield, ast.YieldFrom)) for descendant in body_nodes
    )
    computation_types = (
        ast.AugAssign,
        ast.BinOp,
        ast.BoolOp,
        ast.Call,
        ast.Compare,
        ast.DictComp,
        ast.For,
        ast.GeneratorExp,
        ast.If,
        ast.IfExp,
        ast.ListComp,
        ast.SetComp,
        ast.Subscript,
        ast.Try,
        ast.UnaryOp,
        ast.While,
        ast.With,
    )
    return has_result and any(
        isinstance(descendant, computation_types) for descendant in body_nodes
    )


def _code_bytes_are_relevant_implementation(
    member_name: str,
    content: bytes,
    source: Mapping[str, Any],
    linkage_terms: frozenset[str],
) -> bool:
    """Verify architecture structure and source-to-family linkage in exact bytes.

    Parameters
    ----------
    member_name:
        Archive member name or raw-source locator.
    content:
        Exact fetched candidate source bytes.
    source:
        Bound controlled-fetch manifest row.
    linkage_terms:
        Specific claimed model/family symbols.

    Returns
    -------
    bool
        True only when the bytes are both model-like and linked to the claimed family.
    """

    if not content or len(content) > _MAX_STREAM_INVENTORY_BYTES or not linkage_terms:
        return False
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    normalized_context = re.sub(r"[^a-z0-9]+", "", f"{member_name} {text}".lower())
    if not any(term in normalized_context for term in linkage_terms):
        return False
    if Path(member_name).suffix.lower() in {".py", ".pyx", ""}:
        return _python_has_model_structure(text, linkage_terms)
    architecture_tokens = re.search(
        r"\b(?:attention|conv(?:olution)?|embedding|forward|layer|lstm|model|module|network)\b",
        text,
        flags=re.IGNORECASE,
    )
    definition_tokens = re.search(
        r"\b(?:class|def|function|struct)\s+[A-Za-z_][A-Za-z0-9_]*",
        text,
    )
    return architecture_tokens is not None and definition_tokens is not None


def _validate_structural_slop(facts: Mapping[str, Any], code_paths: Sequence[Path]) -> None:
    """Trip on a plain Sequential/MLP staged as a named exotic family.

    Parameters
    ----------
    facts:
        Proposed fact tree.
    code_paths:
        Closed validated staged model-code manifest.

    Raises
    ------
    ProposalValidationError
        If staged structure is a generic stand-in for an exotic family claim.
    """

    if not code_paths:
        return
    trees: list[ast.AST] = []
    for code_path in code_paths:
        try:
            trees.append(ast.parse(code_path.read_text(encoding="utf-8"), filename=str(code_path)))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise ProposalValidationError(f"cannot inspect staged structure: {exc}") from exc
    defined_classes = {
        node.name for tree in trees for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    module_calls = [
        _call_name(node.func).rsplit(".", 1)[-1]
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            _call_name(node.func).startswith(("nn.", "torch.nn."))
            or _call_name(node.func).rsplit(".", 1)[-1] in _GENERIC_MODEL_CALLS
        )
        and _call_name(node.func).rsplit(".", 1)[-1] not in defined_classes
    ]
    if not module_calls:
        return
    generic_structure = ("Sequential" in module_calls or module_calls.count("Linear") >= 2) and set(
        module_calls
    ) <= _GENERIC_MODEL_CALLS
    if not generic_structure or not _claims_exotic_family(facts):
        return
    raise ProposalValidationError(
        "structural slop tripwire: generic Sequential/MLP staged as an exotic named family"
    )


def _claims_exotic_family(facts: Mapping[str, Any]) -> bool:
    """Return whether authored identity claims more than a generic MLP family.

    Parameters
    ----------
    facts:
        Proposed fact tree.

    Returns
    -------
    bool
        True when family/name/class claims are not explicitly generic feed-forward terms.
    """

    identity = _mapping(facts.get("identity"), "identity")
    metadata = _mapping(facts.get("external_metadata"), "external_metadata")
    architecture_classes = metadata.get("architecture_class")
    names = [
        identity.get("canonical_name"),
        metadata.get("family"),
        *(architecture_classes if isinstance(architecture_classes, list) else []),
    ]
    normalized = {
        _normalize_support_text(str(name))
        for name in names
        if isinstance(name, str) and name.strip()
    }
    return bool(normalized) and not normalized <= _GENERIC_FAMILY_NAMES


def _validate_anti_slop(facts: Mapping[str, Any]) -> None:
    """Reject explicit approximation language on implementation-fidelity surfaces.

    Parameters
    ----------
    facts:
        Proposed fact tree.

    Raises
    ------
    ProposalValidationError
        If implementation-fidelity text admits a generic or simplified stand-in.
    """

    texts = _authored_implementation_texts(facts)
    lowered = " ".join(texts).lower()
    matched = sorted(
        {match.group(0) for pattern in _SLOP_PATTERNS if (match := re.search(pattern, lowered))}
    )
    if matched:
        raise ProposalValidationError(
            f"proposal contains forbidden approximation language: {matched}"
        )


def _authored_implementation_texts(facts: Mapping[str, Any]) -> list[str]:
    """Collect implementation-fidelity prose surfaces that can admit slop.

    Scoped to the surfaces where approximation language means what the tripwire
    thinks it means: the implementation's own rationales, declared choices, patches,
    fidelity reason/deviations, and the source-resolution decision. Descriptions and
    website text are deliberately NOT scanned -- dozens of real roster models are
    *named* with this vocabulary (surrogate-gradient SNNs, Neural Mesh Simplification,
    approximate message passing) and cannot be honestly described without it. The
    five-way fidelity verdict and the structural Sequential/MLP tripwire remain fully
    armed.

    Parameters
    ----------
    facts:
        Proposed fact tree.

    Returns
    -------
    list[str]
        Authored implementation decisions, rationales, and fidelity notes.
    """

    resolution = _mapping(facts.get("source_resolution"), "source_resolution")
    implementation = _mapping(facts.get("implementation"), "implementation")
    fidelity = _mapping(facts.get("fidelity"), "fidelity")
    texts = [
        resolution.get("decision"),
        _mapping(resolution.get("search_report"), "source_resolution.search_report").get(
            "conclusion"
        ),
        fidelity.get("reason"),
    ]
    for collection_name in ("patches", "declared_choices"):
        collection = implementation.get(collection_name)
        if isinstance(collection, list):
            texts.extend(item.get("rationale") for item in collection if isinstance(item, Mapping))
    deviations = fidelity.get("deviations")
    if isinstance(deviations, list):
        texts.extend(deviations)
    return [text for text in texts if isinstance(text, str)]


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    """Return a required mapping.

    Parameters
    ----------
    value:
        Candidate object.
    field:
        Field name used in errors.

    Returns
    -------
    Mapping[str, Any]
        Valid mapping.

    Raises
    ------
    ProposalValidationError
        If the value is not an object.
    """

    if not isinstance(value, Mapping):
        raise ProposalValidationError(f"{field} must be an object")
    return value
