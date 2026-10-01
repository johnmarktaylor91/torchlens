"""The universe registry: the closure object of the composition harness.

Compo memo D1/3.2: closure lives on a COUNTED, OWNED, code-derived universe
registry, not on any taxonomy of bugs. Every universe carries a descriptor
(id, owner, derivation, public boundary, filters, independent census,
adequacy status, review trigger); derived universes print their cardinality
and fail on unexplained drift (baseline:
``data/universe_cardinalities.tsv``); universes whose derivation is not yet
built are DECLARED here as ``OPEN`` rows with owners -- counted as missing,
never silently absent.

Adequacy statuses (memo 3.2 + Dis-3, Sol's enforceable form):

* ``planted`` -- a red-capability plant proves the derivation catches a
  member a weaker derivation would miss.
* ``single_derivation`` -- no independent census exists yet; declared,
  counted, carries reviewer + trigger (never silently assumed adequate).
* ``open`` -- no derivation yet; the row exists so the gap is COUNTED.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tests.composition_expectations import _censuses

DATA_DIR = Path(__file__).resolve().parent / "data"
REPO_ROOT = Path(__file__).resolve().parents[2]

ADEQUACY_STATUSES = ("planted", "single_derivation", "open")


@dataclass(frozen=True)
class UniverseDescriptor:
    """One universe row (memo 3.2 descriptor schema).

    Parameters
    ----------
    universe_id:
        Stable id (``U-...``).
    owner:
        Accountable owner (compo unless the memo routes elsewhere).
    derivation:
        One-line statement of how members are derived (or why OPEN).
    public_boundary:
        What is inside/outside this universe.
    filters:
        Declared narrowings and residuals (``none`` if none). Every
        narrowing is a declared, justified, diffable act (memo D10).
    independent_census:
        The second derivation, or ``single_derivation``.
    adequacy_status:
        One of :data:`ADEQUACY_STATUSES`.
    review_trigger:
        What event forces a review of this row.
    derive:
        Callable producing the member list; ``None`` for OPEN rows.
    """

    universe_id: str
    owner: str
    derivation: str
    public_boundary: str
    filters: str
    independent_census: str
    adequacy_status: str
    review_trigger: str
    derive: Callable[[], tuple[str, ...]] | None = None


def _backends() -> tuple[str, ...]:
    from torchlens.backends.registry import registered_backend_specs

    return tuple(sorted(spec.name for spec in registered_backend_specs()))


def _public_operations() -> tuple[str, ...]:
    import torchlens

    return tuple(sorted(torchlens.__all__))


def _raise_site_keys() -> tuple[str, ...]:
    return tuple(site.site_key for site in _censuses.raise_sites())


def _warn_site_keys() -> tuple[str, ...]:
    return tuple(site.site_key for site in _censuses.warn_sites())


def _save_levels() -> tuple[str, ...]:
    from torchlens._io.tlspec import TLSPEC_VALID_SAVE_LEVELS

    return tuple(sorted(TLSPEC_VALID_SAVE_LEVELS))


def _declared_extras() -> tuple[str, ...]:
    """Extras keys from pyproject (regex idiom; tomllib is 3.11+, floor is 3.10)."""

    lines = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    start = lines.index("[project.optional-dependencies]")
    extras: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("["):
            break
        match = re.match(r"([A-Za-z0-9_-]+)\s*=\s*\[", line)
        if match is not None:
            extras.append(match.group(1))
    return tuple(sorted(extras))


def _open(
    universe_id: str,
    owner: str,
    derivation: str,
    public_boundary: str,
    review_trigger: str,
) -> UniverseDescriptor:
    """An OPEN row: declared, counted-as-missing, owned."""

    return UniverseDescriptor(
        universe_id=universe_id,
        owner=owner,
        derivation=derivation,
        public_boundary=public_boundary,
        filters="none declared yet (row is OPEN)",
        independent_census="single_derivation",
        adequacy_status="open",
        review_trigger=review_trigger,
    )


UNIVERSES: tuple[UniverseDescriptor, ...] = (
    UniverseDescriptor(
        universe_id="U-BACKENDS",
        owner="compo",
        derivation="torchlens.backends.registry.registered_backend_specs()",
        public_boundary="every registered backend runtime, previews included",
        filters="none",
        independent_census="tests/backend_conformance suite roster (diff wired at wave A/M2)",
        adequacy_status="single_derivation",
        review_trigger="a backend registered outside registry.py or a conformance roster drift",
        derive=_backends,
    ),
    UniverseDescriptor(
        universe_id="U-PUBLIC-OPERATIONS",
        owner="compo",
        derivation="sorted(torchlens.__all__)",
        public_boundary="top-level public names; submodule verbs join via the api-surface census",
        filters="submodule surfaces (tl.debug, tl.report, ...) enter at wave A via the surface walk",
        independent_census="tests/oracles reachable-surface walk (diff wired at wave A)",
        adequacy_status="single_derivation",
        review_trigger="any __all__ edit or api-surface census drift",
        derive=_public_operations,
    ),
    UniverseDescriptor(
        universe_id="U-CAPTURE-OPTIONS",
        owner="compo",
        derivation="dataclasses.fields(CaptureOptions) minus the declared private filter",
        public_boundary="the 47 public capture-option fields (48 minus _specified_fields)",
        filters="CAPTURE_OPTIONS_PRIVATE_FIELDS (explicit; red-capability-proved both directions)",
        independent_census="tests/oracles Options-dataclass census (108 fields, all six classes)",
        adequacy_status="planted",
        review_trigger="any CaptureOptions field add/remove or a new private field",
        derive=_censuses.capture_option_fields,
    ),
    UniverseDescriptor(
        universe_id="U-RAISE-SITES",
        owner="compo",
        derivation="S-17: AST raise sites of package-defined *Error classes + factory-call closure",
        public_boundary="raises of TorchLens-defined *Error classes; builtin-exception raises are wave-A review",
        filters=(
            "call sites of always-raising helpers OUTSIDE raise position are a declared "
            "residual; factory resolution is name-keyed across modules"
        ),
        independent_census=(
            "the code=-literal scanner in tests/test_error_contract_lockstep.py covers the "
            "coded subset; full two-way census is the S-17 burn-down"
        ),
        adequacy_status="planted",
        review_trigger="new *Error class, new factory shape, or contract-doc drift",
        derive=_raise_site_keys,
    ),
    UniverseDescriptor(
        universe_id="U-WARN-SITES",
        owner="compo",
        derivation="S-18: alias-resolving AST census of warnings.warn call sites",
        public_boundary="every warnings.warn reachable from package code",
        filters="dynamic warn references (getattr(warnings, 'warn')) are a declared residual",
        independent_census=(
            "the literal warnings.warn scan is the WEAKER census; alias-resolved minus "
            "literal is the live plant set (currently non-empty)"
        ),
        adequacy_status="planted",
        review_trigger="new warn alias/import shape or a warn helper wrapper",
        derive=_warn_site_keys,
    ),
    UniverseDescriptor(
        universe_id="U-ENV-VARS",
        owner="compo",
        derivation="TORCHLENS_* literals at environ reads, declared reader helpers, *_ENV constants",
        public_boundary="environment variables the package READS (not ones it merely names in prose)",
        filters="reader set is declared (ENV_READER_CALLEES); non-literal keys are a residual",
        independent_census="single_derivation",
        adequacy_status="single_derivation",
        review_trigger="a new env reader helper or a TORCHLENS_ literal outside the shapes",
        derive=_censuses.env_var_reads,
    ),
    UniverseDescriptor(
        universe_id="U-SAVE-LEVELS",
        owner="compo",
        derivation="torchlens._io.tlspec.TLSPEC_VALID_SAVE_LEVELS (closed vocabulary)",
        public_boundary="tl.save level= values; storage kinds cross at wave A (save x storage matrix)",
        filters="none",
        independent_census="single_derivation",
        adequacy_status="single_derivation",
        review_trigger="any TLSPEC_VALID_SAVE_LEVELS edit",
        derive=_save_levels,
    ),
    UniverseDescriptor(
        universe_id="U-EXTRAS",
        owner="compo",
        derivation="pyproject [project.optional-dependencies] keys (regex idiom)",
        public_boundary="declared installable extras",
        filters="none",
        independent_census="dependency-declaration clean-env sweep (wave C, C.3)",
        adequacy_status="single_derivation",
        review_trigger="any optional-dependencies edit",
        derive=_declared_extras,
    ),
    # ------------------------------------------------------------------
    # OPEN rows: declared and counted, derivations land in waves A-D.
    # ------------------------------------------------------------------
    _open(
        "U-PRODUCT-STATES",
        "compo",
        "hand registry + two-way public-class census (wave A, feeds M1)",
        "product/lifecycle states (Trace, Recording, PartialTrace, MergedTrace, forks, loaded...)",
        "M1 build start",
    ),
    _open(
        "U-EDIT-VERBS",
        "compo",
        "edit-verb inventory (wave A, feeds the 26x11 edit x product matrix)",
        "public edit verbs on the push/replay engines",
        "edit x product matrix build start",
    ),
    _open(
        "U-RENDER-SURFACES",
        "compo",
        "read/render/export surface inventory (wave A, feeds M1 and the sink matrix)",
        "report/draw/export/agent doors",
        "M1 build start",
    ),
    _open(
        "U-MODEL-TRAITS",
        "compo+testing",
        "declared fixture table (P04 owns rosters; compo consumes trait axes)",
        "model traits the galleries can discharge (tied weights, multi-input, in-place, ...)",
        "R0 fixture roster landing",
    ),
    _open(
        "U-TAUGHT-PATHS",
        "compo",
        "docs harvester (0.9 spike prints counts; executable subset first, prose OPEN)",
        "public paths the docs/docstrings/notebooks/remedies teach",
        "harvestability spike verdict (this wave)",
    ),
    _open(
        "U-DEPRECATED-ALIASES",
        "compo",
        "deprecation inventory (tests/test_deprecation_inventory.py pins the package deprecation-free)",
        "deprecated aliases/shims (currently pinned empty)",
        "any new alias/shim landing",
    ),
    _open(
        "U-FOREIGN-APIS",
        "compo",
        "derived from the foreign package at test time (wave C, C.2)",
        "foreign API families the bridges claim (captum, netron, nnsight, ...)",
        "foreign-signature conformance build start",
    ),
    _open(
        "U-PARAM-NAMES",
        "compo",
        "public parameter names x accepting entries (mechanical half; spellings = naming sprint)",
        "kwarg spellings across sibling entries",
        "default-agreement sweep build start",
    ),
    _open(
        "U-CONTAINER-PROTOCOL",
        "compo",
        "public classes x container/dunder protocol (wave A sweep)",
        "__contains__/__iter__/__len__/__getitem__ agreement domains",
        "container-protocol sweep build start",
    ),
    _open(
        "U-VIEW-VS-COPY",
        "compo",
        "public accessors x view-vs-copy contract (wave A sweep; SG#49 substrate)",
        "accessor mutation semantics",
        "view/copy sweep build start",
    ),
    _open(
        "U-UNTRUSTED-ENTRY",
        "compo (bounds half); security lane (adversarial half)",
        "entry points consuming untrusted external input (loaders, MCP tools, manifest parsers)",
        "declared byte/opcode/token bounds on untrusted-input doors",
        "MCP/agent sweep build start (C.5)",
    ),
    _open(
        "U-GLOSSARY-TERMS",
        "naming sprint (compo owns the mechanizable citation check)",
        "field-glossary terms; citation-resolution half mechanized at wave A",
        "glossary term census",
        "naming sprint kickoff",
    ),
    _open(
        "U-PROSE-CLAIMS",
        "docs panel (compo provides the generated projection as the quotable source)",
        "prose support claims; OPEN until the claim parser + planted-drift test exist",
        "support claims in migration/limitation docs",
        "docs panel kickoff",
    ),
)


def derived_cardinalities() -> dict[str, int | str]:
    """Universe id -> live cardinality (``"OPEN"`` for underived rows)."""

    counts: dict[str, int | str] = {}
    for universe in UNIVERSES:
        counts[universe.universe_id] = len(universe.derive()) if universe.derive else "OPEN"
    return counts


def baseline_cardinalities() -> dict[str, str]:
    """The committed cardinality baseline (universe id -> count or OPEN)."""

    rows: dict[str, str] = {}
    path = DATA_DIR / "universe_cardinalities.tsv"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("universe_id\t"):
            continue
        universe_id, _, count = line.partition("\t")
        rows[universe_id.strip()] = count.strip()
    return rows
