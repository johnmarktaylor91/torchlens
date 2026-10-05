"""The validation tripwire's own exemption ledger (grind r2 row 19 / matrix R08).

``CLAUDE.md`` -> "Validation Integrity (LOCKED PRINCIPLE)": the ``validation/``
pipeline is a TRIPWIRE, and the ONLY legitimate exemption is behavior that is
*correct by design and provably outside the check's contract*, scoped narrowly
enough that it cannot mask the unintended case.

That principle has been enforced case by case in review. This module makes it
enforceable as a CLASS: every exemption the pipeline can emit is registered here
with the contract clause it sits outside, the predicate that proves its
narrowness, and the case it must still refuse. The vocabulary is derived from
the source (AST for the decision constructors, live imports for the registries),
so a new exemption cannot appear without a ledger entry -- which is where the
contract citation is demanded.

This is an AUDIT artifact: the ledger lives here, in the tests, so auditing the
tripwire never edits the tripwire.

Contract clauses cited below
----------------------------
``C1 replay determinism``
    ``validation/CLAUDE.md`` "Forward Replay Flow" step 4: replaying a saved op
    from its saved parents reproduces the saved output. An exemption is outside
    this clause only when the op's output is NOT a deterministic function of its
    saved inputs (uninitialized memory, RNG-determined values).
``C2 perturbation sensitivity``
    ``validation/CLAUDE.md`` step 5: perturbing a parent must change the output.
    An exemption is outside this clause only when that parent's VALUE provably
    cannot affect the output -- proved from the saved call, never assumed from
    the op's name.
``C3 replay-surface completeness``
    ``validation/CLAUDE.md`` step 4 walks every saved op on the output path. An
    exemption is outside this clause only when the op has no traceable function
    BY DESIGN (a genuine boundary/source node, or a user-injected intervention
    tensor) or the user's own ``save=`` predicate excluded the payload. The
    2026-06-02 incident in ``CLAUDE.md`` is the negative example: an
    auto-synthesized placeholder in PLAIN capture must STILL fail.
``C4 numeric resolution``
    A perturbation whose effect is smaller than the output dtype's representable
    spacing cannot be observed at all, so an unchanged output is arithmetic, not
    a dropped dependency.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest
from _source_corpus import package_files, package_source

from torchlens.validation import exemptions as ex

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXEMPTIONS_SRC = _REPO_ROOT / "torchlens" / "validation" / "exemptions.py"
_CORE_SRC = _REPO_ROOT / "torchlens" / "validation" / "core.py"


def _exemption_emitting_sources() -> list[Path]:
    """Every torchlens source file that can emit an exemption decision.

    r8 R08-1 (opus): the closure derivation used to parse EXACTLY the two
    files above, so an exemption emitted from any THIRD module passed both
    closure directions vacuously (never derived, never ledgered -- the whole
    tripwire for the LOCKED principle going silent per-exemption). The file
    set is now DISCOVERED by scanning the package for the emission patterns
    the two scanners recognize; a god-file split of ``core.py`` or a brand
    new emitting module enters the closure automatically.
    """

    emission_markers = (
        "PosthocPerturbDecision(",
        ".exempted(",
        'decision="exempted"',
    )
    sources: list[Path] = []
    for path in package_files():
        text = package_source(path)
        if any(marker in text for marker in emission_markers):
            sources.append(path)
    return sources


_CONTRACTS = frozenset(
    {
        "C1 replay determinism",
        "C2 perturbation sensitivity",
        "C3 replay-surface completeness",
        "C4 numeric resolution",
    }
)

_TIERS = frozenset({"early_exit", "registry", "posthoc", "recorder"})


@dataclass(frozen=True)
class Exemption:
    """One registered validation exemption and its audit record.

    Parameters
    ----------
    code:
        Stable reason code the pipeline emits.
    tier:
        Where the decision is taken: ``early_exit`` (before replay),
        ``registry`` (declared per-op table), ``posthoc`` (after the perturbed
        call ran), or ``recorder`` (decision-log only, no verdict of its own).
    contract:
        Contract clause from the module docstring that the exemption sits
        OUTSIDE. Not "the check we turned off" -- the clause it never covered.
    proof:
        ``module:symbol`` of the predicate that establishes narrowness. Checked
        to exist, so a citation cannot rot.
    refuses:
        The case the exemption must still let FAIL. This is the anti-masking
        record: an exemption whose ``refuses`` is empty is a blanket, not a
        carve-out.
    """

    code: str
    tier: str
    contract: str
    proof: str
    refuses: str


#: EVERY exemption the validation pipeline can emit, with its audit record.
#: Kept exhaustive against the source by the closure tests below.
EXEMPTION_LEDGER: tuple[Exemption, ...] = (
    # ---------------------------------------------------------------- early exit
    Exemption(
        code="functionless_source_or_boundary",
        tier="early_exit",
        contract="C3 replay-surface completeness",
        proof="torchlens.validation.core:_is_provable_functionless_source_or_boundary",
        refuses=(
            "a functionless COMPUTATIONAL op, which returns failed_result "
            "('functionless_computational_op') -- the 2026-06-02 placeholder class"
        ),
    ),
    Exemption(
        code="uninitialized_by_design",
        tier="early_exit",
        contract="C1 replay determinism",
        proof="torchlens.validation.exemptions:SKIP_VALIDATION_ENTIRELY",
        refuses=(
            "any op outside the six uninitialized-memory constructors; the table "
            "carries a per-entry justification string, not a bare name"
        ),
    ),
    Exemption(
        code="not_saved_by_user",
        tier="early_exit",
        contract="C3 replay-surface completeness",
        proof="torchlens.validation.core:_classify_user_excluded_replay_surface",
        refuses=(
            "an omission NOT attributable to the user's own save= predicate: an "
            "exhaustive capture missing a payload still fails"
        ),
    ),
    Exemption(
        code="pre_perturbation_exemption",
        tier="early_exit",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.core:_check_perturbation_exemptions",
        refuses=(
            "any op/arg pair outside the four declared registries; forward replay "
            "still runs for every op this only excuses from PERTURBATION"
        ),
    ),
    Exemption(
        code="ulp_swamped_perturbation",
        tier="early_exit",
        contract="C4 numeric resolution",
        proof="torchlens.validation.core:_perturbation_delta_below_output_spacing",
        refuses=(
            "a delta at or above the output dtype's representable spacing, which "
            "must change the output and otherwise fails"
        ),
    ),
    Exemption(
        code="parent_inplace_rng_bernoulli",
        tier="early_exit",
        contract="C1 replay determinism",
        proof="torchlens.validation.core:_check_whether_func_on_saved_parents_yields_saved_tensor",
        refuses=(
            "a non-RNG in-place parent, and any mismatch the snapshot proof cannot "
            "explain: the exemption fires only when re-replaying the op from the "
            "child's own saved-arg snapshots reproduces the saved output, so a "
            "corrupted child func/kwargs under a bernoulli_ parent still fails"
        ),
    ),
    # ------------------------------------------------------------------ recorder
    Exemption(
        code="intentional_intervention_replacement",
        tier="recorder",
        contract="C3 replay-surface completeness",
        proof="torchlens.validation.core:_is_intentional_intervention_replacement",
        refuses=(
            "a replacement op appearing during PLAIN capture -- the intervention "
            "must be attributable to a user spec, which is the LOCKED narrowing"
        ),
    ),
    Exemption(
        code="skip_perturbation_entirely",
        tier="recorder",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:SKIP_PERTURBATION_ENTIRELY",
        refuses=(
            "forward replay, which still runs for every member; and any op not "
            "literally named in the table"
        ),
    ),
    # ------------------------------------------------------------------- posthoc
    Exemption(
        code="discrete_bool_output",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_posthoc_discrete_output_decision",
        refuses=(
            "a floating/complex output, where a value perturbation must show up; "
            "(R08) an elementwise comparison whose threshold-straddle probe "
            "proves the recorded parent has no value influence -- that op falls "
            "through to the perturbation_insensitive failure; and (R08-2) a "
            "non-comparison predicate op whose substitution battery (sign/zero/"
            "extreme/non-finite/inversion) pins the output across every "
            "substitute -- same fall-through. Bool-output "
            "children also run the negate/zero/NaN + magnitude-ladder retry "
            "rungs BEFORE any exemption is consulted, so a real edge validates "
            "instead of exempting"
        ),
    ),
    Exemption(
        code="discrete_index_output",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_posthoc_discrete_output_decision",
        refuses=(
            "the VALUE half of an index-and-value op (topk/sort values), and any "
            "non-integer output dtype"
        ),
    ),
    Exemption(
        code="discrete_value_output",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_posthoc_discrete_output_decision",
        refuses="a continuous output whose value carries the perturbed parent",
    ),
    Exemption(
        code="type_template_output",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_posthoc_structural_output_decision",
        refuses=(
            "a perturbed parent outside the template slot (args[1]/other=) -- in "
            "particular the perturbed data SOURCE of the cast, whose values flow "
            "through to the output"
        ),
    ),
    Exemption(
        code="integer_cast_quantization",
        tier="posthoc",
        contract="C4 numeric resolution",
        proof="torchlens.validation.exemptions:_posthoc_structural_output_decision",
        refuses="a perturbation large enough to cross an integer quantization boundary",
    ),
    Exemption(
        code="structural_output_template",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_posthoc_structural_output_decision",
        refuses=(
            "a *_like/meshgrid/broadcast op whose output actually carries input "
            "VALUES rather than only shape/dtype/device"
        ),
    ),
    Exemption(
        code="rng_probability_template",
        tier="posthoc",
        contract="C1 replay determinism",
        proof="torchlens.validation.exemptions:_posthoc_structural_output_decision",
        refuses="a non-RNG op, and the RNG state itself (which replay still pins)",
    ),
    Exemption(
        code="uninitialized_destination_overwrite",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_perturbed_parent_is_uninitialized_setitem_dest",
        refuses=(
            "a destination holding REAL written data -- the walk deliberately does "
            "NOT chain through prior in-place writes (B1 review hole, TwoIndexCopyDim2)"
        ),
    ),
    Exemption(
        code="full_destination_overwrite",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_setitem_destination_coverage_is_total",
        refuses=(
            "a partial overwrite, and duplicate advanced indices that would fake "
            "total coverage via numel equality"
        ),
    ),
    Exemption(
        code="scalar_destination_overwrite",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_setitem_destination_coverage_is_total",
        refuses=(
            "the same partial-coverage and duplicate-index cases as the tensor "
            "right-hand side: scalar RHS is held to identical index rigor"
        ),
    ),
    Exemption(
        code="index_domain_value_irrelevant",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_index_domain_value_irrelevance_decision",
        refuses=(
            "a perturbable index domain: indices are rotated IN-DOMAIN and only a "
            "provably degenerate domain (n<=1, or no in-range entry) is excused"
        ),
    ),
    Exemption(
        code="multiplicative_zero_annihilator",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_posthoc_value_proof_decision",
        refuses=(
            "a non-zero (or NaN/Inf-bearing) non-perturbed operand, where the "
            "product does carry the perturbed value"
        ),
    ),
    Exemption(
        code="addcmul_zero_annihilator",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_addcmul_zero_annihilator_decision",
        refuses=(
            "a non-zero (or NaN/Inf-bearing) sibling multiplied operand or "
            "non-zero value=, where the product does carry the perturbed value; "
            "a perturbed input= (the additive, non-multiplied operand) is never "
            "matched and stays strict"
        ),
    ),
    Exemption(
        code="binary_extrema_dominated",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_binary_extrema_nonperturbed_arg_dominates",
        refuses=(
            "any element where the perturbed operand is not dominated; domination "
            "must hold at EVERY output element"
        ),
    ),
    Exemption(
        code="mod_divisor_irrelevant",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_posthoc_value_proof_decision",
        refuses="the DIVIDEND operand, keyed off parent_arg_positions not tensor equality",
    ),
    Exemption(
        code="integer_mod_unit_divisor",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_integer_mod_unit_divisor_decision",
        refuses=(
            "a float dividend, a divisor other than +-1, a nonzero saved output, and "
            "any perturbed parent at the divisor slot"
        ),
    ),
    Exemption(
        code="sdpa_zero_query_uniform_attention",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_sdpa_zero_query_decision",
        refuses=(
            "a non-zero (or NaN/Inf-bearing) saved query, and a perturbed query or "
            "value operand -- both genuinely influence sdpa's output even when query "
            "is zero, so only the key operand is ever exempted"
        ),
    ),
    Exemption(
        code="softmax_singleton_reduction_dim",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_softmax_singleton_dim_decision",
        refuses=(
            "a reduction dimension with more than one element, and a perturbed parent "
            "other than softmax's own (sole) logits input"
        ),
    ),
    Exemption(
        code="layer_norm_singleton_normalized_shape",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_layer_norm_singleton_shape_decision",
        refuses=(
            "a normalized_shape spanning more than one element, and a perturbed parent "
            "other than layer_norm's own input"
        ),
    ),
    Exemption(
        code="locally_constant_by_construction",
        tier="posthoc",
        contract="C2 perturbation sensitivity",
        proof="torchlens.validation.exemptions:_posthoc_value_proof_decision",
        refuses=(
            "a region where the saved values are NOT provably locally constant "
            "(finite operands, or a NaN/Inf pattern that does not absorb the delta)"
        ),
    ),
)


_LEDGER_BY_CODE = {entry.code: entry for entry in EXEMPTION_LEDGER}


# ---------------------------------------------------------------------------
# Source-derived vocabularies (the "generated" side of the audit).
# ---------------------------------------------------------------------------


def posthoc_exempting_reasons(source: str) -> set[str]:
    """Return reason codes that ``PosthocPerturbDecision`` can EXEMPT with.

    Only ``exempt=True`` constructions count: the ``exempt=False`` reasons are
    diagnostics, not carve-outs, and must not need a ledger entry.

    Parameters
    ----------
    source:
        Source text of ``torchlens/validation/exemptions.py``.

    Returns
    -------
    set[str]
        Exempting reason codes, including reasons selected by a conditional
        expression (both branches are collected).
    """

    tree = ast.parse(source)
    bindings = _string_bindings(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "PosthocPerturbDecision"
        ):
            continue
        exempt: object = None
        reason_node: ast.expr | None = None
        if node.args:
            if isinstance(node.args[0], ast.Constant):
                exempt = node.args[0].value
            if len(node.args) > 1:
                reason_node = node.args[1]
        for keyword in node.keywords:
            if keyword.arg == "exempt" and isinstance(keyword.value, ast.Constant):
                exempt = keyword.value.value
            if keyword.arg == "reason":
                reason_node = keyword.value
        if exempt is not True or reason_node is None:
            continue
        found.update(_string_constants(reason_node, bindings))
    return found


def _string_bindings(tree: ast.AST) -> dict[str, set[str]]:
    """Return simple ``name = <string literal or conditional>`` bindings.

    A reason is sometimes computed one line above the decision
    (``reason = "a" if ... else "b"``). Without resolving that binding the
    scanner would silently miss two real exemptions, so the closure would look
    complete while covering less than the source emits.

    Parameters
    ----------
    tree:
        Parsed module.

    Returns
    -------
    dict[str, set[str]]
        Variable name -> string literals it can hold.
    """

    bindings: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        literals = _string_constants(node.value, {})
        if literals:
            bindings.setdefault(target.id, set()).update(literals)
    return bindings


def _string_constants(node: ast.expr, bindings: dict[str, set[str]]) -> set[str]:
    """Return every string literal a reason expression can evaluate to.

    Parameters
    ----------
    node:
        Reason expression (a literal, a conditional over literals, or a name
        bound to either).
    bindings:
        Resolved simple string bindings.

    Returns
    -------
    set[str]
        String literals reachable from the expression. An expression the scanner
        cannot resolve yields nothing, and the closure test then fails loudly on
        the missing ledger entry rather than passing silently.
    """

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _string_constants(node.body, bindings) | _string_constants(node.orelse, bindings)
    if isinstance(node, ast.Name):
        return set(bindings.get(node.id, set()))
    return set()


def core_exempting_reasons(source: str) -> set[str]:
    """Return reason codes ``validation/core.py`` can settle as exempted.

    Covers both spellings: ``ValidationCheckResult.exempted("code", ...)`` and
    decision-recorder rows carrying ``decision="exempted"``. A recorder reason
    built as an f-string (``skip_perturbation_entirely:<func>``) contributes its
    stable prefix.

    Parameters
    ----------
    source:
        Source text of ``torchlens/validation/core.py``.

    Returns
    -------
    set[str]
        Reason codes, prefixes normalized.
    """

    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "exempted"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.add(node.args[0].value)
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        decision = keywords.get("decision")
        if not (isinstance(decision, ast.Constant) and decision.value == "exempted"):
            continue
        reason = keywords.get("reason")
        if isinstance(reason, ast.Constant) and isinstance(reason.value, str):
            found.add(reason.value)
        elif isinstance(reason, ast.JoinedStr):
            prefix = next(
                (
                    value.value
                    for value in reason.values
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                ),
                "",
            )
            found.add(prefix.rstrip(":"))
    return found


def ledger_gaps(derived: set[str], ledgered: set[str]) -> tuple[set[str], set[str]]:
    """Return unledgered and phantom exemption codes.

    Parameters
    ----------
    derived:
        Codes found in the source.
    ledgered:
        Codes registered in :data:`EXEMPTION_LEDGER`.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(unledgered, phantom)``.
    """

    return derived - ledgered, ledgered - derived


# ---------------------------------------------------------------------------
# Closure: the ledger is exhaustive against the source.
# ---------------------------------------------------------------------------


def test_posthoc_exemption_vocabulary_is_fully_ledgered() -> None:
    """Every posthoc exemption reason has an audit record (package-wide)."""

    derived: set[str] = set()
    for source in _exemption_emitting_sources():
        derived |= posthoc_exempting_reasons(source.read_text(encoding="utf-8"))
    ledgered = {entry.code for entry in EXEMPTION_LEDGER if entry.tier == "posthoc"}
    unledgered, phantom = ledger_gaps(derived, ledgered)
    assert not unledgered, (
        "posthoc exemptions with no ledger entry -- each needs the contract clause it "
        f"sits outside and the predicate proving narrowness: {sorted(unledgered)}"
    )
    assert not phantom, f"ledgered posthoc exemptions no longer emitted: {sorted(phantom)}"


def test_core_exemption_vocabulary_is_fully_ledgered() -> None:
    """Every settleable core exemption has an audit record (package-wide)."""

    derived: set[str] = set()
    for source in _exemption_emitting_sources():
        if source == _EXEMPTIONS_SRC:
            # The posthoc constructor file: its ``exempt=True`` reasons are
            # the POSTHOC vocabulary above, not core settlement codes.
            continue
        derived |= core_exempting_reasons(source.read_text(encoding="utf-8"))
    ledgered = {
        entry.code for entry in EXEMPTION_LEDGER if entry.tier in ("early_exit", "recorder")
    }
    unledgered, phantom = ledger_gaps(derived, ledgered)
    assert not unledgered, f"core exemptions with no ledger entry: {sorted(unledgered)}"
    assert not phantom, f"ledgered core exemptions no longer emitted: {sorted(phantom)}"


def test_registry_tables_are_covered_by_a_ledger_entry() -> None:
    """The four declared registries each route through a ledgered exemption.

    Code-level closure only: both remaining registries settle under the single
    ``pre_perturbation_exemption`` code, so this test cannot see a new ENTRY in
    either table. The per-entry closure that can is
    :data:`STRUCTURAL_POSITION_LEDGER` / :data:`CUSTOM_CHECK_LEDGER` below
    (finding B1-06).
    """

    assert "uninitialized_by_design" in _LEDGER_BY_CODE  # SKIP_VALIDATION_ENTIRELY
    assert "skip_perturbation_entirely" in _LEDGER_BY_CODE  # SKIP_PERTURBATION_ENTIRELY
    # STRUCTURAL_ARG_POSITIONS + CUSTOM_EXEMPTION_CHECKS both settle here.
    assert "pre_perturbation_exemption" in _LEDGER_BY_CODE
    assert ex.SKIP_VALIDATION_ENTIRELY and ex.SKIP_PERTURBATION_ENTIRELY
    assert ex.STRUCTURAL_ARG_POSITIONS and ex.CUSTOM_EXEMPTION_CHECKS


# ---------------------------------------------------------------------------
# Quality: every audit record is real, and every citation resolves.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", EXEMPTION_LEDGER, ids=lambda entry: entry.code)
def test_ledger_entry_is_well_formed(entry: Exemption) -> None:
    """Each entry names a known tier, a known contract clause, and a refusal."""

    assert entry.tier in _TIERS, f"{entry.code}: unknown tier {entry.tier!r}"
    assert entry.contract in _CONTRACTS, f"{entry.code}: uncited contract {entry.contract!r}"
    assert len(entry.refuses.strip()) >= 30, (
        f"{entry.code}: 'refuses' must name the case the exemption still lets fail; "
        "an exemption with nothing it refuses is a blanket, not a carve-out"
    )


@pytest.mark.parametrize("entry", EXEMPTION_LEDGER, ids=lambda entry: entry.code)
def test_ledger_proof_symbol_exists(entry: Exemption) -> None:
    """The cited narrowing predicate exists, so a citation cannot rot."""

    module_path, _, symbol = entry.proof.partition(":")
    assert symbol, f"{entry.code}: proof must be 'module:symbol'"
    module = __import__(module_path, fromlist=[symbol])
    assert hasattr(module, symbol), f"{entry.code}: missing proof symbol {entry.proof}"


def test_skip_validation_entirely_entries_carry_justifications() -> None:
    """The widest tier (skip ALL validation) records a reason per entry."""

    for func_name, justification in ex.SKIP_VALIDATION_ENTIRELY.items():
        assert len(justification.strip()) >= 30, (
            f"{func_name}: skipping BOTH replay and perturbation needs a real "
            "by-construction justification"
        )


#: FINDING R19-F1 (r2 row 19; FIXED r3): ``SKIP_PERTURBATION_ENTIRELY`` was the
#: only registry shaped as a bare ``set`` -- its members carried no per-entry
#: justification, only a shared block comment, while every other tier records a
#: reason per row (``SKIP_VALIDATION_ENTIRELY`` maps name -> justification; the
#: posthoc tier returns a reason code; ``STRUCTURAL_ARG_POSITIONS`` carries
#: inline per-entry comments). The r3 fix converted the registry itself into a
#: justification mapping under contract clause C2 (perturbation sensitivity);
#: membership semantics are unchanged. This ledger stays as the independent
#: audit record: the test below pins the live mapping's keys AND values against
#: it, so a registry edit without a matching audit-record edit still fails --
#: the tripwire got stricter, never looser.
SKIP_PERTURBATION_JUSTIFICATIONS: dict[str, str] = {
    "new_zeros": "output is all zeros by construction; no parent value reaches it",
    "new_ones": "output is all ones by construction; no parent value reaches it",
    "zero_": "in-place zero fill; the destination's prior values are discarded",
    "zeros_like": "shape/dtype/device template only; the output value is constant zero",
    "ones_like": "shape/dtype/device template only; the output value is constant one",
    "rand_like": "values are RNG-drawn; the parent supplies shape/dtype/device only",
    "randn_like": "values are RNG-drawn; the parent supplies shape/dtype/device only",
    # meshgrid/broadcast_tensors left this ledger with the R08-2 NARROWING:
    # they now live in CUSTOM_EXEMPTION_CHECKS (_check_zipped_sibling_exempt),
    # so each output's OWN value edge is perturbation-tested again and only
    # provable cross-member zipped-sibling perturbations stay exempt.
    # The six torchvision PyCapsule rows left this ledger with the b1p2 D2
    # adjudicated NARROWING: they now live in STRUCTURAL_ARG_POSITIONS keyed
    # on the coordinate/offset arg only, so feature/score value edges are
    # perturbation-tested again with zero segfault exposure.
    "exponential_": "in-place RNG draw; the output is determined by RNG state, not inputs",
}


def test_skip_perturbation_registry_has_a_per_entry_audit_record() -> None:
    """Each bare-set perturbation skip carries a justification in the ledger."""

    live = set(ex.SKIP_PERTURBATION_ENTIRELY)
    ledgered = set(SKIP_PERTURBATION_JUSTIFICATIONS)
    assert live == ledgered, (
        "SKIP_PERTURBATION_ENTIRELY changed without an audit record; only live: "
        f"{sorted(live - ledgered)}; only ledgered: {sorted(ledgered - live)}"
    )
    # R19-F1 fix: the registry itself now carries the justification per entry.
    # Pin the VALUES against this independent ledger too, so rewriting a
    # justification in the tripwire without updating the audit record fails.
    assert dict(ex.SKIP_PERTURBATION_ENTIRELY) == SKIP_PERTURBATION_JUSTIFICATIONS
    for func_name, justification in SKIP_PERTURBATION_JUSTIFICATIONS.items():
        assert len(justification.strip()) >= 30, f"{func_name} needs a real justification"


def test_structural_arg_positions_are_position_keyed_not_value_keyed() -> None:
    """Structural exemptions key on argument POSITION, never tensor equality.

    The repeated review finding behind several narrowings: a data tensor whose
    contents merely EQUAL a structural argument must not be excused. Every
    registry row is therefore a set of integer positions.
    """

    for func_name, positions in ex.STRUCTURAL_ARG_POSITIONS.items():
        assert positions, f"{func_name}: empty structural position set"
        assert all(isinstance(position, int) for position in positions), (
            f"{func_name}: structural exemptions must be keyed by arg position"
        )


def test_custom_exemption_checks_are_callables_with_the_registry_signature() -> None:
    """Every custom check is a callable taking (trace, op, layers_to_perturb)."""

    import inspect

    for func_name, check in ex.CUSTOM_EXEMPTION_CHECKS.items():
        assert callable(check), f"{func_name}: custom check is not callable"
        parameters = list(inspect.signature(check).parameters)
        assert len(parameters) == 3, f"{func_name}: unexpected custom-check signature"


# ---------------------------------------------------------------------------
# The audit mechanism must be able to go RED.
# ---------------------------------------------------------------------------


class TestLedgerMechanismIsRedCapable:
    """Plant an unledgered exemption and prove the closure reports it."""

    def test_posthoc_scanner_finds_a_planted_exemption(self) -> None:
        """A new exempting reason in the source is picked up by the scanner."""

        planted = "def f():\n    return PosthocPerturbDecision(True, 'planted_new_excuse')\n"
        assert posthoc_exempting_reasons(planted) == {"planted_new_excuse"}

    def test_posthoc_scanner_resolves_a_reason_bound_to_a_local(self) -> None:
        """A reason assigned one line above the decision is still collected."""

        source = (
            "def f():\n"
            "    reason = 'left' if flag else 'right'\n"
            "    return PosthocPerturbDecision(True, reason)\n"
        )
        assert posthoc_exempting_reasons(source) == {"left", "right"}

    def test_posthoc_scanner_ignores_non_exempting_reasons(self) -> None:
        """``exempt=False`` diagnostics are not carve-outs and are not demanded."""

        source = "x = PosthocPerturbDecision(False, 'not_a_carve_out')\n"
        assert posthoc_exempting_reasons(source) == set()

    def test_posthoc_scanner_collects_both_conditional_branches(self) -> None:
        """A reason chosen by a conditional contributes BOTH literals."""

        source = "x = PosthocPerturbDecision(True, 'a' if flag else 'b')\n"
        assert posthoc_exempting_reasons(source) == {"a", "b"}

    def test_core_scanner_finds_both_exemption_spellings(self) -> None:
        """Verdict-settling and recorder-only exemptions are both collected."""

        source = (
            "a = ValidationCheckResult.exempted('planted_verdict')\n"
            "b = rec.record(decision='exempted', reason='planted_recorder')\n"
            "c = rec.record(decision='exempted', reason=f'planted_prefix:{name}')\n"
        )
        assert core_scanner_result(source) == {
            "planted_verdict",
            "planted_recorder",
            "planted_prefix",
        }

    def test_core_scanner_ignores_non_exempted_decisions(self) -> None:
        """A failing or validated decision row is not an exemption."""

        source = "b = rec.record(decision='failed', reason='real_failure')\n"
        assert core_scanner_result(source) == set()

    def test_ledger_gap_checker_reports_both_directions(self) -> None:
        """Unledgered and phantom codes are both surfaced."""

        unledgered, phantom = ledger_gaps({"new_code"}, {"old_code"})
        assert unledgered == {"new_code"}
        assert phantom == {"old_code"}

    def test_proof_citation_checker_rejects_a_missing_symbol(self) -> None:
        """A rotted citation fails rather than passing silently."""

        module = __import__(
            "torchlens.validation.exemptions", fromlist=["SKIP_VALIDATION_ENTIRELY"]
        )
        assert hasattr(module, "SKIP_VALIDATION_ENTIRELY")
        assert not hasattr(module, "_definitely_not_a_real_predicate")


def core_scanner_result(source: str) -> set[str]:
    """Return :func:`core_exempting_reasons` over arbitrary source text.

    Parameters
    ----------
    source:
        Python source to scan.

    Returns
    -------
    set[str]
        Exemption reason codes found.
    """

    return core_exempting_reasons(source)


# ---------------------------------------------------------------------------
# FINDING B1-06: per-entry closure for the two registries the ledger covered
# only by NON-EMPTINESS.
#
# ``test_registry_tables_are_covered_by_a_ledger_entry`` asserted that
# ``STRUCTURAL_ARG_POSITIONS`` and ``CUSTOM_EXEMPTION_CHECKS`` are non-empty and
# that the code they settle under is ledgered. Both facts survive ANY membership
# edit: planting ``STRUCTURAL_ARG_POSITIONS["take_along_dim"] = {1}`` (a genuine
# index-VALUE dependency), ``["where"] = {0}`` (the condition mask), or an
# unconditional ``return True`` custom check left the whole suite green -- the
# protection was a name list that re-catches yesterday's mistake and misses
# tomorrow's.
#
# The two sub-ledgers below close them per ENTRY, on the pattern
# ``SKIP_PERTURBATION_JUSTIFICATIONS`` already uses: name -> justification +
# contract clause + the case the entry must still let FAIL, pinned against the
# live registry by KEY and by VALUE (exact position sets / exact bound check
# function). A new entry, a widened position set, or a rebound check is RED
# until its audit record is written.
#
# Each row also carries a ``proof_kind``, which is the honest ground the entry
# stands on:
#
# ``value_irrelevance_proved``
#     The parent's VALUE provably cannot reach the output (destination fully
#     overwritten, dtype/device/shape template) -- squarely outside C2.
# ``domain_forced``
#     A value dependency with NO admissible alternate value (degenerate index
#     domain): irrelevance is forced by the domain, not assumed.
# ``vehicle_limited``
#     The parent IS value-sensitive and the exemption is a PERTURBATION-VEHICLE
#     limitation, not a proof: no in-domain alternative can be synthesized
#     without crashing the kernel or changing the output shape. These rows are
#     the weak ground in the table; enumerating them is the point -- the class
#     is now bounded and auditable, and a new member cannot join it silently.
#     Each such row must name the check that still guards the value edge.
# ---------------------------------------------------------------------------

_PROOF_KINDS = frozenset({"value_irrelevance_proved", "domain_forced", "vehicle_limited"})


@dataclass(frozen=True)
class StructuralPositionExemption:
    """Audit record for one ``STRUCTURAL_ARG_POSITIONS`` row.

    Parameters
    ----------
    func_name:
        Captured function name, exactly as the registry keys it.
    positions:
        Argument positions the registry declares structural. Pinned EXACTLY
        against the live set, so widening a row is red.
    contract:
        Contract clause from the module docstring the exemption sits outside.
    proof_kind:
        Ground the entry stands on (see the section comment).
    justification:
        Why the declared positions cannot carry output values -- or, for
        ``vehicle_limited``, why no in-domain perturbation can be built.
    refuses:
        The case this row must still let FAIL.
    """

    func_name: str
    positions: frozenset[int]
    contract: str
    proof_kind: str
    justification: str
    refuses: str


@dataclass(frozen=True)
class CustomCheckExemption:
    """Audit record for one ``CUSTOM_EXEMPTION_CHECKS`` row.

    Parameters
    ----------
    func_name:
        Captured function name, exactly as the registry keys it.
    check:
        Attribute name in ``torchlens.validation.exemptions`` of the check the
        registry must be bound to. Pinned by IDENTITY, so rebinding an op to a
        laxer predicate is red.
    contract:
        Contract clause the exemption sits outside.
    proof_kind:
        Ground the entry stands on (see the section comment).
    justification:
        What the check proves from the saved call.
    refuses:
        The case the check must still let FAIL.
    """

    func_name: str
    check: str
    contract: str
    proof_kind: str
    justification: str
    refuses: str


_C2 = "C2 perturbation sensitivity"

_OVERWRITTEN_DESTINATION_REFUSES = (
    "the SOURCE argument, which carries every output value, and any parent "
    "recorded at any other position on the call"
)
_FACTORY_TEMPLATE_REFUSES = (
    "the size/fill/data arguments that actually determine the output values, "
    "and the same tensor appearing at any non-factory position"
)

#: Per-entry audit records for ``STRUCTURAL_ARG_POSITIONS``.
STRUCTURAL_POSITION_LEDGER: tuple[StructuralPositionExemption, ...] = (
    StructuralPositionExemption(
        func_name="new",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "value-bearing Tensor.new(tensor)/new(data): the arg-0 SELF tensor supplies "
            "dtype and device only, per torch's legacy constructor contract -- its "
            "values never reach the output; the size-only overloads are exempted "
            "earlier by uninitialized_by_design_applies and never reach perturbation "
            "(R08-1 narrowing companion)"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="copy_",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "torch's copy_ contract overwrites the arg-0 destination in full from the "
            "arg-1 source, so the destination's prior values cannot reach the output"
        ),
        refuses=_OVERWRITTEN_DESTINATION_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="_foreach_copy_",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "zipped spelling of copy_: each destination member (matched per zipped slot "
            "(0, j)) is totally overwritten by its source member, so no member's prior "
            "values reach the output"
        ),
        refuses=_OVERWRITTEN_DESTINATION_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="foreachcopy",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized TorchLens spelling of _foreach_copy_; identical "
            "totally-overwritten zipped destination argument"
        ),
        refuses=_OVERWRITTEN_DESTINATION_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="fill_",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "the arg-0 destination is overwritten by the scalar/tensor fill VALUE at "
            "arg 1; r31 narrowing moved fill_ out of the whole-op skip precisely so "
            "that fill value stays perturbation-tested"
        ),
        refuses="the fill VALUE at arg 1, which determines every output element",
    ),
    StructuralPositionExemption(
        func_name="expand_as",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "arg 1 is consumed for its SHAPE only; r31 narrowing moved expand_as out of "
            "the whole-op skip because arg 0's values are the output values"
        ),
        refuses="arg 0, whose values are broadcast into the output unchanged",
    ),
    StructuralPositionExemption(
        func_name="expandas",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized TorchLens spelling of expand_as; the arg-1 tensor is a shape "
            "template whose values are never read"
        ),
        refuses="arg 0, whose values are broadcast into the output unchanged",
    ),
    StructuralPositionExemption(
        func_name="reshape_as",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "Tensor.reshape_as(other) is self.reshape(other.sizes()): arg 1 is consumed for its SHAPE only, never its elements"
        ),
        refuses="arg 0, whose values are the output values in a new shape",
    ),
    StructuralPositionExemption(
        func_name="reshapeas",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized TorchLens spelling of reshape_as; the arg-1 tensor is a shape template whose values are never read"
        ),
        refuses="arg 0, whose values are the output values in a new shape",
    ),
    StructuralPositionExemption(
        func_name="view_as",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "Tensor.view_as(other) is self.view(other.size()): arg 1 is consumed for its SHAPE only, never its elements"
        ),
        refuses="arg 0, whose values are the output values in a new shape",
    ),
    StructuralPositionExemption(
        func_name="viewas",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized TorchLens spelling of view_as; the arg-1 tensor is a shape template whose values are never read"
        ),
        refuses="arg 0, whose values are the output values in a new shape",
    ),
    StructuralPositionExemption(
        func_name="type_as",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "arg 1 is consumed for dtype/device only -- Tensor.type_as reads its "
            "template's type, never its elements"
        ),
        refuses="arg 0, the tensor actually being cast, whose values flow to the output",
    ),
    StructuralPositionExemption(
        func_name="new_tensor",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "self at arg 0 is a dtype/device/layout factory template for "
            "Tensor.new_tensor; the output values come from the data argument"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="newtensor",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized Tensor.new_tensor spelling; the self tensor is the same "
            "dtype/device/layout factory template"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="new_full",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "Tensor.new_full reads self only for dtype/device/layout; the output is "
            "determined entirely by the size and fill_value arguments"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="newfull",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized Tensor.new_full spelling; the self tensor is the same "
            "dtype/device/layout factory template"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="new_zeros",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "Tensor.new_zeros reads self only for dtype/device/layout; the output is "
            "constant zero at the requested size"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="newzeros",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized Tensor.new_zeros spelling; the self tensor is the same "
            "dtype/device/layout factory template"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="new_ones",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "Tensor.new_ones reads self only for dtype/device/layout; the output is "
            "constant one at the requested size"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="newones",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized Tensor.new_ones spelling; the self tensor is the same "
            "dtype/device/layout factory template"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="_pack_padded_sequence",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "the arg-1 lengths tensor IS value-sensitive (its entries decide which "
            "timesteps enter the packed data and the batch_sizes output), but the "
            "perturbation vehicle can only step its integer entries, which leaves the "
            "[1, T] / enforce_sorted precondition and aborts inside the native packing "
            "kernel instead of producing a comparable output; forward replay still "
            "reconstructs and re-runs the call from the saved lengths, so a dropped "
            "lengths edge remains visible as a replay mismatch. NARROWING OWED: the "
            "F2-style fix is an in-domain lengths perturbation (a monotone-preserving "
            "shrink), not a position blanket -- tracked as a B1-06 residual finding"
        ),
        refuses=(
            "arg 0, the padded input whose values are packed into the output, and any "
            "lengths tensor reaching the call at another position or keyword"
        ),
    ),
    StructuralPositionExemption(
        func_name="_pad_packed_sequence",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "inverse of _pack_padded_sequence with the same arg-1 lengths descriptor: "
            "stepped lengths leave the admissible domain and abort the native kernel "
            "rather than yielding a comparable output, so no in-domain alternative can "
            "be synthesized by the vehicle; forward replay still re-runs the real call "
            "from the saved lengths. NARROWING OWED with its sibling entry"
        ),
        refuses=(
            "arg 0, the packed data whose values are padded into the output, and any "
            "lengths tensor reaching the call at another position or keyword"
        ),
    ),
    # -- torchvision PyCapsule ops (b1p2 D2 adjudicated NARROWING) ----------
    # Formerly whole-op rows in SKIP_PERTURBATION_ENTIRELY: the skip was wider
    # than its segfault justification. Only the coordinate/offset arg is
    # vehicle-limited (perturbed coordinates index out of bounds inside the
    # native kernel, past Python exception handling); feature and score args
    # returned to strict perturbation.
    StructuralPositionExemption(
        func_name="nms",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "the arg-0 boxes ARE value-sensitive (coordinates decide suppression), but "
            "perturbed coordinates can index out of bounds inside the torchvision "
            "PyCapsule kernel and segfault past Python exception handling, so no safe "
            "in-domain alternative can be synthesized by the vehicle; forward replay "
            "still re-runs the real call from the saved boxes. Scores (arg 1) stay "
            "strictly perturbation-tested (the b1p2 D2 narrowing)"
        ),
        refuses="the arg-1 scores, whose values order and gate every kept box",
    ),
    StructuralPositionExemption(
        func_name="deform_conv2d",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "the arg-1 sampling offsets ARE value-sensitive, but out-of-domain offsets "
            "index outside the feature map inside the native kernel and can segfault, "
            "so the vehicle cannot perturb them safely; forward replay still re-runs "
            "the real call from the saved offsets. Input, weight, bias, and modulation "
            "mask stay strictly perturbation-tested"
        ),
        refuses=(
            "the arg-0 input and the weight/bias/mask arguments, whose values flow "
            "arithmetically into every output element"
        ),
    ),
    StructuralPositionExemption(
        func_name="roi_align",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "the arg-1 boxes ARE value-sensitive (they place the pooling windows), but "
            "perturbed box coordinates can address out-of-bounds feature-map memory in "
            "the PyCapsule kernel; forward replay still re-runs the real call from the "
            "saved boxes. The arg-0 feature map stays strictly perturbation-tested"
        ),
        refuses="the arg-0 feature map, whose values are averaged into every output bin",
    ),
    StructuralPositionExemption(
        func_name="roi_pool",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "same coordinate-safety limitation as roi_align for the arg-1 boxes; the "
            "arg-0 feature map stays strictly perturbation-tested and replay re-runs "
            "the real call from the saved boxes"
        ),
        refuses="the arg-0 feature map, whose values are pooled into every output bin",
    ),
    StructuralPositionExemption(
        func_name="ps_roi_align",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "position-sensitive variant of roi_align with the same arg-1 coordinate "
            "safety limitation; the arg-0 feature map stays strictly tested and replay "
            "re-runs the real call from the saved boxes"
        ),
        refuses="the arg-0 feature map, whose values are averaged into every output bin",
    ),
    StructuralPositionExemption(
        func_name="ps_roi_pool",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "position-sensitive variant of roi_pool with the same arg-1 coordinate "
            "safety limitation; the arg-0 feature map stays strictly tested and replay "
            "re-runs the real call from the saved boxes"
        ),
        refuses="the arg-0 feature map, whose values are pooled into every output bin",
    ),
)

#: Per-entry audit records for ``CUSTOM_EXEMPTION_CHECKS``.
_ZIPPED_SIBLING_JUSTIFICATION = (
    "meshgrid/broadcast_tensors zip N inputs to N outputs: output j carries exactly "
    "input j's values, so a perturbed parent is exempt ONLY when every recorded "
    "position places it at a zipped index other than this output's own "
    "multi_output_index (fail-closed on missing index, unknown position shape, or an "
    "unmapped parent). Replaces the former whole-op skip that also exempted each "
    "output's OWN value edge (R08-2)"
)

_ZIPPED_SIBLING_REFUSES = (
    "the output's own zipped input (index == multi_output_index), any parent whose "
    "recorded position cannot be proven cross-member, and every call on a trace that "
    "did not record multi_output_index"
)

CUSTOM_CHECK_LEDGER: tuple[CustomCheckExemption, ...] = (
    CustomCheckExemption(
        func_name="meshgrid",
        check="_check_zipped_sibling_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=_ZIPPED_SIBLING_JUSTIFICATION,
        refuses=_ZIPPED_SIBLING_REFUSES,
    ),
    CustomCheckExemption(
        func_name="broadcast_tensors",
        check="_check_zipped_sibling_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=_ZIPPED_SIBLING_JUSTIFICATION,
        refuses=_ZIPPED_SIBLING_REFUSES,
    ),
    CustomCheckExemption(
        func_name="broadcasttensors",
        check="_check_zipped_sibling_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            _ZIPPED_SIBLING_JUSTIFICATION
            + "; canonicalized capture spelling -- the old snake_case-keyed skip never "
            "matched it (a silently dead registry row)"
        ),
        refuses=_ZIPPED_SIBLING_REFUSES,
    ),
    CustomCheckExemption(
        func_name="__getitem__",
        check="_check_getitem_exempt",
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "exempts ONLY a parent that occupies no arg-0 slot, i.e. an index/slice "
            "argument of the subscript; keyed on recorded arg POSITION, never on tensor "
            "equality with the index. The indexed data parent (arg 0) stays strictly "
            "perturbed. RESIDUAL: index VALUES do select which data flows out, so this "
            "row is the same class the F2 tightening replaced elsewhere with in-domain "
            "rotation; the value edge stays guarded by forward replay -- tracked as a "
            "B1-06 residual finding"
        ),
        refuses=(
            "the arg-0 data parent, and an index tensor that merely EQUALS the "
            "subscript while being recorded at position 0"
        ),
    ),
    CustomCheckExemption(
        func_name="__setitem__",
        check="_check_setitem_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "exempts a __setitem__ destination only for a structural mask/index slot or "
            "a PROVEN total overwrite of the destination, and the uninitialized-origin "
            "walk deliberately does not chain through prior in-place writes"
        ),
        refuses=(
            "a partial overwrite, a destination holding real written data from an "
            "earlier in-place write (the TwoIndexCopyDim2 hole), and duplicate advanced "
            "indices that would fake total coverage by numel equality"
        ),
    ),
    CustomCheckExemption(
        func_name="index_put",
        check="_check_index_put_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "exempts the index_put destination only when the saved indices provably "
            "cover every destination element, so its prior values cannot survive"
        ),
        refuses=(
            "a partial write, accumulate semantics, and the values argument that "
            "supplies the written data"
        ),
    ),
    CustomCheckExemption(
        func_name="index_put_",
        check="_check_index_put_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "in-place spelling of index_put bound to the same proof: total index "
            "coverage of the destination is required before the destination is excused"
        ),
        refuses=(
            "a partial write, accumulate semantics, and the values argument that "
            "supplies the written data"
        ),
    ),
    CustomCheckExemption(
        func_name="lstm",
        check="_check_lstm_exempt",
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "exempts ONLY the arg-1 (h_0, c_0) initial-state slot, keyed on recorded "
            "arg position and never on equality with a zero-initialized state. "
            "RESIDUAL: the initial state genuinely feeds the recurrence, so this is a "
            "vehicle/position blanket rather than a value-irrelevance proof; the state "
            "edge stays guarded by forward replay, which rebuilds the call from the "
            "saved state -- tracked as a B1-06 residual finding"
        ),
        refuses=(
            "the arg-0 input sequence and the weight arguments, and a data tensor that "
            "merely equals a zero-initialized h0 while sitting at another position"
        ),
    ),
    CustomCheckExemption(
        func_name="interpolate",
        check="_check_interpolate_exempt",
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "exempts ONLY a tensor occupying the scale_factor slot (arg 2 or the "
            "scale_factor keyword), keyed on recorded position. A stepped scale factor "
            "changes the OUTPUT SHAPE rather than producing a comparable output, so the "
            "vehicle cannot express an in-domain alternative; forward replay still "
            "re-runs the real call from the saved scale factor"
        ),
        refuses=(
            "the arg-0 input being resampled, and a scale-factor-valued tensor recorded "
            "at any other position"
        ),
    ),
    CustomCheckExemption(
        func_name="scatter",
        check="_check_scatter_or_index_domain_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "destination is excused only when the saved index provably covers every "
            "slot along the scatter dim (identity-checked against the saved "
            "destination), else the degenerate-index-domain proof must hold"
        ),
        refuses=(
            "a partially-covering index, reduce semantics, and the src argument that "
            "supplies the scattered values"
        ),
    ),
    CustomCheckExemption(
        func_name="scatter_",
        check="_check_scatter_or_index_domain_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "in-place scatter bound to the same total-coverage / degenerate-domain proof as scatter"
        ),
        refuses=(
            "a partially-covering index, reduce semantics, and the src argument that "
            "supplies the scattered values"
        ),
    ),
    CustomCheckExemption(
        func_name="scatter_add",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "scatter_add accumulates, so the destination is never excused; only an "
            "index parent with NO admissible alternate value (domain n<=1, or zero "
            "in-range entries) is exempt"
        ),
        refuses=(
            "any perturbable index domain, which is rotated in-domain instead, and "
            "every non-index parent of the call"
        ),
    ),
    CustomCheckExemption(
        func_name="scatter_add_",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "in-place scatter_add bound to the same degenerate-index-domain proof; "
            "accumulation means the destination stays strictly tested"
        ),
        refuses=(
            "any perturbable index domain, which is rotated in-domain instead, and "
            "every non-index parent of the call"
        ),
    ),
    CustomCheckExemption(
        func_name="scatteradd",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "canonicalized scatter_add spelling bound to the same degenerate-domain "
            "proof, so the canonical and torch spellings cannot drift apart"
        ),
        refuses=(
            "any perturbable index domain, which is rotated in-domain instead, and "
            "every non-index parent of the call"
        ),
    ),
    CustomCheckExemption(
        func_name="embedding",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "the F2 tightening removed embedding's blanket index exemption; the index "
            "parent is now excused only when the vocabulary domain admits no alternate "
            "index at all"
        ),
        refuses=(
            "a perturbable index domain (indices are rotated in-domain by "
            "index_domain_rotation_values) and the weight parent"
        ),
    ),
    CustomCheckExemption(
        func_name="gather",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "gather's index blanket was likewise removed by F2; only a domain with no "
            "admissible alternate index is exempt"
        ),
        refuses=(
            "a perturbable index domain (rotated in-domain) and the arg-0 source "
            "whose values are gathered"
        ),
    ),
    CustomCheckExemption(
        func_name="index_select",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "index_select's index blanket was removed by F2; only a domain with no "
            "admissible alternate index is exempt"
        ),
        refuses=(
            "a perturbable index domain (rotated in-domain) and the arg-0 source "
            "whose values are selected"
        ),
    ),
    CustomCheckExemption(
        func_name="cross_entropy",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "the target blanket was removed by F2; an all-ignore_index target (zero "
            "in-range entries) or a single-class domain admits no alternate target, "
            "which is what this row excuses"
        ),
        refuses=(
            "a perturbable target domain (rotated in-domain with out-of-range "
            "sentinels preserved) and the logits parent"
        ),
    ),
    CustomCheckExemption(
        func_name="where",
        check="_check_where_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "where parents are excused only when the SAVED condition proves the branch "
            "is never taken at any output element (or the one-arg index form applies); "
            "the condition itself is not blanket-exempt"
        ),
        refuses=(
            "a branch the saved condition selects anywhere, and a condition mask whose "
            "perturbation can flip a selected element"
        ),
    ),
    CustomCheckExemption(
        func_name="maskedfill",
        check="_check_masked_fill_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized masked_fill spelling bound to the saved-mask proof: input "
            "excused only when the mask is true everywhere, fill value only when it is "
            "false everywhere, mask only when input already equals the fill value"
        ),
        refuses=(
            "any mixed saved mask, where each parent still carries output values, and "
            "the mask itself unless input == value at every broadcast position"
        ),
    ),
    CustomCheckExemption(
        func_name="masked_fill",
        check="_check_masked_fill_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "the F2 tightening removed masked_fill's structural mask blanket; the mask "
            "is now excused only when the saved input already equals the fill value "
            "everywhere, so every mask yields the same output"
        ),
        refuses=(
            "any mixed saved mask, where each parent still carries output values, and "
            "the mask itself unless input == value at every broadcast position"
        ),
    ),
    CustomCheckExemption(
        func_name="masked_fill_",
        check="_check_masked_fill_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "in-place masked_fill bound to the same saved-value proof as the out-of-place spellings"
        ),
        refuses=(
            "any mixed saved mask, where each parent still carries output values, and "
            "the mask itself unless input == value at every broadcast position"
        ),
    ),
    CustomCheckExemption(
        func_name="batch_norm",
        check="_check_norm_running_stat_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "EITHER (a) in TRAINING mode (proved from saved arg 5 being exactly True) "
            "running_mean/running_var at args 3-4 are update TARGETS: the normalized "
            "output is computed from batch statistics, not from the running buffers; "
            "OR (b) the perturbed parent(s) are a subset of {input, running_mean, "
            "running_var} (args 0/3/4) and the saved weight operand (arg 1, the real "
            "ATen position) is proved all-zero: weight=0 annihilates the whole "
            "normalized term (output = ((input-running_mean)/sqrt(running_var+eps))*"
            "weight+bias), in ANY mode -- timm's zero_init_last convention zero-inits "
            "exactly this operand for the last BatchNorm in many ResNet-family blocks"
        ),
        refuses=(
            "eval-mode running_mean/running_var perturbation when weight is nonzero, "
            "where the running stats DO determine the output and stay strictly tested; "
            "perturbing weight or bias themselves, even when weight is zero (weight "
            "moving off zero, or bias directly, both genuinely change the output); "
            "and every parent outside args 0/3/4"
        ),
    ),
    CustomCheckExemption(
        func_name="instance_norm",
        check="_check_norm_running_stat_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "same running-stat update-target and zero-weight annihilator proofs as "
            "batch_norm, bound to the same predicate so the two cannot drift apart"
        ),
        refuses=(
            "eval-mode running_mean/running_var perturbation when weight is nonzero, "
            "where the running stats DO determine the output and stay strictly tested; "
            "perturbing weight or bias themselves, even when weight is zero; and every "
            "parent outside args 0/3/4"
        ),
    ),
)

#: Kwarg spellings that may match a structural POSITION. Every alias widens the
#: reach of a ``STRUCTURAL_ARG_POSITIONS`` row, so it is audited against that
#: row: an alias for an undeclared func or an undeclared position is red.
STRUCTURAL_KWARG_ALIAS_LEDGER: dict[str, dict[int, frozenset[str]]] = {
    "_pack_padded_sequence": {1: frozenset({"lengths"})},
    "_pad_packed_sequence": {1: frozenset({"lengths"})},
    "type_as": {1: frozenset({"tensor", "other"})},
    "reshape_as": {1: frozenset({"other"})},
    "reshapeas": {1: frozenset({"other"})},
    "view_as": {1: frozenset({"other"})},
    "viewas": {1: frozenset({"other"})},
    # torchvision coordinate args (b1p2 D2 narrowing): keyword spellings of
    # the SAME audited positions above, nothing wider.
    "nms": {0: frozenset({"boxes"})},
    "deform_conv2d": {1: frozenset({"offset"})},
    "roi_align": {1: frozenset({"boxes", "rois"})},
    "roi_pool": {1: frozenset({"boxes", "rois"})},
    "ps_roi_align": {1: frozenset({"boxes", "rois"})},
    "ps_roi_pool": {1: frozenset({"boxes", "rois"})},
}


def structural_position_violations(
    live: dict[str, set[int]],
    ledger: tuple[StructuralPositionExemption, ...],
) -> list[str]:
    """Return audit violations between a structural-position registry and its ledger.

    Parameters
    ----------
    live:
        Registry mapping func name -> structural arg positions.
    ledger:
        Audit records to hold it to.

    Returns
    -------
    list[str]
        One message per unledgered entry, phantom record, or position-set
        mismatch. Empty means the registry is fully audited.
    """

    ledgered = {entry.func_name: entry for entry in ledger}
    violations = [
        f"{name}: structural exemption with no audit record (positions {sorted(positions)})"
        for name, positions in sorted(live.items())
        if name not in ledgered
    ]
    violations.extend(
        f"{name}: audit record for an exemption the registry no longer declares"
        for name in sorted(set(ledgered) - set(live))
    )
    violations.extend(
        f"{name}: registry declares positions {sorted(live[name])} but the audit record "
        f"justifies {sorted(ledgered[name].positions)}"
        for name in sorted(set(live) & set(ledgered))
        if frozenset(live[name]) != ledgered[name].positions
    )
    return violations


def custom_check_violations(
    live: dict[str, object],
    ledger: tuple[CustomCheckExemption, ...],
    module: object,
) -> list[str]:
    """Return audit violations between the custom-check registry and its ledger.

    Parameters
    ----------
    live:
        Registry mapping func name -> bound check callable.
    ledger:
        Audit records to hold it to.
    module:
        Module the cited check symbols must resolve in.

    Returns
    -------
    list[str]
        One message per unledgered entry, phantom record, unresolvable citation,
        or check rebound to a different predicate.
    """

    ledgered = {entry.func_name: entry for entry in ledger}
    violations = [
        f"{name}: custom exemption check with no audit record"
        for name in sorted(set(live) - set(ledgered))
    ]
    violations.extend(
        f"{name}: audit record for a custom check the registry no longer declares"
        for name in sorted(set(ledgered) - set(live))
    )
    for name in sorted(set(live) & set(ledgered)):
        cited = getattr(module, ledgered[name].check, None)
        if cited is None:
            violations.append(f"{name}: audit record cites missing check {ledgered[name].check!r}")
        elif cited is not live[name]:
            violations.append(
                f"{name}: registry is bound to {getattr(live[name], '__name__', live[name])!r} "
                f"but the audit record justifies {ledgered[name].check!r}"
            )
    return violations


def returns_true_unconditionally(source: str) -> bool:
    """Return whether a function's source returns True with no condition at all.

    The planted third defect of finding B1-06 was a custom check whose body is
    ``return True``: a blanket wearing a predicate's signature. A top-level
    (non-branched) ``return True`` is exactly that shape, so it is refused
    structurally rather than by reviewer attention.

    Parameters
    ----------
    source:
        Source text of a single function definition.

    Returns
    -------
    bool
        True when a ``return True`` sits at the function body's top level.
    """

    tree = ast.parse(textwrap.dedent(source))
    definition = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if definition is None:
        return False
    return any(
        isinstance(statement, ast.Return)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is True
        for statement in definition.body
    )


def test_structural_arg_positions_have_a_per_entry_audit_record() -> None:
    """Every structural-position row is justified, by name AND by position set."""

    violations = structural_position_violations(
        dict(ex.STRUCTURAL_ARG_POSITIONS), STRUCTURAL_POSITION_LEDGER
    )
    assert not violations, (
        "STRUCTURAL_ARG_POSITIONS is not fully audited -- each row needs the contract "
        "clause it sits outside, the ground it stands on, and the case it must still "
        "let fail:\n" + "\n".join(violations)
    )


def test_custom_exemption_checks_have_a_per_entry_audit_record() -> None:
    """Every custom-check row is justified and bound to the predicate it cites."""

    violations = custom_check_violations(dict(ex.CUSTOM_EXEMPTION_CHECKS), CUSTOM_CHECK_LEDGER, ex)
    assert not violations, (
        "CUSTOM_EXEMPTION_CHECKS is not fully audited -- a new op or a rebound check "
        "needs its own audit record:\n" + "\n".join(violations)
    )


@pytest.mark.parametrize(
    "entry",
    STRUCTURAL_POSITION_LEDGER + CUSTOM_CHECK_LEDGER,
    ids=lambda entry: entry.func_name,
)
def test_registry_audit_record_is_well_formed(
    entry: StructuralPositionExemption | CustomCheckExemption,
) -> None:
    """Each per-entry record cites a known clause, a ground, and a refusal."""

    assert entry.contract in _CONTRACTS, f"{entry.func_name}: uncited contract"
    assert entry.proof_kind in _PROOF_KINDS, f"{entry.func_name}: unknown proof kind"
    assert len(entry.justification.strip()) >= 60, (
        f"{entry.func_name}: the justification must prove value-irrelevance from the "
        "call (or name the vehicle limit), not restate the op's name"
    )
    assert len(entry.refuses.strip()) >= 30, (
        f"{entry.func_name}: 'refuses' must name the case this entry still lets fail; "
        "an entry with nothing it refuses is a blanket, not a carve-out"
    )
    if entry.proof_kind == "vehicle_limited":
        assert "replay" in entry.justification.lower(), (
            f"{entry.func_name}: a vehicle-limited exemption is NOT a proof, so it must "
            "name the check that still guards the value edge"
        )


def test_structural_kwarg_aliases_are_declared_by_an_audited_position() -> None:
    """A kwarg alias can only widen a position its own audit record justifies."""

    ledgered = {entry.func_name: entry for entry in STRUCTURAL_POSITION_LEDGER}
    live_aliases = {
        name: {position: frozenset(aliases) for position, aliases in by_position.items()}
        for name, by_position in ex.STRUCTURAL_ARG_KWARG_ALIASES.items()
    }
    assert live_aliases == STRUCTURAL_KWARG_ALIAS_LEDGER, (
        "STRUCTURAL_ARG_KWARG_ALIASES changed without an audit record; a kwarg alias "
        "extends a structural exemption to a keyword spelling"
    )
    for name, by_position in live_aliases.items():
        assert name in ledgered, f"{name}: kwarg alias for an unaudited structural func"
        for position in by_position:
            assert position in ledgered[name].positions, (
                f"{name}: kwarg alias widens position {position}, which the audit "
                "record does not justify as structural"
            )


def test_no_custom_exemption_check_is_an_unconditional_blanket() -> None:
    """No registered check can excuse a parent with no condition at all."""

    import inspect

    for func_name, check in ex.CUSTOM_EXEMPTION_CHECKS.items():
        assert getattr(check, "__name__", "<lambda>") != "<lambda>", (
            f"{func_name}: custom checks must be named module-level predicates so the "
            "audit record can cite and pin them"
        )
        assert not returns_true_unconditionally(inspect.getsource(check)), (
            f"{func_name}: custom check returns True with no condition -- that is a "
            "blanket exemption wearing a predicate's signature"
        )


def test_inline_pre_perturbation_predicates_still_exist() -> None:
    """The two non-registry sources of ``pre_perturbation_exemption`` resolve.

    ``_check_perturbation_exemptions`` also excuses empty-tensor parents and
    pure ``out=`` destinations inline. Neither is a registry, so neither can
    grow an unaudited ENTRY, but the citations must not rot.
    """

    from torchlens.validation import core as validation_core

    assert hasattr(validation_core, "_perturbed_parents_only_occupy_out_kwarg")
    assert hasattr(validation_core, "_check_perturbation_exemptions")


class TestRegistryLedgerMechanismIsRedCapable:
    """Replay finding B1-06's three plants and prove each one is now reported."""

    def test_planting_a_value_dependent_structural_position_is_reported(self) -> None:
        """``take_along_dim`` arg 1 is the index tensor -- an unaudited entry."""

        planted = dict(ex.STRUCTURAL_ARG_POSITIONS)
        planted["take_along_dim"] = {1}
        violations = structural_position_violations(planted, STRUCTURAL_POSITION_LEDGER)
        assert any("take_along_dim" in violation for violation in violations)

    def test_planting_the_where_condition_mask_is_reported(self) -> None:
        """``where`` arg 0 is the condition mask -- an unaudited entry."""

        planted = dict(ex.STRUCTURAL_ARG_POSITIONS)
        planted["where"] = {0}
        violations = structural_position_violations(planted, STRUCTURAL_POSITION_LEDGER)
        assert any("where" in violation for violation in violations)

    def test_widening_an_audited_position_set_is_reported(self) -> None:
        """Adding the SOURCE argument to ``copy_`` is a widening, not a rename."""

        planted = dict(ex.STRUCTURAL_ARG_POSITIONS)
        planted["copy_"] = {0, 1}
        violations = structural_position_violations(planted, STRUCTURAL_POSITION_LEDGER)
        assert any("copy_" in violation for violation in violations)

    def test_removing_a_registry_entry_is_reported_as_a_phantom(self) -> None:
        """A narrowing must also update the audit record, in the same change."""

        planted = dict(ex.STRUCTURAL_ARG_POSITIONS)
        planted.pop("type_as")
        violations = structural_position_violations(planted, STRUCTURAL_POSITION_LEDGER)
        assert any("type_as" in violation for violation in violations)

    def test_planting_an_unledgered_custom_check_is_reported(self) -> None:
        """A new op in the custom registry needs its own audit record."""

        planted = dict(ex.CUSTOM_EXEMPTION_CHECKS)
        planted["take_along_dim"] = ex._check_index_domain_degenerate
        violations = custom_check_violations(planted, CUSTOM_CHECK_LEDGER, ex)
        assert any("take_along_dim" in violation for violation in violations)

    def test_rebinding_an_audited_check_to_another_predicate_is_reported(self) -> None:
        """Swapping ``masked_fill``'s proof for a laxer one is reported."""

        planted = dict(ex.CUSTOM_EXEMPTION_CHECKS)
        planted["masked_fill"] = ex._check_getitem_exempt
        violations = custom_check_violations(planted, CUSTOM_CHECK_LEDGER, ex)
        assert any("masked_fill" in violation for violation in violations)

    def test_unconditional_return_true_check_is_detected(self) -> None:
        """The planted blanket shape is caught structurally."""

        assert returns_true_unconditionally(
            "def blanket(trace, op, layers_to_perturb):\n    return True\n"
        )

    def test_a_real_branching_check_is_not_flagged_as_a_blanket(self) -> None:
        """A predicate that can return False is not a blanket."""

        assert not returns_true_unconditionally(
            "def real(trace, op, layers):\n"
            "    if not layers:\n"
            "        return False\n"
            "    if layers == ['x']:\n"
            "        return True\n"
            "    return False\n"
        )
