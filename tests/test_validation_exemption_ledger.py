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
from dataclasses import dataclass
from pathlib import Path

import pytest
from _exemption_contracts import VALIDATION_CONTRACTS
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


_CONTRACTS = VALIDATION_CONTRACTS

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
            "a float dividend, a divisor other than +-1 (compared after widening, so a "
            "uint8 255 is not -1), -1 when any operand or the result is unsigned (it "
            "wraps to the dtype max), a floating result or computation dtype (the "
            "operands' promoted dtype, checked apart from any out= buffer) that cannot "
            "hold the dividend dtype's range (float16), a nonzero saved output, and any perturbed parent "
            "at the divisor slot"
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
    ``STRUCTURAL_POSITION_LEDGER`` / ``CUSTOM_CHECK_LEDGER`` in
    ``test_validation_registry_ledger.py`` (finding B1-06).
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
