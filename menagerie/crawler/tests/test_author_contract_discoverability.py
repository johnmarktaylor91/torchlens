"""Every author-fatal rule must be stated where the author actually reads.

The agentic authoring stage runs exactly ONCE per model across the whole roster.
A rule that is enforced but not communicated therefore does not produce a retry;
it produces a permanently dead record. That makes discoverability a correctness
property of the contract, not a documentation nicety, and it is one nothing
guarded before this module existed.

The 2026-08-03 ten-model diagnostic rung is the evidence. Three of ten models
died on rules that appear nowhere an author reads:

* ``m3671``/``m538``/``m5888`` -- ``R1_LIBRARY must explicitly disable pretrained
  fields``. The schema description for the field carrying that obligation read,
  in full, "Mandatory pretrained disable fields."
* ``m8189`` -- ``staged function 'build_model' must be fully typed``. The canonical
  author prompt's only mention of the symbol showed the UNANNOTATED signature
  ``build_model()``, which is exactly the form the validator refuses; the schema
  said "Mandatory builder symbol."
* ``m9617`` -- a ``BLOCKED`` arm whose thirteen-excerpt pack contained three rows
  citing a source outside the frozen manifest. Terminal evidence resolved
  all-or-nothing, so those three discarded the ten that verified -- including the
  one supporting the blocked predicate -- and the model terminalized as
  disposition-unverifiable. The all-or-nothing amplification was written down
  nowhere.

Two of those three rules turned out to be UNSATISFIABLE rather than merely
undocumented, and were repaired in ``test_unsatisfiable_gate_repair``: a
constructor with no pretrained-capable keyword now asserts absence instead of
dying, and verbatim vendored upstream is exempt from the annotation rule on proof
of provenance. Terminal evidence now resolves per record. Documenting an
unsatisfiable rule perfectly still kills every model of that shape, so this
module's job did not end -- it changed to keeping the CURRENT rules discoverable,
including the two spellings and the one exemption a correct author must know
about.

Each case below pairs the LIVE enforcement site with the surfaces the author is
directed to read, so silencing the guidance fails here even though the rule keeps
working. The surfaces are the ones the pilot run demonstrably reached: the
canonical prompt named by ``request.prompt.path``, the executor stage brief, and
the registered schemas that ``stage2_author.md`` calls authoritative.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.author_executor import _REPAIR_STANDING_RULE
from menagerie.crawler.metadata import AVAILABILITY_BASES

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_DIR = _CRAWLER_ROOT / "schemas"
_AUTHOR_PROMPT = _CRAWLER_ROOT / "prompts" / "claude_crawler_author_v2.txt"
_STAGE2_PROMPT = _CRAWLER_ROOT / "prompts" / "executor" / "stage2_author.md"


def _schema(name: str) -> Mapping[str, Any]:
    """Return one registered schema document.

    Parameters
    ----------
    name:
        Schema filename under the registered schema directory.

    Returns
    -------
    Mapping[str, Any]
        Parsed schema document.
    """

    return json.loads((_SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _description(document: Mapping[str, Any], *path: str) -> str:
    """Return the leaf description a reading author would see at ``path``.

    Parameters
    ----------
    document:
        Parsed schema document.
    path:
        Successive mapping keys down to the leaf owning the description.

    Returns
    -------
    str
        Exact leaf ``description`` text.
    """

    node: Any = document
    for key in path:
        node = node[key]
    description = node["description"]
    assert isinstance(description, str)
    return description


def _implementation_property(name: str) -> Mapping[str, Any]:
    """Return one ``implementation`` property node from the shared model schema."""

    document = _schema("model-common.schema.json")
    return document["$defs"]["implementation"]["properties"][name]


@pytest.mark.smoke
def test_the_pretrained_disable_rule_is_stated_in_the_registered_schema() -> None:
    """``pretrained_disable_fields`` documents every clause that can refuse it.

    ``proposal._validate_source_ladder`` requires a positive disposition, and
    ``recipe.validate_pretrained_disposition`` then requires each named field to
    be a ``kwargs`` key carrying a disabling value, refuses a known pretrained
    keyword left enabled, and -- at load, against the real signature -- refuses an
    enabling default nobody overrode. An author whose constructor exposes no such
    keyword once emitted the honest ``[]`` and died; the alternative spelling that
    now saves it is worthless if it is not written where the author reads.
    """

    assert "library_recipe" in json.dumps(_implementation_property("library_recipe"))
    leaf = _description(
        _schema("model-common.schema.json"),
        "$defs",
        "library_recipe",
        "properties",
        "pretrained_disable_fields",
    )
    lowered = leaf.lower()
    assert "empty" in lowered, "the empty-array rule must be stated, not just enforced"
    assert "kwargs" in lowered, "the kwargs pairing requirement must be stated"
    assert "signature" in lowered, "the constructor-signature check must be stated"
    assert (
        "pretrained_fields_absent" in lowered
    ), "the no-such-keyword spelling must be named where the obligation is stated"
    assert "default" in lowered, "the enabling-default refusal must be stated"
    assert "unlisted" in lowered, "the unlisted-known-key refusal must be stated"

    absent = _description(
        _schema("model-common.schema.json"),
        "$defs",
        "library_recipe",
        "properties",
        "pretrained_fields_absent",
    ).lower()
    assert "checked" in absent, "the assertion must be documented as checked"
    assert "signature" in absent, "the assertion must be documented as verified, not believed"
    assert "cannot be combined" in absent, "the contradiction refusal must be stated"
    assert "contradiction" in absent, "the contradiction refusal must be named as one"


@pytest.mark.smoke
def test_the_pretrained_disable_rule_is_stated_in_the_canonical_prompt() -> None:
    """The prompt names the mechanism, not only the intent.

    The prompt already said "Explicitly disable every pretrained, weights, and
    checkpoint flag" -- an intent an author with no such flag satisfies vacuously.
    It must also name the field that carries the obligation and the fact that an
    empty array is refused.
    """

    text = _AUTHOR_PROMPT.read_text(encoding="utf-8")
    assert "pretrained_disable_fields" in text
    assert "pretrained_fields_absent" in text
    assert "empty" in text.lower()


@pytest.mark.smoke
def test_the_full_annotation_rule_is_stated_in_the_registered_schema() -> None:
    """``code_path`` and both staged symbols document the AST typing check.

    ``proposal._validate_typed_functions`` runs over the recursive model-local
    import closure, so the rule's SCOPE matters as much as its existence -- and
    the scope now has an exemption. An author must be able to read what the
    exemption costs (declare the file, stage it byte-exact) and what it does NOT
    buy (nothing about eval/exec or writes), before spending its one attempt.
    """

    code_path = _implementation_property("code_path")["description"].lower()
    assert "annotat" in code_path, "the annotation requirement must be stated"
    assert "closure" in code_path, "the closure scope must be stated"
    assert "return annotation" in code_path
    assert "vendor" in code_path, "vendored upstream members must be named"
    assert "upstream_files" in code_path, "the exemption's declaration channel must be named"
    assert "content_sha256" in code_path, "the exemption's digest proof must be stated"
    assert "never exempt" in code_path, "the entry point's exclusion must be stated"

    manifest = _implementation_property("code_manifest")["description"].lower()
    assert "annotation" in manifest
    assert "exempt" in manifest

    upstream = _implementation_property("upstream_files")["description"].lower()
    assert "exempt" in upstream, "upstream_files must say what declaring a file there does"

    builder = _implementation_property("builder_symbol")["description"]
    assert "build_model() ->" in builder, "the annotated signature must be shown"
    dummy = _implementation_property("dummy_call_symbol")["description"]
    assert "make_dummy_call(seed: int" in dummy, "the annotated signature must be shown"


@pytest.mark.smoke
def test_the_canonical_prompt_shows_annotated_staged_signatures() -> None:
    """The prompt must not teach the exact form the validator refuses.

    ``m8189`` wrote ``def build_model():`` because the prompt showed
    ``build_model()``. Showing a bare signature beside a rule that rejects bare
    signatures is worse than silence, so the annotated forms must appear.
    """

    text = _AUTHOR_PROMPT.read_text(encoding="utf-8")
    assert "build_model() -> torch.nn.Module:" in text
    assert "make_dummy_call(seed: int, device: str)" in text
    lowered = text.lower()
    assert "annotation" in lowered
    assert "closure" in lowered
    assert "exempt" in lowered, "the vendored-verbatim exemption must be stated"
    assert "byte" in lowered, "the exemption's byte-fidelity condition must be stated"


@pytest.mark.smoke
def test_terminal_evidence_resolution_is_documented_as_per_record() -> None:
    """The pack's per-record settlement is stated on both surfaces.

    ``m9617`` was killed by all-or-nothing resolution AND by the fact that nothing
    warned it. The rule changed; the documentation obligation did not. An author
    reading the old warning would still cite defensively little, which is the
    behaviour the repair exists to stop -- so both the result schema and the stage
    brief must now say that a groundable citation is never worse than silence, and
    that an ungroundable one is named as a gap rather than laundered.
    """

    document = _schema("author-result-v4.schema.json")
    records = _description(document, "$defs", "evidence_records").lower()
    assert "per record" in records
    assert "partially-grounded" in records, "the third resolution must be named"
    assert "source_manifest" in records, "the frozen-manifest restriction must be named"
    assert "unresolved_evidence_ids" in records, "the named-gap channel must be stated"
    assert "never worse than omitting it" in records
    assert "gap" in records, "gaps must be documented as named"
    assert "predicate" in records, "the predicate-grounding floor must be stated"
    assert "never presented as evidence" in records
    assert "all-or-nothing" not in records.replace("not all-or-nothing", "")

    for arm in ("defer_payload", "skip_payload", "blocked_payload"):
        ids = _description(document, "$defs", arm, "properties", "evidence_ids").lower()
        assert "per record" in ids, f"{arm}.evidence_ids must carry the correction"
        assert "partially-grounded" in ids
        assert "gap" in ids, f"{arm}.evidence_ids must state that failures become gaps"

    stage2 = _STAGE2_PROMPT.read_text(encoding="utf-8").lower()
    assert "resolves per record" in stage2
    assert "partially-grounded" in stage2
    assert "named unresolved gap" in stage2
    assert "blocked-prerequisite" in stage2


def _v3_input_contract_property(name: str) -> Mapping[str, Any]:
    """Return one ``input_contract`` property node from the v3 model schema."""

    document = _schema("model-v3.schema.json")
    return document["$defs"]["input_contract"]["properties"][name]


@pytest.mark.smoke
def test_the_builder_symbol_spelling_is_stated_in_the_registered_schema() -> None:
    """``input_contract.builder_symbol`` documents its grammar AND its derivation.

    ``proposal._validate_author_read_grants`` refuses anything outside
    ``identifier('.'identifier)*``. The schema said, in full, "Mandatory builder
    symbol." Seven of nine pilot authors independently landed on
    ``library_recipe.module + '.' + library_recipe.symbol``; the two that did not
    wrote ``timm.models.dla:dla60`` -- the ordinary entry-point spelling of the
    same fact -- and the placeholder ``declarative-library-recipe``, and both died
    permanently. A convention seven of nine agents infer is a real convention, and
    the two that guess a different spelling are owed the rule, not a dead record.
    """

    builder = _v3_input_contract_property("builder_symbol")["description"]
    assert "dotted" in builder.lower(), "the grammar must be named, not just enforced"
    assert "identifier('.'identifier)*" in builder, "the exact accepted grammar must be shown"
    assert ":" in builder and "dla60" in builder, "the refused colon spelling must be shown"
    assert "library_recipe.module" in builder, "the declarative derivation must be stated"
    assert "library_recipe.symbol" in builder
    assert "build_model" in builder, "the staged-rung spelling must be stated"
    assert "placeholder" in builder.lower(), "a placeholder must be named as refused"


@pytest.mark.smoke
def test_the_exact_source_echo_rule_is_stated_in_the_registered_schema() -> None:
    """``source_resolution.sources`` documents that the echo is exact, both ways.

    ``artifact_transactions._validate_context_result`` requires set EQUALITY
    against the frozen manifest plus any supplementary pack. The schema said
    "Mandatory resolved public sources.", which reads like a bibliography. ``m9617``
    read it that way, cited the eight sources it used out of the fifteen we fetched,
    and died on a rule stated nowhere.
    """

    sources = _description(
        _schema("model-common.schema.json"),
        "$defs",
        "source_resolution",
        "properties",
        "sources",
    )
    lowered = sources.lower()
    assert "exactly" in lowered, "the exact-set rule must be stated"
    assert "supplementary" in lowered, "the supplementary half must be named as citable"
    assert "dropped" in lowered, "omitting an unused source must be named as refused"
    assert "verbatim" in lowered, "the per-row verbatim fields must be named"
    assert "source sets differ" in lowered, "the live refusal wording must be quoted"


@pytest.mark.smoke
def test_the_availability_basis_vocabulary_is_declared_where_the_author_reads() -> None:
    """The closed ``basis`` set is an enum in the schema and named in the prompt.

    The 2026-08-05 twenty-model rung lost two models here, both on their first and
    only attempt. ``m8245`` wrote ``bounded-source-read`` and ``m9617`` wrote
    ``bounded-frozen-source-read`` -- honest descriptions of what each had actually
    done -- for ``country`` and ``institution`` absences that were otherwise correct
    and complete. Every other availability record across all 39 archived proposals
    used the canonical ``search-exhausted`` (57 of 67 rows), so the vocabulary was
    reachable; it simply was not DECLARED. ``status`` sat next to it as a real
    ``enum`` and has never lost a model.

    The schema read "Mandatory closed-vocabulary basis for the disposition." -- it
    told the author the set was closed and then withheld the members, which is the
    same shape as the original ``ungrounded claim categories`` wall: strings that
    exist only in Python. Both surfaces must now carry them.
    """

    for schema_name in ("author-proposal-v3.schema.json", "model-v3.schema.json"):
        basis = _schema(schema_name)["$defs"]["availability_claim"]["properties"]["basis"]
        assert "enum" in basis, (
            f"{schema_name} must DECLARE the basis members; a nonempty_string that only "
            "promises a closed vocabulary is what killed m8245 and m9617"
        )
        assert set(basis["enum"]) == set(AVAILABILITY_BASES)
        assert "search-exhausted" in basis["description"], (
            "the description must name the member an absence-by-reading resolves to, "
            "which is the exact substitution both dead models got wrong"
        )

    prompt = _AUTHOR_PROMPT.read_text(encoding="utf-8")
    unwrapped = " ".join(prompt.split())
    assert 'BOTH "status" AND "basis" ARE CLOSED ENUMS' in unwrapped
    assert "bounded-frozen-source-read" in unwrapped, (
        "the prompt must show the exact spelling that killed a model, not a paraphrase"
    )
    assert "bounded-source-read" in unwrapped
    for basis in AVAILABILITY_BASES:
        assert basis in prompt, f"the prompt must state the basis member {basis!r} verbatim"


@pytest.mark.smoke
def test_the_per_author_grounding_rule_is_stated_where_the_author_reads() -> None:
    """``citation.authors`` documents the per-name paper-excerpt requirement.

    The 2026-08-05 twenty-model rung's largest single failure class: four models died
    on ``citation leaves are not grounded verbatim in the fetched paper text:
    ['authors[...]']``, and for three of them -- ``m5273``, ``m538``, ``m5445`` -- the
    full author list sat spelled out in bytes the author had already fetched. Each had
    bound a real paper excerpt (a ``<title>`` element, the abs-page collapse ``by Jian
    Du and 4 other authors``, one author's search-link entry), which is exactly what
    the prompt's "the citation must be grounded on a verbatim excerpt of that page"
    asks for; that EVERY declared author must occur inside such an excerpt existed
    only in Python. The schema description read, in full, "Mandatory authors." Both
    surfaces must now state the rule, name the two summary-line shapes that do not
    satisfy it, and say that documentation-source mentions do not count.
    """

    leaf = _description(
        _schema("model-common.schema.json"),
        "$defs",
        "citation",
        "properties",
        "authors",
    )
    lowered = leaf.lower()
    assert "every listed author" in lowered, "the per-name scope must be stated"
    assert "paper-role" in lowered, "the paper-role restriction must be stated"
    assert "complete author list" in lowered, "the remedy must be stated"
    assert "and N other authors" in leaf, (
        "the abs-page collapse line that killed m538 and m5445 must be shown verbatim"
    )
    assert "documentation source" in lowered, (
        "the ignored-surface trap must be named: m5273's model-doc excerpt named "
        "eleven authors and silently counted for nothing"
    )
    assert "grounds nothing" in lowered, "elsewhere-on-the-page must be named as insufficient"
    assert "\\addauthor" in leaf, (
        "the fused-rendering entailment must be disclosed beside the rule it relaxes, "
        "with the instruction to quote the line as printed"
    )

    prompt = " ".join(_AUTHOR_PROMPT.read_text(encoding="utf-8").split())
    assert "EVERY name in authors[] must occur inside a bound paper-role excerpt" in prompt
    assert "COMPLETE author list" in prompt
    assert "and N other authors" in prompt, (
        "the prompt must show the exact line shape that killed two models"
    )

    proposal_source = (_CRAWLER_ROOT / "proposal.py").read_text(encoding="utf-8")
    assert "not grounded verbatim in the fetched paper text" in proposal_source, (
        "the documented refusal must still be the live one"
    )
    assert "_email_fused_component_grounded" in proposal_source, (
        "the fused-rendering entailment is documented in the schema; if it is gone the "
        "documentation must be revisited rather than left promising a tolerance"
    )


@pytest.mark.smoke
def test_emptying_a_gated_value_on_repair_is_stated_as_half_an_instruction() -> None:
    """Every repair brief carries the rule that a bare removal is refused.

    ``m4334`` and ``m7362`` each received the checker repair "Ground or remove the
    asserted US country value", obeyed it literally on the next attempt, emptied
    ``country``, declared no availability state, and terminalized as permanently dead
    records with ``ungrounded claim categories: ['external_metadata.country']``.
    Neither author was wrong about its evidence; each did exactly what the rejecting
    party asked, and the next gate refused it. ``m7362`` additionally dropped its
    ``website`` and ``input_contract`` tags in the same rewrite, which is why that one
    diagnostic named three claims -- both of which its own attempts 1 and 2 had
    grounded, so neither was a wall.

    The rule is stamped by the machine onto every repair brief rather than requested of
    the checker, because depending on the rejecting party to phrase a repair completely
    is precisely what failed. The checker is asked as well, as a second surface.
    """

    executor_source = (_CRAWLER_ROOT / "author_executor.py").read_text(encoding="utf-8")
    assert "_REPAIR_STANDING_RULE" in executor_source
    assert "lines.extend((\"\", _REPAIR_STANDING_RULE))" in executor_source, (
        "the rule must be appended to the rendered feedback block itself; defining it "
        "without emitting it leaves the trap open"
    )
    rule = " ".join(_REPAIR_STANDING_RULE.split())
    assert "REMOVE, DROP, RETRACT, or NOT ASSERT" in rule
    assert "external_metadata.availability" in rule
    assert "search-exhausted" in rule

    prompt = " ".join(_AUTHOR_PROMPT.read_text(encoding="utf-8").split())
    assert "only half an instruction" in prompt

    checker = " ".join(
        (_CRAWLER_ROOT / "prompts" / "codex_accuracy_checker_v2.txt").read_text(
            encoding="utf-8"
        ).split()
    )
    assert "PHRASE REMOVAL REPAIRS COMPLETELY" in checker
    assert "Ground or remove the asserted country value" in checker, (
        "the checker must be shown the exact phrasing that killed two models"
    )


@pytest.mark.smoke
def test_the_required_check_rule_is_stated_where_the_checker_reads() -> None:
    """``field_check.field`` documents the machine-derived required-check contract.

    Not an author rule, but the same failure shape and the same one-shot cost:
    ``metadata.validate_authored_facts_for_write`` requires exactly one check per
    machine-required gated claim, and the checker is TOLD that set through the
    envelope item's ``required_field_checks`` inventory rather than asked to
    reconstruct it from the proposal. ``m5915`` terminalized on the earlier shape
    of this wall (a SECTION name ``identity``, grouped names like ``citation;
    dates``); the rung-2 census then proved the leaf-granular successor rule was
    unsatisfiable in the other direction -- 10-25 emitted checks against 200+
    demanded leaves that no authored evidence pack could support. Both surfaces
    must carry the closed-inventory rule.
    """

    field = _description(
        _schema("gate-common.schema.json"), "$defs", "field_check", "properties", "field"
    )
    lowered = field.lower()
    assert "required_field_checks" in field, "the machine-derived inventory must be named"
    assert "verbatim" in lowered, "the exact-copy rule must be stated"
    assert "external_metadata.citation" in field, "an exact accepted spelling must be shown"
    assert "proposed_facts." in field, "the tolerated prefix must be stated"
    assert "section" in lowered, "the section-name refusal must be stated"
    assert "'identity'" in field, "the exact spelling that killed m5915 must be shown"
    assert "per-leaf expansion" in lowered, "the leaf-expansion refusal must be stated"
    assert "duplicate" in lowered and "ungated" in lowered, "one-to-one coverage must be stated"

    prompt = (_CRAWLER_ROOT / "prompts" / "codex_accuracy_checker_v2.txt").read_text(
        encoding="utf-8"
    )
    assert "Group only truly identical fields" not in prompt, (
        "the instruction that produced the grouped names must be gone, not merely "
        "contradicted elsewhere in the same prompt"
    )
    unwrapped = " ".join(prompt.split())
    assert "required_field_checks" in prompt, "the checker must be told to read the inventory"
    assert "EXACTLY ONE check per listed string" in unwrapped
    assert "never reconstruct it by inspecting the proposal" in unwrapped


@pytest.mark.smoke
def test_the_guarded_rules_are_still_the_rules_the_code_enforces() -> None:
    """The documented wording still matches live enforcement messages.

    A guard that only reads documentation would keep passing after the rule it
    describes was renamed or removed, leaving prose that misleads the one attempt
    an author gets. These are the exact strings the pilot's quarantined
    diagnostics carried.
    """

    proposal_source = (_CRAWLER_ROOT / "proposal.py").read_text(encoding="utf-8")
    assert "must declare its pretrained disposition" in recipe_or(proposal_source)
    assert "must be fully typed" in proposal_source
    assert "_verbatim_upstream_members" in proposal_source, (
        "the vendored-legibility exemption is documented; if it is gone the documentation "
        "must be revisited rather than left describing a rule that changed"
    )

    recipe_source = (_CRAWLER_ROOT / "recipe.py").read_text(encoding="utf-8")
    assert "is absent from constructor kwargs" in recipe_source
    assert "does not carry a disabling value" in recipe_source
    assert "would resolve pretrained assets" in recipe_source
    assert "contradicted by the pinned constructor signature" in recipe_source

    terminal_source = (_CRAWLER_ROOT / "terminal_evidence.py").read_text(encoding="utf-8")
    assert 'PARTIALLY_GROUNDED = "partially-grounded"' in terminal_source, (
        "terminal evidence is documented as per-record; if this resolution is gone the "
        "documentation must be revisited rather than left describing a rule that changed"
    )
    assert "if len(verified) == len(declared):" in terminal_source, (
        "full grounding must still mean every declared ID, unchanged"
    )
    assert "_any_supports_predicate" in terminal_source, (
        "the predicate-grounding floor is documented; if it is gone the documentation "
        "must be revisited"
    )

    assert "builder_symbol must be a dotted symbol" in proposal_source, (
        "the documented builder-symbol grammar must still be the enforced one"
    )

    binding_source = (_CRAWLER_ROOT / "artifact_transactions.py").read_text(encoding="utf-8")
    assert "proposal and source manifest source sets differ" in binding_source
    assert "citable_rows = manifest_source_rows(source_manifest)" in binding_source, (
        "the documented supplementary-row citability must still come from the widened "
        "reading; reverting it to `sources` alone re-kills a correct proposal"
    )
    assert "source_identity = stable_hash(sources)" in binding_source, (
        "the IDENTITY must still be derived from the frozen rows alone, or the author's "
        "echoed digest stops matching"
    )

    for module, site in (
        ("gates.py", "_manifest_source_ids(source_manifest, include_supplementary=True)"),
        ("terminal_evidence.py", "for source in manifest_source_rows(source_manifest)"),
        ("reducer.py", "self.source_manifest = tuple(manifest_source_rows(manifest))"),
    ):
        text = (_CRAWLER_ROOT / module).read_text(encoding="utf-8")
        assert site in text, (
            f"{module} must keep reading the supplementary half for its COVERAGE question; "
            "narrowing it back to `sources` alone re-kills a terminal arm for citing a "
            "source our own broker fetched for it"
        )
    gates_source = (_CRAWLER_ROOT / "gates.py").read_text(encoding="utf-8")
    assert "source_ids = tuple(sorted(_manifest_source_ids(source_manifest)))" in gates_source, (
        "the BLOCKED arm's DERIVATION must stay on the frozen rows, matching the two "
        "sibling derivations that bind evidence_identity"
    )

    metadata_source = (_CRAWLER_ROOT / "metadata.py").read_text(encoding="utf-8")
    assert "checker gate names a field outside the required gated-claim" in metadata_source, (
        "the extraneous-check refusal is documented; its message must keep naming the "
        "checker gate as the owner rather than reading as an author defect"
    )
    assert "ungated authored facts" in metadata_source, (
        "the exact-coverage refusal must survive the closed-inventory narrowing; a "
        "listed claim without a check still refuses"
    )


def recipe_or(proposal_source: str) -> str:
    """Return the proposal source joined with the recipe source it delegates to.

    The pretrained disposition message moved to ``recipe`` when the two spellings
    were introduced, and ``proposal`` re-raises it. Reading both keeps this guard
    pinned to the live wording wherever it lives.

    Parameters
    ----------
    proposal_source:
        Exact ``proposal.py`` text.

    Returns
    -------
    str
        Concatenated proposal and recipe source text.
    """

    return proposal_source + (_CRAWLER_ROOT / "recipe.py").read_text(encoding="utf-8")


@pytest.mark.smoke
def test_the_wall_exhaustion_spelling_is_commanded_where_the_author_reads() -> None:
    """The stage brief commands the ONE reserved exhaustion spelling, verbatim.

    Rung 7's largest cluster: eight of twenty sessions hit the wall deadline and
    followed the brief's old instruction -- "Emit a valid BLOCKED result ... flows
    through the engine's terminal-disposition gate and can be requeued" -- inventing
    five free-form spellings (``authoring-budget-exhausted``,
    ``author-wall-deadline-reached``, ``authoring-wall-exhausted``, and two
    ``author-wall-deadline-before-*`` forms). Every one parsed as an ordinary
    prerequisite claim, reached the terminal checker, and was correctly rejected or
    found unverifiable: no frozen source can witness a wall clock. Meanwhile the
    honest path existed the whole time: ``_validate_blocked_reason`` refuses the
    reserved spellings as :class:`AuthorEffortExhaustionClaim` and the driver lane
    records ``failed:<stage>`` with a stage-valid effort reason -- requeueable, no
    checker involved. The rung's whole loss was discoverability, so both surfaces
    must now command the exact spelling and the schema must warn beside the field.
    """

    stage2 = _STAGE2_PROMPT.read_text(encoding="utf-8")
    unwrapped = " ".join(stage2.split())
    assert "exactly `wall-exceeded`" in unwrapped, (
        "the brief must command the reserved spelling verbatim, not describe it"
    )
    assert "`stage` naming the stage in flight" in unwrapped
    assert '`["authoring-wall-budget"]`' in unwrapped, (
        "prerequisite_ids is schema-required (minItems 1); the brief must show a "
        "publishable value or the commanded shape is unwritable"
    )
    assert "asks no checker to adjudicate it" in unwrapped, (
        "the brief must say the honest path needs no blocked-prerequisite excerpt, "
        "or an exhausted author will still burn its last minutes grounding one"
    )
    assert "requeue with a larger grant" in unwrapped
    assert "authoring-budget-exhausted" in unwrapped, (
        "the exact spelling that killed real models must be shown as the trap"
    )
    assert "no frozen source can witness your wall clock" in unwrapped

    reason = json.loads(
        (_SCHEMA_DIR / "author-result-v4.schema.json").read_text(encoding="utf-8")
    )["$defs"]["blocked_payload"]["properties"]["reason_code"]["description"]
    assert "wall-exceeded" in reason, "the schema must name the reserved spelling"
    assert "effort-exhausted:*" in reason, "the reserved family must be named"
    assert "requeueable" in reason

    # The commanded spelling must stay lockstep with the live vocabulary: if the
    # reserved set moves, the prompt and schema prose above are lying to the one
    # attempt each author gets.
    from menagerie.crawler.constants import (
        EFFORT_EXHAUSTION_REASON_CODES,
        EXHAUSTION_TERMINAL_REASON_BY_STAGE,
        FAILURE_REASON_CODES,
        TERMINAL_STATUS_CODES,
    )

    assert "wall-exceeded" in EFFORT_EXHAUSTION_REASON_CODES, (
        "the brief commands wall-exceeded; if the reserved vocabulary dropped it, "
        "the commanded path silently became the rejected path"
    )
    routed = EXHAUSTION_TERMINAL_REASON_BY_STAGE["author"]
    assert routed in FAILURE_REASON_CODES["author"]
    assert "failed:author" in TERMINAL_STATUS_CODES


@pytest.mark.smoke
def test_the_exact_value_traps_are_stated_in_the_stage_brief() -> None:
    """Every byte-exact refusal that killed a real session is disclosed as a block.

    Four rung-7 author sessions died on exact-value rules that lived only in
    Python or in an unread schema leaf: ``m3671`` attempt 1 on
    ``source_to_code_map[0].code_path`` null, ``m3671`` attempt 2 and ``m9617``
    attempt 1 on the ``licenses.weights.status`` const, ``m4334`` attempt 1 on the
    ``initialization.policy`` const. The prompt-audit docket added the three
    Python-only cross-field checks (citation object equality, the mandatory-link
    invariant, the ``cas_path`` grant refusal) as the same trap shape waiting to
    fire. Authors run once; each rule must be stated where they read, and each
    stated rule must still be the live one.
    """

    stage2 = " ".join(_STAGE2_PROMPT.read_text(encoding="utf-8").split())
    assert 'exactly `"not-used"`' in stage2, (
        "the licenses.weights.status const must be commanded verbatim"
    )
    assert '`"not-applicable"` killed a real proposal' in stage2, (
        "the near-synonym that killed m3671 and m9617 must be shown as the trap"
    )
    assert "must be **exactly equal**, leaf for leaf" in stage2
    assert "must list **every** source in the frozen manifest" in stage2, (
        "the staging set-equality rule must be stated: m9617 listed the 9 sources "
        "it had used out of 18 and was refused at staging as a session crash"
    )
    assert "no subset, no extras" in stage2
    assert '`source_resolution.mandatory_link_status` must be `"ok"`' in stage2
    assert "Never add `cas_path`" in stage2
    assert "`code_path` is a **non-empty string**" in stage2
    assert "`source_to_code_map: []`" in stage2, (
        "the declarative-R1 shape must be shown beside the row rule"
    )
    assert 'const `"random"`' in stage2
    assert "`source_specified_choices`" in stage2

    # Each disclosed trap must still be the rule the code or schema enforces.
    proposal_source = (_CRAWLER_ROOT / "proposal.py").read_text(encoding="utf-8")
    assert "top-level citation differs from accuracy-checked external_metadata.citation" in (
        proposal_source
    )
    assert "mandatory source link is not satisfied" in proposal_source
    assert "primary_source_id does not name a declared source" in proposal_source
    assert "source_resolution.sources cannot carry author-controlled CAS paths" in (
        proposal_source
    )

    common = _schema("model-common.schema.json")
    weights = common["$defs"]["licenses"]["properties"]["weights"]["properties"]["status"]
    assert weights.get("const") == "not-used"
    initialization = common["$defs"]["initialization"]["properties"]
    assert initialization["policy"].get("const") == "random"
    assert initialization["pretrained_disabled"].get("const") is True
    code_path = common["$defs"]["source_to_code"]["properties"]["code_path"]
    assert code_path.get("$ref", "").endswith("nonempty_string"), (
        "the disclosed row rule must still be the schema's: a null code_path row "
        "is what killed m3671 attempt 1"
    )
    assert "code_path" in common["$defs"]["source_to_code"]["required"]


@pytest.mark.smoke
def test_the_digest_tool_rule_is_stated_in_the_stage_brief() -> None:
    """The brief tells the author to COMPUTE excerpt digests, never to write them.

    ``m10517`` published a proposal whose quoted excerpt was byte-perfect against
    the frozen CAS line while its declared ``text_sha256`` matched no bytes
    anywhere -- a digest written from memory. The existing digest section warned
    about hashing the wrong REGION; it never said to run a real tool, and a
    fabricated digest is exactly the failure a warning about regions does not
    reach.
    """

    stage2 = " ".join(_STAGE2_PROMPT.read_text(encoding="utf-8").split())
    assert "`sha256sum`" in stage2, "a concrete tool must be named"
    assert "A digest written from memory" in stage2
    assert "matched no bytes anywhere" in stage2, (
        "the m10517 failure shape must be described so the instruction reads as "
        "load-bearing rather than pedantry"
    )
