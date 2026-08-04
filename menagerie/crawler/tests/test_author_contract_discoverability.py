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
  citing a source outside the frozen manifest. Terminal evidence resolves
  all-or-nothing, so those three discarded the ten that verified -- including the
  one supporting the blocked predicate -- and the model terminalized as
  disposition-unverifiable. The all-or-nothing amplification was written down
  nowhere.

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

    ``proposal._validate_source_ladder`` refuses an empty array outright, and
    ``recipe.validate_pretrained_disable_fields`` then requires each name to be a
    ``kwargs`` key carrying a disabling value. Neither clause was discoverable, so
    an author whose constructor exposes no such keyword emitted the honest ``[]``
    and died. The schema must now say both, and must say what to do when the
    constructor has no such keyword at all.
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
    assert "empty" in lowered, "the empty-array refusal must be stated, not just enforced"
    assert "kwargs" in lowered, "the kwargs pairing requirement must be stated"
    assert "signature" in lowered, "the constructor-signature check must be stated"
    assert "blocked" in lowered, "the no-such-keyword escape hatch must be named"


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
    assert "empty" in text.lower()


@pytest.mark.smoke
def test_the_full_annotation_rule_is_stated_in_the_registered_schema() -> None:
    """``code_path`` and both staged symbols document the AST typing check.

    ``proposal._validate_typed_functions`` runs over the whole recursive
    model-local import closure, so the rule's SCOPE matters as much as its
    existence: an author vendoring unannotated upstream source needs to know that
    file is checked too, before spending a whole one-shot attempt on it.
    """

    code_path = _implementation_property("code_path")["description"].lower()
    assert "annotat" in code_path, "the annotation requirement must be stated"
    assert "closure" in code_path, "the closure scope must be stated"
    assert "return annotation" in code_path
    assert "vendor" in code_path, "vendored upstream members must be named as in scope"

    manifest = _implementation_property("code_manifest")["description"].lower()
    assert "annotation" in manifest

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


@pytest.mark.smoke
def test_terminal_evidence_resolution_is_documented_as_all_or_nothing() -> None:
    """The pack's all-or-nothing amplification is stated on both surfaces.

    ``terminal_evidence.resolve_terminal_evidence`` returns ``UNRESOLVED`` with an
    EMPTY excerpt tuple unless every declared ID grounds. An author reading only
    "absence is a named gap" reasonably concludes that quoting more is never
    worse; ``m9617`` proved otherwise. Both the result schema and the stage brief
    must carry the correction, including that a source outside the frozen manifest
    is what triggers it.
    """

    document = _schema("author-result-v4.schema.json")
    records = _description(document, "$defs", "evidence_records").lower()
    assert "all-or-nothing" in records
    assert "source_manifest" in records, "the frozen-manifest restriction must be named"
    assert "discards every excerpt that did verify" in records

    for arm in ("defer_payload", "skip_payload", "blocked_payload"):
        ids = _description(document, "$defs", arm, "properties", "evidence_ids").lower()
        assert "all-or-nothing" in ids, f"{arm}.evidence_ids must carry the warning"

    stage2 = _STAGE2_PROMPT.read_text(encoding="utf-8").lower()
    assert "all-or-nothing" in stage2
    assert "blocked-prerequisite" in stage2


@pytest.mark.smoke
def test_the_guarded_rules_are_still_the_rules_the_code_enforces() -> None:
    """The documented wording still matches live enforcement messages.

    A guard that only reads documentation would keep passing after the rule it
    describes was renamed or removed, leaving prose that misleads the one attempt
    an author gets. These are the exact strings the pilot's quarantined
    diagnostics carried.
    """

    proposal_source = (_CRAWLER_ROOT / "proposal.py").read_text(encoding="utf-8")
    assert "R1_LIBRARY must explicitly disable pretrained fields" in proposal_source
    assert "must be fully typed" in proposal_source

    recipe_source = (_CRAWLER_ROOT / "recipe.py").read_text(encoding="utf-8")
    assert "is absent from constructor kwargs" in recipe_source
    assert "does not carry a disabling value" in recipe_source

    terminal_source = (_CRAWLER_ROOT / "terminal_evidence.py").read_text(encoding="utf-8")
    assert "if len(verified) == len(declared):" in terminal_source, (
        "terminal evidence is documented as all-or-nothing; if this equality is gone the "
        "documentation must be revisited rather than left describing a rule that changed"
    )
