"""Drift guard binding ``SCHEMA_REFERENCE.md`` to the executable schemas.

``procedures/SCHEMA_REFERENCE.md`` is author-facing documentation of the proposal
schema, and the agentic authoring stage runs exactly once per model, so a stale line
there is not a documentation bug -- it is a permanently wrong record, produced at
scale, in a stage that cannot be re-run. Nothing generated or cross-checked that file
before, and it drifted: an ``artifact_sha256`` presence claim outlived the leaf
becoming required-whenever-derivable, the ``kwargs`` construct-node contract and the
``entrypoint``/``post_construct``/``code_manifest``/``declared_timeout_seconds``
additions never appeared at all, and two whole top-level blocks
(``family_variant_derivation``, ``untrusted_attempt``) were undocumented.

These tests close that in both directions:

* a schema change not reflected in the document fails
  :func:`test_shipped_document_matches_a_fresh_render` (the committed bytes stop
  equalling a fresh projection) or
  :func:`test_document_layout_covers_every_top_level_schema_field` (a new top-level
  field lands in no section, so it would render nowhere);
* a documented claim with no schema basis fails the same render equality, because the
  generated regions are replaced wholesale by the projection.

Both directions are additionally proven in isolation below on synthetic inputs, so the
guard's own failure behavior is tested rather than assumed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION,
    GATE_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
)
from menagerie.crawler.schema import load_schema
from menagerie.crawler.tools import render_schema_reference
from menagerie.crawler.tools.render_schema_reference import (
    DOCUMENT_SECTIONS,
    DOCUMENTED_SCHEMA_VERSIONS,
    PRESENCE_MANDATORY,
    PRESENCE_OPTIONAL,
    default_document_path,
    field_rows,
    layout_coverage_errors,
    referenced_vocabularies,
    render,
    render_region,
)


def _shipped_document() -> str:
    """Read the committed schema-reference document.

    Returns
    -------
    str
        Exact committed document text.
    """

    return default_document_path().read_text(encoding="utf-8")


# --------------------------------------------------------------------------------
# 1. The shipped document is exactly what the schemas project.
# --------------------------------------------------------------------------------


@pytest.mark.smoke
def test_shipped_document_matches_a_fresh_render() -> None:
    """The committed document equals a fresh projection of the executable schemas.

    This is the drift guard. It fails when a schema leaf is added, removed,
    retyped, made (un)required, or re-described without regenerating the document,
    and equally when a table row is hand-edited into something the schemas do not
    say.
    """

    current = _shipped_document()
    assert render(current) == current, (
        "SCHEMA_REFERENCE.md is stale; regenerate with "
        "python -m menagerie.crawler.tools.render_schema_reference --write"
    )


def test_render_tool_reports_the_shipped_document_as_current() -> None:
    """``render_schema_reference`` with no ``--write`` is a clean drift check."""

    assert render_schema_reference.main([]) == 0


@pytest.mark.smoke
def test_document_layout_covers_every_top_level_schema_field() -> None:
    """Every documented schema's top-level properties are filed in exactly one section.

    Pure generation cannot catch a NEW top-level field on its own: a field named by
    no section simply renders nowhere and the committed bytes still match. This
    coverage check is what makes that case a failure.
    """

    assert layout_coverage_errors() == ()


def test_render_is_deterministic() -> None:
    """Repeated renders are byte-identical, so the drift test cannot flake."""

    current = _shipped_document()
    assert render(current) == render(current)
    assert [render_region(section.region_id) for section in DOCUMENT_SECTIONS] == [
        render_region(section.region_id) for section in DOCUMENT_SECTIONS
    ]


# --------------------------------------------------------------------------------
# 2. Direction one: a schema change the document does not reflect must fail.
# --------------------------------------------------------------------------------


def _patched_schema(monkeypatch: pytest.MonkeyPatch, mutate: Any) -> None:
    """Install a mutated copy of ``model.v2`` for one test.

    Parameters
    ----------
    monkeypatch:
        Pytest patcher scoped to the calling test.
    mutate:
        Callable applied to the deep-copied schema before installation.
    """

    original = load_schema
    patched = deepcopy(load_schema(MODEL_SCHEMA_VERSION))
    mutate(patched)

    def _load(schema_version: str) -> dict[str, Any]:
        """Return the mutated model schema, or the real schema for other versions.

        Parameters
        ----------
        schema_version:
            Requested schema version.

        Returns
        -------
        dict[str, Any]
            Loaded schema document.
        """

        if schema_version == MODEL_SCHEMA_VERSION:
            return patched
        return original(schema_version)

    monkeypatch.setattr(render_schema_reference, "load_schema", _load)


def test_a_changed_schema_description_fails_the_drift_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-describing a schema leaf without regenerating the document fails.

    Parameters
    ----------
    monkeypatch:
        Pytest patcher used to install the mutated schema.
    """

    current = _shipped_document()

    def mutate(schema: dict[str, Any]) -> None:
        """Rewrite one leaf description.

        Parameters
        ----------
        schema:
            Deep-copied model schema to mutate in place.
        """

        schema["properties"]["notes"]["description"] = "Rewritten by a schema change."

    _patched_schema(monkeypatch, mutate)
    assert render(current) != current


def test_a_new_nested_schema_leaf_fails_the_drift_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema leaf the document never lists fails the drift test.

    Parameters
    ----------
    monkeypatch:
        Pytest patcher used to install the mutated schema.
    """

    current = _shipped_document()

    def mutate(schema: dict[str, Any]) -> None:
        """Grow a documented scalar field into an object with a nested leaf.

        Parameters
        ----------
        schema:
            Deep-copied model schema to mutate in place.
        """

        schema["properties"]["notes"] = {
            "type": "object",
            "required": ["newly_added_check"],
            "properties": {
                "newly_added_check": {
                    "type": "boolean",
                    "description": "Synthetic leaf the shipped document cannot know about.",
                }
            },
            "description": "Mandatory free-form record notes.",
        }

    _patched_schema(monkeypatch, mutate)
    rendered = render(current)
    assert rendered != current
    assert "newly_added_check" in rendered


def test_a_changed_required_list_fails_the_drift_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Making a documented leaf optional without regenerating the document fails.

    Parameters
    ----------
    monkeypatch:
        Pytest patcher used to install the mutated schema.
    """

    current = _shipped_document()
    assert f"| `notes` | string | {PRESENCE_MANDATORY} |" in current

    def mutate(schema: dict[str, Any]) -> None:
        """Drop one field from the root required list.

        Parameters
        ----------
        schema:
            Deep-copied model schema to mutate in place.
        """

        schema["required"] = [name for name in schema["required"] if name != "notes"]

    _patched_schema(monkeypatch, mutate)
    rendered = render(current)
    assert rendered != current
    assert f"| `notes` | string | {PRESENCE_OPTIONAL} |" in rendered


def test_a_new_top_level_schema_field_fails_layout_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A top-level schema field filed in no section is a coverage failure.

    Parameters
    ----------
    monkeypatch:
        Pytest patcher used to install the mutated schema.
    """

    def mutate(schema: dict[str, Any]) -> None:
        """Add one undocumented top-level property.

        Parameters
        ----------
        schema:
            Deep-copied model schema to mutate in place.
        """

        schema["properties"]["newly_added_block"] = {
            "type": "string",
            "description": "Synthetic top-level field filed in no document section.",
        }

    _patched_schema(monkeypatch, mutate)
    errors = layout_coverage_errors()
    assert errors
    assert any("newly_added_block" in problem for problem in errors)


# --------------------------------------------------------------------------------
# 3. Direction two: a documented claim with no schema basis must fail.
# --------------------------------------------------------------------------------


def test_an_invented_table_row_fails_the_drift_test() -> None:
    """A row the schemas do not support cannot survive a render."""

    tampered = _shipped_document().replace(
        "| `notes` | string | Mandatory | Mandatory free-form record notes. |",
        "| `notes` | string | Mandatory | Mandatory free-form record notes. |\n"
        "| `notes.invented_leaf` | string | Optional | This field does not exist. |",
        1,
    )
    assert "invented_leaf" in tampered
    rendered = render(tampered)
    assert rendered != tampered
    assert "invented_leaf" not in rendered


def test_a_reworded_meaning_cell_fails_the_drift_test() -> None:
    """A hand-softened presence or meaning claim cannot survive a render."""

    original = "| `notes` | string | Mandatory | Mandatory free-form record notes. |"
    tampered = _shipped_document().replace(
        original,
        "| `notes` | string | Optional | Best-effort free-form record notes. |",
        1,
    )
    assert tampered != _shipped_document()
    assert render(tampered) != tampered


def test_a_documented_field_absent_from_the_schema_fails_layout_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A section naming a field the schema does not declare is a coverage failure.

    Parameters
    ----------
    monkeypatch:
        Pytest patcher used to install the mutated schema.
    """

    def mutate(schema: dict[str, Any]) -> None:
        """Remove a documented top-level property from the schema.

        Parameters
        ----------
        schema:
            Deep-copied model schema to mutate in place.
        """

        del schema["properties"]["notes"]
        schema["required"] = [name for name in schema["required"] if name != "notes"]

    _patched_schema(monkeypatch, mutate)
    errors = layout_coverage_errors()
    assert errors
    assert any("notes" in problem for problem in errors)


def test_an_unknown_generated_region_is_rejected() -> None:
    """A generated region no section declares cannot pass silently."""

    tampered = _shipped_document() + (
        "\n<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/invented-region -->\n"
        "<!-- END GENERATED SCHEMA TABLE: model.v2/invented-region -->\n"
    )
    with pytest.raises(ValueError, match="unknown generated region"):
        render(tampered)


def test_a_removed_region_marker_is_rejected() -> None:
    """A deleted marker pair fails loudly instead of dropping a table."""

    tampered = _shipped_document().replace(
        "<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/evidence -->", "", 1
    )
    with pytest.raises(ValueError, match="no unique region markers"):
        render(tampered)


# --------------------------------------------------------------------------------
# 4. The projection itself stays faithful to the schemas.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema_version",
    [MODEL_SCHEMA_VERSION, ATTEMPT_SCHEMA_VERSION, GATE_SCHEMA_VERSION],
)
def test_every_row_meaning_is_the_verbatim_schema_description(schema_version: str) -> None:
    """No row invents, softens, or truncates a schema description.

    Parameters
    ----------
    schema_version:
        Documented schema whose rows are checked.
    """

    document = _shipped_document()
    for section in DOCUMENT_SECTIONS:
        if section.schema_version != schema_version:
            continue
        for row in field_rows(section.schema_version, section.fields):
            assert row.to_markdown() in document


def test_documented_schema_versions_match_the_declared_sections() -> None:
    """The section table and the documented-version tuple cannot fall out of step."""

    from_sections = {
        section.schema_version for section in DOCUMENT_SECTIONS if section.schema_version
    }
    assert from_sections == set(DOCUMENTED_SCHEMA_VERSIONS)


def test_every_referenced_vocabulary_is_expanded_exactly_once() -> None:
    """Each named closed vocabulary used by a Type cell appears in the appendix."""

    document = _shipped_document()
    vocabularies = referenced_vocabularies()
    assert vocabularies
    for name, members in vocabularies:
        assert members
        assert document.count(f"- `{name}` ({len(members)} members):") == 1
        for member in members:
            assert f"`{member}`" in document
