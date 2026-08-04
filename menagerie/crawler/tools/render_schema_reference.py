"""Render the author-facing schema data dictionary from the executable schemas.

``procedures/SCHEMA_REFERENCE.md`` is what an authoring agent reads to decide what
to emit, and the authoring stage runs once per model, so a stale line there buys a
permanently wrong record. The field tables in that document are not independent
prose: every row is a mechanical projection of one JSON Schema leaf -- the path, the
rendered type, whether the enclosing object requires it, and the leaf's own
``description`` verbatim. Hand-maintaining that projection is what let the
``artifact_sha256`` presence claim, the ``kwargs`` construct-node contract, and the
whole ``entrypoint``/``post_construct`` pair drift away from the schemas unnoticed.

So the tables are GENERATED into delimited regions and re-derived by the test suite
on every run, exactly like :mod:`menagerie.crawler.tools.render_claim_vocabulary`
does for the author prompt's claim vocabulary. Explanatory prose lives outside the
markers and stays hand-written; anything inside a marker is owned by this module.
Changing a schema without regenerating the document is a hard test failure, not a
silent divergence.

The section layout in :data:`DOCUMENT_SECTIONS` is deliberately hand-declared: it is
the one editorial judgement the schemas cannot supply. It is checked for exact
coverage against each documented schema's top-level properties, so a newly added
top-level field cannot be silently omitted from the document either.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION,
    GATE_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
)
from menagerie.crawler.schema import (
    SCHEMA_RESOURCE_FILES,
    _resolve_schema_reference,
    load_schema,
    load_schema_resource,
)

REGION_BEGIN = "<!-- BEGIN GENERATED SCHEMA TABLE: "
REGION_END = "<!-- END GENERATED SCHEMA TABLE: "
REGION_SUFFIX = " -->"

TABLE_HEADER = (
    "| Field | Type | Presence | Meaning |\n| --- | --- | --- | --- |"
)
VOCABULARY_WRAP_WIDTH = 96

PRESENCE_MANDATORY = "Mandatory"
PRESENCE_OPTIONAL = "Optional"
PRESENCE_BRANCH = "Branch-dependent"

_UNDOCUMENTED = "_(no schema description)_"

# Composition keywords whose object branches contribute properties to one instance
# location. ``oneOf``/``anyOf`` are alternative shapes; ``allOf`` is a conjunction.
_BRANCH_KEYWORDS = ("oneOf", "anyOf")

# Keywords that Draft 2020-12 applies alongside a ``$ref``. This projection resolves
# through references, so a reference carrying any of these is refused rather than
# quietly dropped.
_APPLICATOR_KEYWORDS = frozenset(
    {"properties", "items", "oneOf", "anyOf", "allOf", "enum", "const", "type", "required"}
)


@dataclass(frozen=True)
class SectionSpec:
    """One generated table region and the schema fields it projects.

    Parameters
    ----------
    region_id:
        Stable identifier repeated in the document's begin/end markers.
    schema_version:
        Executable schema whose top-level properties the region projects.
    fields:
        Top-level property names rendered in this region, in document order.
    """

    region_id: str
    schema_version: str
    fields: tuple[str, ...]


DOCUMENT_SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec(
        "model.v2/bookkeeping",
        MODEL_SCHEMA_VERSION,
        (
            "schema_version",
            "stable_id",
            "record_seq",
            "record_revision",
            "parent_revision",
            "created_at",
            "revised_by",
            "authored_metadata_state",
            "intake",
            "provenance",
            "budget",
            "flags",
            "notes",
            "scar_history",
            "completeness",
            "untrusted_attempt",
        ),
    ),
    SectionSpec(
        "model.v2/identity-and-taxonomy",
        MODEL_SCHEMA_VERSION,
        ("identity", "taxonomy", "family_variant_derivation"),
    ),
    SectionSpec(
        "model.v2/external-metadata",
        MODEL_SCHEMA_VERSION,
        ("external_metadata",),
    ),
    SectionSpec("model.v2/website", MODEL_SCHEMA_VERSION, ("website",)),
    SectionSpec(
        "model.v2/people-origin-dates-citation",
        MODEL_SCHEMA_VERSION,
        ("people_and_origin", "dates", "citation"),
    ),
    SectionSpec("model.v2/licenses", MODEL_SCHEMA_VERSION, ("licenses",)),
    SectionSpec(
        "model.v2/source-resolution",
        MODEL_SCHEMA_VERSION,
        ("source_resolution",),
    ),
    SectionSpec("model.v2/evidence", MODEL_SCHEMA_VERSION, ("evidence",)),
    SectionSpec(
        "model.v2/implementation",
        MODEL_SCHEMA_VERSION,
        ("implementation",),
    ),
    SectionSpec(
        "model.v2/input-contract",
        MODEL_SCHEMA_VERSION,
        ("input_contract",),
    ),
    SectionSpec("model.v2/observed", MODEL_SCHEMA_VERSION, ("observed",)),
    SectionSpec(
        "model.v2/modes-and-verification-state",
        MODEL_SCHEMA_VERSION,
        ("modes", "fidelity", "accuracy_gate", "execution", "status"),
    ),
    SectionSpec(
        "attempt.v2/receipt",
        ATTEMPT_SCHEMA_VERSION,
        (
            "schema_version",
            "attempt_id",
            "ledger_seq",
            "payload_sha256",
            "work_id",
            "stable_id",
            "attempt_no",
            "parent_attempt_id",
            "actor",
            "stage",
            "mode",
            "started_at",
            "finished_at",
            "result",
            "attempted_rungs",
            "retries",
            "identities",
            "environment",
            "host",
            "invocation",
            "worker_receipt",
            "supervisor_observation",
            "policy_observation",
            "error",
            "defer_evidence",
        ),
    ),
    SectionSpec(
        "gate.v2/verdict",
        GATE_SCHEMA_VERSION,
        (
            "schema_version",
            "gate_id",
            "ledger_seq",
            "payload_sha256",
            "gate_kind",
            "batch_size",
            "gate_round",
            "gate_identity",
            "checker",
            "items",
            "result_envelope_sha256",
        ),
    ),
    SectionSpec(
        "vocabularies/closed",
        "",
        (),
    ),
)

DOCUMENTED_SCHEMA_VERSIONS: tuple[str, ...] = (
    MODEL_SCHEMA_VERSION,
    ATTEMPT_SCHEMA_VERSION,
    GATE_SCHEMA_VERSION,
)


def _schema_document_names() -> dict[int, str]:
    """Map every loadable schema document to a stable short name.

    Returns
    -------
    dict[int, str]
        Identity-keyed short names used to qualify closed-vocabulary references.
    """

    names: dict[int, str] = {}
    for version in DOCUMENTED_SCHEMA_VERSIONS:
        document = load_schema(version)
        names[id(document)] = str(document["$id"]).rsplit("/", 1)[-1].split(".", 1)[0]
    for filename in SCHEMA_RESOURCE_FILES:
        document = load_schema_resource(filename)
        names[id(document)] = filename.split(".", 1)[0]
    return names


def _resolve(node: Mapping[str, Any], root: Mapping[str, Any]) -> tuple[
    Mapping[str, Any], Mapping[str, Any], Optional[str]
]:
    """Follow a ``$ref`` chain to its concrete node.

    Parameters
    ----------
    node:
        Schema node that may be a reference.
    root:
        Document containing ``node``.

    Returns
    -------
    tuple[Mapping[str, Any], Mapping[str, Any], Optional[str]]
        Resolved node, its owning document, and the qualified name of the last
        named ``$defs`` entry traversed, or ``None`` for an inline node.

    Raises
    ------
    ValueError
        If a reference carries an applicator sibling. Draft 2020-12 applies those
        alongside the reference; this projection would silently drop them, so a
        schema that starts using the form must fail here rather than quietly
        under-document a leaf.
    """

    names = _schema_document_names()
    seen: set[str] = set()
    definition_name: Optional[str] = None
    current: Mapping[str, Any] = node
    current_root = root
    while isinstance(current, Mapping) and isinstance(current.get("$ref"), str):
        reference = str(current["$ref"])
        siblings = sorted(set(current) & _APPLICATOR_KEYWORDS)
        if siblings:
            raise ValueError(f"reference {reference!r} carries applicator siblings {siblings}")
        if reference in seen:
            break
        seen.add(reference)
        current_root, resolved = _resolve_schema_reference(current_root, reference)
        fragment = reference.partition("#")[2]
        if fragment.startswith("/$defs/") and fragment.count("/") == 2:
            document = names.get(id(current_root), "schema")
            definition_name = f"{document}.{fragment.rsplit('/', 1)[-1]}"
        current = resolved
    return current, current_root, definition_name


def _render_type(node: Mapping[str, Any], root: Mapping[str, Any]) -> str:
    """Render one schema node as a compact Markdown type expression.

    Parameters
    ----------
    node:
        Schema node at an instance location.
    root:
        Document containing ``node``.

    Returns
    -------
    str
        Markdown-escaped type expression such as ``array<string>`` or
        ``string \\| null``.
    """

    resolved, resolved_root, definition_name = _resolve(node, root)
    if "const" in resolved:
        return f"const `{resolved['const']}`"
    if isinstance(resolved.get("enum"), list):
        if definition_name is not None:
            return f"enum `{definition_name}`"
        members = " \\| ".join(f"`{member}`" for member in resolved["enum"])
        return f"enum: {members}"
    for keyword in _BRANCH_KEYWORDS:
        branches = resolved.get(keyword)
        if isinstance(branches, list) and branches:
            rendered: list[str] = []
            for branch in branches:
                if isinstance(branch, Mapping):
                    text = _render_type(branch, resolved_root)
                    if text not in rendered:
                        rendered.append(text)
            if rendered:
                return " \\| ".join(rendered)
    declared = resolved.get("type")
    types = declared if isinstance(declared, list) else [declared] if declared else []
    parts: list[str] = []
    for declared_type in types:
        if declared_type == "array":
            items = resolved.get("items")
            if isinstance(items, Mapping):
                parts.append(f"array<{_render_type(items, resolved_root)}>")
            else:
                parts.append("array")
        elif declared_type == "object":
            has_properties = isinstance(resolved.get("properties"), Mapping)
            parts.append("object" if has_properties else "object map")
        elif declared_type is not None:
            parts.append(str(declared_type))
    if not parts:
        return "value"
    return " \\| ".join(parts)


def _object_shapes(
    node: Mapping[str, Any], root: Mapping[str, Any]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Return every alternative object shape contributed at one instance location.

    ``allOf`` conjunctions are merged into each alternative; ``oneOf``/``anyOf``
    alternatives stay separate so branch-dependent presence stays visible.

    Parameters
    ----------
    node:
        Schema node at an instance location.
    root:
        Document containing ``node``.

    Returns
    -------
    list[tuple[Mapping[str, Any], Mapping[str, Any]]]
        Object shapes paired with their owning documents, in schema order.
    """

    resolved, resolved_root, _ = _resolve(node, root)
    conjuncts: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    if isinstance(resolved.get("properties"), Mapping):
        conjuncts.append((resolved, resolved_root))
    for member in resolved.get("allOf", []) or []:
        if isinstance(member, Mapping):
            conjuncts.extend(_object_shapes(member, resolved_root))
    alternatives: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for keyword in _BRANCH_KEYWORDS:
        for member in resolved.get(keyword, []) or []:
            if isinstance(member, Mapping):
                alternatives.extend(_object_shapes(member, resolved_root))
    if alternatives and conjuncts:
        return conjuncts + alternatives
    return alternatives or conjuncts


def _presence(declaring: int, requiring: int, total: int) -> str:
    """Classify a property's presence across the alternative shapes that declare it.

    Parameters
    ----------
    declaring:
        Number of alternative shapes declaring the property.
    requiring:
        Number of alternative shapes requiring the property.
    total:
        Number of alternative shapes at the instance location.

    Returns
    -------
    str
        One of ``Mandatory``, ``Optional``, or ``Branch-dependent``.
    """

    if requiring == total and declaring == total:
        return PRESENCE_MANDATORY
    if requiring == 0 and declaring == total:
        return PRESENCE_OPTIONAL
    return PRESENCE_BRANCH


def _escape(text: str) -> str:
    """Escape Markdown table-hostile characters in a cell value.

    Parameters
    ----------
    text:
        Raw description text from a schema leaf.

    Returns
    -------
    str
        Single-line cell text safe inside a Markdown table.
    """

    collapsed = " ".join(text.split())
    return collapsed.replace("\\", "\\\\").replace("|", "\\|")


@dataclass(frozen=True)
class FieldRow:
    """One rendered data-dictionary row.

    Parameters
    ----------
    path:
        Normalized instance path, with collection elements written ``[]``.
    type_expression:
        Rendered Markdown type expression.
    presence:
        Rendered presence classification.
    meaning:
        Verbatim schema description, collapsed to one line.
    """

    path: str
    type_expression: str
    presence: str
    meaning: str

    def to_markdown(self) -> str:
        """Render the row as one Markdown table line.

        Returns
        -------
        str
            Pipe-delimited Markdown table row.
        """

        return f"| `{self.path}` | {self.type_expression} | {self.presence} | {self.meaning} |"


def _walk(
    node: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str,
    rows: list[FieldRow],
    visited: frozenset[str],
) -> None:
    """Append rows for every leaf reachable beneath one instance location.

    Parameters
    ----------
    node:
        Schema node at ``path``.
    root:
        Document containing ``node``.
    path:
        Normalized instance path already emitted for ``node``.
    rows:
        Mutable accumulator, appended in stable schema order.
    visited:
        References already expanded on this path, stopping recursive schemas.
    """

    reference = node.get("$ref") if isinstance(node, Mapping) else None
    if isinstance(reference, str):
        if reference in visited:
            return
        visited = visited | {reference}
    shapes = _object_shapes(node, root)
    if shapes:
        _walk_object(shapes, path, rows, visited)
        return
    resolved, resolved_root, _ = _resolve(node, root)
    items = resolved.get("items")
    if isinstance(items, Mapping):
        _walk(items, resolved_root, f"{path}[]", rows, visited)


def _walk_object(
    shapes: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    path: str,
    rows: list[FieldRow],
    visited: frozenset[str],
) -> None:
    """Append rows for the merged property set of one object location.

    Parameters
    ----------
    shapes:
        Alternative object shapes at this location, in schema order.
    path:
        Normalized instance path of the enclosing object.
    rows:
        Mutable accumulator.
    visited:
        References already expanded on this path.
    """

    ordered: list[str] = []
    for shape, _ in shapes:
        for name in shape.get("properties", {}):
            if name not in ordered:
                ordered.append(name)
    for name in ordered:
        declaring = [
            (shape, shape_root)
            for shape, shape_root in shapes
            if name in shape.get("properties", {})
        ]
        requiring = sum(1 for shape, _ in declaring if name in shape.get("required", []))
        child, child_root = declaring[0][0]["properties"][name], declaring[0][1]
        child_path = f"{path}.{name}" if path else name
        descriptions = [
            str(shape["properties"][name].get("description", "")).strip()
            for shape, _ in declaring
        ]
        meaning = next((text for text in descriptions if text), "")
        type_expressions: list[str] = []
        for shape, shape_root in declaring:
            expression = _render_type(shape["properties"][name], shape_root)
            if expression not in type_expressions:
                type_expressions.append(expression)
        rows.append(
            FieldRow(
                path=child_path,
                type_expression=" \\| ".join(type_expressions),
                presence=_presence(len(declaring), requiring, len(shapes)),
                meaning=_escape(meaning) if meaning else _UNDOCUMENTED,
            )
        )
        _walk(child, child_root, child_path, rows, visited)


def field_rows(schema_version: str, field_names: Iterable[str]) -> tuple[FieldRow, ...]:
    """Project selected top-level schema properties into ordered table rows.

    Parameters
    ----------
    schema_version:
        Executable schema version to project.
    field_names:
        Top-level property names, in the document's editorial order.

    Returns
    -------
    tuple[FieldRow, ...]
        Rows for the named properties and every leaf beneath them.

    Raises
    ------
    KeyError
        If a named property is absent from the schema.
    """

    schema = load_schema(schema_version)
    properties = schema["properties"]
    required = set(schema.get("required", []))
    rows: list[FieldRow] = []
    for name in field_names:
        if name not in properties:
            raise KeyError(f"{schema_version} has no top-level property {name!r}")
        child = properties[name]
        description = str(child.get("description", "")).strip()
        rows.append(
            FieldRow(
                path=name,
                type_expression=_render_type(child, schema),
                presence=PRESENCE_MANDATORY if name in required else PRESENCE_OPTIONAL,
                meaning=_escape(description) if description else _UNDOCUMENTED,
            )
        )
        _walk(child, schema, name, rows, frozenset())
    return tuple(rows)


def referenced_vocabularies() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Collect every named closed vocabulary referenced by the generated tables.

    Returns
    -------
    tuple[tuple[str, tuple[str, ...]], ...]
        Name-sorted vocabulary names paired with their exact members.
    """

    names = _schema_document_names()
    collected: dict[str, tuple[str, ...]] = {}
    for document in [load_schema(version) for version in DOCUMENTED_SCHEMA_VERSIONS] + [
        load_schema_resource(filename) for filename in SCHEMA_RESOURCE_FILES
    ]:
        short_name = names[id(document)]
        for definition_name, definition in document.get("$defs", {}).items():
            resolved, _, _ = _resolve(definition, document)
            members = resolved.get("enum")
            if isinstance(members, list):
                collected[f"{short_name}.{definition_name}"] = tuple(
                    str(member) for member in members
                )
    used = {
        row.type_expression
        for section in DOCUMENT_SECTIONS
        if section.schema_version
        for row in field_rows(section.schema_version, section.fields)
    }
    referenced = {
        name for name in collected if any(f"enum `{name}`" in text for text in used)
    }
    return tuple((name, collected[name]) for name in sorted(referenced))


def render_region(region_id: str) -> str:
    """Render the body of one generated region.

    Parameters
    ----------
    region_id:
        Stable region identifier declared in :data:`DOCUMENT_SECTIONS`.

    Returns
    -------
    str
        Markdown body, without the surrounding markers.

    Raises
    ------
    KeyError
        If no section declares ``region_id``.
    """

    section = next((item for item in DOCUMENT_SECTIONS if item.region_id == region_id), None)
    if section is None:
        raise KeyError(f"undeclared generated region: {region_id!r}")
    if not section.schema_version:
        lines: list[str] = []
        for name, members in referenced_vocabularies():
            lines.append(f"- `{name}` ({len(members)} members):")
            body = ", ".join(f"`{member}`" for member in members)
            lines.extend(
                textwrap.wrap(
                    body,
                    width=VOCABULARY_WRAP_WIDTH,
                    initial_indent="  ",
                    subsequent_indent="  ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        return "\n".join(lines)
    lines = [TABLE_HEADER]
    lines.extend(row.to_markdown() for row in field_rows(section.schema_version, section.fields))
    return "\n".join(lines)


def render(document_text: str) -> str:
    """Return the document with every generated region freshly derived.

    Parameters
    ----------
    document_text:
        Complete current document text.

    Returns
    -------
    str
        Document whose generated regions match the executable schemas.

    Raises
    ------
    ValueError
        If a declared region is absent, duplicated, or malformed, or if the
        document carries a generated region no section declares.
    """

    rendered = document_text
    for section in DOCUMENT_SECTIONS:
        begin = f"{REGION_BEGIN}{section.region_id}{REGION_SUFFIX}"
        end = f"{REGION_END}{section.region_id}{REGION_SUFFIX}"
        if rendered.count(begin) != 1 or rendered.count(end) != 1:
            raise ValueError(f"document has no unique region markers for {section.region_id!r}")
        start = rendered.index(begin) + len(begin)
        stop = rendered.index(end)
        if stop < start:
            raise ValueError(f"document region markers are inverted for {section.region_id!r}")
        body = render_region(section.region_id)
        rendered = f"{rendered[:start]}\n{body}\n{rendered[stop:]}"
    declared = {section.region_id for section in DOCUMENT_SECTIONS}
    for line in rendered.splitlines():
        stripped = line.strip()
        if stripped.startswith(REGION_BEGIN):
            found = stripped[len(REGION_BEGIN) : -len(REGION_SUFFIX)]
            if found not in declared:
                raise ValueError(f"document declares an unknown generated region: {found!r}")
    return rendered


def layout_coverage_errors() -> tuple[str, ...]:
    """Report every documented schema whose top-level properties are not covered.

    Returns
    -------
    tuple[str, ...]
        Human-readable coverage failures, empty when the layout is exact.
    """

    errors: list[str] = []
    for schema_version in DOCUMENTED_SCHEMA_VERSIONS:
        documented: list[str] = []
        for section in DOCUMENT_SECTIONS:
            if section.schema_version == schema_version:
                documented.extend(section.fields)
        declared = set(load_schema(schema_version)["properties"])
        duplicated = sorted({name for name in documented if documented.count(name) > 1})
        missing = sorted(declared - set(documented))
        extraneous = sorted(set(documented) - declared)
        if duplicated:
            errors.append(f"{schema_version} documents duplicate fields: {duplicated}")
        if missing:
            errors.append(f"{schema_version} has undocumented top-level fields: {missing}")
        if extraneous:
            errors.append(f"{schema_version} documents absent top-level fields: {extraneous}")
    return tuple(errors)


def default_document_path() -> Path:
    """Return the bundled schema-reference document path.

    Returns
    -------
    Path
        Absolute path to ``procedures/SCHEMA_REFERENCE.md``.
    """

    return Path(__file__).resolve().parents[1] / "procedures" / "SCHEMA_REFERENCE.md"


def build_parser() -> argparse.ArgumentParser:
    """Build the schema-reference rendering parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with an overridable document path for tests and audits.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", type=Path, default=default_document_path())
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the generated regions in place instead of only checking them.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check or rewrite the schema reference's generated regions.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero when the document already matches, or when it was rewritten.
    """

    args = build_parser().parse_args(argv)
    coverage = layout_coverage_errors()
    if coverage:
        for problem in coverage:
            print(f"schema-reference layout is incomplete: {problem}", file=sys.stderr)
        return 1
    try:
        current = args.document.read_text(encoding="utf-8")
        rendered = render(current)
    except (OSError, ValueError, KeyError) as exc:
        print(f"schema-reference rendering failed: {exc}", file=sys.stderr)
        return 1
    if rendered == current:
        print("schema reference is current")
        return 0
    if not args.write:
        print("schema reference is stale; rerun with --write", file=sys.stderr)
        return 1
    args.document.write_text(rendered, encoding="utf-8")
    print(f"rewrote generated regions in {args.document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
