"""Generate ``docs/maintainers/persisted_fields.md``, the persisted field set of the saved format.

One command: ``python tools/docs/persisted_fields.py``. It reads the writer contract of
record (``torchlens/schemas/writer_contract_v11.json``), the frozen previous contract
(``writer_contract_v10.json``) for the key diff, the generated storage bindings
(``torchlens/_data_classes/_schema_bindings.py``) for each key's storage kind and type
annotation, and the record-class modules under ``torchlens/_data_classes/`` with ``ast``
(nothing is imported) for the public accessor that reads each private store. It writes
one table per record contract: the stored key, its type, whether every save level writes
it or only levels with a payload flag on, its field-policy class, its tier and the public
accessor. ``--check`` exits 1 when the committed page differs from what the inputs
produce, so a contract change without a regenerated page fails the pinning test. The
output is deterministic: sorted keys, no timestamps, no torchlens version.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCHEMAS = REPO / "torchlens" / "schemas"
CONTRACT = SCHEMAS / "writer_contract_v11.json"
PREVIOUS_CONTRACT = SCHEMAS / "writer_contract_v10.json"
DATA_CLASSES = REPO / "torchlens" / "_data_classes"
BINDINGS = DATA_CLASSES / "_schema_bindings.py"
GROUPS_MODULE = DATA_CLASSES / "_trace_groups.py"
OUTPUT = REPO / "docs" / "maintainers" / "persisted_fields.md"
COMMAND = "python tools/docs/persisted_fields.py"

#: Record contracts documented first, in this order; the rest follow alphabetically.
LEAD_RECORDS = ("trace", "op", "layer")
#: The storage bindings key where it differs from the record contract key.
BINDING_KEY_ALIASES = {"aten_op": "primitive_op"}
#: Trace concern groups: a stored key ``_<group>_<leaf>`` reads as ``Trace.<group>.<leaf>``.
TRACE_GROUPS = ("autograd", "cost", "graph", "health", "options", "provenance", "source")
#: Hops through private members a public getter is followed when it reaches a store.
ACCESSOR_DEPTH = 3
#: Accessors shown per key before the count of the rest.
ACCESSOR_LIMIT = 4
#: The four save levels (``torchlens/_io/tlspec.py``, ``TLSPEC_VALID_SAVE_LEVELS``).
SAVE_LEVELS = ("audit", "executable_with_callables", "portable", "runnable")
#: Payload include flags and the levels whose defaults turn them on
#: (``torchlens/_io/save_policy.py``, ``_LEVEL_INCLUDE_FLAG_DEFAULTS``; the two opt-out
#: channels default on at every level).
FLAG_LEVELS = {
    "include_outs": ("executable_with_callables", "portable"),
    "include_grads": ("executable_with_callables", "portable"),
    "include_saved_args": ("executable_with_callables",),
    "include_rng_states": ("executable_with_callables",),
    "include_buffer_values": SAVE_LEVELS,
    "include_custom_attributes": SAVE_LEVELS,
}
#: Stored keys each payload flag gates (``torchlens/_io/scrub.py``, the policy resolver).
FLAG_FIELDS = {
    "include_outs": ("out", "transformed_out"),
    "include_grads": ("grad", "transformed_grad"),
    "include_saved_args": (
        "forward_args",
        "forward_kwargs",
        "out_versions_by_child",
        "saved_args",
        "saved_kwargs",
    ),
    "include_rng_states": ("func_rng_states",),
    "include_buffer_values": ("_buffer_initial_values",),
    "include_custom_attributes": ("custom_attributes",),
}
_CLASS_REPR = re.compile(r"^<class '([\w.]+)'>$")


def _load(path: Path) -> dict[str, Any]:
    """Return the parsed JSON document at ``path``."""

    return json.loads(path.read_text(encoding="utf-8"))


def _literal(node: ast.expr) -> Any:
    """Return the Python value of a constant node, or ``None``."""

    return node.value if isinstance(node, ast.Constant) else None


def _binding_fields(call: ast.expr) -> tuple[str, str | None]:
    """Return ``(kind, annotation)`` from one ``StorageBinding(...)`` call node."""

    if not isinstance(call, ast.Call):
        return ("?", None)
    kind = "?"
    if call.args and isinstance(call.args[0], ast.Attribute):
        kind = call.args[0].attr.lower()
    annotation = None
    for keyword in call.keywords:
        if keyword.arg == "annotation":
            annotation = _literal(keyword.value)
    return (kind, annotation)


def read_bindings(path: Path = BINDINGS) -> dict[str, dict[str, tuple[str, str | None]]]:
    """Return ``{binding key: {field: (kind, annotation)}}`` from the generated bindings."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id != "STORAGE_BINDINGS" or not isinstance(node.value, ast.Dict):
            continue
        table: dict[str, dict[str, tuple[str, str | None]]] = {}
        for record_node, fields_node in zip(node.value.keys, node.value.values):
            if record_node is None or not isinstance(fields_node, ast.Dict):
                continue
            fields = {}
            for field_node, binding_node in zip(fields_node.keys, fields_node.values):
                if field_node is not None:
                    fields[str(_literal(field_node))] = _binding_fields(binding_node)
            table[str(_literal(record_node))] = fields
        return table
    raise SystemExit(f"{path}: no STORAGE_BINDINGS table found")


def _type_text(binding: tuple[str, str | None] | None) -> str:
    """Render the type column from a storage binding."""

    if binding is None:
        return "(no binding)"
    kind, annotation = binding
    if annotation is None:
        return kind
    match = _CLASS_REPR.match(annotation)
    text = match.group(1) if match else annotation
    text = text.replace("typing.", "").replace("torchlens.", "")
    return f"{kind}: `{text}`"


def _self_reads(function: ast.AST) -> set[str]:
    """Return attribute names read as ``self.<name>`` or ``self._trace.<name>`` in a body."""

    names: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        if (
            isinstance(base, ast.Name)
            and base.id == "self"
            or (
                isinstance(base, ast.Attribute)
                and base.attr == "_trace"
                and isinstance(base.value, ast.Name)
                and base.value.id == "self"
            )
        ):
            names.add(node.attr)
    return names


def _is_property(function: ast.FunctionDef) -> bool:
    """Return whether a function definition is decorated as a property getter."""

    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id in {"property", "cached_property"}:
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "cached_property":
            return True
    return False


def read_accessors(root: Path = DATA_CLASSES) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Scan the record-class modules for public property getters and the stores they reach.

    Every function in the package contributes the ``self.<name>`` and
    ``self._trace.<name>`` attributes it touches; a public property getter reaches a
    private store directly or through private members, followed up to
    ``ACCESSOR_DEPTH`` hops by attribute name. Returns ``(reads, members)``: ``reads``
    maps a private attribute name to the ``Class.property`` spellings that reach it;
    ``members`` maps a class name to its public property names.
    """

    touched: dict[str, set[str]] = defaultdict(set)
    getters: list[tuple[str, str, set[str]]] = []
    members: dict[str, set[str]] = defaultdict(set)
    for module in sorted(root.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for class_node in tree.body:
            if not isinstance(class_node, ast.ClassDef):
                continue
            for item in class_node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                names = _self_reads(item)
                touched[item.name] |= names
                if item.name.startswith("_") or not _is_property(item):
                    continue
                members[class_node.name].add(item.name)
                getters.append((class_node.name, item.name, names))
    reads: dict[str, set[str]] = defaultdict(set)
    for class_name, getter, names in getters:
        for name in _closure(names, touched):
            reads[name].add(f"{class_name}.{getter}")
    return (reads, members)


def _closure(
    names: set[str], touched: dict[str, set[str]], depth: int = ACCESSOR_DEPTH
) -> set[str]:
    """Return the private names reachable from ``names`` through private members."""

    seen = {name for name in names if name.startswith("_")}
    frontier = set(seen)
    for _ in range(depth):
        frontier = {
            name
            for member in frontier
            for name in touched.get(member, ())
            if name.startswith("_") and name not in seen
        }
        if not frontier:
            break
        seen |= frontier
    return seen


def _group_leaf(key: str) -> tuple[str, str] | None:
    """Split a Trace group store key ``_<group>_<leaf>`` into its parts."""

    for group in TRACE_GROUPS:
        prefix = f"_{group}_"
        if key.startswith(prefix) and len(key) > len(prefix):
            return (group, key[len(prefix) :])
    return None


def _accessor(
    owner: str, key: str, reads: dict[str, set[str]], members: dict[str, set[str]]
) -> str:
    """Render the public accessor column for one stored key."""

    if not key.startswith("_"):
        return f"`{owner}.{key}` (field)"
    parts = _group_leaf(key) if owner == "Trace" else None
    if parts is not None:
        group, leaf = parts
        if leaf in members.get(f"Trace{group.capitalize()}", set()):
            return f"`Trace.{group}.{leaf}`"
    found = sorted(reads.get(key, ()), key=lambda name: (_rank(owner, name), name))
    if not found:
        return "none (storage only)"
    shown = ", ".join(f"`{_public_spelling(name)}`" for name in found[:ACCESSOR_LIMIT])
    extra = len(found) - ACCESSOR_LIMIT
    return shown if extra <= 0 else f"{shown} and {extra} more"


def _rank(owner: str, accessor: str) -> int:
    """Order accessors: the owner class first, then its group views, then other classes."""

    class_name = accessor.split(".", 1)[0]
    if class_name == owner:
        return 0
    return 1 if class_name.startswith(owner) else 2


def _public_spelling(accessor: str) -> str:
    """Spell a ``Trace<Group>.member`` accessor as the public ``Trace.<group>.member``."""

    class_name, member = accessor.split(".", 1)
    for group in TRACE_GROUPS:
        if class_name == f"Trace{group.capitalize()}":
            return f"Trace.{group}.{member}"
    return accessor


def _written(key: str) -> str:
    """Render the save-level column for one stored key."""

    for flag, fields in FLAG_FIELDS.items():
        if key in fields:
            levels = FLAG_LEVELS[flag]
            where = "every level, opt-out" if levels == SAVE_LEVELS else ", ".join(levels)
            return f"optional: `{flag}` ({where})"
    return "required: every level"


def _short(value: str) -> str:
    """Return a SHA-256 digest shortened for a table."""

    return value[:16]


def _record_order(records: dict[str, Any]) -> list[str]:
    """Return record keys with the lead records first."""

    rest = sorted(key for key in records if key not in LEAD_RECORDS)
    return [key for key in LEAD_RECORDS if key in records] + rest


def _owner(record: dict[str, Any]) -> str:
    """Return the class short name of a record contract."""

    return str(record["class"]).rsplit(".", 1)[-1]


def _diff_rows(current: dict[str, Any], previous: dict[str, Any]) -> list[str]:
    """Render the previous-to-current persisted key diff, one row per changed key."""

    rows = []
    for key in sorted(set(current) | set(previous)):
        before = set(previous.get(key, {}).get("persisted_field_names", ()))
        after = set(current.get(key, {}).get("persisted_field_names", ()))
        changes = [(name, "removed") for name in before - after]
        changes += [(name, "added") for name in after - before]
        for name, change in sorted(changes):
            rows.append(f"| `{key}` | `{name}` | {change} |")
    return rows


def _envelope(contract: dict[str, Any]) -> list[str]:
    """Render the artifact-level facts the record tables do not carry."""

    lines = [
        "## Envelope",
        "",
        "A bundle or trace directory has no record contract of its own: its shape is the",
        "manifest plus the component specs below, and the record tables that follow.",
        "",
        "| manifest schema | digest |",
        "| --- | --- |",
    ]
    for name, digest in sorted(contract["manifest_schema_ids"].items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "Writer identity fields: "
        + ", ".join(f"`{name}`" for name in contract["writer_identity_fields"])
        + ".",
        "Payload codecs: " + ", ".join(f"`{name}`" for name in contract["payload_codec_ids"]) + ".",
        "",
        "| component | schema version |",
        "| --- | ---: |",
    ]
    for name, version in sorted(contract["component_schema_versions"].items()):
        lines.append(f"| `{name}` | {version} |")
    lines.append("")
    return lines


def _header(current: dict[str, Any], previous: dict[str, Any], total: int) -> list[str]:
    """Render the page header and column key."""

    contract = current["contract"]
    records = contract["record_contracts"]
    return [
        f"# Persisted fields of the saved format (tlspec {contract['tlspec_version']})",
        "",
        f"Generated by `{COMMAND}`; do not edit by hand. Rerun it after the writer contract,",
        "the storage bindings or a record class changes and commit the result;",
        "`tests/test_persisted_fields_doc.py` fails when this page and the generator disagree.",
        "",
        f"Contract of record: `torchlens/schemas/{CONTRACT.name}`, digest",
        f"`{current['writer_contract_digest']}`.",
        f"Previous contract: `torchlens/schemas/{PREVIOUS_CONTRACT.name}`, digest",
        f"`{previous['writer_contract_digest']}` (tlspec"
        f" {previous['contract']['tlspec_version']}).",
        f"Reader floor: tlspec {contract['min_tlspec_version']}.",
        "",
        f"{len(records)} record contracts, {total} persisted keys.",
        "",
        "## Columns",
        "",
        "- **key**: the stored spelling in the artifact; a leading underscore marks private",
        "  storage read through an accessor.",
        "- **type**: storage kind from `torchlens/_data_classes/_schema_bindings.py`, with the",
        "  public annotation where the class declares one.",
        "- **written**: `required` keys are written at every save level (`audit`,",
        "  `executable_with_callables`, `portable`, `runnable`); `optional` keys are written",
        "  only when their payload flag is on, with the levels whose defaults turn it on",
        "  (`torchlens/_io/save_policy.py`). A structure-only capture additionally drops buffer",
        "  value payloads at every level.",
        "- **policy**: the field-policy class (`keep` is written and read back, `blob` and",
        "  `blob_recursive` are written as payload blobs, `weakref_strip` is written with",
        "  weak references removed). `drop` keys are session-only and listed after each table.",
        "- **tier**: `fact` is captured, `derived` is computed from facts, `enrichment` is",
        "  added after capture.",
        "- **accessor**: the public spelling that reads the key: the field itself, a",
        "  `Trace.<group>.<leaf>` view member for a `_<group>_<leaf>` store, or the public",
        "  properties whose getters reach the store directly or through up to",
        f"  {ACCESSOR_DEPTH} private members (an `ast` scan of `torchlens/_data_classes/`, matched",
        "  by attribute name). `none (storage only)` means no public getter reaches it.",
        "",
    ]


def render(
    contract_path: Path = CONTRACT,
    previous_path: Path = PREVIOUS_CONTRACT,
    bindings_path: Path = BINDINGS,
    data_classes: Path = DATA_CLASSES,
) -> str:
    """Return the page text produced from the inputs."""

    current = _load(contract_path)
    previous = _load(previous_path)
    records = current["contract"]["record_contracts"]
    bindings = read_bindings(bindings_path)
    reads, members = read_accessors(data_classes)
    total = sum(len(record["persisted_field_names"]) for record in records.values())
    lines = _header(current, previous, total)
    lines += [
        *_envelope(current["contract"]),
        "## Records",
        "",
        "| record | class | persisted keys | session-only keys |",
        "| --- | --- | ---: | ---: |",
    ]
    for key in _record_order(records):
        record = records[key]
        dropped = sum(1 for policy in record["field_policies"].values() if policy == "drop")
        lines.append(
            f"| `{key}` | `{record['class']}` | {len(record['persisted_field_names'])} | {dropped} |"
        )
    lines += [
        "",
        f"## Key diff from tlspec {previous['contract']['tlspec_version']}",
        "",
        "Persisted keys removed or added between the previous contract and the contract of",
        "record, by record; a rename appears as one removed and one added row.",
        "",
        "| record | key | change |",
        "| --- | --- | --- |",
        *(_diff_rows(records, previous["contract"]["record_contracts"]) or ["| - | - | - |"]),
        "",
    ]
    storage_only: list[str] = []
    for key in _record_order(records):
        record = records[key]
        owner = _owner(record)
        binding = bindings.get(BINDING_KEY_ALIASES.get(key, key), {})
        persisted = sorted(record["persisted_field_names"])
        lines += [
            f"## `{key}` ({owner}, {len(persisted)} persisted keys)",
            "",
            "| key | type | written | policy | tier | accessor |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name in persisted:
            accessor = _accessor(owner, name, reads, members)
            if accessor.startswith("none"):
                storage_only.append(f"`{key}.{name}`")
            lines.append(
                f"| `{name}` | {_type_text(binding.get(name))} | {_written(name)} | "
                f"{record['field_policies'].get(name, '')} | "
                f"{record['field_tiers'].get(name, '')} | {accessor} |"
            )
        session_only = sorted(k for k, p in record["field_policies"].items() if p == "drop")
        if session_only:
            lines += ["", f"Session-only (`drop`): {', '.join(f'`{k}`' for k in session_only)}."]
        lines.append("")
    lines += _findings(records, members, storage_only)
    return "\n".join(lines).rstrip("\n") + "\n"


def _findings(
    records: dict[str, Any], members: dict[str, set[str]], storage_only: list[str]
) -> list[str]:
    """Render the two reconciliation lists: keys without accessors, accessors without keys."""

    trace_keys = set(records.get("trace", {}).get("persisted_field_names", ()))
    unkeyed = []
    for group in TRACE_GROUPS:
        for leaf in sorted(members.get(f"Trace{group.capitalize()}", ())):
            if f"_{group}_{leaf}" not in trace_keys:
                unkeyed.append(f"`Trace.{group}.{leaf}`")
    return [
        "## Private keys with no public accessor",
        "",
        f"{len(storage_only)} keys: " + (", ".join(storage_only) if storage_only else "none") + ".",
        "",
        "## Group view members with no stored key",
        "",
        "Members of the seven `Trace` group views whose name is not a persisted",
        "`_<group>_<leaf>` store (computed from other keys, or read from a store spelled",
        f"differently). {len(unkeyed)} members: "
        + (", ".join(unkeyed) if unkeyed else "none")
        + ".",
    ]


def main(argv: list[str] | None = None) -> int:
    """Write or check the page."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when the page is stale")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print(f"{OUTPUT.relative_to(REPO)} is stale; run {COMMAND}")
            return 1
        print(f"{OUTPUT.relative_to(REPO)} is current")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
