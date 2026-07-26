"""Typed fail-closed loading and deterministic expansion for conformance catalogs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


COMPACT_SCHEMA_VERSION = "menagerie.crawler.round21-conformance.compact.v1"
EXPANDED_SCHEMA_VERSION = "menagerie.crawler.round21-conformance.v1"
DEFAULT_CATALOG_PATH = Path(__file__).with_name("conformance-round21.json")

_ROOT_FIELDS = frozenset(
    {
        "aliases",
        "expanded_envelope",
        "record_defaults",
        "record_fields",
        "records",
        "schema_version",
    }
)
_ALIAS_FIELDS = frozenset({"hosts", "nodes", "source_locators"})
_EXPANDED_ENVELOPE = {
    "no_waivers": True,
    "schema_version": EXPANDED_SCHEMA_VERSION,
    "status": "complete",
}
_RECORD_DEFAULTS = {
    "expected_outcome": "passed",
    "real_prefix": True,
    "shipped_compiler": True,
}
_RECORD_FIELDS = (
    "clause_id",
    "source_locator_alias",
    "invariant_ids",
    "finding_ids",
    "real_node_aliases",
    "structural_node_aliases",
    "host_alias",
    "deliberate_reversion_ids",
)
_ALLOWED_HOSTS = frozenset({"both", "linux", "macos"})


class ConformanceErrorCode(str, Enum):
    """Stable error categories emitted by the compact catalog loader."""

    DUPLICATE_KEY = "duplicate_key"
    DUPLICATE_VALUE = "duplicate_value"
    INVALID_JSON = "invalid_json"
    INVALID_SHAPE = "invalid_shape"
    INVALID_VALUE = "invalid_value"
    MISSING_FIELD = "missing_field"
    UNDEFINED_ALIAS = "undefined_alias"
    UNUSED_ALIAS = "unused_alias"


class ConformanceCatalogError(ValueError):
    """A typed, fail-closed compact conformance catalog error."""

    def __init__(self, code: ConformanceErrorCode, message: str) -> None:
        """Initialize a catalog error.

        Parameters
        ----------
        code:
            Stable machine-readable error category.
        message:
            Human-readable diagnostic detail.
        """

        self.code = code
        super().__init__(f"{code.value}: {message}")


@dataclass(frozen=True)
class ConformanceRecord:
    """One fully expanded conformance clause record."""

    clause_id: str
    source_locator: str
    invariant_ids: tuple[str, ...]
    finding_ids: tuple[str, ...]
    real_node_ids: tuple[str, ...]
    structural_node_ids: tuple[str, ...]
    host: str
    deliberate_reversion_ids: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical expanded JSON mapping."""

        return {
            "clause_id": self.clause_id,
            "deliberate_reversion_ids": list(self.deliberate_reversion_ids),
            "expected_outcome": _RECORD_DEFAULTS["expected_outcome"],
            "finding_ids": list(self.finding_ids),
            "host": self.host,
            "invariant_ids": list(self.invariant_ids),
            "real_node_ids": list(self.real_node_ids),
            "real_prefix": _RECORD_DEFAULTS["real_prefix"],
            "shipped_compiler": _RECORD_DEFAULTS["shipped_compiler"],
            "source_locator": self.source_locator,
            "structural_node_ids": list(self.structural_node_ids),
        }


@dataclass(frozen=True)
class ConformanceRegistry:
    """A complete validated expanded conformance registry."""

    records: tuple[ConformanceRecord, ...]

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical expanded registry envelope."""

        return {
            "no_waivers": _EXPANDED_ENVELOPE["no_waivers"],
            "records": [record.to_mapping() for record in self.records],
            "schema_version": _EXPANDED_ENVELOPE["schema_version"],
            "status": _EXPANDED_ENVELOPE["status"],
        }

    def to_json_bytes(self) -> bytes:
        """Return deterministic expanded JSON bytes."""

        rendered = json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"
        return rendered.encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate keys.

    Parameters
    ----------
    pairs:
        Ordered key-value pairs supplied by the JSON decoder.

    Returns
    -------
    dict[str, object]
        The unique-key JSON object.

    Raises
    ------
    ConformanceCatalogError
        If a key occurs more than once.
    """

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConformanceCatalogError(
                ConformanceErrorCode.DUPLICATE_KEY,
                f"duplicate JSON object key {key!r}",
            )
        result[key] = value
    return result


def _require_mapping(value: object, context: str) -> dict[str, object]:
    """Require and return a JSON object.

    Parameters
    ----------
    value:
        Candidate decoded JSON value.
    context:
        Diagnostic location for failures.

    Returns
    -------
    dict[str, object]
        The validated JSON object.

    Raises
    ------
    ConformanceCatalogError
        If ``value`` is not an object.
    """

    if not isinstance(value, dict):
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_SHAPE,
            f"{context} must be an object",
        )
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    """Require an object to contain exactly the declared fields.

    Parameters
    ----------
    value:
        Candidate object.
    expected:
        Required closed field set.
    context:
        Diagnostic location for failures.

    Raises
    ------
    ConformanceCatalogError
        If a required field is missing or an unknown field is present.
    """

    actual = set(value)
    missing = sorted(expected - actual)
    if missing:
        raise ConformanceCatalogError(
            ConformanceErrorCode.MISSING_FIELD,
            f"{context} is missing fields {missing!r}",
        )
    extra = sorted(actual - expected)
    if extra:
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_SHAPE,
            f"{context} has unknown fields {extra!r}",
        )


def _require_string_list(
    value: object,
    context: str,
    *,
    allow_empty: bool,
    require_sorted: bool = True,
) -> tuple[str, ...]:
    """Require a sorted list of unique non-empty strings.

    Parameters
    ----------
    value:
        Candidate decoded JSON value.
    context:
        Diagnostic location for failures.
    allow_empty:
        Whether an empty list is valid.
    require_sorted:
        Whether values must use lexical order.

    Returns
    -------
    tuple[str, ...]
        The validated immutable values.

    Raises
    ------
    ConformanceCatalogError
        If the value has the wrong shape, contains invalid strings, or is not canonical.
    """

    if not isinstance(value, list):
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_SHAPE,
            f"{context} must be an array",
        )
    if not allow_empty and not value:
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            f"{context} must not be empty",
        )
    if any(not isinstance(item, str) or not item for item in value):
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            f"{context} must contain only non-empty strings",
        )
    strings = tuple(value)
    if len(strings) != len(set(strings)):
        raise ConformanceCatalogError(
            ConformanceErrorCode.DUPLICATE_VALUE,
            f"{context} contains duplicate values",
        )
    if require_sorted and strings != tuple(sorted(strings)):
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            f"{context} must be sorted",
        )
    return strings


def _require_alias_table(value: object, context: str) -> dict[str, str]:
    """Require a sorted, bijective string alias table.

    Parameters
    ----------
    value:
        Candidate alias-table value.
    context:
        Diagnostic location for failures.

    Returns
    -------
    dict[str, str]
        The validated alias-to-value mapping.

    Raises
    ------
    ConformanceCatalogError
        If the alias table is empty, malformed, unordered, or non-bijective.
    """

    mapping = _require_mapping(value, context)
    if not mapping:
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            f"{context} must not be empty",
        )
    if list(mapping) != sorted(mapping):
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            f"{context} aliases must be sorted",
        )
    if any(not key or not isinstance(item, str) or not item for key, item in mapping.items()):
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            f"{context} must map non-empty aliases to non-empty strings",
        )
    aliases = {key: item for key, item in mapping.items() if isinstance(item, str)}
    if len(aliases) != len(set(aliases.values())):
        raise ConformanceCatalogError(
            ConformanceErrorCode.DUPLICATE_VALUE,
            f"{context} contains duplicate expanded values",
        )
    return aliases


def _resolve_aliases(
    aliases: tuple[str, ...],
    table: Mapping[str, str],
    context: str,
) -> tuple[str, ...]:
    """Resolve one validated alias sequence.

    Parameters
    ----------
    aliases:
        Alias names to resolve.
    table:
        Validated alias table.
    context:
        Diagnostic location for failures.

    Returns
    -------
    tuple[str, ...]
        Expanded values in catalog order.

    Raises
    ------
    ConformanceCatalogError
        If an alias is undefined.
    """

    missing = sorted(set(aliases) - set(table))
    if missing:
        raise ConformanceCatalogError(
            ConformanceErrorCode.UNDEFINED_ALIAS,
            f"{context} references undefined aliases {missing!r}",
        )
    return tuple(table[alias] for alias in aliases)


def _require_literal_mapping(
    value: object,
    expected: Mapping[str, object],
    context: str,
) -> None:
    """Require a mapping to equal one fixed contract mapping.

    Parameters
    ----------
    value:
        Candidate decoded JSON value.
    expected:
        Required literal mapping.
    context:
        Diagnostic location for failures.

    Raises
    ------
    ConformanceCatalogError
        If a field is missing or the mapping differs from the fixed contract.
    """

    mapping = _require_mapping(value, context)
    _require_exact_fields(mapping, frozenset(expected), context)
    if mapping != expected:
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            f"{context} does not match the compact schema contract",
        )


def _parse_record(
    value: object,
    index: int,
    *,
    hosts: Mapping[str, str],
    nodes: Mapping[str, str],
    source_locators: Mapping[str, str],
) -> tuple[ConformanceRecord, str, tuple[str, ...], str]:
    """Parse and expand one compact record row.

    Parameters
    ----------
    value:
        Candidate compact row.
    index:
        Zero-based row index.
    hosts:
        Validated host aliases.
    nodes:
        Validated pytest-node aliases.
    source_locators:
        Validated source-locator aliases.

    Returns
    -------
    tuple[ConformanceRecord, str, tuple[str, ...], str]
        Expanded record and the aliases consumed by the row.

    Raises
    ------
    ConformanceCatalogError
        If the row is incomplete, malformed, or references an undefined alias.
    """

    context = f"records[{index}]"
    if not isinstance(value, list) or len(value) != len(_RECORD_FIELDS):
        raise ConformanceCatalogError(
            ConformanceErrorCode.MISSING_FIELD
            if isinstance(value, list) and len(value) < len(_RECORD_FIELDS)
            else ConformanceErrorCode.INVALID_SHAPE,
            f"{context} must contain exactly {len(_RECORD_FIELDS)} columns",
        )
    (
        clause_id_value,
        locator_alias_value,
        invariant_ids_value,
        finding_ids_value,
        real_aliases_value,
        structural_aliases_value,
        host_alias_value,
        reversion_ids_value,
    ) = value
    if not isinstance(clause_id_value, str) or not clause_id_value:
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            f"{context}.clause_id must be a non-empty string",
        )
    if not isinstance(locator_alias_value, str) or not locator_alias_value:
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            f"{context}.source_locator_alias must be a non-empty string",
        )
    if not isinstance(host_alias_value, str) or not host_alias_value:
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            f"{context}.host_alias must be a non-empty string",
        )
    if locator_alias_value not in source_locators:
        raise ConformanceCatalogError(
            ConformanceErrorCode.UNDEFINED_ALIAS,
            f"{context} references undefined source locator {locator_alias_value!r}",
        )
    if host_alias_value not in hosts:
        raise ConformanceCatalogError(
            ConformanceErrorCode.UNDEFINED_ALIAS,
            f"{context} references undefined host {host_alias_value!r}",
        )
    invariant_ids = _require_string_list(
        invariant_ids_value,
        f"{context}.invariant_ids",
        allow_empty=True,
    )
    finding_ids = _require_string_list(
        finding_ids_value,
        f"{context}.finding_ids",
        allow_empty=True,
    )
    real_aliases = _require_string_list(
        real_aliases_value,
        f"{context}.real_node_aliases",
        allow_empty=False,
    )
    structural_aliases = _require_string_list(
        structural_aliases_value,
        f"{context}.structural_node_aliases",
        allow_empty=True,
    )
    reversion_ids = _require_string_list(
        reversion_ids_value,
        f"{context}.deliberate_reversion_ids",
        allow_empty=True,
    )
    real_nodes = _resolve_aliases(real_aliases, nodes, f"{context}.real_node_aliases")
    structural_nodes = _resolve_aliases(
        structural_aliases,
        nodes,
        f"{context}.structural_node_aliases",
    )
    record = ConformanceRecord(
        clause_id=clause_id_value,
        source_locator=source_locators[locator_alias_value],
        invariant_ids=invariant_ids,
        finding_ids=finding_ids,
        real_node_ids=real_nodes,
        structural_node_ids=structural_nodes,
        host=hosts[host_alias_value],
        deliberate_reversion_ids=reversion_ids,
    )
    return record, locator_alias_value, real_aliases + structural_aliases, host_alias_value


def load_conformance_catalog(path: Path = DEFAULT_CATALOG_PATH) -> ConformanceRegistry:
    """Load a compact catalog completely or raise a typed error.

    The loader validates the closed schema, all row fields, uniqueness and canonical
    ordering, every alias reference, and complete alias use before returning any records.

    Parameters
    ----------
    path:
        Compact catalog path.

    Returns
    -------
    ConformanceRegistry
        Fully validated and expanded immutable registry.

    Raises
    ------
    ConformanceCatalogError
        If any part of the catalog is malformed, partial, duplicated, or inconsistent.
    """

    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ConformanceCatalogError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_JSON,
            f"cannot decode {path}: {exc}",
        ) from exc

    root = _require_mapping(raw, "catalog")
    _require_exact_fields(root, _ROOT_FIELDS, "catalog")
    if root["schema_version"] != COMPACT_SCHEMA_VERSION:
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            "catalog.schema_version is unsupported",
        )
    _require_literal_mapping(
        root["expanded_envelope"],
        _EXPANDED_ENVELOPE,
        "catalog.expanded_envelope",
    )
    _require_literal_mapping(
        root["record_defaults"],
        _RECORD_DEFAULTS,
        "catalog.record_defaults",
    )
    record_fields = _require_string_list(
        root["record_fields"],
        "catalog.record_fields",
        allow_empty=False,
        require_sorted=False,
    )
    if record_fields != _RECORD_FIELDS:
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            "catalog.record_fields does not match the positional row contract",
        )

    alias_root = _require_mapping(root["aliases"], "catalog.aliases")
    _require_exact_fields(alias_root, _ALIAS_FIELDS, "catalog.aliases")
    hosts = _require_alias_table(alias_root["hosts"], "catalog.aliases.hosts")
    nodes = _require_alias_table(alias_root["nodes"], "catalog.aliases.nodes")
    source_locators = _require_alias_table(
        alias_root["source_locators"],
        "catalog.aliases.source_locators",
    )
    if any(host not in _ALLOWED_HOSTS for host in hosts.values()):
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            "catalog.aliases.hosts contains an unsupported host",
        )

    rows = root["records"]
    if not isinstance(rows, list) or not rows:
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_SHAPE,
            "catalog.records must be a non-empty array",
        )
    records: list[ConformanceRecord] = []
    locator_uses: set[str] = set()
    node_uses: set[str] = set()
    host_uses: set[str] = set()
    for index, row in enumerate(rows):
        record, locator_alias, node_aliases, host_alias = _parse_record(
            row,
            index,
            hosts=hosts,
            nodes=nodes,
            source_locators=source_locators,
        )
        records.append(record)
        locator_uses.add(locator_alias)
        node_uses.update(node_aliases)
        host_uses.add(host_alias)

    clause_ids = tuple(record.clause_id for record in records)
    if len(clause_ids) != len(set(clause_ids)):
        raise ConformanceCatalogError(
            ConformanceErrorCode.DUPLICATE_VALUE,
            "catalog.records contains duplicate clause IDs",
        )
    if clause_ids != tuple(sorted(clause_ids)):
        raise ConformanceCatalogError(
            ConformanceErrorCode.INVALID_VALUE,
            "catalog.records must be sorted by clause ID",
        )
    for aliases, uses, context in (
        (hosts, host_uses, "catalog.aliases.hosts"),
        (nodes, node_uses, "catalog.aliases.nodes"),
        (source_locators, locator_uses, "catalog.aliases.source_locators"),
    ):
        unused = sorted(set(aliases) - uses)
        if unused:
            raise ConformanceCatalogError(
                ConformanceErrorCode.UNUSED_ALIAS,
                f"{context} contains unused aliases {unused!r}",
            )
    return ConformanceRegistry(tuple(records))


def expand_conformance_catalog(
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    output_path: Path | None = None,
) -> bytes:
    """Expand a compact catalog to deterministic canonical JSON bytes.

    Parameters
    ----------
    catalog_path:
        Compact catalog to validate and expand.
    output_path:
        Optional destination. When omitted, the bytes are returned without writing.

    Returns
    -------
    bytes
        Canonical expanded registry bytes.

    Raises
    ------
    ConformanceCatalogError
        If the entire compact catalog does not validate.
    """

    expanded = load_conformance_catalog(catalog_path).to_json_bytes()
    if output_path is not None:
        output_path.write_bytes(expanded)
    return expanded


def _build_parser() -> argparse.ArgumentParser:
    """Build the deterministic catalog-expansion command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help="compact catalog to validate and expand",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="expanded JSON destination; stdout when omitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic conformance expansion command.

    Parameters
    ----------
    argv:
        Optional command-line arguments excluding the program name.

    Returns
    -------
    int
        Process exit status.
    """

    args = _build_parser().parse_args(argv)
    expanded = expand_conformance_catalog(args.catalog, args.output)
    if args.output is None:
        sys.stdout.buffer.write(expanded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
