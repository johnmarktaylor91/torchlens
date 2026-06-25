"""Data-dictionary generation for the public menagerie CSV schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from menagerie.op_taxonomy import OP_TAXONOMY_VERSION
from menagerie.trace_summary import TRACE_SUMMARY_VERSION


SCHEMA_VERSION = "2.0"
DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / ".research" / "menagerie-csv-schema" / "SCHEMA_v2.md"
)


@dataclass(frozen=True)
class SchemaColumn:
    """One flagship schema column from ``SCHEMA_v2.md``.

    Parameters
    ----------
    index:
        One-based column position.
    name:
        CSV column name.
    type_unit:
        Type and unit text from the schema.
    description:
        Plain-language column meaning.
    availability:
        Availability tier.
    source:
        Source or operational provenance.
    priority:
        Schema priority.
    nullable:
        ``"Y"`` when empty values are allowed, otherwise ``"N"``.
    """

    index: int
    name: str
    type_unit: str
    description: str
    availability: str
    source: str
    priority: str
    nullable: str


_COLUMN_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_BOOLEAN_COLUMNS = {
    "is_trustworthy",
    "is_multimodal",
    "is_generative",
    "is_recurrent",
    "is_branching",
    "is_dynamic_graph",
    "has_attention",
    "has_conv",
    "has_residual",
    "has_embedding",
    "validated_on_current_release",
    "catalog_verified_hint",
}
_EXAMPLES = {
    "stable_id": "m8840",
    "display_name": "ResNet-18",
    "family_normalized": "ResNet",
    "variant": "",
    "zoo": "torchvision",
    "domain": "vision/classification-backbone",
    "task_primary": "image-classification",
    "modality_primary": "image",
    "architecture_paradigm": "CNN",
    "n_params": "11689512",
    "n_params_source": "traced",
    "publication_year": "2015",
    "era": "2015",
    "validation_status": "passed",
    "is_trustworthy": "1",
    "n_ops": "69",
    "family": "resnet",
    "model_scale_bucket": "10-100M",
    "top_op_types_json": '["conv2d","batch_norm","relu_"]',
    "graph_depth": "63",
    "pct_conv": "28.985507246376812",
    "has_conv": "1",
    "norm_type": "batch_norm",
    "activation_fn_type": "relu",
    "model_class_name": "ResNet",
    "structural_barcode_json": '{"graph_depth":63,"has_conv":true}',
    "total_flops_forward": "3628877824",
    "total_macs_forward": "1814438912",
    "input_shape": "(1,3,224,224)",
    "input_dtype": "float32",
    "reference_batch_size": "1",
    "trust_tier": "current_verified",
    "validated_on_current_release": "1",
    "catalog_verified_hint": "1",
}
_OPERATIONAL_DEFINITIONS = {
    "graph_depth": "graph_depth = max(op.max_distance_from_input) over compute ops (op hops).",
    "graph_max_width": (
        "graph_max_width = max count of compute ops sharing one op.max_distance_from_input level."
    ),
    "pct_conv": (
        f"pct_conv = 100 * (#ops classified conv by op_taxonomy v{OP_TAXONOMY_VERSION}) "
        "/ n_compute_ops."
    ),
    "pct_linear": (
        f"pct_linear = 100 * (#ops classified linear by op_taxonomy v{OP_TAXONOMY_VERSION}) "
        "/ n_compute_ops."
    ),
    "pct_attention": (
        f"pct_attention = 100 * (#ops classified attention by op_taxonomy "
        f"v{OP_TAXONOMY_VERSION}) / n_compute_ops."
    ),
    "pct_norm": (
        f"pct_norm = 100 * (#ops classified norm by op_taxonomy v{OP_TAXONOMY_VERSION}) "
        "/ n_compute_ops."
    ),
    "pct_elementwise": (
        f"pct_elementwise = 100 * (#ops classified elementwise by op_taxonomy "
        f"v{OP_TAXONOMY_VERSION}) / n_compute_ops."
    ),
    "is_trustworthy": (
        "is_trustworthy = forward_pass=1 AND metadata_ok=1 AND n_ops NOT NULL AND "
        "graph_shape_hash NOT NULL on a compatible torchlens version."
    ),
    "validated_on_current_release": (
        "validated_on_current_release = is_trustworthy AND torchlens_version = <current> "
        "AND recipe hash matches the current catalog row."
    ),
    "model_scale_bucket": (
        "model_scale_bucket = bucketed n_params with edges <1M, 1-10M, 10-100M, 100M-1B, and 1B+."
    ),
    "has_residual": (
        "has_residual = exists an add-op whose parents include a skip-connected ancestor "
        "(topology)."
    ),
    "has_attention": "has_attention = pct_attention > 0.",
    "has_conv": "has_conv = pct_conv > 0.",
    "reference_batch_size": "reference_batch_size = first dimension of the primary tensor input.",
    "dedup_architecture_key": "dedup_architecture_key = trace-summary graph_shape_hash when present.",
}


def _strip_markdown(value: str) -> str:
    """Normalize inline Markdown used in schema cells.

    Parameters
    ----------
    value:
        Raw Markdown table cell.

    Returns
    -------
    str
        Human-readable cell text.
    """

    stripped = value.strip()
    stripped = _BACKTICK_RE.sub(r"\1", stripped)
    return re.sub(r"\s+", " ", stripped)


def parse_flagship_schema(schema_path: Path = DEFAULT_SCHEMA_PATH) -> list[SchemaColumn]:
    """Parse flagship column metadata from ``SCHEMA_v2.md``.

    Parameters
    ----------
    schema_path:
        Path to the authoritative schema Markdown.

    Returns
    -------
    list[SchemaColumn]
        Ordered flagship schema columns.

    Raises
    ------
    ValueError
        If the schema does not contain exactly 78 flagship columns.
    """

    columns: list[SchemaColumn] = []
    for line in schema_path.read_text(encoding="utf-8").splitlines():
        if _COLUMN_ROW_RE.match(line) is None:
            continue
        cells = [_strip_markdown(cell) for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8:
            continue
        columns.append(
            SchemaColumn(
                index=int(cells[0]),
                name=cells[1],
                type_unit=cells[2],
                description=cells[3],
                availability=cells[4],
                source=cells[5],
                priority=cells[6],
                nullable=cells[7],
            )
        )
    if len(columns) != 78:
        raise ValueError(f"Expected 78 flagship columns in {schema_path}, found {len(columns)}")
    expected = list(range(1, 79))
    observed = [column.index for column in columns]
    if observed != expected:
        raise ValueError(f"Flagship column positions are not contiguous: {observed}")
    return columns


def _nullability_text(column: SchemaColumn) -> str:
    """Return public nullability wording for a schema column.

    Parameters
    ----------
    column:
        Schema column.

    Returns
    -------
    str
        Nullability explanation.
    """

    if column.nullable == "N":
        return "Not nullable; every row must carry a value."
    return "Nullable; an empty CSV cell means unknown, absent, or not yet exported."


def _boolean_text(column: SchemaColumn) -> str:
    """Return boolean encoding wording for a schema column.

    Parameters
    ----------
    column:
        Schema column.

    Returns
    -------
    str
        Boolean encoding text, or an empty string for non-booleans.
    """

    if column.name not in _BOOLEAN_COLUMNS:
        return ""
    empty = "empty = unknown/not determined" if column.nullable == "Y" else "empty is invalid"
    return f"Boolean encoding: 1 = true, 0 = false, {empty}."


def _example_for(column: SchemaColumn) -> str:
    """Return an example value for a schema column.

    Parameters
    ----------
    column:
        Schema column.

    Returns
    -------
    str
        Example value text.
    """

    if column.name in _EXAMPLES:
        return _EXAMPLES[column.name]
    if column.type_unit.startswith("int"):
        return "123"
    if column.type_unit.startswith("float"):
        return "12.5"
    if "JSON array" in column.type_unit:
        return '["conv2d","relu"]'
    if "JSON object" in column.type_unit:
        return '{"has_conv":true}'
    if column.type_unit.startswith("bool"):
        return "1"
    return "example"


def _column_entry(column: SchemaColumn) -> str:
    """Render one Markdown dictionary entry.

    Parameters
    ----------
    column:
        Schema column.

    Returns
    -------
    str
        Markdown entry.
    """

    lines = [
        f"## {column.index}. `{column.name}`",
        f"- Type/unit: {column.type_unit}",
        f"- Description: {column.description}",
        f"- Availability: {column.availability}",
        f"- Source: {column.source}",
        f"- Nullability: {_nullability_text(column)}",
        f"- Example: `{_example_for(column)}`",
    ]
    boolean_text = _boolean_text(column)
    if boolean_text:
        lines.append(f"- Boolean semantics: {boolean_text}")
    definition = _OPERATIONAL_DEFINITIONS.get(column.name)
    if definition:
        lines.append(f"- Operational definition: {definition}")
    return "\n".join(lines)


def render_data_dictionary(columns: Iterable[SchemaColumn]) -> str:
    """Render the public Markdown data dictionary.

    Parameters
    ----------
    columns:
        Ordered flagship schema columns.

    Returns
    -------
    str
        Markdown data dictionary.
    """

    entries = [_column_entry(column) for column in columns]
    header = [
        "# Menagerie CSV Data Dictionary",
        "",
        f"- Schema version: {SCHEMA_VERSION}",
        f"- Trace-summary version: {TRACE_SUMMARY_VERSION}",
        f"- Op-taxonomy version: {OP_TAXONOMY_VERSION}",
        "- Join key: `stable_id` is the primary key of `menagerie.csv` and the foreign key of "
        "every side-table.",
        "- Side-table cardinality: `trace_metrics.parquet`, `trace_histograms.jsonl`, "
        "`papers.csv`, and `artifacts.csv` are 1:1 left joins by `stable_id`; an absent "
        "histogram/paper row means no retrace or paper curation is available. `lineage.csv` "
        "is 1:N by `stable_id`; an absent row means no curated lineage is available.",
        "",
    ]
    return "\n".join([*header, *entries, ""])


def write_data_dictionary(
    output_path: Path,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> Path:
    """Write the public Markdown data dictionary.

    Parameters
    ----------
    output_path:
        Destination Markdown path.
    schema_path:
        Path to the authoritative schema Markdown.

    Returns
    -------
    Path
        Written output path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_data_dictionary(parse_flagship_schema(schema_path)),
        encoding="utf-8",
    )
    return output_path
