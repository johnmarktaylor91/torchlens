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


@dataclass(frozen=True)
class SideTableColumn:
    """One public side-table schema column.

    Parameters
    ----------
    table:
        Side-table file name.
    name:
        Column name.
    type_unit:
        Public type and unit text.
    description:
        Plain-language column meaning.
    availability:
        Availability tier.
    source:
        Source or operational provenance.
    nullable:
        ``"Y"`` when empty values are allowed, otherwise ``"N"``.
    example:
        Example value text.
    operational_definition:
        Exact formula or predicate for computed columns.
    """

    table: str
    name: str
    type_unit: str
    description: str
    availability: str
    source: str
    nullable: str
    example: str = "example"
    operational_definition: str = ""


_COLUMN_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
TRACE_METRICS_COLUMNS: tuple[str, ...] = (
    "stable_id",
    "n_compute_ops",
    "n_unique_op_types",
    "n_inplace_ops",
    "has_custom_op",
    "dominant_op_type",
    "pct_conv",
    "pct_linear",
    "pct_attention",
    "pct_norm",
    "pct_elementwise",
    "pct_reduction",
    "pct_reshape",
    "pct_embedding",
    "pct_pooling",
    "graph_depth",
    "graph_max_width",
    "branching_factor",
    "max_fan_out",
    "max_fan_in",
    "is_branching",
    "is_recurrent",
    "max_recurrence_iters",
    "n_recurrent_layers",
    "has_conditional_branching",
    "n_conditionals",
    "is_dynamic_graph",
    "n_modules",
    "n_module_calls",
    "module_max_depth",
    "n_top_level_modules",
    "n_unique_module_types",
    "model_class_name",
    "model_class_qualname",
    "n_params",
    "n_params_trainable",
    "n_params_frozen",
    "n_param_tensors",
    "param_memory_bytes",
    "primary_param_dtype",
    "param_dtype_set_json",
    "quantized_param_tensor_count",
    "has_frozen_params",
    "n_buffers",
    "buffer_memory_bytes",
    "buffer_overwrite_count",
    "total_flops_forward",
    "total_macs_forward",
    "total_flops_backward",
    "total_macs_backward",
    "flops_coverage_pct",
    "n_unknown_flops_ops",
    "activation_memory_bytes",
    "forward_peak_memory_bytes",
    "largest_activation_bytes",
    "param_memory_mb",
    "activation_memory_mb",
    "forward_peak_memory_mb",
)
TRACE_HISTOGRAM_COLUMNS: tuple[str, ...] = (
    "stable_id",
    "op_type_histogram",
    "module_type_histogram",
    "top_module_types_json",
    "top_level_block_sequence_json",
    "flops_by_op_type",
    "macs_by_op_type",
    "structural_barcode",
)
PAPERS_COLUMNS: tuple[str, ...] = (
    "stable_id",
    "paper_title",
    "paper_url",
    "arxiv_id",
    "doi",
    "publication_venue",
    "venue_type",
    "authors_json",
    "first_author",
    "first_preprint_date",
    "first_publication_date",
    "huggingface_id",
    "model_card_url",
    "semantic_scholar_paper_id",
    "n_citations",
    "citations_snapshot_date",
    "citation_tier",
    "is_milestone",
)
LINEAGE_COLUMNS: tuple[str, ...] = (
    "stable_id",
    "relation",
    "related_stable_id",
    "related_name",
    "variant_dimension",
    "is_canonical_variant",
    "source_note",
)
ARTIFACTS_COLUMNS: tuple[str, ...] = (
    "stable_id",
    "model_page_url",
    "svg_path",
    "svg_url",
    "svg_sha256",
    "has_svg",
    "tlspec_path",
    "model_card_url",
    "added_wave",
    "row_quality_flags_json",
    "dataset_schema_version",
    "dataset_as_of_date",
    "csv_generation_commit_sha",
    "op_taxonomy_version",
)
SIDE_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "trace_metrics.parquet": TRACE_METRICS_COLUMNS,
    "trace_histograms.jsonl": TRACE_HISTOGRAM_COLUMNS,
    "papers.csv": PAPERS_COLUMNS,
    "lineage.csv": LINEAGE_COLUMNS,
    "artifacts.csv": ARTIFACTS_COLUMNS,
}
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
_SIDE_BOOLEAN_COLUMNS = {
    "has_custom_op",
    "is_branching",
    "is_recurrent",
    "has_conditional_branching",
    "is_dynamic_graph",
    "has_frozen_params",
    "is_milestone",
    "is_canonical_variant",
    "has_svg",
}
_SIDE_INT_COLUMNS = {
    "n_compute_ops",
    "n_unique_op_types",
    "n_inplace_ops",
    "max_fan_out",
    "max_fan_in",
    "max_recurrence_iters",
    "n_recurrent_layers",
    "n_conditionals",
    "n_modules",
    "n_module_calls",
    "n_top_level_modules",
    "n_unique_module_types",
    "n_params",
    "n_params_trainable",
    "n_params_frozen",
    "n_param_tensors",
    "quantized_param_tensor_count",
    "n_buffers",
    "buffer_overwrite_count",
    "total_flops_forward",
    "total_macs_forward",
    "total_flops_backward",
    "total_macs_backward",
    "n_unknown_flops_ops",
    "n_output_tensors",
    "n_citations",
    "graph_depth",
    "graph_max_width",
    "module_max_depth",
}
_SIDE_BYTE_COLUMNS = {
    "param_memory_bytes",
    "buffer_memory_bytes",
    "activation_memory_bytes",
    "forward_peak_memory_bytes",
    "largest_activation_bytes",
}
_SIDE_FLOAT_COLUMNS = {
    "pct_conv",
    "pct_linear",
    "pct_attention",
    "pct_norm",
    "pct_elementwise",
    "pct_reduction",
    "pct_reshape",
    "pct_embedding",
    "pct_pooling",
    "branching_factor",
    "flops_coverage_pct",
    "param_memory_mb",
    "activation_memory_mb",
    "forward_peak_memory_mb",
}
_SIDE_JSON_COLUMNS = {
    "param_dtype_set_json",
    "op_type_histogram",
    "module_type_histogram",
    "top_module_types_json",
    "top_level_block_sequence_json",
    "flops_by_op_type",
    "macs_by_op_type",
    "structural_barcode",
    "authors_json",
    "row_quality_flags_json",
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
    "display_name": "display_name = catalog name with underscores replaced by spaces.",
    "size_variant": (
        "size_variant = first natural size token parsed from catalog name + variant; tokens are "
        "nano/tiny/small/base/large/xl/xxl/huge/giant/mini/micro."
    ),
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
    "install_difficulty": (
        "install_difficulty = runtime dependency classification from menagerie.runtime: core for "
        "base-environment rows, pip for rows with pip-installable dependency packages, "
        "source-build for runtime-classified local/source recipes, unavailable for "
        "runtime-classified unsupported rows, and empty when runtime classification is unavailable."
    ),
    "trust_tier": (
        "trust_tier = current_verified when the latest trustworthy ledger row matches current "
        "TorchLens and recipe; stale_recipe when trustworthy but not current; legacy_seed when only "
        "the catalog verified hint exists; otherwise failed/deferred/unsupported/not_verified from "
        "the latest ledger status."
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


def _nullability_text(column: SchemaColumn | SideTableColumn) -> str:
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


def _side_type_unit(column: str) -> str:
    """Return public type text for one side-table column.

    Parameters
    ----------
    column:
        Side-table column name.

    Returns
    -------
    str
        Type and unit text.
    """

    if column in _SIDE_BOOLEAN_COLUMNS:
        return "bool"
    if column in _SIDE_BYTE_COLUMNS:
        return "int (bytes)"
    if column in _SIDE_INT_COLUMNS:
        return "int"
    if column in _SIDE_FLOAT_COLUMNS:
        if column.startswith("pct_") or column == "flops_coverage_pct":
            return "float (0-100)"
        return "float"
    if column in _SIDE_JSON_COLUMNS:
        return (
            "JSON object str" if "histogram" in column or column.endswith("_type") else "JSON str"
        )
    return "str"


def _side_description(table: str, column: str) -> str:
    """Return a concise public description for a side-table column.

    Parameters
    ----------
    table:
        Side-table file name.
    column:
        Side-table column name.

    Returns
    -------
    str
        Plain-language description.
    """

    if column == "stable_id":
        return f"Foreign key joining `{table}` to the flagship `menagerie.csv` row."
    descriptions = {
        "op_taxonomy_version": "Version of the deterministic op-taxonomy classifier used.",
        "model_page_url": "Public model page URL when a website page exists.",
        "svg_path": "Relative artifact path for the rendered SVG graph.",
        "svg_url": "Public URL for the rendered SVG graph.",
        "svg_sha256": "SHA-256 hash of the rendered SVG graph.",
        "has_svg": "Whether an SVG hash/path is available for this model.",
        "tlspec_path": "Relative portable trace artifact path when exported.",
        "model_card_url": "External model card URL when curated.",
        "added_wave": "Catalog discovery/import wave that added the row.",
        "row_quality_flags_json": "JSON array of non-authoritative quality flags for the row.",
        "dataset_schema_version": "Public dataset schema version used for this export.",
        "dataset_as_of_date": "UTC date when the dataset export was generated.",
        "csv_generation_commit_sha": "Git commit hash used to generate the CSV export.",
    }
    if column in descriptions:
        return descriptions[column]
    return column.replace("_", " ")


def _side_availability(table: str) -> str:
    """Return side-table availability tier.

    Parameters
    ----------
    table:
        Side-table file name.

    Returns
    -------
    str
        Availability tier.
    """

    if table in {"trace_metrics.parquet", "trace_histograms.jsonl"}:
        return "retrace_required"
    if table in {"papers.csv", "lineage.csv"}:
        return "external"
    return "persisted (computed)"


def _side_source(table: str, column: str) -> str:
    """Return side-table source text.

    Parameters
    ----------
    table:
        Side-table file name.
    column:
        Side-table column name.

    Returns
    -------
    str
        Source or provenance text.
    """

    if column == "stable_id":
        return "catalog"
    if table in {"trace_metrics.parquet", "trace_histograms.jsonl"}:
        return "trace_summary"
    if table == "artifacts.csv":
        if column == "op_taxonomy_version":
            return "menagerie.op_taxonomy"
        if column in {"svg_sha256"}:
            return "ledger"
        if column == "added_wave":
            return "catalog added_wave"
        return "computed"
    return "external/curated"


def _side_example(column: str) -> str:
    """Return an example value for one side-table column.

    Parameters
    ----------
    column:
        Side-table column name.

    Returns
    -------
    str
        Example value text.
    """

    if column == "stable_id":
        return "m8840"
    if column.endswith("_sha256"):
        return "0" * 64
    if column.endswith("_json") or column in _SIDE_JSON_COLUMNS:
        return '{"example":1}'
    if column in _SIDE_BOOLEAN_COLUMNS:
        return "1"
    if column in _SIDE_INT_COLUMNS or column in _SIDE_BYTE_COLUMNS:
        return "123"
    if column in _SIDE_FLOAT_COLUMNS:
        return "12.5"
    return "example"


def _side_operational_definition(table: str, column: str) -> str:
    """Return an operational definition for a side-table column when needed.

    Parameters
    ----------
    table:
        Side-table file name.
    column:
        Side-table column name.

    Returns
    -------
    str
        Operational definition or an empty string.
    """

    definitions = {
        "pct_conv": (
            f"pct_conv = 100 * (#ops classified conv by op_taxonomy v{OP_TAXONOMY_VERSION}) "
            "/ n_compute_ops."
        ),
        "graph_depth": "graph_depth = max(op.max_distance_from_input) over compute ops.",
        "graph_max_width": (
            "graph_max_width = max count of compute ops sharing one "
            "op.max_distance_from_input level."
        ),
        "module_max_depth": "module_max_depth = max(op.module_call_depth) over trace ops.",
        "has_svg": "has_svg = svg_sha256 is present.",
        "svg_url": "svg_url = svg_path plus a sha256 query token; empty when no SVG hash exists.",
        "added_wave": "added_wave = matching typed catalog record added_wave; empty when absent.",
        "op_taxonomy_version": (
            f"op_taxonomy_version = menagerie.op_taxonomy.OP_TAXONOMY_VERSION "
            f"({OP_TAXONOMY_VERSION})."
        ),
    }
    if table == "trace_metrics.parquet" and column.startswith("pct_"):
        category = column.removeprefix("pct_")
        return (
            f"{column} = 100 * (#ops classified {category} by op_taxonomy "
            f"v{OP_TAXONOMY_VERSION}) / n_compute_ops."
        )
    return definitions.get(column, "")


def side_table_schema_columns() -> list[SideTableColumn]:
    """Return dictionary entries for every emitted side-table column.

    Returns
    -------
    list[SideTableColumn]
        Side-table dictionary columns in emitted table order.
    """

    columns: list[SideTableColumn] = []
    for table, names in SIDE_TABLE_COLUMNS.items():
        for name in names:
            columns.append(
                SideTableColumn(
                    table=table,
                    name=name,
                    type_unit=_side_type_unit(name),
                    description=_side_description(table, name),
                    availability=_side_availability(table),
                    source=_side_source(table, name),
                    nullable="N" if name == "stable_id" else "Y",
                    example=_side_example(name),
                    operational_definition=_side_operational_definition(table, name),
                )
            )
    return columns


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


def _side_column_entry(column: SideTableColumn, index: int) -> str:
    """Render one side-table Markdown dictionary entry.

    Parameters
    ----------
    column:
        Side-table column.
    index:
        One-based side-table entry index.

    Returns
    -------
    str
        Markdown entry.
    """

    lines = [
        f"## Side {index}. `{column.table}.{column.name}`",
        f"- Table: `{column.table}`",
        f"- Type/unit: {column.type_unit}",
        f"- Description: {column.description}",
        f"- Availability: {column.availability}",
        f"- Source: {column.source}",
        f"- Nullability: {_nullability_text(column)}",
        f"- Example: `{column.example}`",
    ]
    if column.name in _SIDE_BOOLEAN_COLUMNS:
        empty = "empty = unknown/not determined" if column.nullable == "Y" else "empty is invalid"
        lines.append(f"- Boolean semantics: Boolean encoding: 1 = true, 0 = false, {empty}.")
    if column.operational_definition:
        lines.append(f"- Operational definition: {column.operational_definition}")
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
    side_entries = [
        _side_column_entry(column, index)
        for index, column in enumerate(side_table_schema_columns(), start=1)
    ]
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
    return "\n".join([*header, *entries, *side_entries, ""])


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
