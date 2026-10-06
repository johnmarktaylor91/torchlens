"""The summary configuration grammar (F08; summary memo item 13, 3.10).

Orthogonal axes, one column registry, nothing silent: granularity
(level/depth/fold_repeats/buffers) x presentation (view/columns) x
selection (filter) x numbers (flop_convention/units) x output (style).
The legacy spellings are REMOVED (clean break, no aliases): each one
refuses typed and names its successor (``REMOVED_SUMMARY_OPTIONS`` /
``REMOVED_SUMMARY_LEVELS``); contradictions raise typed
``summary_option_conflict``; unknown names get a nearest-match teaching
refusal. Nothing is ever accepted-and-ignored.

Spellings DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any

from .._errors import InvalidArgumentError
from ._summary_ladder import DEFAULT_ROW_BUDGET

#: New-grammar level vocabulary (row grain; memo 3.10).
LEVELS: tuple[str, ...] = ("auto", "module", "op")

#: View presets implemented natively by the rebuilt renderer.
NATIVE_VIEWS: tuple[str, ...] = ("overview", "compute")

#: Removed legacy ``level=`` preset names -> what replaces each one. They
#: refuse typed (``summary_level_invalid``) naming the replacement; there is
#: no alias and no historical renderer behind them.
REMOVED_SUMMARY_LEVELS: dict[str, str] = {
    "overview": "view='overview' (the default column bundle)",
    "compute": "view='compute'",
    "cost": "view='compute'",
    "graph": "trace.to_agent_json() (op rows, edges, module hierarchy) or trace.draw()",
    "memory": (
        "trace.profile(sort_by='activation_memory') for per-op tensor memory; "
        "the summary footer carries the memory totals"
    ),
    "control_flow": (
        "trace.conditional_records and the conditional_* columns of trace.to_pandas() "
        "(multi-pass layers: num_passes)"
    ),
    "waterfall": (
        "trace.profile(level='op') for per-op time and activation memory, "
        "or trace.to_pandas() in execution order"
    ),
    "output": "trace.output_table()",
}

#: Removed legacy keyword spellings -> what replaces each one. They refuse
#: typed (``summary_option_invalid``) naming the replacement; there is no
#: alias and no historical renderer behind them.
REMOVED_SUMMARY_OPTIONS: dict[str, str] = {
    "preset": "view= ('overview' | 'compute') for the column bundle, level= for row grain",
    "fields": "columns= (bundle name, exact ordered list, or +name/-name deltas)",
    "show_ops": "level='op' (one row per executed op pass)",
    "include_ops": "level='op' (one row per executed op pass)",
    "mode": (
        "level='op' (one row per executed op pass) or fold_repeats=False (unfold repeated runs)"
    ),
    "print_to": "report.print(file=...) on the returned report, or print_to(str(report))",
    "count_fma_as_two": "flop_convention='fma2' (was True) or flop_convention='fma1' (was False)",
    "show_input_preprocessing_details": (
        "trace.provenance() and trace.input_preprocessor (verified, source, identifier)"
    ),
}

#: Style vocabulary (charset contract, memo 3.9).
STYLES: tuple[str, ...] = ("auto", "ascii", "unicode")

#: Units vocabulary for rendered numerics (raw data is ALWAYS plain ints).
UNITS: tuple[str, ...] = ("human", "raw")

#: Buffers tri-value (rows are gated on qualified-name plumbing; memo 3.10).
BUFFER_MODES: tuple[str, ...] = ("hide", "summary")

#: FLOP display conventions (the stored capture convention is fma=2).
FLOP_CONVENTIONS: tuple[str, ...] = ("fma2", "fma1")


@dataclass(frozen=True)
class ColumnSpec:
    """One registered column: semantics, formatting, and export typing."""

    name: str
    header: str
    semantic_type: str
    align: str
    applicability: str
    export_dtype: str


#: The one column registry (memo 3.10): applicability, semantic type,
#: formatter routing (by semantic_type), alignment, and export dtype.
COLUMN_REGISTRY: dict[str, ColumnSpec] = {
    "name": ColumnSpec("name", "name (type)", "text", "left", "always", "object"),
    "output": ColumnSpec("output", "output", "shape", "left", "always", "object"),
    "params": ColumnSpec("params", "params", "count", "right", "always", "Int64"),
    "params_pct": ColumnSpec("params_pct", "(%)", "percent", "right", "always", "float64"),
    "flops": ColumnSpec("flops", "fwd flops", "flops", "right", "always", "Int64"),
    "macs": ColumnSpec("macs", "macs", "macs", "right", "always", "Int64"),
    "train": ColumnSpec("train", "train", "tristate", "left", "mixed_only", "object"),
    "passes": ColumnSpec("passes", "passes", "count", "right", "always", "Int64"),
    "evidence": ColumnSpec("evidence", "evidence", "text", "left", "always", "object"),
}

#: View presets are COLUMN BUNDLES and never change row granularity.
VIEW_BUNDLES: dict[str, tuple[str, ...]] = {
    "overview": ("name", "output", "params", "params_pct", "flops"),
    "compute": ("name", "output", "params", "flops", "macs", "evidence"),
}


@dataclass(frozen=True)
class SummaryConfig:
    """The resolved new-grammar configuration (pure data, hashable)."""

    level: str = "auto"
    view: str = "overview"
    depth: Any = "auto"
    columns: tuple[str, ...] | None = None
    filter: Any = None
    buffers: str = "summary"
    fold_repeats: Any = "auto"
    max_rows: int = DEFAULT_ROW_BUDGET
    flop_convention: str = "fma2"
    units: str = "human"
    style: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_columns(self, mixed_trainability: bool) -> tuple[ColumnSpec, ...]:
        """The ordered column specs this config renders.

        Column presence is a pure function of (config, trace facts):
        ``train`` auto-appears exactly when trainability is mixed, so
        byte-stability holds per capture.
        """

        names = list(self.columns if self.columns is not None else VIEW_BUNDLES[self.view])
        if mixed_trainability and "train" not in names:
            names.append("train")
        return tuple(COLUMN_REGISTRY[name] for name in names)


def _did_you_mean(name: str, valid: tuple[str, ...]) -> str:
    """A teaching suffix naming the nearest valid choice."""

    close = difflib.get_close_matches(name, valid, n=1)
    hint = f" (did you mean {close[0]!r}?)" if close else ""
    return f"{hint} Valid choices: {', '.join(valid)}."


def _refuse_choice(
    axis: str,
    value: Any,
    valid: tuple[str, ...],
    code: str = "summary_option_invalid",
    successor: str | None = None,
) -> None:
    """Typed teaching refusal for a closed-vocabulary axis.

    ``successor`` names what replaces a REMOVED value (the legacy level
    presets); the refusal then teaches the replacement, not a near match.
    """

    if successor is not None:
        problem = f"summary() no longer accepts {axis}={value!r} (removed); use {successor}."
        remedy = f"use {successor}"
    else:
        problem = f"summary() got invalid {axis}={value!r}.{_did_you_mean(str(value), valid)}"
        remedy = f"pass one of: {', '.join(valid)}"
    raise InvalidArgumentError(problem, code=code, remedy=remedy)


def _apply_column_deltas(columns: Any, names: list[str]) -> tuple[str, ...]:
    """Apply ``+name``/``-name`` delta items to the view bundle's columns."""

    for item in columns:
        name = item[1:]
        if name not in COLUMN_REGISTRY:
            _refuse_choice("columns", name, tuple(COLUMN_REGISTRY))
        if item[0] == "+" and name not in names:
            names.append(name)
        if item[0] == "-" and name in names:
            names.remove(name)
    return tuple(names)


def _resolve_columns(columns: Any, view: str) -> tuple[str, ...] | None:
    """Resolve columns=: bundle name, exact ordered list, or +/- deltas."""

    if columns is None:
        return None
    if isinstance(columns, str):
        if columns in VIEW_BUNDLES:
            return VIEW_BUNDLES[columns]
        _refuse_choice("columns", columns, tuple(VIEW_BUNDLES) + tuple(COLUMN_REGISTRY))
    if all(isinstance(item, str) and item[:1] in "+-" for item in columns):
        return _apply_column_deltas(columns, list(VIEW_BUNDLES[view]))
    resolved: list[str] = []
    for item in columns:
        if not isinstance(item, str) or item not in COLUMN_REGISTRY:
            _refuse_choice("columns", item, tuple(COLUMN_REGISTRY))
        resolved.append(item)
    return tuple(resolved)


def _resolve_level(level: Any) -> Any:
    """Validate level=, keeping the historical refusal code (taxonomy pin)."""

    if level not in LEVELS:
        successor = REMOVED_SUMMARY_LEVELS.get(level) if isinstance(level, str) else None
        _refuse_choice("level", level, LEVELS, code="summary_level_invalid", successor=successor)
    return level


def _resolve_depth(depth: Any) -> Any:
    """Validate the depth= axis: auto, all, None, or an int cut."""

    if depth not in ("auto", "all", None) and not isinstance(depth, int):
        _refuse_choice("depth", depth, ("auto", "all", "<int>"))
    return depth


def _resolve_buffers(buffers: Any) -> Any:
    """Validate buffers=, teaching the gated 'rows' spelling separately."""

    if buffers not in BUFFER_MODES:
        if buffers == "rows":
            raise InvalidArgumentError(
                "buffers='rows' is gated on qualified buffer-name plumbing and is "
                "not available yet; footer buffer totals ship via buffers='summary'.",
                code="summary_option_invalid",
                remedy="use buffers='summary' (default) or buffers='hide'",
            )
        _refuse_choice("buffers", buffers, BUFFER_MODES)
    return buffers


def _resolve_fold_repeats(fold_repeats: Any) -> Any:
    """Validate the fold_repeats= tri-state."""

    if fold_repeats not in ("auto", True, False, None):
        _refuse_choice("fold_repeats", fold_repeats, ("auto", "True", "False"))
    return fold_repeats


def _resolve_max_rows(max_rows: Any) -> int:
    """Validate max_rows=; None means unbudgeted."""

    if max_rows is None:
        max_rows = 1 << 30
    if not isinstance(max_rows, int) or max_rows < 1:
        _refuse_choice("max_rows", max_rows, ("<positive int>", "None"))
    return max_rows


#: Uniform closed-vocabulary axes: (name, default, valid choices).
_CLOSED_VOCABULARY_AXES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("view", "overview", NATIVE_VIEWS),
    ("style", "auto", STYLES),
    ("units", "human", UNITS),
    ("flop_convention", "fma2", FLOP_CONVENTIONS),
)


#: Every option name of the rebuilt grammar (the did-you-mean vocabulary).
GRAMMAR_OPTIONS: tuple[str, ...] = (
    "level",
    "view",
    "depth",
    "columns",
    "filter",
    "buffers",
    "fold_repeats",
    "max_rows",
    "flop_convention",
    "units",
    "style",
)


def _refuse_unknown_options(unknown: list[str]) -> None:
    """Typed refusal for option names outside the grammar.

    A removed legacy spelling teaches its successor; any other unknown name
    gets the nearest grammar option.
    """

    removed = [name for name in unknown if name in REMOVED_SUMMARY_OPTIONS]
    if removed:
        successors = "; ".join(f"{name}= -> {REMOVED_SUMMARY_OPTIONS[name]}" for name in removed)
        problem = (
            f"summary() no longer accepts {', '.join(f'{name}=' for name in removed)} "
            f"(removed legacy spelling); use {successors}."
        )
        remedy = f"replace {successors}"
    else:
        problem = f"summary() got unknown option(s): {', '.join(unknown)}." + _did_you_mean(
            unknown[0], GRAMMAR_OPTIONS
        )
        remedy = "see docs/reference/summary.md for the option grammar"
    raise InvalidArgumentError(problem, code="summary_option_invalid", remedy=remedy)


def resolve_config(**kwargs: Any) -> SummaryConfig:
    """Validate and freeze one new-grammar configuration.

    Raises
    ------
    InvalidArgumentError
        ``summary_option_invalid`` for unknown closed-vocabulary values;
        ``summary_option_conflict`` for contradictory axes.
    """

    level = _resolve_level(kwargs.pop("level", "auto"))
    closed: dict[str, str] = {}
    for axis, default, valid in _CLOSED_VOCABULARY_AXES:
        value = kwargs.pop(axis, default)
        if value not in valid:
            _refuse_choice(axis, value, valid)
        closed[axis] = value
    view = closed["view"]
    depth = _resolve_depth(kwargs.pop("depth", "auto"))
    buffers = _resolve_buffers(kwargs.pop("buffers", "summary"))
    fold_repeats = _resolve_fold_repeats(kwargs.pop("fold_repeats", "auto"))
    max_rows = _resolve_max_rows(kwargs.pop("max_rows", DEFAULT_ROW_BUDGET))
    columns = _resolve_columns(kwargs.pop("columns", None), view)
    filter_ = kwargs.pop("filter", None)
    if kwargs:
        _refuse_unknown_options(sorted(kwargs))
    if level == "op" and depth not in ("auto", "all", None):
        raise InvalidArgumentError(
            "level='op' renders one row per executed op pass; depth= applies to "
            "module-tree views only -- the two axes contradict.",
            code="summary_option_conflict",
            remedy="drop depth= or use level='module'/'auto'",
        )
    return SummaryConfig(
        level=level,
        view=view,
        depth=depth,
        columns=columns,
        filter=filter_,
        buffers=buffers,
        fold_repeats=fold_repeats,
        max_rows=max_rows,
        flop_convention=closed["flop_convention"],
        units=closed["units"],
        style=closed["style"],
    )
