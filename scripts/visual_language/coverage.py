"""Coverage of the TorchLens visual language deck, checked from the renderer.

The check enumerates what the renderer can emit (draw parameters, closed vocabularies,
legend rows, Graphviz tokens, label templates and emission sites) and demands that the
deck teaches each item or that a committed table classifies it with a reason. A new item
fails with its name, its source and the remedy; nothing is accepted automatically.

The torch-free parts (the AST scans) read the package source through the shared test
corpus when the test suite has loaded it, so a test session parses each file once.

Usage::

    python -m scripts.visual_language.coverage check
    python -m scripts.visual_language.coverage discover
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
import sys
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from types import UnionType
from typing import Any

from scripts.visual_language import slides
from scripts.visual_language.coverage_data import (
    ENTRY_CHEATS,
    LEGEND_ROW_WITNESS,
    LEGEND_SECTION_ROWS,
    NONVISUAL_DRAW_PARAMS,
    ROWS,
    VIS_OPTION_ALIASES,
    VOCABULARY_EXEMPT,
    Row,
)
from scripts.visual_language.coverage_scan import (
    DrawParam,
    EmissionSite,
    LegendText,
    SourceItem,
    VocabValue,
    emission_site_universe,
    encoding_legend_texts,
    label_template_universe,
    visual_token_universe,
)
from scripts.visual_language.coverage_tables import EMISSION_SITES, LABEL_TEMPLATES, VISUAL_TOKENS

__all__ = [
    "EMISSION_SITES",
    "LABEL_TEMPLATES",
    "LEGEND_ROW_WITNESS",
    "ROWS",
    "VISUAL_TOKENS",
    "Row",
    "check_all",
    "discover",
    "receipt_rows",
]

VIS_PREFIX = "torchlens.visualization."
EXEMPT_PREFIX = "exempt: "

DRAW_ENTRY_POINTS: Mapping[str, str] = {
    "Trace.draw": "torchlens.data_classes._trace_viz:TraceVisualizationMixin.draw",
    "Trace.draw_backward": "torchlens.data_classes._trace_viz:TraceVisualizationMixin.draw_backward",
    "Trace.draw_combined": "torchlens.data_classes._trace_viz:TraceVisualizationMixin.draw_combined",
    "Module.draw": "torchlens.data_classes.module:Module.draw",
}
CALL_ENTRY = {
    "draw": "Trace.draw",
    "lens": "Trace.draw",
    "draw_backward": "Trace.draw_backward",
    "draw_combined": "Trace.draw_combined",
}

# ---------------------------------------------------------------------------
# Torch-importing helpers
# ---------------------------------------------------------------------------


def resolve_locator(locator: str) -> object:
    """Import ``module:symbol`` (module relative to ``torchlens.visualization`` when short)."""

    module_name, _, symbol = locator.partition(":")
    if not module_name.startswith("torchlens"):
        module_name = VIS_PREFIX + module_name
    target: object = importlib.import_module(module_name)
    for part in symbol.split("."):
        target = getattr(target, part)
    return target


def _literal_values(annotation: object) -> list[object] | None:
    if isinstance(annotation, str):
        match = re.search(r"Literal\[(.*?)\]", annotation)
        return list(ast.literal_eval(f"({match.group(1)},)")) if match else None
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return list(typing.get_args(annotation))
    if origin in (typing.Union, UnionType):
        found = [_literal_values(arg) for arg in typing.get_args(annotation)]
        values = [v for group in found if group for v in group]
        return values or None
    return None


def _vocab(name: str, values: Iterable[object], source: str) -> list[VocabValue]:
    return [VocabValue(name, str(value), source) for value in values]


def _literal_alias_vocab() -> list[VocabValue]:
    from torchlens import _literals

    out: list[VocabValue] = []
    for name, value in vars(_literals).items():
        values = _literal_values(value) if not name.startswith("_") else None
        if values is not None:
            out += _vocab(f"_literals.{name}", values, f"torchlens/_literals.py:{name}")
    return out


def _signature_literal_vocab(aliases: list[VocabValue]) -> list[VocabValue]:
    alias_sets: dict[str, set[str]] = {}
    for item in aliases:
        alias_sets.setdefault(item.vocab, set()).add(item.value)
    out: list[VocabValue] = []
    for entry, locator in DRAW_ENTRY_POINTS.items():
        for param in inspect.signature(resolve_locator(locator)).parameters.values():  # type: ignore[arg-type]
            annotation = param.annotation
            text = annotation if isinstance(annotation, str) else repr(annotation)
            values = _literal_values(annotation) if "Literal[" in text else None
            if values is None:
                continue
            if any({str(v) for v in values} <= known for known in alias_sets.values()):
                continue
            if values is not None:
                out += _vocab(f"{entry}.{param.name}", values, f"{entry}({param.name})")
    return out


def _module_vocab() -> list[VocabValue]:
    from torchlens.visualization import (
        _encoding,
        _surgery_diff,
        code_panel,
        collapse_patterns,
        fastlog_preview,
        modes,
        overlays,
        surgery_visuals,  # imported before list_lenses(): registers the surgery lens
        theme_registry,
        themes,
    )
    from torchlens.visualization.lenses import _filter, _nonfinite, _roster

    rows = [str(v) for k, v in vars(_encoding).items() if k.startswith("ROW_")]
    lenses = set(_roster.ROSTER) | {lens.name for lens in theme_registry.list_lenses()}
    groups: list[tuple[str, Iterable[object]]] = [
        ("code_panel.CodePanelMode", typing.get_args(code_panel.CodePanelMode)),
        ("modes.MODE_REGISTRY", modes.MODE_REGISTRY),
        ("modes.COLLAPSED_MODE_REGISTRY", modes.COLLAPSED_MODE_REGISTRY),
        ("themes.THEME_PRESETS", themes.THEME_PRESETS),
        ("lenses", lenses),
        ("lenses._roster.COMPOSITIONS", _roster.COMPOSITIONS),
        ("lenses._filter.FILTER_TOKENS", _filter.FILTER_TOKENS),
        ("collapse_patterns.IDIOMATIC_PATTERNS", collapse_patterns.IDIOMATIC_PATTERNS),
        ("overlays.SUPPORTED_OVERLAYS", overlays.SUPPORTED_OVERLAYS),
        ("_encoding.SCALAR_BUILTIN_SOURCES", _encoding.SCALAR_BUILTIN_SOURCES),
        ("_encoding.COLOR_TRANSFORM_VOCABULARY", _encoding.COLOR_TRANSFORM_VOCABULARY),
        ("_encoding.SIZE_SCALE_VOCABULARY", _encoding.SIZE_SCALE_VOCABULARY),
        ("_encoding.ROW_*", rows),
        ("lenses._nonfinite.NONFINITE_STATES", _nonfinite.NONFINITE_STATES),
        ("surgery_visuals.MARK_KINDS", surgery_visuals.MARK_KINDS),
        ("surgery_visuals.MARK_BASES", surgery_visuals.MARK_BASES),
        ("_surgery_diff.DIFF_JOIN_KINDS", _surgery_diff.DIFF_JOIN_KINDS),
        ("fastlog_preview.Decision", (d.value for d in fastlog_preview.Decision)),
    ]
    out: list[VocabValue] = []
    for name, values in groups:
        out += _vocab(name, values, f"torchlens.visualization.{name}")
    return out


# ---------------------------------------------------------------------------
# Universes 1-3
# ---------------------------------------------------------------------------


def draw_surface_universe() -> tuple[DrawParam, ...]:
    """Every parameter of the draw entry points plus every ``VisualizationOptions`` field."""

    import dataclasses

    from torchlens.options import VisualizationOptions

    items: list[DrawParam] = []
    for entry, locator in DRAW_ENTRY_POINTS.items():
        for param in inspect.signature(resolve_locator(locator)).parameters.values():  # type: ignore[arg-type]
            if param.name != "self" and param.kind not in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                items.append(DrawParam(entry, param.name))
    for field in dataclasses.fields(VisualizationOptions):
        if not field.name.startswith("_"):
            items.append(DrawParam("VisualizationOptions", field.name))
    return tuple(sorted(items))


def vocabulary_universe() -> tuple[VocabValue, ...]:
    """Every value of every closed vocabulary the renderer accepts or emits."""

    aliases = _literal_alias_vocab()
    items = aliases + _signature_literal_vocab(aliases) + _module_vocab()
    return tuple(sorted(set(items)))


def legend_universe() -> tuple[LegendText, ...]:
    """Every legend row text the forward, backward and encoding legends can show."""

    from torchlens.visualization import _legend, themes

    sections: list[tuple[str, Any]] = []
    for name, theme in themes.THEME_PRESETS.items():
        sections += [
            (f"theme_role_sections({name})", s)
            for s in _legend.theme_role_sections(theme, mutated_parameter=True)
        ]
    for passes in (1, 2):
        built = _legend.backward_key_sections(
            has_higher_order=True,
            has_intervening=True,
            has_accumulation=True,
            has_custom=True,
            num_backward_passes=passes,
        )
        sections += [(f"backward_key_sections(passes={passes})", s) for s in built]
    items: set[LegendText] = set()
    first: dict[tuple[str, str], str] = {}
    for source, section in sections:
        for text in (section.title, *(row.text for row in section.rows)):
            first.setdefault((section.title, text), source)
    items = {LegendText(sec, text, src) for (sec, text), src in first.items()}
    items |= set(encoding_legend_texts())
    dedup: dict[tuple[str, str], LegendText] = {}
    for item in sorted(items):
        dedup.setdefault((item.section, item.text), item)
    return tuple(sorted(dedup.values()))


# ---------------------------------------------------------------------------
# What the deck says
# ---------------------------------------------------------------------------


def filled_deck_text() -> str:
    """``slides.deck_text()`` with caption templates filled line by line."""

    lines = []
    for line in slides.deck_text().splitlines():
        try:
            lines.append(slides.fill(line))
        except (KeyError, ValueError, IndexError, AttributeError):
            lines.append(line)
    return "\n".join(lines)


def _flatten(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, sub in value.items():
            yield str(key)
            yield from _flatten(sub)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for sub in value:
            yield from _flatten(sub)
    else:
        yield str(value)


def panel_kwarg_values() -> frozenset[str]:
    """Every explicit panel kwarg value (flattened) and every convention value."""

    values = {v for *_ignored, value in slides.iter_panel_kwargs() for v in _flatten(value)}
    values |= set(_flatten(dict(slides.CONVENTIONS)))
    return frozenset(values)


def row_variants(rows: Mapping[str, Row] = ROWS) -> frozenset[str]:
    """Every variant any row lists as needing a witness."""

    return frozenset(v for row in rows.values() for v in (*row.variants, *row.caption_variants))


def draw_lessons() -> dict[str, frozenset[str]]:
    """Per entry point, the parameters a slide, a convention or a cheat list teaches."""

    lessons: dict[str, set[str]] = {entry: set(slides.CONVENTIONS) for entry in DRAW_ENTRY_POINTS}
    for _slide_id, panel, key, _value in slides.iter_panel_kwargs():
        entry = CALL_ENTRY.get(panel.call)
        if entry is not None:
            lessons[entry].add(key)
    for _see, _pass, names in slides.CHEAT_GROUPS:
        lessons["Trace.draw"].update(names)
    for entry, names in ENTRY_CHEATS.items():
        lessons.setdefault(entry, set()).update(names)
    lessons["Module.draw"] |= lessons["Trace.draw"]
    return {entry: frozenset(names) for entry, names in lessons.items()}


# ---------------------------------------------------------------------------
# Checkers (each a plain function over its two sides)
# ---------------------------------------------------------------------------


def check_draw_surface(
    universe: Iterable[DrawParam],
    lessons: Mapping[str, frozenset[str]],
    cheat_groups: Iterable[tuple[str, str, tuple[str, ...]]] = slides.CHEAT_GROUPS,
    aliases: Mapping[str, str] = VIS_OPTION_ALIASES,
    nonvisual: Mapping[str, str] = NONVISUAL_DRAW_PARAMS,
) -> list[str]:
    """Report draw parameters no slide, convention or cheat list teaches.

    Parameters
    ----------
    universe:
        Draw-surface parameters (:func:`draw_surface_universe`).
    lessons:
        Per entry point, the parameters the deck teaches (:func:`draw_lessons`).
    cheat_groups:
        Cheat-sheet groups; each ``Trace.draw`` parameter sits in exactly one.
    aliases:
        ``VisualizationOptions`` field to ``Trace.draw`` kwarg.
    nonvisual:
        ``"<entry>.<param>"`` that changes nothing visible, with the reason.

    Returns
    -------
    list[str]
        One failure line per finding, naming the item, its source and the remedy.
    """

    failures: list[str] = []
    universe = tuple(universe)
    trace_draw = {p.name for p in universe if p.entry == "Trace.draw"}
    for param in universe:
        name = param.name
        if param.entry == "VisualizationOptions":
            if name not in aliases:
                failures.append(
                    f"New VisualizationOptions field `{name}` has no draw kwarg mapping: add it "
                    "to coverage_data.VIS_OPTION_ALIASES with the Trace.draw kwarg it becomes"
                )
                continue
            name = aliases[name]
            if name not in trace_draw:
                failures.append(
                    f"VisualizationOptions field `{param.name}` maps to `{name}`, which "
                    "Trace.draw does not accept: fix coverage_data.VIS_OPTION_ALIASES"
                )
                continue
            entry = "Trace.draw"
        else:
            entry = param.entry
        if name in lessons.get(entry, frozenset()) or f"{entry}.{name}" in nonvisual:
            continue
        failures.append(
            f"New draw option `{name}` ({param.entry}) has no lesson: add it to a slide's "
            "kwargs or to slides.CHEAT_GROUPS (coverage_data.ENTRY_CHEATS for backward and "
            "combined), or classify it as nonvisual in coverage_data.NONVISUAL_DRAW_PARAMS "
            "with a reason"
        )
    seen: dict[str, str] = {}
    for see, _pass, names in cheat_groups:
        for name in names:
            if name not in trace_draw:
                failures.append(
                    f"Cheat group `{see}` lists `{name}`, which Trace.draw does not accept: "
                    "remove it from slides.CHEAT_GROUPS"
                )
            elif name in seen:
                failures.append(
                    f"Trace.draw option `{name}` is in two cheat groups (`{seen[name]}`, "
                    f"`{see}`): keep it in exactly one"
                )
            seen.setdefault(name, see)
    for name in sorted(trace_draw - set(seen)):
        failures.append(
            f"Trace.draw option `{name}` is in no cheat group: add it to slides.CHEAT_GROUPS"
        )
    return failures


def _word_in(value: str, text: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(value)}(?![\w])", text) is not None


def check_vocabularies(
    universe: Iterable[VocabValue],
    deck_text: str,
    kwarg_values: frozenset[str],
    variants: frozenset[str],
    exempt: Mapping[str, str] = VOCABULARY_EXEMPT,
) -> list[str]:
    """Report closed-vocabulary values no slide call, caption or variant list witnesses.

    Parameters
    ----------
    universe:
        Vocabulary values (:func:`vocabulary_universe`).
    deck_text:
        Filled deck words; a value counts when it appears as a whole word.
    kwarg_values:
        Flattened panel kwarg and convention values.
    variants:
        Every row's variants.
    exempt:
        Vocabulary name to the reason it is not a drawing choice.

    Returns
    -------
    list[str]
        One failure line per finding, naming the item, its source and the remedy.
    """

    failures = []
    for item in universe:
        if item.vocab in exempt:
            continue
        value = item.value
        if value in kwarg_values or value in variants or _word_in(value, deck_text):
            continue
        failures.append(
            f"New value `{value}` of {item.vocab} ({item.source}) has no witness: use it in a "
            "slide's kwargs, state it in a slide caption, or list it in the variants of the "
            "coverage_data.ROWS row that teaches it (or exempt the vocabulary in "
            "coverage_data.VOCABULARY_EXEMPT with a reason)"
        )
    return failures


def check_legend_rows(
    universe: Iterable[LegendText],
    deck_text: str,
    witness: Mapping[str, str] = LEGEND_ROW_WITNESS,
    section_rows: Mapping[str, str] = LEGEND_SECTION_ROWS,
    rows: Mapping[str, Row] = ROWS,
    slide_ids: Iterable[str] | None = None,
) -> list[str]:
    """Report legend row texts that no mapped row and no slide shows.

    Parameters
    ----------
    universe:
        Legend row texts (:func:`legend_universe`).
    deck_text:
        Filled deck words; a row counts when its text appears (any case).
    witness:
        Row text to the slide whose render draws it.
    section_rows:
        Legend section title to the row id that teaches it.
    rows:
        The inventory.
    slide_ids:
        Existing slide ids (default: the live slide table).

    Returns
    -------
    list[str]
        One failure line per finding, naming the item, its source and the remedy.
    """

    ids = set(slide_ids if slide_ids is not None else slides.slide_ids())
    lowered = deck_text.lower()
    failures = []
    for item in universe:
        row_id = section_rows.get(item.section)
        if row_id is None or row_id not in rows:
            failures.append(
                f"Legend section `{item.section}` ({item.source}) maps to no row: add it to "
                "coverage_data.LEGEND_SECTION_ROWS with the row id that teaches it"
            )
            continue
        if item.text.lower() in lowered:
            continue
        slide_id = witness.get(item.text)
        if slide_id is not None and slide_id in ids:
            continue
        failures.append(
            f"Legend row `{item.text}` ({item.source}) is shown by no slide: state it in a "
            "slide's text or add it to coverage_data.LEGEND_ROW_WITNESS with the slide whose "
            "render draws it"
        )
    return failures


def _classification_failure(kind: str, key: str, value: str, rows: Mapping[str, Row]) -> str | None:
    if value.startswith(EXEMPT_PREFIX):
        if value[len(EXEMPT_PREFIX) :].strip():
            return None
        return f"{kind} `{key}` is exempt without a reason: write the reason after 'exempt: '"
    if value in rows:
        return None
    return f"{kind} `{key}` is classified to unknown row `{value}`: use a coverage_data.ROWS id"


def _check_table(
    kind: str,
    table_name: str,
    universe: Iterable[SourceItem],
    table: Mapping[str, str],
    rows: Mapping[str, Row],
) -> list[str]:
    failures = []
    keys = set()
    for item in universe:
        keys.add(item.key)
        value = table.get(item.key)
        if value is None:
            failures.append(
                f"New {kind} `{item.key}` ({item.source}) is unclassified: add it to "
                f"coverage_data.{table_name} with the row id it belongs to or 'exempt: <reason>' "
                "(`python -m scripts.visual_language.coverage discover` lists suggestions)"
            )
            continue
        problem = _classification_failure(kind, item.key, value, rows)
        if problem is not None:
            failures.append(problem)
    for key in sorted(set(table) - keys):
        failures.append(
            f"Stale {kind} `{key}` in coverage_data.{table_name}: the renderer no longer "
            "emits it; remove the entry"
        )
    return failures


def check_visual_tokens(
    universe: Iterable[SourceItem],
    table: Mapping[str, str] = VISUAL_TOKENS,
    rows: Mapping[str, Row] = ROWS,
) -> list[str]:
    """Report Graphviz tokens and hex colours with no row and no exempt reason.

    Parameters
    ----------
    universe:
        Tokens found in the renderer source.
    table:
        Token to row id or ``"exempt: <reason>"``.
    rows:
        The inventory.

    Returns
    -------
    list[str]
        One failure line per finding, naming the item, its source and the remedy.
    """

    return _check_table("visual token", "VISUAL_TOKENS", universe, table, rows)


def check_label_templates(
    universe: Iterable[SourceItem],
    table: Mapping[str, str] = LABEL_TEMPLATES,
    rows: Mapping[str, Row] = ROWS,
) -> list[str]:
    """Report label templates with no row and no exempt reason.

    Parameters
    ----------
    universe:
        Template prefixes found in the label builders.
    table:
        Template to row id or ``"exempt: <reason>"``.
    rows:
        The inventory.

    Returns
    -------
    list[str]
        One failure line per finding, naming the item, its source and the remedy.
    """

    return _check_table("label template", "LABEL_TEMPLATES", universe, table, rows)


def check_emission_sites(
    universe: Iterable[EmissionSite],
    table: Mapping[str, tuple[str, str]] = EMISSION_SITES,
    rows: Mapping[str, Row] = ROWS,
) -> list[str]:
    """Report new or changed emission sites; they need classification, never auto-accept.

    Parameters
    ----------
    universe:
        Emitting functions with their vocabulary fingerprints.
    table:
        Site to ``(fingerprint, row id or "exempt: <reason>")``.
    rows:
        The inventory.

    Returns
    -------
    list[str]
        One failure line per finding, naming the item, its source and the remedy.
    """

    failures = []
    seen = set()
    for site in universe:
        seen.add(site.site)
        entry = table.get(site.site)
        remedy = (
            "review what it draws, then record it in coverage_data.EMISSION_SITES as "
            f'"{site.site}": ("{site.fingerprint}", "<row id or exempt: reason>")'
        )
        if entry is None:
            failures.append(
                f"New emission site `{site.site}` ({site.source}) needs classification: {remedy}"
            )
            continue
        if entry[0] != site.fingerprint:
            failures.append(
                f"Emission site `{site.site}` ({site.source}) changed its emitted vocabulary "
                f"({entry[0]} -> {site.fingerprint}) and needs classification: {remedy}"
            )
        problem = _classification_failure("Emission site", site.site, entry[1], rows)
        if problem is not None:
            failures.append(problem)
    for name in sorted(set(table) - seen):
        failures.append(
            f"Stale emission site `{name}` in coverage_data.EMISSION_SITES: no longer emits; "
            "remove the entry"
        )
    return failures


def gate_namespace() -> dict[str, Any]:
    """Modules a row's gating expression may name, by short name."""

    from torchlens import _literals
    from torchlens.visualization import _render_common, modes, themes

    return {
        "themes": themes,
        "modes": modes,
        "_literals": _literals,
        "_render_common": _render_common,
    }


def check_inactive_rows(rows: Mapping[str, Row], namespace: Mapping[str, Any]) -> list[str]:
    """Report gated rows whose flag flipped while the row keeps its non-active status.

    Parameters
    ----------
    rows:
        The inventory.
    namespace:
        Short module names the gate expressions use (:func:`gate_namespace`).

    Returns
    -------
    list[str]
        One failure line per finding, naming the item, its source and the remedy.
    """

    failures = []
    for row_id, row in rows.items():
        if row.gate is None or row.status == "active":
            continue
        try:
            holds = bool(eval(row.gate, {"__builtins__": {}}, dict(namespace)))  # noqa: S307
        except Exception as exc:  # noqa: BLE001  (a broken gate is itself a failure)
            failures.append(f"Row {row_id} gate `{row.gate}` does not evaluate: {exc!r}")
            continue
        if not holds:
            failures.append(
                f"Row {row_id} ({row.slug}) is still {row.status} but its gate `{row.gate}` "
                "no longer holds: mark it active, give it a slide and a witness"
            )
    return failures


def check_map_integrity(
    rows: Mapping[str, Row],
    deck: Iterable[slides.Slide],
    resolve: Callable[[str], object] = resolve_locator,
) -> list[str]:
    """Report broken locators, unknown slides and rows the deck does not claim.

    Parameters
    ----------
    rows:
        The inventory.
    deck:
        The slides.
    resolve:
        Locator resolver (default: import).

    Returns
    -------
    list[str]
        One failure line per finding, naming the item, its source and the remedy.
    """

    failures = []
    deck = tuple(deck)
    by_id: dict[str, slides.Slide] = {}
    for slide in deck:
        if slide.id in by_id:
            failures.append(f"Duplicate slide id `{slide.id}` in slides.SLIDES")
        by_id[slide.id] = slide
    slugs: dict[str, str] = {}
    for row_id, row in rows.items():
        if row.slug in slugs:
            failures.append(f"Rows {slugs[row.slug]} and {row_id} share the slug `{row.slug}`")
        slugs.setdefault(row.slug, row_id)
        try:
            resolve(row.locator)
        except (ImportError, AttributeError) as exc:
            failures.append(f"Row {row_id} locator `{row.locator}` does not resolve: {exc}")
        if row.slide is None:
            if row.status != "inactive":
                failures.append(f"Row {row_id} has no slide: only inactive rows may omit one")
            continue
        slide = by_id.get(row.slide)
        if slide is None:
            failures.append(
                f"Row {row_id} names slide `{row.slide}`, which slides.SLIDES does not have: "
                "point the row at an existing slide"
            )
        elif row_id not in slide.rows:
            failures.append(
                f"Row {row_id} is not claimed by its primary slide `{row.slide}`: add it to "
                "that slide's rows"
            )
    for slide in deck:
        for row_id in slide.rows:
            if row_id not in rows:
                failures.append(
                    f"Slide `{slide.id}` claims row `{row_id}`, which coverage_data.ROWS lacks"
                )
    return failures


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def universes() -> dict[str, tuple[Any, ...]]:
    """Every universe, freshly derived."""

    return {
        "draw_surface": draw_surface_universe(),
        "vocabularies": vocabulary_universe(),
        "legend_rows": legend_universe(),
        "visual_tokens": visual_token_universe(),
        "label_templates": label_template_universe(),
        "emission_sites": emission_site_universe(),
    }


def check_all(derived: Mapping[str, tuple[Any, ...]] | None = None) -> list[str]:
    """Run the eight checkers against the live universes and return every failure line."""

    derived = derived if derived is not None else universes()
    text = filled_deck_text()
    return [
        *check_draw_surface(derived["draw_surface"], draw_lessons()),
        *check_vocabularies(derived["vocabularies"], text, panel_kwarg_values(), row_variants()),
        *check_legend_rows(derived["legend_rows"], text),
        *check_visual_tokens(derived["visual_tokens"]),
        *check_label_templates(derived["label_templates"]),
        *check_emission_sites(derived["emission_sites"]),
        *check_inactive_rows(ROWS, gate_namespace()),
        *check_map_integrity(ROWS, slides.SLIDES),
    ]


def _suggest_row(source: str) -> str:
    stem = Path(source.split(":")[0]).stem
    for row_id, row in ROWS.items():
        if row.locator.split(":")[0].rsplit(".", 1)[-1] == stem:
            return row_id
    return "exempt: <reason>"


def discover(torch_free: bool = False) -> dict[str, Any]:
    """Every universe's size and the unclassified items, with suggested table entries."""

    report: dict[str, Any] = {}
    tables = {"visual_tokens": VISUAL_TOKENS, "label_templates": LABEL_TEMPLATES}
    for name, items in (
        ("visual_tokens", visual_token_universe()),
        ("label_templates", label_template_universe()),
    ):
        missing = [i for i in items if i.key not in tables[name]]
        report[name] = {
            "count": len(items),
            "suggested": {i.key: f"{_suggest_row(i.source)}  # {i.source}" for i in missing},
        }
    sites = emission_site_universe()
    report["emission_sites"] = {
        "count": len(sites),
        "suggested": {
            s.site: [s.fingerprint, EMISSION_SITES.get(s.site, ("", _suggest_row(s.source)))[1]]
            for s in sites
            if EMISSION_SITES.get(s.site, ("",))[0] != s.fingerprint
        },
    }
    if not torch_free:
        derived = universes()
        report["counts"] = {name: len(items) for name, items in derived.items()}
        report["failures"] = check_all(derived)
    return report


def receipt_rows() -> list[dict[str, Any]]:
    """Per row: slide, witness kind and the key selectors on that slide (DOT filled later)."""

    by_id = {slide.id: slide for slide in slides.SLIDES}
    out = []
    for row_id, row in ROWS.items():
        slide = by_id.get(row.slide) if row.slide else None
        keys = [
            {"panel": key.panel, "select": key.select, "text": key.text}
            for key in (slide.keys if slide else ())
            if key.select
        ]
        out.append(
            {
                "row": row_id,
                "slug": row.slug,
                "slide": row.slide,
                "witness": row.witness,
                "status": row.status,
                "variants": list(row.variants),
                "caption_variants": list(row.caption_variants),
                "keys": keys,
                "dot_matches": [],
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: ``check`` prints failures (exit 1 if any); ``discover`` prints suggestions."""

    args = sys.argv[1:] if argv is None else argv
    command = args[0] if args else "check"
    if command == "discover":
        print(json.dumps(discover(torch_free="--torch-free" in args), indent=2, sort_keys=True))
        return 0
    failures = check_all()
    for line in failures:
        print(line)
    print(f"{len(failures)} coverage failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
