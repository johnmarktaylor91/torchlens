"""Build the deck: render every panel, compose every slide, write the deck and receipts.

``build(out)`` writes into ``out``: ``raw/`` (each panel's SVG, DOT and Graphviz JSON),
``Sxx-<id>.svg`` (one composed picture per slide), ``visual-language.deck.md``,
``slides.json`` (per-slide status, legibility and the marks each key matched),
``coverage-receipt.json`` (per inventory row: slide, witness kind, DOT matches) and
``environment.json`` (versions, commit, build date). A slide that fails a selector, the
legibility check or its render is recorded as failed; the deck is still written so every
failure can be read, and the caller decides whether to promote it.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import traceback
from pathlib import Path
from typing import Any

import graphviz

from scripts.visual_language import capture, compose, coverage
from scripts.visual_language.dot import parse_json, select, strip_tags
from scripts.visual_language.slides import (
    ALPHABET,
    CHEAT_GROUPS,
    OTHER_PICTURES,
    SLIDES,
    Slide,
    constants,
    fill,
)

DECK_NAME = "visual-language.deck.md"
_NODE_KEYS = ("shape", "style", "fillcolor", "color", "penwidth", "peripheries", "fontcolor")
_EDGE_KEYS = ("style", "color", "arrowhead", "arrowsize", "dir", "penwidth", "fontcolor")
_EDGE_LABEL_KEYS = ("label", "headlabel", "taillabel", "xlabel")
_CLUSTER_KEYS = ("style", "color", "pencolor", "penwidth", "fillcolor")


def _quote(value: str) -> str:
    if value.startswith("<") and value.endswith(">") or ("<" in value and "</" in value):
        inner = value[1:-1] if value.startswith("<") and value.endswith(">") else value
        return f"<{inner}>"
    return '"' + value.replace('"', '\\"') + '"'


def _attrs(attrs: dict[str, str]) -> str:
    return " ".join(f"{k}={_quote(v)}" for k, v in attrs.items())


def alphabet_dot(slide_id: str, raw: Path, order: dict[str, int]) -> tuple[str, list[str]]:
    """A DOT sheet whose cells copy each exemplar's exact attributes from its slide."""

    entries = ALPHABET[slide_id]
    cols = 4
    cell_w, cell_h = 216.0, 54.0
    lines = [
        "digraph alphabet {",
        'graph [bgcolor=white fontname=Helvetica splines=true outputorder=edgesfirst pad="0.1"]',
        "node [fontname=Helvetica fontsize=14]",
        "edge [fontname=Helvetica]",
    ]
    missing = []
    for i, (meaning, source, panel, selector) in enumerate(entries):
        r, c = divmod(i, cols)
        x, y = c * cell_w + cell_w / 2, -r * cell_h
        stem = raw / f"{source}-{panel}.json"
        matches = select(parse_json(stem.read_text()), fill(selector)) if stem.exists() else []
        if not matches:
            missing.append(f"{meaning}: {selector} found nothing on {source}-{panel}")
            continue
        mark = matches[0]
        number = order.get(source, 0)
        label = f"{meaning} ({number})"
        if mark.kind == "node":
            attrs = {k: mark.attrs[k] for k in _NODE_KEYS if mark.attrs.get(k)}
            if attrs.get("shape") in ("plaintext", "none"):
                attrs["fontcolor"] = mark.attrs.get("fontcolor", "#777777")
            attrs.update({"label": label, "pos": f"{x:.1f},{y:.1f}!", "margin": "0.08,0.04"})
            lines.append(f"n{i} [{_attrs(attrs)}]")
        elif mark.kind == "edge":
            attrs = {k: mark.attrs[k] for k in _EDGE_KEYS if mark.attrs.get(k)}
            for key in _EDGE_LABEL_KEYS:
                if mark.attrs.get(key):
                    attrs[key] = mark.attrs[key]
                    for size in ("fontsize", "labelfontsize"):
                        if mark.attrs.get(size):
                            attrs[size] = mark.attrs[size]
            a, b = f"a{i}", f"b{i}"
            lines.append(f'{a} [shape=point width=0.06 pos="{x - 100:.1f},{y:.1f}!"]')
            lines.append(f'{b} [shape=point width=0.06 pos="{x - 20:.1f},{y:.1f}!"]')
            lines.append(f"{a} -> {b} [{_attrs(attrs)}]")
            lines.append(
                f'm{i} [shape=plaintext label="{label}" pos="{x + 40:.1f},{y:.1f}!" fontsize=13]'
            )
        else:
            attrs = {k: mark.attrs[k] for k in _CLUSTER_KEYS if mark.attrs.get(k)}
            if "pencolor" in attrs:
                attrs["color"] = attrs.pop("pencolor")
            style = attrs.get("style", "")
            if "filled" not in style and attrs.get("fillcolor"):
                attrs["style"] = f"{style},filled".strip(",")
            attrs.update({"shape": "box", "label": label, "pos": f"{x:.1f},{y:.1f}!"})
            attrs.setdefault("fillcolor", "white")
            lines.append(f"n{i} [{_attrs(attrs)}]")
    lines.append("}")
    return "\n".join(lines), missing


def render_alphabet(slide: Slide, raw: Path, order: dict[str, int]) -> tuple[str, list[str]]:
    """Render an alphabet sheet with pinned positions; write its SVG and JSON; return stem."""

    source, missing = alphabet_dot(slide.id, raw, order)
    stem = f"{slide.id}-a"
    (raw / f"{stem}.dot").write_text(source)
    graph = graphviz.Source(source, engine="neato")
    (raw / f"{stem}.svg").write_bytes(graph.pipe(format="svg", neato_no_op=2))
    (raw / f"{stem}.json").write_bytes(graph.pipe(format="json", neato_no_op=2))
    return stem, missing


def render_all(raw: Path, only: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """Capture and draw every panel serially; record errors instead of stopping."""

    raw.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for slide in SLIDES:
        if only and slide.id not in only:
            continue
        for panel in slide.panels:
            stem = f"{slide.id}-{panel.name}"
            try:
                records[stem] = capture.render_panel(slide, panel, raw)
            except Exception as exc:  # recorded per panel; the slide then fails
                records[stem] = {
                    "slide": slide.id,
                    "panel": panel.name,
                    "error": f"{type(exc).__name__}: {exc}"[:800],
                    "traceback": traceback.format_exc()[-2500:],
                }
            print(f"panel {stem}: {'error' if 'error' in records[stem] else 'ok'}", flush=True)
    return records


def _environment() -> dict[str, Any]:
    import torch

    import torchlens

    def run(cmd: list[str]) -> str:
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            return (done.stdout + done.stderr).strip()
        except OSError as exc:
            return f"unavailable: {exc}"

    return {
        "torchlens": getattr(torchlens, "__version__", "unknown"),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "graphviz": run(["dot", "-V"]),
        "commit": run(["git", "rev-parse", "HEAD"]),
        "built": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }


def _table_slide(slide: Slide, height: float, env: dict[str, Any], probe: dict[str, Any]) -> Any:
    import inspect

    from torchlens.data_classes._trace_viz import TraceVisualizationMixin

    params = inspect.signature(TraceVisualizationMixin.draw).parameters
    footer = f"TorchLens {env['torchlens']}, commit {env['commit'][:10]}, built {env['built']}"
    if slide.id == "cheat-sheet":
        rows = [[see, how, ", ".join(names)] for see, how, names in CHEAT_GROUPS]
        return compose.compose_table(
            rows, ["To see", "Pass", "All parameters in this group"], [0.25, 0.32, 0.43], height
        )
    if slide.id == "draw-parameters":
        items = []
        for _see, _how, names in CHEAT_GROUPS:
            for name in names:
                default = params[name].default
                shown = (
                    "(alias)"
                    if repr(default) == "<MISSING>" or "MISSING" in repr(default)
                    else repr(default)
                )
                items.append(f"{name}={shown}")
        per_col = -(-len(items) // 3)
        cols = [items[i * per_col : (i + 1) * per_col] for i in range(3)]
        rows = [[col[r] if r < len(col) else "" for col in cols] for r in range(per_col)]
        return compose.compose_table(
            rows, ["Parameter=default", "", ""], [0.34, 0.33, 0.33], height, footer
        )
    if slide.id == "other-pictures":
        rows = [list(row[:3]) for row in OTHER_PICTURES]
        return compose.compose_table(
            rows, ["Picture", "What it shows", "How to call it"], [0.22, 0.48, 0.30], height
        )
    rows = [[k, str(v)] for k, v in probe.items()]
    return compose.compose_table(
        rows, ["Export choice", "Measured on this build"], [0.4, 0.6], height
    )


def compose_slide(
    slide: Slide,
    raw: Path,
    out: Path,
    number: int,
    order: dict[str, int],
    env: dict[str, Any],
    probe: dict[str, Any],
) -> dict[str, Any]:
    """Compose one slide picture; return its record (status, problems, witnesses)."""

    height = compose.room_height(slide)
    record: dict[str, Any] = {"id": slide.id, "number": number, "room_h": round(height, 1)}
    problems: list[str] = []
    if slide.layout == "sheet":
        stem, missing = render_alphabet(slide, raw, order)
        problems.extend(missing)
        canvas = compose.compose_grid(
            Slide(slide.id, slide.title, slide.rule), {"a": compose.Render.load(raw, stem)}, height
        )
    elif slide.layout == "table" and slide.id != "export" or slide.layout == "table":
        canvas = _table_slide(slide, height, env, probe)
    else:
        renders = {}
        for panel in slide.panels:
            stem = f"{slide.id}-{panel.name}"
            if not (raw / f"{stem}.json").exists():
                problems.append(f"panel {panel.name} did not render")
                continue
            renders[panel.name] = compose.Render.load(raw, stem, fill(panel.label))
        if problems:
            record.update(status="failed", problems=problems)
            return record
        if slide.layout == "grid":
            canvas = compose.compose_grid(slide, renders, height)
        elif slide.layout == "text":
            canvas = compose.compose_wide(slide, list(renders.values()), height)
        else:
            canvas = compose.compose_key(slide, list(renders.values()), height)
    problems.extend(canvas.problems)
    picture = out / f"S{number:02d}-{slide.id}.svg"
    canvas.write(picture)
    size = picture.stat().st_size
    if size > 150_000:
        problems.append(f"picture is {size // 1000} KB, over 150 KB")
    record.update(
        status="failed" if problems else "ok",
        problems=problems,
        picture=picture.name,
        bytes=size,
        min_css_px=round(canvas.min_css_px, 2),
        witnesses=canvas.witnesses,
    )
    return record


def deck_markdown(records: list[dict[str, Any]]) -> str:
    """The deck file: one ``##`` title, the rule line and one picture per slide."""

    blocks = []
    for slide, record in zip(SLIDES, records, strict=True):
        lines = [f"## {slide.title}", "", fill(slide.rule)]
        if record.get("picture"):
            lines += ["", f"![{slide.title}]({record['picture']})"]
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks) + "\n"


def receipt(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per inventory row: slide, witness kind and the DOT marks its slide's keys matched."""

    by_slide = {r["id"]: r for r in records}
    rows = coverage.receipt_rows()
    for row in rows:
        witnesses = by_slide.get(row["slide"] or "", {}).get("witnesses", {})
        row["dot_matches"] = {
            key["select"]: witnesses.get(key["select"], []) for key in row["keys"]
        }
        row["proven_by_picture"] = row["witness"] == "picture" and any(row["dot_matches"].values())
    counts = {
        "picture": sum(1 for r in rows if r["proven_by_picture"]),
        "caption": sum(1 for r in rows if r["witness"] == "caption"),
        "inactive": sum(1 for r in rows if r["status"] == "inactive"),
        "picture_claimed_unproven": [
            r["row"] for r in rows if r["witness"] == "picture" and not r["proven_by_picture"]
        ],
    }
    return {"counts": counts, "rows": rows}


def build(out: Path, only: set[str] | None = None) -> int:
    """Render, compose and write the deck into ``out``; return the number of failed slides."""

    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    panel_records = render_all(raw, only)
    probe = capture.probe_export(raw)
    env = _environment()
    order = {slide.id: i for i, slide in enumerate(SLIDES, start=1)}
    records = []
    for number, slide in enumerate(SLIDES, start=1):
        if only and slide.id not in only and slide.layout != "table":
            records.append({"id": slide.id, "number": number, "status": "skipped"})
            continue
        try:
            record = compose_slide(slide, raw, out, number, order, env, probe)
        except Exception as exc:  # recorded; the slide fails
            record = {
                "id": slide.id,
                "number": number,
                "status": "failed",
                "problems": [f"{type(exc).__name__}: {exc}"[:600]],
                "traceback": traceback.format_exc()[-2500:],
            }
        record["panels"] = {k: v for k, v in panel_records.items() if v.get("slide") == slide.id}
        records.append(record)
        print(f"slide {number:02d} {slide.id}: {record['status']}", flush=True)
    (out / DECK_NAME).write_text(deck_markdown(records))
    (out / "slides.json").write_text(json.dumps(records, indent=1, default=str))
    (out / "coverage-receipt.json").write_text(json.dumps(receipt(records), indent=1))
    (out / "environment.json").write_text(
        json.dumps({**env, "constants": constants()}, indent=1, default=str)
    )
    failed = [r["id"] for r in records if r["status"] == "failed"]
    print(json.dumps({"slides": len(records), "failed": failed}), flush=True)
    return len(failed)


def visible_text(panel_json: Path) -> str:
    """Every label word of a rendered panel (used by the slides test)."""

    dot = parse_json(panel_json.read_text())
    marks = [*dot.nodes, *dot.edges, *dot.clusters, dot.graph]
    return " ".join(
        strip_tags(
            " ".join(m.attrs.get(k, "") for k in ("label", "xlabel", "headlabel", "taillabel"))
        )
        for m in marks
    )
