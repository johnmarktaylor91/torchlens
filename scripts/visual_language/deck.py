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
import os
import platform
import re
import subprocess
import traceback
from pathlib import Path
from typing import Any

import graphviz

from scripts.visual_language import capture, compose, coverage
from scripts.visual_language.at_scale import scale_slides
from scripts.visual_language.coverage import _flatten
from scripts.visual_language.dot import parse_json, select, strip_tags
from scripts.visual_language.sentinels import sentinel_slides
from scripts.visual_language.slides import (
    ALPHABET,
    CHEAT_GROUPS,
    OTHER_PICTURES,
    SLIDES,
    UNENCODED,
    Slide,
    constants,
    fill,
)

DECK_NAME = "visual-language.deck.md"
#: Words a deck viewer's file server refuses in any path part, as names that may hold a
#: credential; it also refuses hidden parts, ``.pem`` files, ``id_`` prefixes and ``state``.
_REFUSED_WORDS = ("key", "secret", "token")
_NODE_KEYS = ("shape", "style", "fillcolor", "color", "penwidth", "peripheries", "fontcolor")
_EDGE_KEYS = ("style", "color", "arrowhead", "arrowsize", "dir", "penwidth", "fontcolor")
_EDGE_LABEL_KEYS = ("label", "headlabel", "taillabel", "xlabel")
_CLUSTER_KEYS = ("style", "color", "pencolor", "penwidth", "fillcolor")
#: Environment variable naming the commit a build renders, for trees with no Git checkout
#: (a ``git archive`` export) or a checkout that is not the commit itself.
COMMIT_ENV = "TORCHLENS_DECK_COMMIT"
#: Trailer a tree-snapshot commit carries naming the commit whose tree it copies.
_SNAPSHOT_TRAILER = re.compile(r"^Source-Commit: ([0-9a-f]{40})$", re.MULTILINE)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def deck_slides() -> tuple[Slide, ...]:
    """Every slide a full build writes: teaching slides, the sentinel strip, then at scale."""

    return (*SLIDES, *sentinel_slides(), *scale_slides())


def _quote(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def _attrs(attrs: dict[str, str]) -> str:
    return " ".join(f"{k}={_quote(v)}" for k, v in attrs.items())


def alphabet_dot(slide_id: str, raw: Path, order: dict[str, int]) -> tuple[str, list[str]]:
    """A DOT sheet whose cells copy each exemplar's exact attributes from its slide."""

    # Essential complexity: one branch per mark kind (node, edge, cluster), each copying a
    # different attribute set into a pinned cell.

    entries = ALPHABET[slide_id]
    # A sheet of lines needs room for a short line, its words and a left-aligned meaning;
    # a sheet of nodes fits four across.
    has_lines = any("edge(" in selector for _m, _s, _p, selector in entries)
    cols = 3 if has_lines else 4
    cell_w, cell_h = (330.0, 50.0) if has_lines else (250.0, 54.0)
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
        left = x - cell_w / 2
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
                    # The words, not their 8 pt annotation size: the sheet shows the line.
                    attrs[key] = strip_tags(mark.attrs[key])
                    attrs.update(fontsize="13", labelfontsize="13")
            a, b = f"a{i}", f"b{i}"
            lines.append(f'{a} [shape=point width=0.06 pos="{left + 4:.1f},{y:.1f}!"]')
            lines.append(f'{b} [shape=point width=0.06 pos="{left + 76:.1f},{y:.1f}!"]')
            lines.append(f"{a} -> {b} [{_attrs(attrs)}]")
            # Plaintext nodes are centred on pos: shift by half the estimated width so the
            # meaning starts just right of the line.
            mid = left + 88 + 0.55 * 13 * len(label) / 2
            lines.append(
                f'm{i} [shape=plaintext label="{label}" pos="{mid:.1f},{y:.1f}!" fontsize=13 '
                'margin="0,0"]'
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


def render_all(
    raw: Path, only: set[str] | None = None, slides: tuple[Slide, ...] | None = None
) -> dict[str, dict[str, Any]]:
    """Capture and draw every panel serially; record errors instead of stopping."""

    raw.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for slide in deck_slides() if slides is None else slides:
        if only and slide.id not in only:
            continue
        for panel in slide.panels:
            stem = f"{slide.id}-{panel.name}"
            try:
                records[stem] = capture.render_panel(slide, panel, raw)
            except Exception as exc:  # noqa: BLE001 -- recorded per panel; the slide then fails
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
        **build_commit(),
        # timezone.utc, not datetime.UTC: the pinned environment's Python predates 3.11.
        "built": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30, check=False
        )
    except OSError:
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def build_commit(root: Path = _REPO_ROOT) -> dict[str, str]:
    """The commit this build renders, and where that answer came from.

    In order: the ``TORCHLENS_DECK_COMMIT`` environment variable (a tree exported with
    ``git archive`` has no Git data, so its caller names the commit); the ``Source-Commit``
    trailer of a tree-snapshot commit, which copies another commit's tree (the commit that
    matters is the one it copies, not the snapshot); the checkout's own ``HEAD``.
    """

    named = os.environ.get(COMMIT_ENV, "").strip()
    if named:
        return {"commit": named, "commit_source": "environment"}
    head = _git(root, "rev-parse", "HEAD")
    if head is None:
        return {"commit": "unknown", "commit_source": "no git checkout and no commit named"}
    trailer = _SNAPSHOT_TRAILER.search(_git(root, "log", "-1", "--format=%B", head) or "")
    if trailer and trailer.group(1) != head:
        return {
            "commit": trailer.group(1),
            "commit_source": "snapshot trailer",
            "checkout_commit": head,
        }
    return {"commit": head, "commit_source": "checkout"}


def _table_slide(slide: Slide, height: float, env: dict[str, Any], probe: dict[str, Any]) -> Any:
    import inspect

    from torchlens.data_classes._trace_viz import TraceVisualizationMixin

    params = inspect.signature(TraceVisualizationMixin.draw).parameters
    footer = f"TorchLens {env['torchlens']}, commit {env['commit'][:10]}, built {env['built'][:10]}"
    half = (len(CHEAT_GROUPS) + 1) // 2
    groups = CHEAT_GROUPS[half:] if slide.id.endswith("-2") else CHEAT_GROUPS[:half]
    if slide.id.startswith("cheat-sheet"):
        rows = [[see, how, ", ".join(names)] for see, how, names in groups]
        return compose.compose_table(
            rows, ["To see", "Pass", "All parameters in this group"], [0.25, 0.32, 0.43], height
        )
    if slide.id.startswith("draw-parameters"):
        items = []
        for _see, _how, names in groups:
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
    if slide.id == "unencoded":
        rows = [list(line) for line in UNENCODED]
        return compose.compose_table(
            rows, ["Legend line", "When TorchLens writes it"], [0.52, 0.48], height
        )
    if slide.id == "other-pictures":
        rows = [list(row[:3]) for row in OTHER_PICTURES]
        return compose.compose_table(
            rows, ["Picture", "What it shows", "How to call it"], [0.22, 0.48, 0.30], height
        )
    rows = [[k, str(v)] for k, v in probe.items()]
    return compose.compose_table(
        rows, ["Export choice", "What this build does"], [0.4, 0.6], height
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
    elif slide.layout == "table":
        canvas = _table_slide(slide, height, env, probe)
    else:
        renders = {}
        for panel in slide.panels:
            stem = f"{slide.id}-{panel.name}"
            if not (raw / f"{stem}.json").exists():
                problems.append(f"panel {panel.name} did not render")
                continue
            renders[panel.name] = compose.Render.load(raw, stem, panel.label, panel.crop)
        if problems:
            record.update(status="failed", problems=problems)
            return record
        if slide.layout == "grid":
            canvas = compose.compose_grid(slide, renders, height)
        else:
            canvas = compose.compose_auto(slide, list(renders.values()), height)
    problems.extend(canvas.problems)
    picture = out / f"S{number:02d}-{slide.id}.svg"
    canvas.write(picture)
    size = picture.stat().st_size
    if size > 150_000:
        problems.append(f"picture is {size // 1000} KB, over 150 KB")
    if slide.id.startswith(("sentinel-", "scale-")):
        # The strip and the at-scale pages are overviews: their full-size renders are in raw/
        # and the review PDF, so small text or a large page is a warning, never a failure.
        overview = [p for p in problems if "CSS px" in p or "KB, over" in p]
        problems = [p for p in problems if p not in overview]
        canvas.warnings.extend(overview)
    record.update(
        status="failed" if problems else "ok",
        problems=problems,
        picture=picture.name,
        bytes=size,
        min_css_px=round(canvas.min_css_px, 2),
        warnings=getattr(canvas, "warnings", []),
        witnesses=canvas.witnesses,
    )
    return record


def deck_markdown(records: list[dict[str, Any]], slides: tuple[Slide, ...] | None = None) -> str:
    """The deck file: one ``##`` title, the rule line and one picture per slide."""

    blocks = []
    for slide, record in zip(deck_slides() if slides is None else slides, records, strict=True):
        lines = [f"## {slide.title}", "", fill(slide.rule)]
        if record.get("picture"):
            lines += ["", f"![{slide.title}]({record['picture']})"]
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks) + "\n"


def _rendered_panels(records: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    """``(slide id, panel)`` for every panel whose render was written without error."""

    by_id = {slide.id: slide for slide in deck_slides()}
    out = []
    for record in records:
        slide = by_id.get(record["id"])
        if slide is None:
            continue
        for panel in slide.panels:
            panel_record = record.get("panels", {}).get(f"{slide.id}-{panel.name}")
            if panel_record is not None and "error" not in panel_record:
                out.append((slide.id, panel))
    return out


def _panel_words(panel: Any) -> set[str]:
    words = set(_flatten(dict(panel.kwargs))) | set(re.findall(r"[\w+]+", panel.label))
    return {str(word) for word in words}


def _variant_proof(
    variant: str, witnesses: dict[str, list[str]], slide: Slide | None, panels: list
) -> str | None:
    """How a row's variant is shown: a matched grid cell, or a rendered panel drawing it."""

    for cell in slide.cells if slide else ():
        if cell.select and witnesses.get(cell.select) and cell.label.split()[0] == variant:
            return f"cell {cell.select}"
    for slide_id, panel in panels:
        if variant in _panel_words(panel):
            return f"panel {slide_id}-{panel.name}"
    return None


def _row_proof(row: dict[str, Any], by_slide: dict[str, Any], panels: list) -> dict[str, Any]:
    """The evidence that a picture-witnessed row is drawn (finding F40).

    A row is proven by a keyed mark or grid cell its slide's selectors matched, or else by
    whole panels: every variant drawn by a rendered panel (or a matched cell) when the row
    lists variants; otherwise every panel of its slide rendered and the slide keys no mark.
    ``variants_unshown`` lists the variants no rendered panel or cell draws, for every row.
    The export slide's table is measured from real draws, so it proves its rows when built.
    """

    record = by_slide.get(row["slide"] or "", {})
    witnesses = record.get("witnesses", {})
    slide = next((s for s in SLIDES if s.id == row["slide"]), None)
    keyed = [k for k, names in row["dot_matches"].items() if names]
    cells = [
        c.select for c in (slide.cells if slide else ()) if c.select and witnesses.get(c.select)
    ]
    variants = {v: _variant_proof(v, witnesses, slide, panels) for v in row["variants"]}
    missing = sorted(v for v, how in variants.items() if how is None)
    own = [p for sid, p in panels if sid == row["slide"]]
    selectors = bool(row["keys"]) or any(c.select for c in (slide.cells if slide else ()))
    if keyed or cells:
        kind = "key" if keyed else "cell"
    elif row["variants"]:
        kind = None if missing else "panel"
    elif slide is not None and slide.layout == "table":
        kind = "measured" if record.get("status") == "ok" else None
    elif slide is not None and own and len(own) == len(slide.panels) and not selectors:
        kind = "panel"
    else:
        kind = None
    return {"proof": kind, "variant_proof": variants, "variants_unshown": missing}


def receipt(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per inventory row: slide, witness kind, the DOT marks its slide's keys matched, proof."""

    by_slide = {r["id"]: r for r in records}
    panels = _rendered_panels(records)
    rows = coverage.receipt_rows()
    for row in rows:
        witnesses = by_slide.get(row["slide"] or "", {}).get("witnesses", {})
        row["dot_matches"] = {
            key["select"]: witnesses.get(key["select"], []) for key in row["keys"]
        }
        row.update(_row_proof(row, by_slide, panels))
        row["proven_by_picture"] = row["witness"] == "picture" and row["proof"] is not None
    picture = [r for r in rows if r["proven_by_picture"]]
    counts = {
        "picture": len(picture),
        "picture_by_proof": {
            kind: sum(1 for r in picture if r["proof"] == kind)
            for kind in ("key", "cell", "panel", "measured")
        },
        "caption": sum(1 for r in rows if r["witness"] == "caption"),
        "inactive": sum(1 for r in rows if r["status"] == "inactive"),
        "picture_claimed_unproven": [
            r["row"] for r in rows if r["witness"] == "picture" and not r["proven_by_picture"]
        ],
    }
    return {"counts": counts, "rows": rows}


def name_problem(part: str) -> str | None:
    """Why a deck viewer would refuse to serve a path part with this name, or None."""

    lower = part.lower()
    if part.startswith("."):
        return "hidden name"
    if part == "state":
        return "named state"
    if any(word in lower for word in _REFUSED_WORDS) or lower.endswith(".pem"):
        return "may name a credential"
    if lower.startswith("id_"):
        return "may name a credential"
    return None


def generated_names() -> list[str]:
    """Every file and folder name a full build writes into its output folder."""

    names = [DECK_NAME, "slides.json", "coverage-receipt.json", "environment.json", "raw"]
    for number, slide in enumerate(deck_slides(), start=1):
        names.append(f"S{number:02d}-{slide.id}.svg")
        stems = [f"{slide.id}-{panel.name}" for panel in slide.panels]
        if slide.id in ALPHABET:
            stems.append(f"{slide.id}-a")
        names += [f"{stem}.{ext}" for stem in stems for ext in ("svg", "dot", "json")]
    names += [f"{stem}.{ext}" for stem in capture.EXPORT_STEMS for ext in ("png", "svg", "pdf")]
    return names


def build(out: Path, only: set[str] | None = None, slides: tuple[Slide, ...] | None = None) -> int:
    """Render, compose and write the deck into ``out``; return the number of failed slides.

    ``slides`` builds a part of the deck (the sentinel strip, say); the default is all of it.
    """

    slides = deck_slides() if slides is None else slides
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    panel_records = render_all(raw, only, slides)
    probe = capture.probe_export(raw)
    env = _environment()
    order = {slide.id: i for i, slide in enumerate(SLIDES, start=1)}
    records = []
    for number, slide in enumerate(slides, start=1):
        if only and slide.id not in only and slide.layout != "table":
            records.append({"id": slide.id, "number": number, "status": "skipped"})
            continue
        try:
            record = compose_slide(slide, raw, out, number, order, env, probe)
        except Exception as exc:  # noqa: BLE001 -- recorded; the slide fails
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
    (out / DECK_NAME).write_text(deck_markdown(records, slides))
    (out / "slides.json").write_text(json.dumps(records, indent=1, default=str))
    (out / "coverage-receipt.json").write_text(json.dumps(receipt(records), indent=1))
    (out / "environment.json").write_text(
        json.dumps({**env, "constants": constants()}, indent=1, default=str)
    )
    failed = [r["id"] for r in records if r["status"] == "failed"]
    # Fail closed on any written name a deck viewer would refuse to serve.
    refused = sorted(str(p.relative_to(out)) for p in out.rglob("*") if name_problem(p.name))
    print(json.dumps({"slides": len(records), "failed": failed, "refused_names": refused}))
    return len(failed) + len(refused)


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
