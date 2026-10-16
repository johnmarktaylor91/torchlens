"""Review mode: render a branch against its base and write one "what changed" PDF.

``review(repo, base, head, out)`` exports each ref with ``git archive`` into its own tree
inside ``out`` (the working tree is never switched), builds the deck in each tree with that
tree's own generator, builds the sentinel strip twice with one strip definition (the
head's, or ``strip_from``'s when the refs predate the strip), once over each side's
``torchlens``, compares the DOT per slide id and writes ``changes.json`` and ``changes.pdf``: every slide whose DOT changed, base and head
side by side; then the sentinel strip before and after; then one line naming the unchanged
slides.

The comparison ignores only what differs between two builds by construction, never by
content: each rule in ``NORMALIZE_RULES`` names one such source. Everything else that
differs is a change.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

#: The parts of a ref the deck needs: the package it draws and the generator.
ARCHIVE_PATHS = ("torchlens", "scripts")
GENERATOR = "scripts/render_visual_language_deck.py"
COMMIT_ENV = "TORCHLENS_DECK_COMMIT"
SENTINEL_PREFIX = "sentinel-"
_FOOTER = re.compile(r"commit [0-9a-f]{7,40}, built \d{4}-\d\d-\d\d")


def _renumber_grad_fns(text: str) -> tuple[str, int]:
    """Backward node names carry ``id()`` of an autograd node: renumber by first appearance."""

    names: dict[str, str] = {}

    def swap(match: re.Match[str]) -> str:
        return names.setdefault(match.group(0), f"grad_fn_#{len(names) + 1}")

    out = re.sub(r"grad_fn_\d+", swap, text)
    return out, len(names)


def _sub(pattern: str, replacement: str) -> Callable[[str], tuple[str, int]]:
    compiled = re.compile(pattern)

    def apply(text: str) -> tuple[str, int]:
        return compiled.subn(replacement, text)

    return apply


#: (rule, why it differs by construction, function returning the text and a hit count).
NORMALIZE_RULES: tuple[tuple[str, str, Callable[[str], tuple[str, int]]], ...] = (
    (
        "grad_fn ids",
        "backward node names embed the Python id() of each autograd node, new every process",
        _renumber_grad_fns,
    ),
    (
        "measured durations",
        "wall-clock times (t=53.41 us, 135 us) are measured anew in every build",
        _sub(r"(?<![\w.])\d+(?:\.\d+)?\s?(?:ns|us|µs|ms)\b|(?<=t=)\d+(?:\.\d+)?\s?s\b", "<t>"),
    ),
    (
        "object addresses",
        "Python reprs print the memory address of the object, new every process",
        _sub(r" at 0x[0-9a-fA-F]{6,}", " at 0x<addr>"),
    ),
    (
        "build footer",
        "table slides print the commit and the build date, which differ between refs by design",
        _sub(_FOOTER.pattern, "commit <sha>, built <date>"),
    ),
)


def normalize(text: str, roots: tuple[str, ...] = ()) -> tuple[str, dict[str, int]]:
    """Text with every by-construction difference replaced; hit counts per rule."""

    hits: dict[str, int] = {}
    for root in roots:
        # Each side is built in its own folder: a printed path names that folder.
        count = text.count(root)
        if count:
            text = text.replace(root, "<tree>")
            hits["tree paths"] = hits.get("tree paths", 0) + count
    for name, _why, rule in NORMALIZE_RULES:
        text, count = rule(text)
        if count:
            hits[name] = count
    return text, hits


@dataclass
class SlideRender:
    """One slide as one build drew it."""

    id: str
    status: str = "ok"
    #: panel name to (normalized DOT, path of the original DOT)
    panels: dict[str, tuple[str, Path]] = field(default_factory=dict)
    text: str | None = None
    hits: dict[str, int] = field(default_factory=dict)

    def signature(self) -> tuple[Any, ...]:
        return (tuple(sorted((k, v[0]) for k, v in self.panels.items())), self.text)


def _svg_text(path: Path) -> str:
    texts = [el.text or "" for el in ET.parse(path).iter() if el.tag.endswith("text")]
    return "\n".join(t.strip() for t in texts if t.strip())


def read_build(deck: Path, roots: tuple[str, ...] = ()) -> tuple[list[str], dict[str, SlideRender]]:
    """Slide order and renders of one build, from ``slides.json`` and ``raw/*.dot``.

    A build that stopped before writing ``slides.json`` still has its panels in ``raw/``;
    they are read by stem (``<slide id>-<panel>``) so a failed build is compared, not lost.
    """

    renders: dict[str, SlideRender] = {}
    records = []
    if (deck / "slides.json").exists():
        records = json.loads((deck / "slides.json").read_text())
    for dot in sorted((deck / "raw").glob("*.dot")):
        slide_id, panel = dot.stem.rsplit("-", 1)
        render = renders.setdefault(slide_id, SlideRender(slide_id))
        text, hits = normalize(dot.read_text(), roots)
        render.panels[panel] = (text, dot)
        for name, count in hits.items():
            render.hits[name] = render.hits.get(name, 0) + count
    for record in records:
        render = renders.setdefault(record["id"], SlideRender(record["id"]))
        render.status = record.get("status", "ok")
        picture = record.get("picture")
        if not render.panels and picture and (deck / picture).exists():
            render.text, hits = normalize(_svg_text(deck / picture), roots)
            render.hits.update(hits)
    order = [r["id"] for r in records] or sorted(renders)
    order += [slide_id for slide_id in sorted(renders) if slide_id not in order]
    return order, renders


def compare(
    base: tuple[list[str], dict[str, SlideRender]], head: tuple[list[str], dict[str, SlideRender]]
) -> dict[str, list[str]]:
    """Slide ids changed, added, removed and unchanged, in head order then base-only order."""

    (base_order, base_r), (head_order, head_r) = base, head
    order = head_order + [s for s in base_order if s not in head_order]
    out: dict[str, list[str]] = {"changed": [], "added": [], "removed": [], "unchanged": []}
    for slide_id in order:
        if slide_id not in base_r:
            out["added"].append(slide_id)
        elif slide_id not in head_r:
            out["removed"].append(slide_id)
        elif (
            base_r[slide_id].signature() != head_r[slide_id].signature()
            or base_r[slide_id].status != head_r[slide_id].status
        ):
            out["changed"].append(slide_id)
        else:
            out["unchanged"].append(slide_id)
    return out


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if done.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def export_tree(repo: Path, sha: str, dest: Path, paths: tuple[str, ...] = ARCHIVE_PATHS) -> None:
    """Write ``paths`` of commit ``sha`` into ``dest`` with ``git archive``; no checkout moves."""

    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(
        ["git", "-C", str(repo), "archive", "--format=tar", sha, "--", *paths],
        stdout=subprocess.PIPE,
    )
    untar = subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout, check=False)
    if archive.stdout:
        archive.stdout.close()
    if archive.wait() or untar.returncode:
        raise SystemExit(f"git archive of {sha} {paths} into {dest} failed")


def _env(tree: Path, sha: str) -> dict[str, str]:
    path = os.environ.get("PYTHONPATH", "")
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in (str(tree), path) if p),
        COMMIT_ENV: sha,
    }


def check_import_root(tree: Path, sha: str) -> str:
    """Prove the tree's own ``torchlens`` is the one its build imports; return its path."""

    done = subprocess.run(
        [sys.executable, "-c", "import torchlens; print(torchlens.__file__)"],
        cwd=tree,
        env=_env(tree, sha),
        capture_output=True,
        text=True,
        check=False,
    )
    where = done.stdout.strip()
    if done.returncode or not Path(where).resolve().is_relative_to(tree.resolve()):
        raise SystemExit(f"the build in {tree} would import torchlens from {where or done.stderr}")
    return where


def run_generator(tree: Path, scripts: Path, command: str, out: Path, sha: str, log: Path) -> int:
    """Run ``scripts``'s generator inside ``tree``; the exit code (the log keeps the output)."""

    with log.open("w") as handle:
        done = subprocess.run(
            [sys.executable, str(scripts / GENERATOR), command, "--out", str(out)],
            cwd=tree,
            env=_env(tree, sha),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return done.returncode


def review(
    repo: Path, base: str, head: str, out: Path, strip_from: str | None = None
) -> dict[str, Any]:
    """Build both refs, compare them and write ``changes.json`` and ``changes.pdf``.

    The sentinel strip is drawn by ``strip_from``'s generator (default ``head``) over each
    side's own ``torchlens``, so both sides draw the same fixed set of renders.
    """

    from scripts.visual_language.review_pdf import write_changes_pdf

    out = out.resolve()

    def sha_of(ref: str) -> str:
        return _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")

    shas = {"base": sha_of(base), "head": sha_of(head)}
    strip_sha = sha_of(strip_from or head)
    trees = {"base": out / "base" / "tree", "head": out / "head" / "tree"}
    for side, tree in trees.items():
        export_tree(repo, shas[side], tree)
    strip_trees = {}
    for side in ("base", "head"):
        if strip_sha == shas[side]:
            strip_trees[side] = trees[side]
            continue
        strip_trees[side] = out / side / "strip-tree"
        export_tree(repo, shas[side], strip_trees[side], ("torchlens",))
        export_tree(repo, strip_sha, strip_trees[side], ("scripts",))
    sides: dict[str, Any] = {}
    builds, strips = {}, {}
    for side in ("base", "head"):
        root = out / side
        imported = check_import_root(trees[side], shas[side])
        check_import_root(strip_trees[side], shas[side])
        deck_rc = run_generator(
            trees[side], trees[side], "deck", root / "deck", shas[side], root / "deck.log"
        )
        strip_rc = run_generator(
            strip_trees[side],
            strip_trees[side],
            "sentinels",
            root / "strip",
            shas[side],
            root / "strip.log",
        )
        roots = (str(trees[side]), str(strip_trees[side]), str(root))
        builds[side] = read_build(root / "deck", roots)
        strips[side] = read_build(root / "strip", roots)
        sides[side] = {
            "ref": base if side == "base" else head,
            "sha": shas[side],
            "deck_exit": deck_rc,
            "strip_exit": strip_rc,
            "torchlens": imported,
        }
    teach = [_without_sentinels(builds[s]) for s in ("base", "head")]
    slides = compare(teach[0], teach[1])
    strip = compare(strips["base"], strips["head"])
    result = {
        "base": sides["base"],
        "head": sides["head"],
        **slides,
        "sentinel": strip,
        "strip_from": strip_sha,
        "normalization": {
            "tree paths": "each side is built in its own folder",
            **{name: why for name, why, _ in NORMALIZE_RULES},
        },
        "normalized": {
            side: {sid: r.hits for sid, r in builds[side][1].items() if r.hits}
            for side in ("base", "head")
        },
    }
    (out / "changes.json").write_text(json.dumps(result, indent=1))
    write_changes_pdf(out / "changes.pdf", result, teach, strips, strip_trees, trees)
    return result


def _without_sentinels(
    build: tuple[list[str], dict[str, SlideRender]],
) -> tuple[list[str], dict[str, SlideRender]]:
    order, renders = build
    keep = [s for s in order if not s.startswith(SENTINEL_PREFIX)]
    return keep, {k: v for k, v in renders.items() if not k.startswith(SENTINEL_PREFIX)}
