"""Round-2 options codemod: door keywords into ``CaptureOptions`` / ``RecordingOptions``.

Rewrites call sites of the ``trace`` and ``record`` doors in Python sources:

* ``halt=`` becomes ``stop_after=`` and ``streaming=`` becomes ``storage=``;
* ``profile=`` is dropped;
* the capture keywords the doors no longer take (``save_mode=``, ``grad_transform=``,
  ``lookback=``, ...) move into ``capture=CaptureOptions(...)``, merging into an existing
  ``CaptureOptions(...)`` constructor when the call already has one;
* ``save=SaveOptions(...)`` spills its keywords into ``capture=CaptureOptions(...)``;
* ``CaptureOptions(layers_to_save=...)`` moves to the enclosing door's ``save=``;
  standalone options move when all uses are visible in the same function; conflicts
  and ``save_preview=True`` require review, while ``save_preview=False`` is dropped;
* ``record`` recording keywords (``default_op=``, ``history_size=``, ...) move into
  ``recording=RecordingOptions(...)``, merging into an existing ``RecordingOptions(...)``
  constructor; an existing ``recording=`` stays as it is (``tl.recording.Recorder(...)``
  keeps its flat keywords);
* a third positional ``trace`` argument becomes ``input_kwargs=``;
* inside any ``CaptureOptions(...)`` constructor, ``raise_on_nan=`` / ``track_nonfinite=``
  become the one ``on_nonfinite=`` vocabulary.

A ``capture=`` whose value is not a constructor call (a variable) cannot be merged into and
is reported instead. The codemod is idempotent.

Usage::

    python tools/codemods/r2_options.py tests            # dry run, prints counts
    python tools/codemods/r2_options.py tests --apply    # write the edits
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

TRACE_CAPTURE_MOVES = (
    "save_mode",
    "grad_transform",
    "reconstruction_ready",
    "lookback",
    "lookback_payload_policy",
    "chunk_size",
    "grouping",
    "jax_static_argnums",
    "grad_options",
)
RECORD_CAPTURE_MOVES = (
    "lookback",
    "lookback_payload_policy",
    "seed",
    "activation_transform",
    "save_raw_activations",
    "save_grads",
    "grad_transform",
    "save_raw_grads",
    "backward_ready",
    "detach_saved_activations",
)
# ``record`` recorder keywords: grouped in ``recording=RecordingOptions(...)``.
RECORD_RECORDING_MOVES = (
    "default_op",
    "default_module",
    "history_size",
    "include_source_events",
    "max_predicate_failures",
    "on_predicate_error",
    "on_forward_error",
    "default_grad",
    "postprocess",
)
RENAMES = {"halt": "stop_after", "streaming": "storage"}
# RecordingOptions fields that left the public class (round 2, r2-options-2):
# the capture knobs are CaptureOptions fields, the rest are door keywords.
RECORDING_OPTIONS_CAPTURE_MOVES = (
    "lookback",
    "lookback_payload_policy",
    "seed",
    "activation_transform",
    "save_raw_activations",
    "save_grads",
    "grad_transform",
    "save_raw_grads",
    "detach_saved_activations",
)
RECORDING_OPTIONS_DOOR_MOVES = {"halt": "stop_after", "narrate": "narrate", "streaming": "storage"}
RECORDER_RENAMES = {"streaming": "storage"}
# Former public builders of CaptureOptions / RecordingOptions, now private with no
# public replacement: reported, never rewritten.
PRIVATE_BUILDERS = ("from_values", "is_field_explicit")
OPTION_CLASSES = ("CaptureOptions", "RecordingOptions")
DROPS = ("profile",)
DOORS = {
    "trace": "trace",
    "record": "record",
    "trace_fn": "trace",
    "record_fn": "record",
    "capture_trace": "trace",
}


def _door_name(func: ast.expr) -> str | None:
    # Cohesion: recognize the supported call syntax together, including lazy public doors.
    if isinstance(func, ast.Call):
        if isinstance(func.func, ast.Name) and func.func.id == "cast" and func.args:
            return _door_name(func.args[-1])
        if isinstance(func.func, ast.Name) and func.func.id == "_resolve_top_level" and func.args:
            name = func.args[0]
            return DOORS.get(name.value) if isinstance(name, ast.Constant) else None
    if isinstance(func, ast.Name):
        return DOORS.get(func.id)
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Call)
        and isinstance(func.value.func, ast.Name)
        and func.value.func.id == "_tl"
        and not func.value.args
    ):
        return DOORS.get(func.attr)
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id in {"tl", "torchlens", "_tl", "_user_funcs", "tl.bridge.transformers"}
    ):
        return DOORS.get(func.attr)
    return None


def _ctor_name(func: ast.expr, wanted: str) -> bool:
    if isinstance(func, ast.Name):
        return func.id == wanted
    if isinstance(func, ast.Attribute):
        return func.attr == wanted
    return False


class _Edits:
    def __init__(self, source: str) -> None:
        self.source = source
        self.lines = source.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))
        self.edits: list[tuple[int, int, str]] = []

    def offset(self, lineno: int, col: int) -> int:
        return self.starts[lineno - 1] + col

    def span(self, node: ast.AST) -> tuple[int, int]:
        return (
            self.offset(node.lineno, node.col_offset),
            self.offset(node.end_lineno, node.end_col_offset),  # type: ignore[arg-type]
        )

    def text(self, node: ast.AST) -> str:
        a, b = self.span(node)
        return self.source[a:b]

    def remove_arg(self, node: ast.AST, call: ast.Call) -> None:
        """Remove one argument with its separating comma."""
        a, b = self.span(node)
        src = self.source
        j = b
        while j < len(src) and src[j] in " \t\r\n":
            j += 1
        if j < len(src) and src[j] == ",":
            j += 1
            while j < len(src) and src[j] in " \t":
                j += 1
            if j < len(src) and src[j] == "\n":
                j += 1
                # drop the indentation of the removed line when it stood alone
                k = a
                while k > 0 and src[k - 1] in " \t":
                    k -= 1
                if k > 0 and src[k - 1] == "\n":
                    a = k
            self.edits.append((a, j, ""))
            return
        # last argument: eat the preceding comma
        i = a
        while i > 0 and src[i - 1] in " \t\r\n":
            i -= 1
        if i > 0 and src[i - 1] == ",":
            i -= 1
        self.edits.append((i, b, ""))

    def replace(self, a: int, b: int, text: str) -> None:
        self.edits.append((a, b, text))

    def apply(self) -> str:
        out = self.source
        for a, b, text in sorted(self._coalesced(), key=lambda e: (e[0], e[1]), reverse=True):
            out = out[:a] + text + out[b:]
        return out

    def _coalesced(self) -> list[tuple[int, int, str]]:
        """Merge overlapping deletions (adjacent removed arguments share a comma)."""

        merged: list[tuple[int, int, str]] = []
        for a, b, text in sorted(self.edits, key=lambda e: (e[0], e[1])):
            if merged and text == "" and merged[-1][2] == "" and a < merged[-1][1]:
                pa, pb, _ = merged[-1]
                merged[-1] = (pa, max(pb, b), "")
                continue
            merged.append((a, b, text))
        return merged


def _rewrite_nonfinite(
    kwargs: list[ast.keyword], ed: _Edits, call: ast.Call, counts: Counter
) -> None:
    raise_kw = next((k for k in kwargs if k.arg == "raise_on_nan"), None)
    track_kw = next((k for k in kwargs if k.arg == "track_nonfinite"), None)
    if raise_kw is None and track_kw is None:
        return

    def literal(kw: ast.keyword) -> bool | None:
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
            return kw.value.value
        return None

    r = literal(raise_kw) if raise_kw is not None else False
    t = literal(track_kw) if track_kw is not None else False
    if raise_kw is not None and r is None:
        # non-literal: forward the expression as the mode (type tests)
        a, b = ed.span(raise_kw)
        ed.replace(a, b, f"on_nonfinite={ed.text(raise_kw.value)}")
        if track_kw is not None:
            ed.remove_arg(track_kw, call)
        counts["nonfinite_expr"] += 1
        return
    mode = "raise" if r else ("track" if t else "ignore")
    first = raise_kw if raise_kw is not None else track_kw
    a, b = ed.span(first)
    ed.replace(a, b, f'on_nonfinite="{mode}"')
    other = track_kw if first is raise_kw else raise_kw
    if other is not None and other is not first:
        ed.remove_arg(other, call)
    counts["nonfinite"] += 1


def _ctor_kw(call: ast.Call, name: str, ctor: str) -> ast.keyword | None:
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Call) and _ctor_name(kw.value.func, ctor):
            return kw
    return None


def _append_into_ctor(ed: _Edits, ctor_call: ast.Call, pieces: list[str]) -> None:
    _, b = ed.span(ctor_call)
    inner_end = b - 1  # before ')'
    text = ed.source[ed.span(ctor_call)[0] : inner_end]
    has_args = bool(ctor_call.args or ctor_call.keywords)
    sep = ", " if has_args and not text.rstrip().endswith(",") else (" " if has_args else "")
    ed.replace(inner_end, inner_end, sep + ", ".join(pieces))


def _style(source: str, name: str) -> str:
    if f"tl.options.{name}" in source or "import torchlens as tl" in source:
        return f"tl.options.{name}"
    return name


def _rewrite_door(
    call: ast.Call, door: str, ed: _Edits, counts: Counter, reports: list[str], path: Path
) -> None:
    capture_moves = TRACE_CAPTURE_MOVES if door == "trace" else RECORD_CAPTURE_MOVES
    recording_moves = RECORD_RECORDING_MOVES if door == "record" else ()
    moved_capture: list[str] = []
    recording_kws: list[ast.keyword] = []
    for kw in list(call.keywords):
        if kw.arg is None:
            continue
        if kw.arg in RENAMES:
            a, _ = ed.span(kw)
            ed.replace(a, a + len(kw.arg), RENAMES[kw.arg])
            counts[f"rename:{kw.arg}"] += 1
        elif kw.arg in DROPS:
            ed.remove_arg(kw, call)
            counts[f"drop:{kw.arg}"] += 1
        elif kw.arg in capture_moves:
            moved_capture.append(f"{kw.arg}={ed.text(kw.value)}")
            ed.remove_arg(kw, call)
            counts[f"move:{kw.arg}"] += 1
        elif kw.arg in recording_moves:
            recording_kws.append(kw)
        elif (
            kw.arg == "save"
            and isinstance(kw.value, ast.Call)
            and _ctor_name(kw.value.func, "SaveOptions")
        ):
            for inner in kw.value.keywords:
                moved_capture.append(f"{inner.arg}={ed.text(inner.value)}")
            ed.remove_arg(kw, call)
            counts["save_options"] += 1
    if door == "record":
        for kw in call.keywords:
            if (
                kw.arg == "recording"
                and isinstance(kw.value, ast.Call)
                and _ctor_name(kw.value.func, "RecordingOptions")
            ):
                moved_capture.extend(
                    _rewrite_recording_options(kw.value, call, ed, counts, reports, path)
                )
    if door == "trace" and len(call.args) == 3:
        third = call.args[2]
        a, _ = ed.span(third)
        ed.replace(a, a, "input_kwargs=")
        counts["positional_input_kwargs"] += 1
    for pieces, name, ctor in ((moved_capture, "capture", "CaptureOptions"),):
        if not pieces:
            continue
        existing = next((k for k in call.keywords if k.arg == name), None)
        if (
            existing is not None
            and isinstance(existing.value, ast.Call)
            and _ctor_name(existing.value.func, ctor)
        ):
            _append_into_ctor(ed, existing.value, pieces)
        elif existing is not None and name == "capture":
            # a capture= variable: merge the moved fields through the public merge helper
            a, b = ed.span(existing.value)
            merge = _style(ed.source, "merge_capture_options")
            ed.replace(a, b, f"{merge}(capture={ed.text(existing.value)}, {', '.join(pieces)})")
            counts["merge_capture_variable"] += 1
        elif existing is not None:
            reports.append(
                f"{path}:{call.lineno}: {name}= is not a constructor; move by hand: {pieces}"
            )
        else:
            style = _style(ed.source, ctor)
            # append as a new keyword after the last argument
            last = (list(call.args) + list(call.keywords))[-1]
            _, b = ed.span(last)
            ed.replace(b, b, f", {name}={style}({', '.join(pieces)})")
    if recording_kws:
        _fold_recording_keywords(call, recording_kws, ed, counts, reports, path)


def _fold_recording_keywords(
    door_call: ast.Call,
    flat: list[ast.keyword],
    ed: _Edits,
    counts: Counter,
    reports: list[str],
    path: Path,
) -> None:
    """Fold a ``record`` door's flat recorder keywords into ``recording=RecordingOptions``.

    Merges into an existing ``RecordingOptions(...)`` constructor (a field it already
    sets is a conflict for review) or adds a new ``recording=`` keyword. A ``recording=``
    whose value is not a constructor call cannot be merged into and is reported. A
    reported call keeps its flat keywords untouched.
    """

    pieces = [f"{kw.arg}={ed.text(kw.value)}" for kw in flat]
    existing = next((k for k in door_call.keywords if k.arg == "recording"), None)
    ctor = existing.value if existing is not None else None
    if existing is not None and not (
        isinstance(ctor, ast.Call) and _ctor_name(ctor.func, "RecordingOptions")
    ):
        reports.append(
            f"{path}:{door_call.lineno}: recording= is not a constructor; move by hand: {pieces}"
        )
        return
    present = {k.arg for k in ctor.keywords} if isinstance(ctor, ast.Call) else set()
    clashes = [str(kw.arg) for kw in flat if kw.arg in present]
    if clashes:
        reports.append(
            f"{path}:{door_call.lineno}: {', '.join(clashes)} set both on record and in "
            "recording=RecordingOptions(...); move by hand"
        )
        return
    for kw in flat:
        ed.remove_arg(kw, door_call)
        counts[f"recording_fold:{kw.arg}"] += 1
    if isinstance(ctor, ast.Call):
        _append_into_ctor(ed, ctor, pieces)
        return
    style = _style(ed.source, "RecordingOptions")
    last = (list(door_call.args) + list(door_call.keywords))[-1]
    _, b = ed.span(last)
    ed.replace(b, b, f", recording={style}({', '.join(pieces)})")


def _rewrite_recording_options(
    call: ast.Call,
    door_call: ast.Call | None,
    ed: _Edits,
    counts: Counter,
    reports: list[str],
    path: Path,
) -> list[str]:
    """Move the removed ``RecordingOptions`` fields to their doors.

    Returns the capture pieces to fold into the enclosing ``record`` door's
    ``capture=``; door keywords are appended to the door call directly. A
    constructor outside a ``record`` call is reported for a hand move.
    """

    moved_capture: list[str] = []
    for kw in list(call.keywords):
        if kw.arg is None:
            continue
        if kw.arg in RECORDING_OPTIONS_CAPTURE_MOVES:
            if door_call is None:
                reports.append(
                    f"{path}:{call.lineno}: RecordingOptions({kw.arg}=) is a CaptureOptions "
                    "field now; move by hand"
                )
                continue
            moved_capture.append(f"{kw.arg}={ed.text(kw.value)}")
            ed.remove_arg(kw, call)
            counts[f"recording_options_move:{kw.arg}"] += 1
        elif kw.arg in RECORDING_OPTIONS_DOOR_MOVES:
            target = RECORDING_OPTIONS_DOOR_MOVES[kw.arg]
            if door_call is None or any(k.arg == target for k in door_call.keywords):
                reports.append(
                    f"{path}:{call.lineno}: RecordingOptions({kw.arg}=) is the door keyword "
                    f"{target}= now; move by hand"
                )
                continue
            last = (list(door_call.args) + list(door_call.keywords))[-1]
            _, b = ed.span(last)
            ed.replace(b, b, f", {target}={ed.text(kw.value)}")
            ed.remove_arg(kw, call)
            counts[f"recording_options_door:{kw.arg}"] += 1
    return moved_capture


def _report_private_builders(tree: ast.Module, reports: list[str], path: Path) -> None:
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in PRIVATE_BUILDERS:
            continue
        owner = node.func.value
        owner_name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", None)
        looks_like_option = owner_name is not None and re.search(
            "capture|recording", owner_name, re.IGNORECASE
        )
        if owner_name in OPTION_CLASSES or looks_like_option:
            reports.append(
                f"{path}:{node.lineno}: .{node.func.attr}() is private on CaptureOptions and "
                "RecordingOptions (no public replacement); review"
            )


def _capture_door_use(
    node: ast.AST, parents: dict[int, ast.AST]
) -> tuple[ast.Call, ast.keyword] | None:
    """Recognize a whole capture argument at a supported public door."""

    keyword = parents.get(id(node))
    door = parents.get(id(keyword))
    if (
        isinstance(keyword, ast.keyword)
        and keyword.arg == "capture"
        and isinstance(door, ast.Call)
        and _door_name(door.func) is not None
    ):
        return door, keyword
    return None


def _visible_capture_uses(
    ctor: ast.Call, parents: dict[int, ast.AST]
) -> list[tuple[ast.Call, ast.keyword]]:
    """Require one binding and exclusively visible same-function capture uses."""

    # Safety: binding, scope and use checks stay together so ambiguous rewrites fail closed.

    direct = _capture_door_use(ctor, parents)
    if direct is not None:
        return [direct]
    owner = parents.get(id(ctor))
    if not isinstance(owner, (ast.Assign, ast.AnnAssign)):
        return []
    targets = owner.targets if isinstance(owner, ast.Assign) else [owner.target]
    if len(targets) != 1 or not isinstance(targets[0], ast.Name):
        return []
    scope: ast.AST = owner
    while id(scope) in parents and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        scope = parents[id(scope)]
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    name = targets[0].id
    refs = [node for node in ast.walk(scope) if isinstance(node, ast.Name) and node.id == name]
    if sum(isinstance(node.ctx, ast.Store) for node in refs) != 1:
        return []
    uses = []
    for ref in (node for node in refs if isinstance(node.ctx, ast.Load)):
        context: ast.AST = ref
        while id(context) in parents and not isinstance(
            context, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ):
            context = parents[id(context)]
        use = _capture_door_use(ref, parents)
        if context is not scope or use is None or ref.lineno <= owner.lineno:
            return []
        uses.append(use)
    return uses


def _move_capture_selection(
    ctor: ast.Call,
    selection: ast.keyword,
    removed: list[ast.keyword],
    uses: list[tuple[ast.Call, ast.keyword]],
    ed: _Edits,
) -> None:
    """Keep whole-argument replacements separate from nested argument deletions."""

    remaining = [kw for kw in ctor.keywords if kw not in removed]
    owner = uses[0][1]
    if not remaining and not ctor.args and owner.value is ctor:
        a, b = ed.span(owner)
        ed.edits = [edit for edit in ed.edits if not (a <= edit[0] < b)]
        ed.replace(a, b, f"save={ed.text(selection.value)}")
        return
    ed.remove_arg(selection, ctor)
    for _door, keyword in uses:
        _, end = ed.span(keyword)
        ed.replace(end, end, f", save={ed.text(selection.value)}")


def _extract_capture_selection(
    tree: ast.Module, ed: _Edits, counts: Counter, reports: list[str], path: Path
) -> None:
    """Move removed CaptureOptions fields, reporting every ambiguous migration."""

    # Cohesion: coordinate nested edits in one pass so deletions cannot erase save= additions.

    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    aliases = {
        alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name == "CaptureOptions" and alias.asname
    }
    for ctor in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        imported_alias = isinstance(ctor.func, ast.Name) and ctor.func.id in aliases
        if not imported_alias and not _ctor_name(ctor.func, "CaptureOptions"):
            continue
        selection = next((kw for kw in ctor.keywords if kw.arg == "layers_to_save"), None)
        preview = next((kw for kw in ctor.keywords if kw.arg == "save_preview"), None)
        removed = [selection] if selection is not None else []
        if preview is not None:
            if isinstance(preview.value, ast.Constant) and preview.value.value is False:
                ed.remove_arg(preview, ctor)
                removed.append(preview)
                counts["drop:save_preview"] += 1
            else:
                reports.append(f"{path}:{ctor.lineno}: save_preview=True/expression; REVIEW")
        if selection is None:
            continue
        uses = _visible_capture_uses(ctor, parents)
        if not uses or any(any(kw.arg == "save" for kw in door.keywords) for door, _ in uses):
            reports.append(
                f"{path}:{ctor.lineno}: layers_to_save has no unique visible save= door or conflicts with save=; REVIEW"
            )
            continue
        _move_capture_selection(ctor, selection, removed, uses, ed)
        counts["capture_selection_to_save"] += 1


def _add_options_import(source: str, new: str) -> str:
    """Import the option classes a rewrite introduced when ``tl`` is not imported."""

    if "import torchlens as tl" in new:
        return new
    added = [name for name in OPTION_CLASSES if f"{name}(" in new and name not in source]
    if not added:
        return new
    return new.replace(
        "\nimport torch\n",
        f"\nimport torch\n\nfrom torchlens.options import {', '.join(added)}\n",
        1,
    )


def rewrite(path: Path, apply: bool, counts: Counter, reports: list[str]) -> bool:
    source = path.read_text()
    return rewrite_source(source, path, apply, counts, reports)


def rewrite_source(
    source: str,
    path: Path,
    apply: bool,
    counts: Counter,
    reports: list[str],
    result: list[str] | None = None,
) -> bool:
    """Rewrite one Python source block, preserving surrounding document text."""

    # Essential edit state machine: all migrations share one ordered edit accumulator.

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    ed = _Edits(source)
    _extract_capture_selection(tree, ed, counts, reports, path)
    nested_recording: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _door_name(node.func) != "record":
            continue
        for kw in node.keywords:
            if (
                kw.arg == "recording"
                and isinstance(kw.value, ast.Call)
                and _ctor_name(kw.value.func, "RecordingOptions")
            ):
                nested_recording.add(id(kw.value))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        door = _door_name(node.func)
        if door is not None:
            _rewrite_door(node, door, ed, counts, reports, path)
        if _ctor_name(node.func, "CaptureOptions"):
            _rewrite_nonfinite(node.keywords, ed, node, counts)
        if _ctor_name(node.func, "Recorder"):
            for kw in node.keywords:
                if kw.arg in RECORDER_RENAMES:
                    a, _ = ed.span(kw)
                    ed.replace(a, a + len(kw.arg), RECORDER_RENAMES[kw.arg])
                    counts[f"recorder_rename:{kw.arg}"] += 1
        if _ctor_name(node.func, "RecordingOptions") and id(node) not in nested_recording:
            _rewrite_recording_options(node, None, ed, counts, reports, path)
    _report_private_builders(tree, reports, path)
    if not ed.edits:
        return False
    new = _add_options_import(source, ed.apply())
    try:
        ast.parse(new)
    except SyntaxError as exc:
        reports.append(f"{path}: rewrite does not parse ({exc}); left untouched")
        return False
    if result is not None:
        result.append(new)
    elif apply:
        path.write_text(new)
    return True


def rewrite_document(path: Path, apply: bool, counts: Counter, reports: list[str]) -> bool:
    """Apply the Python migration to Markdown fences and notebook code cells."""

    # Cohesion: retain each container format while sharing the Python block rewrite.

    source = path.read_text()
    changed = False

    def block(text: str) -> str:
        nonlocal changed
        result: list[str] = []
        if rewrite_source(text, path, False, counts, reports, result):
            changed = True
            return result[0]
        return text

    if path.suffix == ".md":
        new = re.sub(
            r"(```(?:python|py)\n)(.*?)(\n```)",
            lambda m: m[1] + block(m[2]) + m[3],
            source,
            flags=re.DOTALL,
        )
    else:
        notebook = json.loads(source)
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                original = cell.get("source", [])
                rewritten = block("".join(original) if isinstance(original, list) else original)
                cell["source"] = (
                    rewritten.splitlines(keepends=True) if isinstance(original, list) else rewritten
                )
        new = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    if changed and apply:
        path.write_text(new)
    return changed


def main(argv: list[str] | None = None) -> int:
    # Cohesion: traverse source containers and aggregate one review report for the invocation.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    counts: Counter = Counter()
    reports: list[str] = []
    changed = 0
    for root in args.roots:
        root_path = Path(root)
        paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("*"))
        for path in paths:
            if path.suffix not in {".py", ".md", ".ipynb"} or not path.is_file():
                continue
            rewrite_file = rewrite if path.suffix == ".py" else rewrite_document
            if rewrite_file(path, args.apply, counts, reports):
                changed += 1
    for key, n in sorted(counts.items()):
        print(f"{key}: {n}")
    print(f"files changed: {changed}")
    for line in reports:
        print("REVIEW", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
