"""Round-2 options codemod: door keywords into ``CaptureOptions`` / ``RecordingOptions``.

Rewrites call sites of the ``trace`` and ``record`` doors in Python sources:

* ``halt=`` becomes ``stop_after=`` and ``streaming=`` becomes ``storage=``;
* ``profile=`` is dropped;
* the capture keywords the doors no longer take (``save_mode=``, ``grad_transform=``,
  ``lookback=``, ...) move into ``capture=CaptureOptions(...)``, merging into an existing
  ``CaptureOptions(...)`` constructor when the call already has one;
* ``save=SaveOptions(...)`` spills its keywords into ``capture=CaptureOptions(...)``;
* ``record`` recording keywords (``default_op=``, ``history_size=``, ...) move into
  ``recording=RecordingOptions(...)``;
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
DROPS = ("profile",)
DOORS = {"trace": "trace", "record": "record", "trace_fn": "trace", "record_fn": "record"}


def _door_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return DOORS.get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id in {"tl", "torchlens"}:
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
        for a, b, text in sorted(self.edits, key=lambda e: (e[0], e[1]), reverse=True):
            out = out[:a] + text + out[b:]
        return out


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
    moved_capture: list[str] = []
    moved_recording: list[str] = []
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
        elif door == "record" and kw.arg in RECORD_RECORDING_MOVES:
            moved_recording.append(f"{kw.arg}={ed.text(kw.value)}")
            ed.remove_arg(kw, call)
            counts[f"move:{kw.arg}"] += 1
        elif (
            kw.arg == "save"
            and isinstance(kw.value, ast.Call)
            and _ctor_name(kw.value.func, "SaveOptions")
        ):
            for inner in kw.value.keywords:
                moved_capture.append(f"{inner.arg}={ed.text(inner.value)}")
            ed.remove_arg(kw, call)
            counts["save_options"] += 1
    if door == "trace" and len(call.args) >= 3:
        third = call.args[2]
        a, _ = ed.span(third)
        ed.replace(a, a, "input_kwargs=")
        counts["positional_input_kwargs"] += 1
    for pieces, name, ctor in (
        (moved_capture, "capture", "CaptureOptions"),
        (moved_recording, "recording", "RecordingOptions"),
    ):
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


def rewrite(path: Path, apply: bool, counts: Counter, reports: list[str]) -> bool:
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    ed = _Edits(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        door = _door_name(node.func)
        if door is not None:
            _rewrite_door(node, door, ed, counts, reports, path)
        if _ctor_name(node.func, "CaptureOptions"):
            _rewrite_nonfinite(node.keywords, ed, node, counts)
    if not ed.edits:
        return False
    new = ed.apply()
    if "CaptureOptions(" in new and "CaptureOptions" not in source:
        if "import torchlens as tl" not in new:
            new = new.replace(
                "\nimport torch\n",
                "\nimport torch\n\nfrom torchlens.options import CaptureOptions\n",
                1,
            )
    if (
        "RecordingOptions(" in new
        and "RecordingOptions" not in source
        and "import torchlens as tl" not in new
    ):
        new = new.replace(
            "\nimport torch\n",
            "\nimport torch\n\nfrom torchlens.options import RecordingOptions\n",
            1,
        )
    try:
        ast.parse(new)
    except SyntaxError as exc:
        reports.append(f"{path}: rewrite does not parse ({exc}); left untouched")
        return False
    if apply:
        path.write_text(new)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    counts: Counter = Counter()
    reports: list[str] = []
    changed = 0
    for root in args.roots:
        for path in sorted(Path(root).rglob("*.py")):
            if rewrite(path, args.apply, counts, reports):
                changed += 1
    for key, n in sorted(counts.items()):
        print(f"{key}: {n}")
    print(f"files changed: {changed}")
    for line in reports:
        print("REVIEW", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
