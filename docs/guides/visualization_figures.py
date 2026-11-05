"""Regenerate every image on the visualization guide (``docs/guides/visualization.md``).

Each figure is one row of :data:`FIGURES`: a small model, its input, and the exact
``trace.draw(...)`` keywords the page shows beside the image. The page's code lines
are built from the same rows (:func:`draw_call`), and a test checks that the page and
this table agree, so the code under each picture is the code that drew it.

Usage::

    python docs/guides/visualization_figures.py              # every figure
    python docs/guides/visualization_figures.py --only rolled unrolled
    python docs/guides/visualization_figures.py --dot-dir /tmp/dot   # also save DOT source

Images land in ``docs/images/visualization/``, as PNG except the ``show_cone`` figure,
which is SVG: TorchLens draws the downstream shading into SVG itself, while a PNG needs
the optional ``cairosvg`` package and its native library. Rendering needs Graphviz. A
drawing that warns it left out something its keywords asked for fails instead of being
written.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensError

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "docs" / "images" / "visualization"
DPI = 96


# ---------------------------------------------------------------------------
# Models: each one small enough that its whole drawing reads at page width.
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    """Two linear layers with a ReLU between them."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class ConvBlock(nn.Module):
    """Convolution, batch norm and ReLU."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SmallCNN(nn.Module):
    """Two nested convolution blocks and a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(ConvBlock(3, 8), ConvBlock(8, 16))
        self.head = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x.mean(dim=(2, 3)))


class NormNet(nn.Module):
    """A linear layer, a batch norm and a ReLU: one module with buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.norm = nn.BatchNorm1d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(self.fc(x)))


class Block(nn.Module):
    """A residual block: layer norm, then a two-layer MLP submodule."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class Stack(nn.Module):
    """An embedding layer, a stack of residual blocks and a readout."""

    def __init__(self, num_blocks: int = 4) -> None:
        super().__init__()
        self.embed = nn.Linear(8, 16)
        self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])
        self.readout = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.blocks(self.embed(x)))


class LoopNet(nn.Module):
    """One recurrent cell applied three times to the same input."""

    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(8, 8)
        self.rec = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros_like(x)
        for _ in range(3):
            h = torch.tanh(self.inp(x) + self.rec(h))
        return h


class Gate(nn.Module):
    """Takes one of two branches depending on the sign of the input's mean."""

    def __init__(self) -> None:
        super().__init__()
        self.pos = nn.Linear(8, 8)
        self.neg = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.mean() > 0:
            return torch.relu(self.pos(x))
        return -self.neg(x)


class FillDemo(nn.Module):
    """One node for each default fill: trainable, frozen, mixed weights and a branch test."""

    def __init__(self) -> None:
        super().__init__()
        self.trainable = nn.Linear(8, 8)
        self.frozen = nn.Linear(8, 8)
        self.mixed = nn.Linear(8, 8)
        self.frozen.requires_grad_(False)
        self.mixed.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.trainable(x))
        x = self.mixed(self.frozen(x))
        if x.sum() > 0:
            x = x * 2
        return x


# ---------------------------------------------------------------------------
# The figure table.
# ---------------------------------------------------------------------------


def _mlp() -> tuple[nn.Module, torch.Tensor]:
    return TinyMLP(), torch.randn(2, 8)


def _cnn() -> tuple[nn.Module, torch.Tensor]:
    return SmallCNN(), torch.randn(1, 3, 8, 8)


def _norm() -> tuple[nn.Module, torch.Tensor]:
    return NormNet(), torch.randn(4, 8)


def _stack() -> tuple[nn.Module, torch.Tensor]:
    return Stack(), torch.randn(2, 8)


def _stack8() -> tuple[nn.Module, torch.Tensor]:
    return Stack(num_blocks=8), torch.randn(2, 8)


def _deep_stack() -> tuple[nn.Module, torch.Tensor]:
    return Stack(num_blocks=40), torch.randn(2, 8)


def _loop() -> tuple[nn.Module, torch.Tensor]:
    return LoopNet(), torch.randn(2, 8)


def _gate() -> tuple[nn.Module, torch.Tensor]:
    return Gate(), torch.ones(2, 8)


def _fills() -> tuple[nn.Module, torch.Tensor]:
    return FillDemo(), torch.ones(2, 8)


@dataclass(frozen=True)
class Figure:
    """One image on the guide.

    Attributes
    ----------
    name:
        File stem under ``docs/images/visualization``.
    build:
        Returns a fresh model and its input.
    kwargs:
        The ``trace.draw`` keywords, exactly as the page shows them.
    edit:
        Optional ``(module address, edit name)`` applied to a fork before
        drawing (the intervention figure).
    backward:
        Run ``trace.backward`` on the summed output before drawing.
    file_format:
        Image format, also the file extension.
    """

    name: str
    build: Callable[[], tuple[nn.Module, torch.Tensor]]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    edit: tuple[str, str] | None = None
    backward: bool = False
    file_format: str = "png"

    @property
    def filename(self) -> str:
        """The image's file name under ``docs/images/visualization``."""

        return f"{self.name}.{self.file_format}"


FIGURES: tuple[Figure, ...] = (
    Figure("first_drawing", _cnn),
    Figure("node_anatomy", _mlp),
    Figure("fills", _fills),
    Figure("legend", _fills, {"show_legend": True}),
    Figure("branch", _gate),
    Figure("unrolled", _loop),
    Figure("rolled", _loop, {"rolled": True}),
    Figure("depth_1", _cnn, {"depth": 1}),
    Figure("depth_2", _cnn, {"depth": 2}),
    Figure("collapse_quarter", _stack8, {"collapse": 0.25}),
    Figure("collapse_half", _stack8, {"collapse": 0.5}),
    Figure("collapse_max", _stack8, {"collapse": "max"}),
    Figure("collapse_auto", _deep_stack, {"collapse": "auto"}),
    Figure("module_focus", _stack, {"module": "blocks.1"}),
    Figure("profiling", _cnn, {"node_style": "profiling"}),
    Figure("color_size", _cnn, {"color_by": "time", "size_by": "bytes"}),
    Figure(
        "interventions",
        _mlp,
        {"view": "interventions"},
        edit=("fc1", "zero_ablate"),
    ),
    Figure("edited", _mlp, edit=("fc1", "zero_ablate")),
    Figure(
        "show_cone",
        _mlp,
        {"show_cone": True},
        edit=("fc1", "zero_ablate"),
        file_format="svg",
    ),
    Figure("leftright", _mlp, {"direction": "leftright"}),
    Figure("buffers_always", _norm, {"show_buffers": "always"}),
    Figure("label_fields", _mlp, {"node_label_fields": ["name", "shape", "dtype"]}),
    Figure("saved_for_backward", _mlp, {"show_saved_for_backward": True}),
    Figure(
        "backward_paths",
        _mlp,
        {"graph": "backward", "show_module_paths": True},
        backward=True,
    ),
    Figure("theme_paper", _mlp, {"theme": "paper"}),
    Figure("theme_dark", _mlp, {"theme": "dark"}),
    Figure("theme_colorblind", _fills, {"theme": "colorblind"}),
    Figure("theme_high_contrast", _mlp, {"theme": "high_contrast"}),
)


def draw_call(figure: Figure) -> str:
    """Return the ``trace.draw(...)`` line the page shows for ``figure``.

    Parameters
    ----------
    figure:
        A row of :data:`FIGURES`.

    Returns
    -------
    str
        The call, keywords in table order, values as Python literals.
    """

    target = "edited" if figure.edit is not None else "trace"
    args = ", ".join(f"{key}={value!r}" for key, value in figure.kwargs.items())
    return f"{target}.draw({args})"


#: Warning codes meaning the image lacks something its keywords asked for.
_DROPPED_FEATURE_CODES = frozenset({"viz_highlighter_not_drawn"})


def _refuse_dropped_features(caught: list[warnings.WarningMessage]) -> None:
    """Raise when a drawing warned that it left out a requested feature.

    Parameters
    ----------
    caught:
        Warnings recorded while drawing.

    Raises
    ------
    RuntimeError
        A warning carried a code in :data:`_DROPPED_FEATURE_CODES`.
    """

    for item in caught:
        fields = getattr(item.message, "fields", None)
        if isinstance(fields, dict) and fields.get("code") in _DROPPED_FEATURE_CODES:
            raise RuntimeError(str(item.message))


def _move_sidecars(figure: Figure, dot_dir: Path | None) -> None:
    """Keep only images in the image folder.

    ``view="interventions"`` also writes the full edit record as
    ``<name>.census.txt`` beside the image. It moves to ``dot_dir`` when one is
    given and is deleted otherwise (the page shows the caption, not the file).

    Parameters
    ----------
    figure:
        The row just rendered.
    dot_dir:
        Folder for non-image outputs, or ``None``.
    """

    sidecar = OUTPUT_DIR / f"{figure.name}.census.txt"
    if not sidecar.exists():
        return
    if dot_dir is not None:
        sidecar.replace(dot_dir / sidecar.name)
    else:
        sidecar.unlink()


def render(figure: Figure, dot_dir: Path | None) -> Path:
    """Trace ``figure``'s model and write its PNG.

    Parameters
    ----------
    figure:
        The row to render.
    dot_dir:
        When given, the DOT source is also written there as ``<name>.gv``.

    Returns
    -------
    Path
        The written image.
    """

    torch.manual_seed(0)
    model, x = figure.build()
    trace = tl.trace(model, x)
    target = trace
    try:
        if figure.backward:
            trace.backward(trace.output.sum())
        if figure.edit is not None:
            address, edit_name = figure.edit
            target = trace.fork()
            result = target.do(tl.module(address), getattr(tl, edit_name)())
            if isinstance(result, tl.Trace):
                target = result
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            drawing = target.draw(
                str(OUTPUT_DIR / figure.name),
                file_format=figure.file_format,
                dpi=DPI,
                **figure.kwargs,
            )
        _refuse_dropped_features(caught)
        if dot_dir is not None:
            dot_dir.mkdir(parents=True, exist_ok=True)
            (dot_dir / f"{figure.name}.gv").write_text(drawing.dot, encoding="utf-8")
        _move_sidecars(figure, dot_dir)
        return OUTPUT_DIR / figure.filename
    finally:
        if target is not trace:
            target.cleanup()
        trace.cleanup()


def main(argv: list[str] | None = None) -> int:
    """Render the selected figures; return 1 when any failed.

    Parameters
    ----------
    argv:
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit status.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", default=None, help="figure names to render")
    parser.add_argument("--dot-dir", type=Path, default=None, help="also save DOT source here")
    args = parser.parse_args(argv)
    wanted = set(args.only) if args.only else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for figure in FIGURES:
        if wanted is not None and figure.name not in wanted:
            continue
        try:
            path = render(figure, args.dot_dir)
        except (TorchLensError, RuntimeError, ValueError, TypeError, OSError) as exc:
            # Report every failing figure, then exit 1.
            failures.append(figure.name)
            print(f"FAILED {figure.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print(f"rendered {path.relative_to(REPO_ROOT)}  {draw_call(figure)}")
    if failures:
        print(f"{len(failures)} figure(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
