"""Regenerate the checked-in smart-collapse reference gallery.

Run from the repository root:

    PYTHONPATH=$PWD python scripts/render_collapse_reference.py

The default destination is ``docs/images/collapse``.  The script also copies
the finished gallery to ``/tmp/collapse_renders`` for visual review.
"""

from __future__ import annotations

import argparse
import html
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "notebooks" / "audit"
VISUAL_DIR = AUDIT_DIR / "visual"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "images" / "collapse"
REVIEW_OUT_DIR = Path("/tmp/collapse_renders")
IMAGE_NAMES = (
    "mode-none.svg",
    "mode-auto.svg",
    "mode-max.svg",
    "schedule-0.svg",
    "schedule-25.svg",
    "schedule-50.svg",
    "schedule-75.svg",
    "schedule-100.svg",
    "fold-off.svg",
    "fold-on.svg",
    "segments.svg",
    "recurrence.svg",
    "ellipsis.svg",
    "diagnostics.svg",
)

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AUDIT_DIR))
sys.path.insert(0, str(VISUAL_DIR))

import torch  # noqa: E402
import torchlens as tl  # noqa: E402
from _models import ZOO  # noqa: E402
from _visual_models import VZOO  # noqa: E402


def _verify_checkout() -> None:
    """Ensure renders use the TorchLens source in this checkout.

    Raises
    ------
    RuntimeError
        If Python resolved TorchLens from another checkout.
    """

    torchlens_path = Path(tl.__file__).resolve()
    if REPO_ROOT not in torchlens_path.parents:
        raise RuntimeError(f"torchlens imported from {torchlens_path}, not {REPO_ROOT}")


def _trace(model_key: str) -> Any:
    """Capture the named visual-audit model in evaluation mode.

    Parameters
    ----------
    model_key:
        Key from the visual audit's model zoo.

    Returns
    -------
    Trace
        The captured model execution.
    """

    model, example = {**ZOO, **VZOO}[model_key]()
    model.eval()
    with torch.no_grad():
        return tl.trace(model, example)


def _draw(trace: Any, output: Path, **kwargs: object) -> None:
    """Render one compact SVG graph.

    Parameters
    ----------
    trace:
        Captured trace to render.
    output:
        SVG path without its extension.
    **kwargs:
        Rendering options forwarded to :meth:`Trace.draw`.
    """

    trace.draw(
        vis_outpath=str(output.with_suffix("")),
        vis_fileformat="svg",
        vis_save_only=True,
        show_containers=False,
        **kwargs,
    )


def _diagnostics_svg(trace: Any, output: Path) -> None:
    """Write a text SVG showing actual plan and schedule diagnostics.

    Parameters
    ----------
    trace:
        Trace whose collapse diagnostics should be shown.
    output:
        Destination SVG path.
    """

    schedule = trace.collapse_schedule()
    rows = ["collapse_plan(mode='auto')", repr(trace.collapse_plan(mode="auto")), ""]
    rows.extend(["collapse_plan(mode='max')", repr(trace.collapse_plan(mode="max")), ""])
    rows.append("t      target  visible  collapsed module addresses")
    for step in schedule.steps:
        rows.append(
            f"{step.t:0.3f}  {step.target_count:>6}  {step.visible_count:>7}"
            f"  {len(step.collapsed_addresses):>5}"
        )
    width = 1180
    line_height = 18
    height = 48 + line_height * len(rows)
    text = "\n".join(
        f'<text x="20" y="{36 + line_height * index}">{html.escape(row)}</text>'
        for index, row in enumerate(rows)
    )
    output.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                '<rect width="100%" height="100%" fill="#fbfbfb" stroke="#d0d0d0"/>',
                "<style>text { font: 14px monospace; fill: #202020; }</style>",
                text,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def _copy_for_review(out_dir: Path) -> None:
    """Copy the gallery to the stable human-review staging directory.

    Parameters
    ----------
    out_dir:
        Directory containing the regenerated SVG files.
    """

    REVIEW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_name in IMAGE_NAMES:
        shutil.copy2(out_dir / image_name, REVIEW_OUT_DIR / image_name)


def render(out_dir: Path) -> None:
    """Render every reference image and copy it for human review.

    Parameters
    ----------
    out_dir:
        Directory where checked-in SVG images are written.
    """

    _verify_checkout()
    torch.manual_seed(1234)
    out_dir.mkdir(parents=True, exist_ok=True)

    resnet18 = _trace("resnet18")
    try:
        for mode in ("none", "auto", "max"):
            _draw(resnet18, out_dir / f"mode-{mode}.svg", collapse=mode)
        for label, value in (("0", 0.0), ("25", 0.25), ("50", 0.5), ("75", 0.75), ("100", 1.0)):
            _draw(resnet18, out_dir / f"schedule-{label}.svg", collapse=value)
        _diagnostics_svg(resnet18, out_dir / "diagnostics.svg")
    finally:
        resnet18.cleanup()

    block_stack = _trace("block_stack")
    try:
        _draw(block_stack, out_dir / "fold-off.svg", collapse="none", fold_repeats=False)
        _draw(block_stack, out_dir / "fold-on.svg", collapse="none", fold_repeats=True)
        _draw(block_stack, out_dir / "ellipsis.svg", collapse="auto", fold_repeats=True)
    finally:
        block_stack.cleanup()

    recurrence = _trace("rnn_cell_loop")
    try:
        _draw(recurrence, out_dir / "recurrence.svg", vis_mode="rolled")
    finally:
        recurrence.cleanup()

    resnet50 = _trace("resnet50")
    try:
        _draw(resnet50, out_dir / "segments.svg", collapse="max")
    finally:
        resnet50.cleanup()

    missing = [image_name for image_name in IMAGE_NAMES if not (out_dir / image_name).is_file()]
    if missing:
        raise RuntimeError(f"Missing rendered images: {', '.join(missing)}")
    _copy_for_review(out_dir)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and regenerate or verify the gallery.

    Parameters
    ----------
    argv:
        Optional command-line arguments, excluding the executable name.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--check", action="store_true", help="verify the expected checked-in image set"
    )
    args = parser.parse_args(argv)
    if args.check:
        missing = [name for name in IMAGE_NAMES if not (args.out_dir / name).is_file()]
        if missing:
            raise SystemExit(f"Missing collapse reference images: {', '.join(missing)}")
        return
    render(args.out_dir)


if __name__ == "__main__":
    main()
