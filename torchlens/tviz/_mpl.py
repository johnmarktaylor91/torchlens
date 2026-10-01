"""matplotlib gate + shared figure chrome (tviz memo D3/D4, FORK-T1 branch A).

matplotlib is resolved at CALL time through :func:`require_matplotlib`; when
absent, every save-format request refuses typed (``tv_matplotlib_missing``)
printing the exact install command. The zero-dependency SVG/HTML emitters in
``_strip.py`` keep first contact alive on a bare install.

Rendering rules implemented here for every numeric picture:

- rect meshes, never ``imshow`` (an ``imshow`` can embed a raster inside an
  "SVG" file; Stage 0 rejects any ``<image>`` element in a numeric SVG);
- ``svg.fonttype`` defaults to ``'path'`` (renders identically everywhere);
  ``'none'`` is the opt-in for Illustrator/Inkscape editing;
- PDF is the documented paper format (vector, self-contained,
  text-extractable);
- figures are built on explicit ``Figure`` objects (never pyplot global
  state).

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._errors import refuse

__all__ = [
    "ATTENTION_CMAP",
    "DIVERGING_CMAP",
    "SUPPORTED_FORMATS",
    "figure",
    "require_matplotlib",
    "save_figure",
]

#: Sequential colormap for [0, 1] attention mass (colorblind-safe ramp).
ATTENTION_CMAP = "viridis"

#: Diverging colormap for signed scores/effects (zero-centered).
DIVERGING_CMAP = "RdBu_r"

#: The advertised save formats (tviz memo D1: genuine paper-ready outputs).
SUPPORTED_FORMATS = ("png", "svg", "pdf")

#: Fill color for masked cells -- OUTSIDE the data colormap, so a masked
#: position can never read as a small-but-present attention weight.
MASKED_CELL_COLOR = "#d9d9d9"

_INSTALL_COMMAND = 'pip install "torchlens[viz]"'


def require_matplotlib() -> Any:
    """Return the matplotlib package, refusing typed when it is absent.

    Returns
    -------
    Any
        The imported ``matplotlib`` package.

    Raises
    ------
    TvizError
        ``tv_matplotlib_missing`` with the exact install command when
        matplotlib is not importable (FORK-T1 branch A: call-time
        resolution; the dependency-free token-strip emitters still work).
    """

    try:
        import matplotlib
    except ImportError:
        refuse(
            code="tv_matplotlib_missing",
            message="This picture needs matplotlib, which is not installed.",
            remedy=f"run {_INSTALL_COMMAND} (the zero-dependency SVG/HTML token-strip "
            "emitters work without it)",
            install_command=_INSTALL_COMMAND,
        )
    return matplotlib


def figure(*, width: float, height: float) -> Any:
    """Return a fresh Agg-backed ``Figure`` (no pyplot global state).

    Parameters
    ----------
    width:
        Figure width in inches.
    height:
        Figure height in inches.

    Returns
    -------
    Any
        A ``matplotlib.figure.Figure`` with an attached Agg canvas.
    """

    require_matplotlib()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    return fig


def colorbar(fig: Any, mesh: Any, ax: Any, *, label: str) -> Any:
    """Attach a VECTOR colorbar (matplotlib rasterizes its solids by default).

    A default colorbar embeds an ``<image>`` element in SVG output -- exactly
    the raster-inside-vector class Stage 0 rejects -- so every tviz colorbar
    goes through this wrapper, which de-rasterizes the solids.

    Parameters
    ----------
    fig:
        The owning figure.
    mesh:
        The mappable (QuadMesh) the bar describes.
    ax:
        Axes (or list of axes) to steal space from.
    label:
        Colorbar label.

    Returns
    -------
    Any
        The ``Colorbar`` instance.
    """

    bar = fig.colorbar(mesh, ax=ax, label=label)
    bar.solids.set_rasterized(False)
    return bar


def _validate_format(fmt: str) -> None:
    """Refuse unknown save formats with the supported roster."""

    if fmt not in SUPPORTED_FORMATS:
        refuse(
            code="tv_record_invalid",
            message=f"Save format {fmt!r} is not supported.",
            remedy=f"pick one of {SUPPORTED_FORMATS}; PDF is the documented paper format",
            format=fmt,
        )


def save_figure(
    fig: Any,
    path: Path | str,
    *,
    svg_fonttype: str = "path",
    dpi: int = 200,
) -> Path:
    """Save a figure with the tviz output contract applied.

    Parameters
    ----------
    fig:
        The matplotlib ``Figure`` to save.
    path:
        Output path; the suffix selects the format (``.png``/``.svg``/
        ``.pdf``).
    svg_fonttype:
        ``'path'`` (default: text as paths, identical everywhere) or
        ``'none'`` (real text elements, the Illustrator/Inkscape opt-in).
    dpi:
        Raster resolution for PNG output.

    Returns
    -------
    Path
        The written path.
    """

    matplotlib = require_matplotlib()
    target = Path(path)
    fmt = target.suffix.lstrip(".").lower()
    _validate_format(fmt)
    if svg_fonttype not in ("path", "none"):
        refuse(
            code="tv_record_invalid",
            message=f"svg_fonttype {svg_fonttype!r} is not 'path' or 'none'.",
            remedy="use 'path' (default) or 'none' for editable text",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": svg_fonttype}):
        fig.savefig(target, format=fmt, dpi=dpi, bbox_inches="tight")
    return target


def paged_paths(path: Path | str, pages: int) -> list[Path]:
    """Return numbered page paths for multi-page file output (D21).

    Parameters
    ----------
    path:
        The requested output path.
    pages:
        Total page count; one page returns the path unchanged.

    Returns
    -------
    list[Path]
        ``name-page01.ext``-style paths in page order.
    """

    target = Path(path)
    if pages <= 1:
        return [target]
    width = max(2, len(str(pages)))
    return [
        target.with_name(f"{target.stem}-page{page + 1:0{width}d}{target.suffix}")
        for page in range(pages)
    ]
