"""Lazy capability probe for ``cairosvg``, the SVG rasterizer behind PDF and PNG composition.

``cairosvg`` is a pip package wrapping the system ``libcairo``; ``import cairosvg``
raises ``ImportError`` when the package is absent and ``OSError`` when the package
is present but the shared library is not (a stock macOS without Homebrew's cairo).
Every renderer that needs it reads :func:`get_cairosvg_support` instead of importing
it, so one probe answers for the highlighter, side-by-side code panels and the
doctor row, and the reason it is unavailable names the missing piece.
"""

from __future__ import annotations

HAS_CAIROSVG: bool = False
_CAIROSVG_PROBED: bool = False
_CAIROSVG_UNAVAILABLE_REASON: str = "not probed yet"


def _probe_cairosvg() -> tuple[bool, str]:
    """Import ``cairosvg`` once and return ``(available, reason)``.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` when the import succeeds; otherwise ``False`` with the first
        line of the import error, which names the missing package or library.
    """

    try:
        import cairosvg  # noqa: F401
    except ImportError as exc:
        return False, f"cairosvg package not installed ({_first_line(exc)})"
    except OSError as exc:
        return False, f"cairosvg installed but its cairo library is missing ({_first_line(exc)})"
    return True, ""


def _first_line(exc: BaseException) -> str:
    """Return the first line of an exception message (the part that names the library)."""

    return str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__


def get_cairosvg_support(*, force_probe: bool = False) -> bool:
    """Return whether ``cairosvg`` can rasterize SVG on this install, lazily.

    Parameters
    ----------
    force_probe:
        Re-run the import probe even after the first call (diagnostics).

    Returns
    -------
    bool
        ``HAS_CAIROSVG``: ``True`` when ``import cairosvg`` succeeds. Cached after
        the first call so no render path pays the import twice.
    """

    global HAS_CAIROSVG, _CAIROSVG_PROBED, _CAIROSVG_UNAVAILABLE_REASON

    if not _CAIROSVG_PROBED or force_probe:
        HAS_CAIROSVG, _CAIROSVG_UNAVAILABLE_REASON = _probe_cairosvg()
        _CAIROSVG_PROBED = True
    return HAS_CAIROSVG


def cairosvg_unavailable_reason() -> str:
    """Return why ``cairosvg`` is unavailable (empty when it is), probing if needed."""

    get_cairosvg_support()
    return _CAIROSVG_UNAVAILABLE_REASON
