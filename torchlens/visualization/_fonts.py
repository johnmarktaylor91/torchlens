"""Bundled open font delivery for Graphviz renders.

The typography record pins the family name ``"Helvetica"`` (vizmech D29), but
which face a machine's fontconfig resolves that name to used to depend on the
machine: Nimbus Sans on one, DejaVu Sans (wider metrics) on another, so the
same graph laid out differently. FK3 ruled for bundling an open font over
fingerprinting a system one. TorchLens ships Liberation Sans 2.1.5 (SIL Open
Font License 1.1, unmodified files plus ``fonts/OFL.txt``), which is
metric-compatible with Helvetica/Arial, and every Graphviz subprocess gets a
fontconfig configuration that resolves the pinned family -- and ``Arial``, the
family Graphviz's PostScript alias table actually asks fontconfig for -- to it.
The emitted DOT keeps ``fontname="Helvetica"``, so no DOT byte changes; only
which face measures and draws the text is now fixed.

The configuration includes the caller's own fontconfig file (``FONTCONFIG_FILE``
when set, else the platform's system file), so every other family still
resolves as before. When the bundled files are missing (a stripped install)
or fontconfig is absent, renders fall back to the system fonts and
:func:`bundled_font_status` says so.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import atexit
import functools
import os
import shutil
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

__all__ = [
    "BUNDLED_FONT_FAMILY",
    "BUNDLED_FONT_FILES",
    "bundled_font_status",
    "graphviz_font_env",
]

#: The bundled face the pinned family resolves to.
BUNDLED_FONT_FAMILY = "Liberation Sans"

#: Font files shipped under ``torchlens/visualization/fonts``.
BUNDLED_FONT_FILES = ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf")

#: Family names the configuration maps onto the bundled face. The typography
#: record emits ``Helvetica`` (``Helvetica-Bold`` on the high-contrast skin),
#: and Graphviz's PostScript alias table turns that into the family ``Arial``
#: (at bold weight for ``-Bold``) before its pango plugin asks fontconfig --
#: measured: ``fontname: "Helvetica" resolved to: (ps:pango Arial, REGULAR)``.
#: Both names map, so the bundled face wins whichever spelling arrives.
_PINNED_FAMILIES = ("Helvetica", "Arial")

_FONT_DIR = Path(__file__).with_name("fonts")

#: System fontconfig files, first existing wins (Linux, Homebrew arm64/x86).
_SYSTEM_FONTCONFIG_FILES = (
    "/etc/fonts/fonts.conf",
    "/opt/homebrew/etc/fonts/fonts.conf",
    "/usr/local/etc/fonts/fonts.conf",
)


def _bundled_files_present() -> bool:
    """Return whether every bundled font file exists on disk."""

    return all((_FONT_DIR / name).is_file() for name in BUNDLED_FONT_FILES)


def _base_fontconfig_file() -> str | None:
    """Return the fontconfig file the bundled configuration layers on, if any."""

    user_file = os.environ.get("FONTCONFIG_FILE")
    if user_file:
        return user_file
    for candidate in _SYSTEM_FONTCONFIG_FILES:
        if os.path.isfile(candidate):
            return candidate
    return None


def _fontconfig_document(base_file: str | None) -> str:
    """Return the fontconfig XML that adds the bundled face and the family alias.

    Parameters
    ----------
    base_file:
        Fontconfig file to include first, so every other family keeps its
        system resolution; ``None`` adds a cache directory instead.

    Returns
    -------
    str
        The configuration document.
    """

    lines = [
        '<?xml version="1.0"?>',
        '<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">',
        "<fontconfig>",
    ]
    if base_file is not None:
        lines.append(f'  <include ignore_missing="yes">{escape(base_file)}</include>')
    else:
        lines.append('  <cachedir prefix="xdg">fontconfig</cachedir>')
    lines.append(f"  <dir>{escape(str(_FONT_DIR))}</dir>")
    for family in _PINNED_FAMILIES:
        # Prepended with a strong binding AFTER the included system rules,
        # so the bundled face wins over the system metric aliases.
        lines.extend(
            [
                '  <match target="pattern">',
                f'    <test qual="any" name="family"><string>{escape(family)}</string></test>',
                '    <edit name="family" mode="prepend" binding="strong">',
                f"      <string>{escape(BUNDLED_FONT_FAMILY)}</string>",
                "    </edit>",
                "  </match>",
            ]
        )
    lines.append("</fontconfig>")
    return "\n".join(lines) + "\n"


@functools.lru_cache(maxsize=4)
def _fontconfig_file_for(base_file: str | None) -> str:
    """Write the bundled configuration once per base file; return its path.

    Parameters
    ----------
    base_file:
        Fontconfig file the configuration includes.

    Returns
    -------
    str
        Path of the written configuration (removed at interpreter exit).
    """

    scratch = tempfile.mkdtemp(prefix="torchlens_fontconfig_")
    atexit.register(shutil.rmtree, scratch, True)
    path = os.path.join(scratch, "fonts.conf")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_fontconfig_document(base_file))
    return path


def graphviz_font_env() -> dict[str, str] | None:
    """Return the environment a Graphviz subprocess renders the bundled font with.

    Returns
    -------
    dict[str, str] | None
        A copy of ``os.environ`` with ``FONTCONFIG_FILE`` pointing at the
        bundled configuration, or ``None`` (inherit the caller's environment
        unchanged) when the bundled files are missing.
    """

    if not _bundled_files_present():
        return None
    env = dict(os.environ)
    env["FONTCONFIG_FILE"] = _fontconfig_file_for(_base_fontconfig_file())
    return env


def bundled_font_status() -> dict[str, object]:
    """Report the bundled-font delivery state (FK3 disclosure).

    Returns
    -------
    dict[str, object]
        ``family`` (the bundled face), ``available`` (files present),
        ``font_dir``, and ``base_fontconfig`` (the file layered under it).
    """

    return {
        "family": BUNDLED_FONT_FAMILY,
        "available": _bundled_files_present(),
        "font_dir": str(_FONT_DIR),
        "base_fontconfig": _base_fontconfig_file(),
    }
