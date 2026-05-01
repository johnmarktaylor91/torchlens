"""Source-location link formatting helpers."""

from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import quote


def file_line_text(file_path: str, line_number: int | str | None) -> str:
    """Return a plain file-line label.

    Parameters
    ----------
    file_path:
        Source file path.
    line_number:
        Source line number, or ``None`` when unknown.

    Returns
    -------
    str
        ``path:line`` label when a line is known, otherwise ``path``.
    """

    if line_number in (None, "unknown"):
        return str(file_path)
    return f"{file_path}:{line_number}"


def terminal_file_line_link(file_path: str, line_number: int | str | None) -> str:
    """Return an OSC 8 terminal hyperlink for a source location.

    Parameters
    ----------
    file_path:
        Source file path.
    line_number:
        Source line number, or ``None`` when unknown.

    Returns
    -------
    str
        OSC 8 hyperlink whose visible text is ``path:line``.
    """

    label = file_line_text(file_path, line_number)
    resolved = Path(file_path).expanduser()
    try:
        resolved = resolved.resolve()
    except OSError:
        pass
    uri = f"file://{quote(str(resolved))}"
    if line_number not in (None, "unknown"):
        uri = f"{uri}:{line_number}"
    return f"\033]8;;{uri}\033\\{label}\033]8;;\033\\"


def vscode_file_line_link(
    file_path: str, line_number: int | str | None, label: str | None = None
) -> str:
    """Return an HTML anchor for opening a source location in VS Code.

    Parameters
    ----------
    file_path:
        Source file path.
    line_number:
        Source line number, or ``None`` when unknown.
    label:
        Optional visible link label.

    Returns
    -------
    str
        HTML ``<a>`` tag with a ``vscode://file`` target.
    """

    visible_label = label or file_line_text(file_path, line_number)
    resolved = Path(file_path).expanduser()
    try:
        resolved = resolved.resolve()
    except OSError:
        pass
    suffix = f":{line_number}" if line_number not in (None, "unknown") else ""
    href = f"vscode://file/{quote(str(resolved))}{suffix}"
    return f'<a href="{href}">{escape(visible_label)}</a>'


__all__ = ["file_line_text", "terminal_file_line_link", "vscode_file_line_link"]
