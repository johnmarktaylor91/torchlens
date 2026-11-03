"""``torchlens agent init``: point a project's agent instruction file at the agent index.

The agent index (``torchlens/_docs/AGENT_INDEX.md``, generated from the task cards by
``tools/docs/build_agent_index.py``) ships in the wheel, so it always matches the installed
version. ``init`` writes a short marked block into an instruction file (``AGENTS.md`` by
default) naming the installed index; ``--remove`` deletes exactly that block. Text outside the
markers is never changed, and a file whose markers are damaged is refused, not rewritten.

Torch-free: the CLI runs this as a tier-0 verb.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path

#: Markers around the block ``init`` owns. Everything between them is replaced on re-run.
BLOCK_START = "<!-- torchlens-agent-index:start -->"
BLOCK_END = "<!-- torchlens-agent-index:end -->"
#: Instruction file written when no ``--file`` is given.
DEFAULT_INSTRUCTION_FILE = "AGENTS.md"


@dataclass(frozen=True)
class InitResult:
    """What ``init`` or ``--remove`` did to one instruction file.

    Attributes
    ----------
    path:
        The instruction file.
    action:
        ``created``, ``added``, ``updated`` or ``unchanged`` for ``init``;
        ``removed``, ``deleted`` (the file held only the block) or ``absent`` for ``--remove``.
    """

    path: str
    action: str


def installed_version() -> str:
    """Return the installed torchlens version (``unknown`` for a bare source checkout).

    Returns
    -------
    str
        The distribution version.

    Examples
    --------
    >>> isinstance(installed_version(), str)
    True
    """

    try:
        return _dist_version("torchlens")
    except PackageNotFoundError:  # pragma: no cover - source checkout without metadata
        return "unknown"


def index_path() -> Path:
    """Return the installed agent index file.

    Returns
    -------
    Path
        Absolute path of ``AGENT_INDEX.md`` inside the installed package.

    Examples
    --------
    >>> index_path().name
    'AGENT_INDEX.md'
    """

    return Path(__file__).resolve().parent.parent / "_docs" / "AGENT_INDEX.md"


def _version_note(version: str) -> str:
    """Return `` for torchlens <version>``, or nothing when the version is unknown."""

    return "" if not version or version == "unknown" else f" for torchlens {version}"


def render_block(index: Path, version: str) -> str:
    """Return the marked block that points an agent at ``index``.

    Parameters
    ----------
    index:
        The agent index file the block names.
    version:
        The installed torchlens version, stated in the block.

    Returns
    -------
    str
        The block, markers included, ending in a newline.

    Examples
    --------
    >>> render_block(Path("/x/AGENT_INDEX.md"), "2.0").splitlines()[0] == BLOCK_START
    True
    """

    return "\n".join(
        [
            BLOCK_START,
            "## TorchLens",
            "",
            f"Before writing TorchLens code, read the agent index{_version_note(version)}:",
            f"`{index}`",
            "It names the call to reach for first for each task and the wrong turns to avoid.",
            "Written by `torchlens agent init`; `torchlens agent init --remove` deletes it.",
            BLOCK_END,
            "",
        ]
    )


def _block_span(text: str) -> tuple[int, int] | None:
    """Return the ``(start, end)`` character span of the block, or ``None`` when absent."""

    starts, ends = text.count(BLOCK_START), text.count(BLOCK_END)
    if starts == 0 and ends == 0:
        return None
    start = text.find(BLOCK_START)
    end = text.find(BLOCK_END)
    if starts != 1 or ends != 1 or end < start:
        raise ValueError(
            "the torchlens block markers are damaged (expected one start marker before one end "
            "marker); fix or delete them by hand, then re-run torchlens agent init"
        )
    end += len(BLOCK_END)
    if text[end : end + 1] == "\n":
        end += 1
    return start, end


def install_block(path: Path, block: str) -> InitResult:
    """Write ``block`` into ``path``, creating the file or replacing only the old block.

    Parameters
    ----------
    path:
        The instruction file.
    block:
        The marked block (from :func:`render_block`).

    Returns
    -------
    InitResult
        What changed.

    Raises
    ------
    ValueError
        When the file's markers are damaged.

    Examples
    --------
    >>> import tempfile
    >>> target = Path(tempfile.mkdtemp()) / "AGENTS.md"
    >>> install_block(target, render_block(Path("/x/AGENT_INDEX.md"), "2.0")).action
    'created'
    """

    if not path.exists():
        path.write_text(block, encoding="utf-8")
        return InitResult(str(path), "created")
    text = path.read_text(encoding="utf-8")
    span = _block_span(text)
    if span is None:
        separator = (
            "" if not text or text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        )
        path.write_text(text + separator + block, encoding="utf-8")
        return InitResult(str(path), "added")
    start, end = span
    if text[start:end] == block:
        return InitResult(str(path), "unchanged")
    path.write_text(text[:start] + block + text[end:], encoding="utf-8")
    return InitResult(str(path), "updated")


def remove_block(path: Path) -> InitResult:
    """Delete the block from ``path``; delete the file too when nothing else is left.

    Parameters
    ----------
    path:
        The instruction file.

    Returns
    -------
    InitResult
        What changed.

    Raises
    ------
    ValueError
        When the file's markers are damaged.

    Examples
    --------
    >>> import tempfile
    >>> remove_block(Path(tempfile.mkdtemp()) / "AGENTS.md").action
    'absent'
    """

    if not path.exists():
        return InitResult(str(path), "absent")
    text = path.read_text(encoding="utf-8")
    span = _block_span(text)
    if span is None:
        return InitResult(str(path), "absent")
    start, end = span
    remaining = text[:start].rstrip("\n") + ("\n" if text[:start].strip() else "") + text[end:]
    if not remaining.strip():
        path.unlink()
        return InitResult(str(path), "deleted")
    path.write_text(remaining, encoding="utf-8")
    return InitResult(str(path), "removed")


def run_init(
    files: list[str] | None, *, remove: bool, torchlens_version: str | None = None
) -> list[InitResult]:
    """Apply ``init`` (or ``--remove``) to every named instruction file.

    Parameters
    ----------
    files:
        Instruction files; ``None`` or empty means ``AGENTS.md`` in the current directory.
    remove:
        Delete the block instead of writing it.
    torchlens_version:
        Version stated in the block (default: :func:`installed_version`).

    Returns
    -------
    list[InitResult]
        One result per file, in order.

    Raises
    ------
    ValueError
        When a file's markers are damaged (files before it are already done).

    Examples
    --------
    >>> import os, tempfile
    >>> os.chdir(tempfile.mkdtemp())
    >>> [r.action for r in run_init(None, remove=False, torchlens_version="2.0")]
    ['created']
    """

    targets = [Path(name).expanduser() for name in (files or [DEFAULT_INSTRUCTION_FILE])]
    if remove:
        return [remove_block(target) for target in targets]
    block = render_block(index_path(), torchlens_version or installed_version())
    return [install_block(target, block) for target in targets]
