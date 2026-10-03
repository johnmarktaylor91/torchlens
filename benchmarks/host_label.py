"""Neutral machine identity for recorded benchmark artifacts.

Benchmark records are committed to a public repository, so they name the machine
with a neutral label instead of its hostname. Comparisons that need hardware
context read ``cpu_model`` and ``cpu_count``, which the writers record separately.
Absolute checkout paths in captured subprocess output are rewritten relative to
the repository for the same reason.
"""

from __future__ import annotations

import os
import platform
import socket
from pathlib import Path

HOST_LABEL_ENV = "TORCHLENS_BENCH_HOST_LABEL"
DEFAULT_HOST_LABEL = "benchmark-host"
REPO_PLACEHOLDER = "<repo>"


def _real_host_names() -> set[str]:
    """Return the names this machine answers to, lowercased.

    Returns
    -------
    set[str]
        The socket hostname, its short form, and the platform node name.
    """

    names = {socket.gethostname(), platform.node()}
    names |= {name.split(".", 1)[0] for name in names}
    return {name.lower() for name in names if name}


def benchmark_host_label() -> str:
    """Return the neutral label a benchmark record uses for this machine.

    The label comes from ``TORCHLENS_BENCH_HOST_LABEL`` and defaults to
    ``"benchmark-host"``. A label equal to one of the machine's real host names
    falls back to the default, so a record never carries the hostname.

    Returns
    -------
    str
        The label to record in the ``hostname`` field.
    """

    label = os.environ.get(HOST_LABEL_ENV, "").strip()
    if not label or label.lower() in _real_host_names():
        return DEFAULT_HOST_LABEL
    return label


def redact_local_paths(text: str, repo_root: Path) -> str:
    """Rewrite absolute checkout and home paths in recorded text.

    Parameters
    ----------
    text:
        Captured output or a path string.
    repo_root:
        The repository root whose absolute spelling is replaced by ``<repo>``.

    Returns
    -------
    str
        Text with the repository root spelled ``<repo>`` and the home directory ``~``.
    """

    for root in {str(repo_root), str(repo_root.resolve())}:
        text = text.replace(root, REPO_PLACEHOLDER)
    home = str(Path.home())
    if home not in {"", "/"}:
        text = text.replace(home, "~")
    return text
