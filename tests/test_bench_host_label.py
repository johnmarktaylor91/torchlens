"""Benchmark records never carry the machine's hostname or local paths.

Benchmark results and baselines are committed to a public repository. The writers
record a neutral host label (``TORCHLENS_BENCH_HOST_LABEL``, default
``"benchmark-host"``) and spell the checkout as ``<repo>``; these tests pin that,
and pin that no benchmark module reads the hostname except the label helper.
"""

from __future__ import annotations

import platform
import re
import socket
from pathlib import Path

from benchmarks import host_label

REPO_ROOT = Path(__file__).resolve().parents[1]
HOSTNAME_READERS = re.compile(
    r"platform\.node\(|gethostname\(|getfqdn\(|os\.uname\(|platform\.uname\("
)


def _real_names() -> set[str]:
    """Return the machine's hostname spellings (full and short)."""

    names = {socket.gethostname(), platform.node()}
    return {name for name in names | {n.split(".", 1)[0] for n in names} if name}


def test_only_the_label_helper_reads_the_hostname() -> None:
    """No benchmark writer calls a hostname API directly."""

    helper = Path(host_label.__file__).resolve()
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted((REPO_ROOT / "benchmarks").rglob("*.py"))
        if path.resolve() != helper and HOSTNAME_READERS.search(path.read_text())
    ]
    assert not offenders, f"benchmark modules reading the hostname directly: {offenders}"
