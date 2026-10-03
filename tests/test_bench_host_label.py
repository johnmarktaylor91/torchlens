"""Benchmark records never carry the machine's hostname or local paths.

Benchmark results and baselines are committed to a public repository. The writers
record a neutral host label (``TORCHLENS_BENCH_HOST_LABEL``, default
``"benchmark-host"``) and spell the checkout as ``<repo>``; these tests pin that,
and pin that no benchmark module reads the hostname except the label helper.
"""

from __future__ import annotations

import json
import platform
import re
import socket
from pathlib import Path

import pytest

from benchmarks import host_label
from benchmarks.host_label import (
    DEFAULT_HOST_LABEL,
    HOST_LABEL_ENV,
    benchmark_host_label,
    redact_local_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HOSTNAME_READERS = re.compile(
    r"platform\.node\(|gethostname\(|getfqdn\(|os\.uname\(|platform\.uname\("
)


def _real_names() -> set[str]:
    """Return the machine's hostname spellings (full and short)."""

    names = {socket.gethostname(), platform.node()}
    return {name for name in names | {n.split(".", 1)[0] for n in names} if name}


@pytest.mark.smoke
def test_default_label_is_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no override the label is the neutral default."""

    monkeypatch.delenv(HOST_LABEL_ENV, raising=False)
    assert benchmark_host_label() == DEFAULT_HOST_LABEL


@pytest.mark.smoke
def test_env_override_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """A neutral override from the environment is recorded as given."""

    monkeypatch.setenv(HOST_LABEL_ENV, "ci-linux-4core")
    assert benchmark_host_label() == "ci-linux-4core"


@pytest.mark.smoke
def test_override_equal_to_hostname_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """An override that spells the real hostname never reaches the record."""

    for name in _real_names():
        monkeypatch.setenv(HOST_LABEL_ENV, name.upper())
        assert benchmark_host_label() == DEFAULT_HOST_LABEL


@pytest.mark.smoke
def test_runner_environment_record_omits_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner's environment record names the machine only by the neutral label."""

    from benchmarks.perf_runner import _env_metadata

    monkeypatch.delenv(HOST_LABEL_ENV, raising=False)
    record = _env_metadata("cpu")
    assert record["hostname"] == DEFAULT_HOST_LABEL
    assert record["cpu_count"] is not None
    text = json.dumps(record, default=str)
    for name in _real_names():
        assert not re.search(rf"\b{re.escape(name)}\b", text), "hostname leaked into the record"


@pytest.mark.smoke
def test_local_paths_are_redacted() -> None:
    """Absolute checkout and home paths become ``<repo>`` and ``~``."""

    tail = f"{REPO_ROOT}/torchlens/x.py:12: warning\n{Path.home()}/cache/y"
    redacted = redact_local_paths(tail, REPO_ROOT)
    assert str(REPO_ROOT) not in redacted
    assert "<repo>/torchlens/x.py:12" in redacted
    assert "~/cache/y" in redacted


@pytest.mark.smoke
def test_only_the_label_helper_reads_the_hostname() -> None:
    """No benchmark writer calls a hostname API directly."""

    helper = Path(host_label.__file__).resolve()
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted((REPO_ROOT / "benchmarks").rglob("*.py"))
        if path.resolve() != helper and HOSTNAME_READERS.search(path.read_text())
    ]
    assert not offenders, f"benchmark modules reading the hostname directly: {offenders}"
