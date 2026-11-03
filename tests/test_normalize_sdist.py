"""r7 R84 (opus b10 MED): unit coverage for the load-bearing artifact normalizer.

``scripts/normalize_sdist.py`` decides the published bytes of BOTH release
artifacts, yet its only exercise anywhere was the nightly double-build gate
(two full builds, never on PRs): deleting the ``dist/*.whl`` argument or
regressing ``_normalized_mode`` passed every PR gate, and the first signal
would have been a nightly red AFTER a release already shipped the broken
bytes. These are the missing unit tests: mode mapping, umask-invariance and
idempotence on synthetic two-mode artifacts, the directory-entry type-bit
guard (opus LOW: the rewriter stamped S_IFREG unconditionally), and the
SOURCE_DATE_EPOCH refusal.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import stat
import tarfile
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_normalizer():
    spec = importlib.util.spec_from_file_location(
        "_normalize_sdist_under_test", _REPO_ROOT / "scripts" / "normalize_sdist.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_sdist(path: Path, mode: int, uid: int, mtime: int) -> None:
    """Write a synthetic .tar.gz with machine-dependent metadata."""

    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, payload in (("pkg-1.0/PKG-INFO", b"Name: pkg\n"), ("pkg-1.0/mod.py", b"x = 1\n")):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mode = mode
            member.uid = uid
            member.gid = uid
            member.uname = "builder"
            member.gname = "staff"
            member.mtime = mtime
            tar.addfile(member, io.BytesIO(payload))
    with open(path, "wb") as output:
        with gzip.GzipFile(filename="orig", mode="wb", fileobj=output, mtime=mtime) as gz:
            gz.write(tar_bytes.getvalue())


def _make_wheel(path: Path, file_mode: int) -> None:
    """Write a synthetic .whl with a directory entry and two-mode files."""

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as wheel:
        directory = zipfile.ZipInfo("pkg/", date_time=(2020, 1, 1, 0, 0, 0))
        directory.create_system = 3
        directory.external_attr = (stat.S_IFDIR | 0o755) << 16
        wheel.writestr(directory, b"")
        for name, mode in (("pkg/a.py", file_mode), ("pkg/b.sh", file_mode | 0o111)):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | mode) << 16
            wheel.writestr(info, b"payload\n")


def test_sdist_normalization_is_machine_invariant_and_idempotent(tmp_path: Path) -> None:
    """Two builds differing only in umask/uid/wall-clock converge byte-identically."""

    normalizer = _load_normalizer()
    first = tmp_path / "one.tar.gz"
    second = tmp_path / "two.tar.gz"
    _make_sdist(first, mode=0o644, uid=1000, mtime=1_700_000_000)
    _make_sdist(second, mode=0o664, uid=1001, mtime=1_700_009_999)
    assert first.read_bytes() != second.read_bytes()

    normalizer.normalize_sdist(str(first), 1_600_000_000)
    normalizer.normalize_sdist(str(second), 1_600_000_000)
    normalized = first.read_bytes()
    assert normalized == second.read_bytes()

    normalizer.normalize_sdist(str(first), 1_600_000_000)
    assert first.read_bytes() == normalized  # idempotent

    with tarfile.open(first) as tar:
        members = tar.getmembers()
    assert {m.mode for m in members} == {0o644}
    assert {(m.uid, m.gid, m.uname, m.gname) for m in members} == {(0, 0, "", "")}
    assert {m.mtime for m in members} == {1_600_000_000}
