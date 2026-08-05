"""Bounded stability retry for the in-flight environment content seal.

External hardlink activity on inodes shared with a sealed prefix (conda's package
cache pattern) bumps ``st_ctime_ns`` inside the tree without changing any content.
The two-walk stability comparison in ``_seal_environment_content`` correctly
observes that as churn; these regressions prove a TRANSIENT blip now costs one
discarded derivation attempt instead of the environment cycle, while SUSTAINED
churn still exhausts the fixed bound and refuses with the exact historical
error type and message. The acceptance predicate itself is unchanged: every
accepted seal passed two consecutive identical walks plus the semantic digest
checks within one fresh attempt.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import menagerie.crawler.authority as authority_module
from menagerie.crawler.authority import (
    _SEAL_CONTENT_STABILITY_ATTEMPTS,
    AuthorityDerivationError,
    EnvironmentAuthorityCache,
    EnvironmentContentEntry,
    EnvironmentExternalTarget,
    _seal_environment_content,
    _SealInstabilityError,
)
from menagerie.crawler.tests.conftest import hardlink_bytes

pytestmark = pytest.mark.smoke

_ScanResult = tuple[
    tuple[EnvironmentContentEntry, ...],
    tuple[EnvironmentExternalTarget, ...],
    str,
]


def _churnable_prefix(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return one sealable prefix, its interpreter, and a shared-inode staging file.

    Parameters
    ----------
    tmp_path:
        Isolated test root.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        Prefix root, selected interpreter, and the staging member whose inode is
        shared with a prefix member so external link churn perturbs the seal.
    """

    prefix = tmp_path / "prefix"
    staging = tmp_path / "staging"
    hardlink_bytes(staging / "python", prefix / "bin" / "python", b"python")
    (prefix / "bin" / "python").chmod(0o755)
    hardlink_bytes(
        staging / "member.py",
        prefix / "lib" / "python3.11" / "site-packages" / "member.py",
        b"VALUE = 1\n",
    )
    return prefix, prefix / "bin" / "python", staging / "member.py"


def _bump_shared_inode_ctime(staging_member: Path, tick: int) -> None:
    """Bump ``st_ctime_ns`` on the prefix member via an external hardlink event.

    Parameters
    ----------
    staging_member:
        Staging path sharing its inode with a sealed prefix member.
    tick:
        Monotone counter making each churn link name unique.
    """

    churn_link = staging_member.with_name(f"churn-{tick}")
    os.link(staging_member, churn_link)
    churn_link.unlink()


def test_transient_external_ctime_churn_costs_one_attempt_not_the_seal(
    tmp_path: Path,
) -> None:
    """One ctime blip between the two walks discards one attempt and then seals.

    Parameters
    ----------
    tmp_path:
        Isolated prefix root.
    """

    prefix, interpreter, staging_member = _churnable_prefix(tmp_path)
    quiet_manifest = _seal_environment_content(prefix, interpreter)

    real_scan = authority_module._scan_environment_tree  # noqa: SLF001
    state: dict[str, int] = {"calls": 0}
    retries: list[int] = []

    def churning_scan(prefix: Path, *, hash_files: bool) -> _ScanResult:
        """Inject one real external hardlink event before attempt 1's second walk."""

        state["calls"] += 1
        if state["calls"] == 2:
            _bump_shared_inode_ctime(staging_member, state["calls"])
        return real_scan(prefix, hash_files=hash_files)

    authority_module._scan_environment_tree = churning_scan  # noqa: SLF001
    try:
        manifest = _seal_environment_content(
            prefix,
            interpreter,
            on_stability_retry=lambda: retries.append(state["calls"]),
        )
    finally:
        authority_module._scan_environment_tree = real_scan  # noqa: SLF001

    # Exactly one discarded pair, then one complete fresh accepted pair.
    assert state["calls"] == 4
    assert len(retries) == 1
    # The accepted seal is the SAME semantic identity a churn-free seal derives:
    # the blip changed no content, and nothing from the discarded attempt leaked in.
    assert manifest.content_manifest_sha256 == quiet_manifest.content_manifest_sha256
    assert manifest.selected_interpreter_digest == quiet_manifest.selected_interpreter_digest
    # The cheap fingerprint honestly re-baselines to the post-blip ctime, so the
    # next cheap validation of the untouched tree passes against it.
    _entries, _external, current_fingerprint = real_scan(prefix, hash_files=False)
    assert manifest.cheap_tree_fingerprint == current_fingerprint


def test_sustained_external_ctime_churn_exhausts_bound_and_refuses_exactly(
    tmp_path: Path,
) -> None:
    """Churn before every walk fails all attempts with the historical error.

    Parameters
    ----------
    tmp_path:
        Isolated prefix root.
    """

    prefix, interpreter, staging_member = _churnable_prefix(tmp_path)
    real_scan = authority_module._scan_environment_tree  # noqa: SLF001
    state: dict[str, int] = {"calls": 0}

    def churning_scan(prefix: Path, *, hash_files: bool) -> _ScanResult:
        """Inject a real external hardlink event before EVERY walk."""

        state["calls"] += 1
        _bump_shared_inode_ctime(staging_member, state["calls"])
        return real_scan(prefix, hash_files=hash_files)

    authority_module._scan_environment_tree = churning_scan  # noqa: SLF001
    try:
        with pytest.raises(AuthorityDerivationError) as caught:
            _seal_environment_content(prefix, interpreter)
    finally:
        authority_module._scan_environment_tree = real_scan  # noqa: SLF001

    # Bounded means bounded: exactly N complete pairs, then refusal with the exact
    # historical type (ledger failure_type vocabulary) and message.
    assert state["calls"] == 2 * _SEAL_CONTENT_STABILITY_ATTEMPTS
    assert type(caught.value) is AuthorityDerivationError
    assert str(caught.value) == "environment tree changed during content sealing"


def test_per_file_ctime_instability_is_retried_by_the_outer_bound(
    tmp_path: Path,
) -> None:
    """A mid-hash ctime double-blip on one member discards the attempt, not the seal.

    Parameters
    ----------
    tmp_path:
        Isolated prefix root.
    """

    prefix, interpreter, _staging_member = _churnable_prefix(tmp_path)
    real_hash = authority_module._hash_regular_file_stably  # noqa: SLF001
    state: dict[str, int] = {"failures": 0}

    def blipping_hash(path: Path, before: os.stat_result) -> tuple[str, os.stat_result]:
        """Report per-file ctime instability once, then delegate to the real hash."""

        if state["failures"] == 0:
            state["failures"] += 1
            raise _SealInstabilityError(f"environment member changed while sealing: {path}")
        return real_hash(path, before)

    authority_module._hash_regular_file_stably = blipping_hash  # noqa: SLF001
    try:
        manifest = _seal_environment_content(prefix, interpreter)
    finally:
        authority_module._hash_regular_file_stably = real_hash  # noqa: SLF001

    assert state["failures"] == 1
    assert manifest.selected_interpreter_relative_path == "bin/python"


def test_semantic_member_change_and_forbidden_entries_never_retry(
    tmp_path: Path,
) -> None:
    """Non-stability refusals stay immediate: no retry masks a real change.

    Parameters
    ----------
    tmp_path:
        Isolated prefix root.
    """

    prefix, interpreter, _staging_member = _churnable_prefix(tmp_path)
    real_scan = authority_module._scan_environment_tree  # noqa: SLF001
    state: dict[str, int] = {"calls": 0}

    def counting_scan(prefix: Path, *, hash_files: bool) -> _ScanResult:
        """Count walks so a single-shot refusal is provable."""

        state["calls"] += 1
        return real_scan(prefix, hash_files=hash_files)

    fifo = prefix / "integrity-fifo"
    os.mkfifo(fifo)
    authority_module._scan_environment_tree = counting_scan  # noqa: SLF001
    try:
        with pytest.raises(AuthorityDerivationError, match="forbidden special entry"):
            _seal_environment_content(prefix, interpreter)
    finally:
        authority_module._scan_environment_tree = real_scan  # noqa: SLF001
        fifo.unlink()
    assert state["calls"] == 1

    real_hash = authority_module._hash_regular_file_stably  # noqa: SLF001

    hash_calls: dict[str, int] = {"calls": 0}

    def semantic_change_hash(path: Path, before: os.stat_result) -> tuple[str, os.stat_result]:
        """Report a semantic (writer-evidencing) change on every hashed member."""

        hash_calls["calls"] += 1
        raise AuthorityDerivationError(f"environment member changed while sealing: {path}")

    authority_module._hash_regular_file_stably = semantic_change_hash  # noqa: SLF001
    try:
        with pytest.raises(AuthorityDerivationError, match="changed while sealing") as caught:
            _seal_environment_content(prefix, interpreter)
    finally:
        authority_module._hash_regular_file_stably = real_hash  # noqa: SLF001
    # A retrying loop would have re-walked and hashed again; the semantic refusal
    # propagates from the FIRST hashed member of the FIRST walk.
    assert hash_calls["calls"] == 1
    assert type(caught.value) is AuthorityDerivationError


def test_cache_accounts_seal_stability_retries_in_walk_ceiling(
    tmp_path: Path,
) -> None:
    """The deterministic walk counter stays an honest ceiling under one retry.

    Parameters
    ----------
    tmp_path:
        Isolated prefix root.
    """

    prefix, interpreter, staging_member = _churnable_prefix(tmp_path)
    cache = EnvironmentAuthorityCache()
    real_scan = authority_module._scan_environment_tree  # noqa: SLF001
    state: dict[str, int] = {"calls": 0}

    def churning_scan(prefix: Path, *, hash_files: bool) -> _ScanResult:
        """Inject one real external hardlink event before attempt 1's second walk."""

        state["calls"] += 1
        if state["calls"] == 2:
            _bump_shared_inode_ctime(staging_member, state["calls"])
        return real_scan(prefix, hash_files=hash_files)

    authority_module._scan_environment_tree = churning_scan  # noqa: SLF001
    try:
        cache.bind(
            prefix=prefix,
            selected_interpreter=interpreter,
            base_environment_generation="sha256:" + "0" * 64,
        )
    finally:
        authority_module._scan_environment_tree = real_scan  # noqa: SLF001

    assert cache.seal_stability_retries == 1
    assert cache.lstat_tree_walks == (
        2 * cache.full_seals + 2 * cache.seal_stability_retries + cache.cheap_tree_walks
    )
