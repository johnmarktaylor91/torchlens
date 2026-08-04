"""Regressions for the three worker read authorities that must not drift apart.

A crawler worker's file reads are confined by THREE independent authorities, and a
read must be accepted by every authority live on the host or the worker never
reaches its constructor:

1. The **OS sandbox profile**. On macOS that is the Seatbelt profile built by
   :func:`menagerie.crawler.policy.generate_macos_sandbox_profile`, whose opening
   forced-report ``(deny file-read-data ...)`` SIGKILLs the child on the first read
   it does not grant. On Linux it is the bubblewrap mount set.
2. The **in-process ``ExecutionPolicy`` audit hook**, which runs *inside* the worker
   subprocess and keys on :data:`menagerie.crawler.policy._SYSTEM_READ_FILES` plus
   the shared :func:`menagerie.crawler.policy._runtime_import_metadata_path_allowed`
   classifier. A read it rejects poisons the attempt as a checkpoint read.
3. The **parent-side denial classifier** in
   :mod:`menagerie.crawler.worker_supervisor`, which reads the same shared
   classifier through its own mirror of ``_SYSTEM_READ_FILES``.

Two macOS-only failures shipped because these authorities disagreed and Linux
allowed both reads unconditionally, so no test on either platform noticed:

* **Defect A.** In v3-manifest mode the supervisor collapsed the macOS runtime read
  roots to the environment prefix, so the generated profile carried no confined
  ``.dist-info``/``.egg-info`` metadata-name grant for the worker's source working
  directory. The declarative rung uniquely calls
  ``importlib.metadata.packages_distributions()``, which scans every ``sys.path``
  entry -- including that working directory, which ``python -m`` puts on the path --
  and the first ``torchlens.egg-info/top_level.txt`` read was denied and SIGKILLed.
  Authority 1 was stricter than authorities 2 and 3.
* **Defect C.** ``_SYSTEM_READ_FILES`` listed only Linux files, so
  ``platform.mac_ver()`` -- called during ordinary ``huggingface_hub``/``timm``
  imports -- tripped authority 2, even though the Seatbelt profile allows every
  ``/System`` read. Authority 2 was stricter than authority 1.

The tests below therefore assert the *agreement*, not a proxy for it. Every one of
them is meaningful on Linux -- there is no ``sys.platform`` guard in this module at
all. They read the policy tables and the generated profile text rather than
requiring a macOS host to deny something, and the last one ties the v3 grant to the
pattern family that ``test_os_sandbox.py`` already proves against a real Seatbelt
matcher on a macOS host.

Defect A additionally only bites when the worker's working directory really
contains a metadata directory -- true in the development repo, false in a fresh
pilot clone. A test that leaned on the ambient working directory would therefore be
vacuous on some hosts, so every test here builds its own synthetic source root with
its own ``.egg-info``/``.dist-info`` under ``tmp_path``.
"""

from __future__ import annotations

import inspect
import platform
import re
from pathlib import Path

import pytest

from menagerie.crawler import policy as policy_module
from menagerie.crawler import worker_supervisor as supervisor_module
from menagerie.crawler.policy import generate_macos_sandbox_profile
from menagerie.crawler.tests.test_environment_authority_composition import (
    _macos_profile_manifest,
)

_REGEX_GRANT = re.compile(r'\(allow file-read-data \(regex #"(?P<pattern>.*)"\)\)\Z')


def _regex_read_grants(profile: str) -> tuple[re.Pattern[str], ...]:
    """Return the plain file-read regex grants of a profile, compiled for matching.

    Parameters
    ----------
    profile:
        Complete generated Seatbelt profile text.

    Returns
    -------
    tuple[re.Pattern[str], ...]
        One compiled pattern per unqualified ``(allow file-read-data (regex ...))``
        rule. Directory-qualified ``require-all`` rules are excluded because they
        grant entry-name visibility, never file contents.
    """

    grants = []
    for line in profile.splitlines():
        match = _REGEX_GRANT.fullmatch(line)
        if match is not None:
            grants.append(re.compile(match.group("pattern")))
    return tuple(grants)


def _granted(profile: str, path: Path) -> bool:
    """Return whether any regex read grant in the profile covers one exact path."""

    return any(pattern.search(str(path)) is not None for pattern in _regex_read_grants(profile))


def _source_root_with_metadata(tmp_path: Path) -> Path:
    """Build a synthetic worker source working directory carrying an egg-info.

    Parameters
    ----------
    tmp_path:
        Isolated tree.

    Returns
    -------
    pathlib.Path
        Resolved root standing in for the ``python -m`` working directory. It is
        synthesized rather than taken from the ambient working directory so the
        test is never vacuous on a checkout that happens to carry no egg-info.
    """

    root = tmp_path / "worker-source-root"
    metadata = root / "torchlens.egg-info"
    metadata.mkdir(parents=True)
    (metadata / "top_level.txt").write_text("torchlens\n", encoding="utf-8")
    (root / "model_payload.pt").write_bytes(b"weights")
    (root / "undeclared_module.py").write_text("SECRET = True\n", encoding="utf-8")
    return root.resolve()


def _v3_profile(tmp_path: Path, source_root: Path) -> tuple[str, dict[str, Path]]:
    """Generate one v3 profile exactly as the supervisor composes it on macOS."""

    manifest, members = _macos_profile_manifest(tmp_path)
    discovered_roots = (source_root,)
    runtime_read_roots, package_data_paths = supervisor_module._macos_runtime_read_capabilities(  # noqa: SLF001
        manifest, discovered_roots
    )
    profile = generate_macos_sandbox_profile(
        (tmp_path / "scratch", tmp_path / "result"),
        allowed_read_paths=(members["request"], *package_data_paths),
        runtime_read_roots=runtime_read_roots,
        execution_read_manifest=manifest,
    )
    return profile, members


@pytest.mark.smoke
def test_v3_profile_grants_source_root_import_metadata_but_no_payload(
    tmp_path: Path,
) -> None:
    """Defect A: the v3 profile must grant the source root's confined metadata names.

    ``importlib.metadata.packages_distributions()`` reads
    ``<source root>/torchlens.egg-info/top_level.txt`` during ordinary declarative
    resolution. Without a grant the profile's forced-report deny SIGKILLs the worker
    before the constructor runs, which surfaces as a confusing downstream failure
    rather than a read denial.

    The confinement is asserted in the same breath: the grant must cover the closed
    metadata names inside a metadata directory and nothing else -- no package
    payload, no source file, and no payload smuggled into the metadata directory.
    """

    source_root = _source_root_with_metadata(tmp_path)
    profile, _members = _v3_profile(tmp_path, source_root)

    assert _granted(profile, source_root / "torchlens.egg-info" / "top_level.txt")
    assert _granted(profile, source_root / "torchlens.egg-info" / "PKG-INFO")
    assert _granted(profile, source_root / "pkg" / "demo-1.0.dist-info" / "METADATA")

    # Nothing beyond the closed metadata names becomes readable outside the prefix.
    assert not _granted(profile, source_root / "model_payload.pt")
    assert not _granted(profile, source_root / "undeclared_module.py")
    assert not _granted(profile, source_root / "torchlens.egg-info" / "payload.pt")
    assert not _granted(profile, source_root / "torchlens.egg-info" / "SOURCES.txt")
    # A non-prefix root must not be promoted to a subtree grant either.
    assert f'(subpath "{source_root}")' not in profile


@pytest.mark.smoke
def test_v3_profile_grants_every_metadata_name_the_shared_classifier_accepts(
    tmp_path: Path,
) -> None:
    """Authority 1 must not be stricter than authorities 2 and 3 on metadata names.

    The shared classifier ``_runtime_import_metadata_path_allowed`` -- the single
    function behind both the in-process hook and the parent-side denial classifier --
    accepts every name in ``_RUNTIME_METADATA_NAMES`` inside a ``.dist-info`` or
    ``.egg-info`` directory, on any path and unconditionally. Each such name is
    therefore a read the worker can genuinely perform, so the Seatbelt profile must
    grant it for the source root as well. This is the anti-drift test: adding a name
    to the classifier's table, or narrowing the profile generator's pattern, breaks
    it immediately instead of at the next real macOS run.
    """

    source_root = _source_root_with_metadata(tmp_path)
    profile, _members = _v3_profile(tmp_path, source_root)
    metadata_names = sorted(policy_module._RUNTIME_METADATA_NAMES)  # noqa: SLF001
    assert metadata_names, "the shared metadata-name table must not be empty"

    disagreements: list[str] = []
    for directory in ("torchlens.egg-info", "demo-1.0.dist-info"):
        for name in metadata_names:
            candidate = source_root / directory / name
            classifier_allows = policy_module._runtime_import_metadata_path_allowed(  # noqa: SLF001
                candidate
            )
            if classifier_allows and not _granted(profile, candidate):
                disagreements.append(str(candidate))
    assert disagreements == []

    # The editable-install path hook is probed by name on every sys.path entry and is
    # accepted by the same classifier, so it carries the same agreement obligation.
    hook = source_root / "__editable__.torchlens-2.0.0.__path_hook__"
    assert policy_module._runtime_import_metadata_path_allowed(hook)  # noqa: SLF001
    assert _granted(profile, hook)


@pytest.mark.smoke
def test_supervisor_keeps_the_discovered_source_roots_for_a_v3_manifest(
    tmp_path: Path,
) -> None:
    """Defect A, supervisor half: v3 mode must not drop the discovered roots.

    The profile grant above can only be emitted for roots the supervisor actually
    hands to the generator. Collapsing the v3 roots to the sealed prefix is what
    made the grant unreachable, and this asserts the composition directly rather
    than through a spawned worker, so it is meaningful on Linux too.
    """

    manifest, members = _macos_profile_manifest(tmp_path)
    source_root = _source_root_with_metadata(tmp_path)
    discovered_roots = (source_root, tmp_path / "interpreter-root")

    roots, package_data = supervisor_module._macos_runtime_read_capabilities(  # noqa: SLF001
        manifest, discovered_roots
    )

    assert roots == (members["prefix"], *discovered_roots)
    assert package_data == ()
    # The discovery-only path keeps its own roots unchanged, so the v3 branch is the
    # only one that could have dropped them.
    discovery_roots, _discovery_data = supervisor_module._macos_runtime_read_capabilities(  # noqa: SLF001
        None, discovered_roots
    )
    assert discovery_roots == discovered_roots


@pytest.mark.smoke
def test_the_worker_source_working_directory_is_always_a_discovered_root(
    tmp_path: Path,
) -> None:
    """The ``python -m`` working directory is on ``sys.path`` and must be a read root.

    This is the link that makes the two tests above load-bearing: if the working
    directory ever stopped being a discovered root, the metadata grant would be
    emitted for the wrong root and the worker would die exactly as it did before.
    """

    working_directory = _source_root_with_metadata(tmp_path)
    argv = ("/some/env/bin/python", "-m", "menagerie.crawler.worker")

    roots = supervisor_module._runtime_read_roots(argv, working_directory)  # noqa: SLF001

    assert working_directory in roots


@pytest.mark.smoke
def test_system_read_files_mirror_cannot_drift_between_the_python_authorities() -> None:
    """Authorities 2 and 3 must name exactly the same system-read allowlist.

    ``policy._SYSTEM_READ_FILES`` is consulted by the in-process hook inside the
    worker subprocess; ``worker_supervisor._SYSTEM_READ_FILES`` is the parent-side
    mirror consulted by the denial classifier. They are separate objects, so they
    can silently diverge -- and did. A week of probe failures was spent patching the
    parent mirror, which is unreachable from inside the worker subprocess, while the
    in-process table stayed strict. Equality is the cheap structural guard.
    """

    assert policy_module._SYSTEM_READ_FILES == supervisor_module._SYSTEM_READ_FILES  # noqa: SLF001


@pytest.mark.smoke
def test_platform_mac_ver_os_version_file_is_admitted_by_both_python_authorities() -> None:
    """Defect C: the file ``platform.mac_ver()`` reads must not poison an attempt.

    ``huggingface_hub`` calls ``platform.mac_ver()`` while ``timm``/``transformers``
    import, and on macOS that reads the OS version record. The Seatbelt profile
    already allows the whole ``/System`` subtree as benign OS metadata, so the two
    Python authorities were strictly stricter than the OS sandbox and killed the
    attempt as a checkpoint read before the constructor started.

    The expected path is derived from the running interpreter's own ``platform``
    source rather than hard-coded, which makes the test meaningful on Linux -- it
    fails if CPython ever reads a different file, instead of passing vacuously and
    leaving the real macOS gap open.
    """

    source = inspect.getsource(platform)
    referenced = sorted(set(re.findall(r"/System/[A-Za-z0-9_/.\-]+", source)))
    assert referenced == ["/System/Library/CoreServices/SystemVersion.plist"], (
        "the platform module no longer reads the expected macOS version record; "
        f"update the read allowlists to match {referenced!r}"
    )
    version_record = Path(referenced[0])

    assert version_record in policy_module._SYSTEM_READ_FILES  # noqa: SLF001
    assert version_record in supervisor_module._SYSTEM_READ_FILES  # noqa: SLF001
    # The OS sandbox authority grants the same read as an unconditional subtree.
    assert '(allow file-read-data (subpath "/System"))' in generate_macos_sandbox_profile(())


@pytest.mark.smoke
def test_every_allowlisted_system_read_file_really_names_a_closed_file() -> None:
    """The shared allowlist stays a closed set of absolute files, never a subtree.

    Both Python authorities compare a candidate path for exact membership, so a
    relative entry or a directory entry would be a silently inert allowance --
    the same class of failure as a narrowed Seatbelt regex.
    """

    for entry in policy_module._SYSTEM_READ_FILES:  # noqa: SLF001
        assert entry.is_absolute()
        assert entry == Path(*entry.parts)
        assert not str(entry).endswith("/")


@pytest.mark.smoke
def test_the_v3_metadata_grant_reuses_the_seatbelt_proven_pattern_family(
    tmp_path: Path,
) -> None:
    """The v3 clauses must stay a literal subset of the discovery-mode patterns.

    Seatbelt's matcher is not Python's: ``_sbpl_regex_literal`` documents two
    spellings Seatbelt silently mis-compiles into a grant that looks right in the
    profile and matches nothing at runtime. A text assertion alone therefore cannot
    prove the clause works on a real host.

    It does not have to. ``_macos_runtime_root_read_patterns`` -- the discovery-mode
    pattern family -- is already exercised against a real Seatbelt matcher by
    ``test_os_sandbox.py::test_macos_seatbelt_allows_authorized_pyc_and_kills_unauthorized_pyc``
    on a macOS host. The fix for defect A deliberately factored the metadata clauses
    into a helper shared by both callers, so pinning the subset relation ties the v3
    grant to that live proof. A refactor that lets the two spellings drift apart
    breaks this test on every platform rather than silently losing the coverage.
    """

    root = (tmp_path / "worker-source-root").resolve()
    metadata_patterns = policy_module._macos_root_import_metadata_patterns(root)  # noqa: SLF001
    discovery_patterns = policy_module._macos_runtime_root_read_patterns(root)  # noqa: SLF001

    assert metadata_patterns, "the v3 metadata grant must not be empty"
    assert set(metadata_patterns) <= set(discovery_patterns)
    # Both spellings must survive the Seatbelt quoting rules unchanged.
    for pattern in metadata_patterns:
        assert policy_module._sbpl_regex_literal(pattern) == f'#"{pattern}"'  # noqa: SLF001
