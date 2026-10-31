"""Package-data globs and MANIFEST rules must name the package's real data paths.

A package move (``autoroute`` to ``_autoroute``) once left the
``[tool.setuptools.package-data]`` globs pointing at a directory that no longer
existed: setuptools silently matches nothing for a stale glob, so the data
file the package reads at runtime through ``importlib.resources`` would have
dropped out of the wheel with every other gate green. These checks bind the
three places that must agree without building anything:

1. every data file under ``torchlens/`` (everything that is not Python source,
   bytecode, or an excluded agent doc) is matched by a package-data glob AND by
   a MANIFEST.in rule (``python -m build`` makes the wheel from the sdist, so a
   file the sdist drops is missing from the wheel too);
2. every package-data glob and every MANIFEST.in data rule matches at least one
   real file, so a stale key goes red instead of matching nothing;
3. every ``importlib.resources.files("torchlens...")`` reader in the package
   names an existing package directory and, where it joins a literal file
   name, an existing file.

The built-wheel counterpart (the same data set asserted on real wheel members)
is ``tests/test_packaging_diet.py::test_built_wheel_manifest_is_diet``.

pyproject parsing is line-based on purpose: ``tomllib`` only exists on 3.11+
and the suite still runs a 3.10 leg (the test_coverage_floor_governance
precedent).
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _PROJECT_ROOT / "torchlens"

#: Files under torchlens/ that are never package data: source, bytecode, and
#: the agent docs [tool.setuptools.exclude-package-data] keeps out of the wheel.
_SOURCE_SUFFIXES = (".py", ".pyc", ".pyo")
_AGENT_DOCS = ("AGENTS.md", "CLAUDE.md")


def runtime_data_files(package_root: Path = _PACKAGE_ROOT) -> list[str]:
    """Return every data file under the package, relative to the package root.

    Parameters
    ----------
    package_root
        The ``torchlens/`` directory to walk.

    Returns
    -------
    list[str]
        Sorted POSIX paths relative to ``package_root``.
    """

    files = []
    for path in package_root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.name.endswith(_SOURCE_SUFFIXES) or path.name in _AGENT_DOCS:
            continue
        files.append(path.relative_to(package_root).as_posix())
    return sorted(files)


def package_data_globs(pyproject_text: str) -> list[str]:
    """Return the ``torchlens`` globs of ``[tool.setuptools.package-data]``.

    Parameters
    ----------
    pyproject_text
        Raw ``pyproject.toml`` text.

    Returns
    -------
    list[str]
        The quoted glob strings of the ``torchlens = [...]`` list, comments dropped.
    """

    section = re.search(
        r"^\[tool\.setuptools\.package-data\]\s*$(.*?)^\[", pyproject_text, re.M | re.S
    )
    assert section, "pyproject.toml lost its [tool.setuptools.package-data] table"
    body = re.search(r"^torchlens\s*=\s*\[(.*?)^\]", section.group(1), re.M | re.S)
    assert body, "[tool.setuptools.package-data] lost its `torchlens = [...]` list"
    lines = (line.split("#", 1)[0] for line in body.group(1).splitlines())
    return [glob for line in lines for glob in re.findall(r'"([^"]+)"', line)]


def manifest_data_rules(manifest_text: str) -> list[tuple[str, str, bool]]:
    """Return MANIFEST.in data rules under torchlens/ as ``(directory, pattern, recursive)``.

    ``include torchlens/X`` becomes ``(dirname, basename, False)``;
    ``recursive-include torchlens/D p1 p2`` becomes one recursive rule per pattern.
    The blanket ``*.py`` source rule is skipped: it ships source, not data.

    Parameters
    ----------
    manifest_text
        Raw ``MANIFEST.in`` text.

    Returns
    -------
    list[tuple[str, str, bool]]
        Directory relative to ``torchlens/`` ("" for the package root), its glob,
        and whether the rule reaches subdirectories.
    """

    rules = []
    for raw in manifest_text.splitlines():
        words = raw.split("#", 1)[0].split()
        if len(words) < 2 or not words[1].startswith("torchlens"):
            continue
        directory = words[1].removeprefix("torchlens").strip("/")
        if words[0] == "include":
            for target in words[1:]:
                rel = target.removeprefix("torchlens/")
                rules.append((rel.rpartition("/")[0], rel.rpartition("/")[2], False))
        elif words[0] == "recursive-include":
            rules.extend((directory, pattern, True) for pattern in words[2:] if pattern != "*.py")
    return rules


def _glob_matches(glob: str, rel_path: str) -> bool:
    """Return whether a package-data glob (relative to torchlens/) matches a file."""

    return Path(rel_path).match(glob) and rel_path.count("/") == glob.count("/")


def _manifest_matches(rule: tuple[str, str, bool], rel_path: str) -> bool:
    """Return whether a MANIFEST.in data rule ships a file."""

    directory, pattern, recursive = rule
    head, _, name = rel_path.rpartition("/")
    below = recursive and (not directory or head.startswith(directory + "/"))
    return (head == directory or below) and fnmatch.fnmatch(name, pattern)


def _read(name: str) -> str:
    return (_PROJECT_ROOT / name).read_text(encoding="utf-8")


def test_every_package_data_glob_names_real_files() -> None:
    """A package-data glob that matches nothing is a stale path (a moved package)."""

    files = runtime_data_files()
    stale = [
        g
        for g in package_data_globs(_read("pyproject.toml"))
        if not any(_glob_matches(g, f) for f in files)
    ]
    assert not stale, (
        f"[tool.setuptools.package-data] glob(s) match no file under torchlens/: {stale}. "
        "A package was moved or renamed; re-key the glob to the package's current path."
    )


def test_every_data_file_is_shipped_by_a_package_data_glob() -> None:
    """Every data file the package carries is in the wheel's package-data set."""

    globs = package_data_globs(_read("pyproject.toml"))
    unshipped = [f for f in runtime_data_files() if not any(_glob_matches(g, f) for g in globs)]
    assert not unshipped, (
        f"data file(s) under torchlens/ match no [tool.setuptools.package-data] glob: "
        f"{unshipped}. The installed wheel would lack them."
    )


def test_manifest_rules_name_real_files_and_cover_every_data_file() -> None:
    """MANIFEST.in mirrors package-data: no stale rule, no data file left out."""

    files = runtime_data_files()
    rules = manifest_data_rules(_read("MANIFEST.in"))
    assert rules, "MANIFEST.in lost its enumerated torchlens/ data rules"
    stale = [r for r in rules if not any(_manifest_matches(r, f) for f in files)]
    assert not stale, f"MANIFEST.in data rule(s) match no file under torchlens/: {stale}"
    unshipped = [f for f in files if not any(_manifest_matches(r, f) for r in rules)]
    assert not unshipped, (
        f"data file(s) under torchlens/ match no MANIFEST.in rule: {unshipped}. "
        "`python -m build` builds the wheel from the sdist, so they would miss both."
    )


_RESOURCE_READER = re.compile(
    r'resources\.files\(\s*"(?P<package>torchlens[\w.]*)"\s*\)'
    r'(?:\s*\.joinpath\(\s*"(?P<name>[^"]+)"\s*\))?'
)


def test_importlib_resources_readers_name_existing_package_paths() -> None:
    """Every runtime resources reader points at a package path that exists."""

    readers = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        for match in _RESOURCE_READER.finditer(path.read_text(encoding="utf-8")):
            readers.append((path.relative_to(_PROJECT_ROOT).as_posix(), match))
    assert len(readers) >= 3, (
        f"expected the autoroute, mechinterp and schema resources readers, found {len(readers)}"
    )
    broken = []
    for source, match in readers:
        target = _PROJECT_ROOT.joinpath(*match["package"].split("."))
        if match["name"]:
            target = target / match["name"]
        if not target.exists():
            broken.append(f"{source}: {match.group(0)!r}")
    assert not broken, f"resources reader(s) name a missing package path: {broken}"


def test_stale_glob_detection_is_red_capable() -> None:
    """The matchers flag the pre-move autoroute keys and accept the current ones."""

    files = ["_autoroute/data/imagenet1k_labels.json", "_autoroute/data/README.md"]
    assert not any(_glob_matches("autoroute/data/*.json", f) for f in files)
    assert _glob_matches("_autoroute/data/*.json", files[0])
    assert not _glob_matches("*.json", files[0]), "a root glob must not match nested files"
    assert not _manifest_matches(("autoroute/data", "*.json", True), files[0])
    assert _manifest_matches(("_autoroute/data", "*.json", True), files[0])
    assert not _manifest_matches(("", "py.typed", False), "sub/py.typed")
    assert _manifest_matches(("", "py.typed", False), "py.typed")
    planted = '[tool.setuptools.package-data]\ntorchlens = [\n    "a/*.json",  # c\n]\n[x]\n'
    assert package_data_globs(planted) == ["a/*.json"]
