"""The conda-forge recipe stays in sync with pyproject.toml.

``packaging/conda/recipe.yaml`` is the canonical copy of the conda-forge
recipe (see ``packaging/conda/README.md``). It restates the package version,
the build backend floor and the runtime dependencies that ``pyproject.toml``
declares, so a change to one without the other is drift. These tests fail on
that drift: a version bump that skips the recipe, a runtime dependency added,
removed or re-floored on one side only, or an optional extra leaking into the
recipe's run requirements.

The v1 recipe format is plain YAML (its ``${{ ... }}`` expressions are string
values), so the file is read with ``yaml.safe_load`` and no rendering step.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

if sys.version_info >= (3, 11):
    import tomllib
else:  # pytest itself requires tomli below 3.11
    import tomli as tomllib

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RECIPE_PATH = _REPO_ROOT / "packaging" / "conda" / "recipe.yaml"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"

# PyPI distribution name (canonicalized) -> conda-forge package name, for the
# names that differ. Every other runtime dependency keeps its PyPI name.
_CONDA_NAME = {
    "torch": "pytorch",
    "graphviz": "python-graphviz",
    "typing-extensions": "typing_extensions",
}

_REQ_LINE = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?:\s+(?P<spec>.+))?$")


def _recipe() -> dict[str, Any]:
    with _RECIPE_PATH.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict), f"{_RECIPE_PATH} is not a YAML mapping"
    return loaded


def _pyproject() -> dict[str, Any]:
    with _PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _split_conda_requirement(line: str) -> tuple[str, str]:
    match = _REQ_LINE.match(line.strip())
    assert match is not None, f"unparseable recipe requirement {line!r}"
    return match["name"], (match["spec"] or "").strip()


def _floor_of(spec: str, where: str) -> Version:
    """Return the single ``>=`` floor in ``spec``; anything else is drift to review."""
    clauses = [clause.strip() for clause in spec.split(",") if clause.strip()]
    assert len(clauses) == 1 and clauses[0].startswith(">="), (
        f"{where}: expected exactly one '>=' floor, got {spec!r}; "
        "extend tests/test_conda_recipe_sync.py and packaging/conda/README.md "
        "if the dependency now needs another constraint form"
    )
    return Version(clauses[0][2:].strip())


def _expected_run_floors() -> dict[str, Version]:
    """Map each pyproject runtime dependency to its conda name and lowest floor.

    A dependency declared several times under python_version markers (torch)
    collapses to the lowest floor: conda-forge's solver pairs each pytorch
    build with the interpreters it was built for, so the markers have no
    conda counterpart.
    """
    floors: dict[str, Version] = {}
    for raw in _pyproject()["project"]["dependencies"]:
        requirement = Requirement(raw)
        assert not requirement.extras, f"runtime dependency with extras: {raw!r}"
        name = canonicalize_name(requirement.name)
        conda_name = _CONDA_NAME.get(name, name)
        floor = _floor_of(str(requirement.specifier), f"pyproject dependency {raw!r}")
        floors[conda_name] = min(floor, floors.get(conda_name, floor))
    return floors


def test_recipe_version_matches_pyproject() -> None:
    recipe_version = str(_recipe()["context"]["version"])
    project_version = _pyproject()["project"]["version"]
    assert recipe_version == project_version, (
        f"packaging/conda/recipe.yaml context.version is {recipe_version!r} but "
        f"pyproject.toml is {project_version!r}: bump the recipe's version and "
        "source sha256 together (packaging/conda/README.md, 'Keeping it in sync')"
    )


def test_recipe_source_is_the_pypi_sdist_of_that_version() -> None:
    recipe = _recipe()
    assert recipe["package"]["name"] == "${{ name }}"
    assert recipe["context"]["name"] == _pyproject()["project"]["name"]
    assert recipe["package"]["version"] == "${{ version }}"
    source = recipe["source"]
    assert source["url"].startswith("https://pypi.org/packages/source/")
    assert source["url"].endswith("${{ name }}-${{ version }}.tar.gz")
    assert re.fullmatch(r"[0-9a-f]{64}", str(source["sha256"])), "sha256 must be 64 hex"


def test_recipe_is_noarch_python_installed_by_pip() -> None:
    build = _recipe()["build"]
    assert build["noarch"] == "python"
    assert "pip install ." in build["script"]


def test_recipe_run_requirements_match_pyproject_dependencies() -> None:
    run_floors: dict[str, Version] = {}
    for line in _recipe()["requirements"]["run"]:
        name, spec = _split_conda_requirement(line)
        if name == "python":
            assert spec == ">=${{ python_min }}", f"python run requirement {line!r}"
            continue
        assert name not in run_floors, f"duplicate recipe run requirement {name!r}"
        run_floors[name] = _floor_of(spec, f"recipe run requirement {line!r}")
    expected = _expected_run_floors()
    drift = {
        name: (str(run_floors.get(name)), str(expected.get(name)))
        for name in sorted(set(run_floors) | set(expected))
        if run_floors.get(name) != expected.get(name)
    }
    assert not drift, (
        f"recipe run requirements drifted from pyproject dependencies "
        f"(name: (recipe floor, pyproject floor)): {drift}"
    )


def test_recipe_host_setuptools_floor_matches_build_system() -> None:
    build_floor = None
    for raw in _pyproject()["build-system"]["requires"]:
        requirement = Requirement(raw)
        if canonicalize_name(requirement.name) == "setuptools":
            build_floor = _floor_of(str(requirement.specifier), f"build requirement {raw!r}")
    assert build_floor is not None, "pyproject build-system no longer names setuptools"
    host = dict(_split_conda_requirement(line) for line in _recipe()["requirements"]["host"])
    assert host["python"] == "${{ python_min }}.*"
    assert "pip" in host
    assert _floor_of(host["setuptools"], "recipe host setuptools") == build_floor


def test_recipe_license_matches_pyproject() -> None:
    about = _recipe()["about"]
    assert about["license"] == _pyproject()["project"]["license"]
    assert about["license_file"] == "LICENSE"
    assert (_REPO_ROOT / "LICENSE").is_file()


@pytest.mark.parametrize("extra", sorted(_pyproject()["project"]["optional-dependencies"]))
def test_optional_extras_stay_out_of_run_requirements(extra: str) -> None:
    run_names = {_split_conda_requirement(line)[0] for line in _recipe()["requirements"]["run"]}
    base = set(_expected_run_floors())
    for raw in _pyproject()["project"]["optional-dependencies"][extra]:
        name = canonicalize_name(Requirement(raw).name)
        if name == "torchlens":
            continue  # self-referencing aggregate extras
        conda_name = _CONDA_NAME.get(name, name)
        if conda_name in base:
            continue  # a base dependency an extra restates
        assert conda_name not in run_names, (
            f"optional extra [{extra}] dependency {raw!r} is a recipe run requirement; "
            "optional extras stay out of the conda package's run requirements"
        )
