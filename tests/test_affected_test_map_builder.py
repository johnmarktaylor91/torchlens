"""Unit tests for ``scripts/build_affected_test_map.py`` on a small synthetic repository.

They cover what the committed-map contract tests cannot show cheaply: two
builds are byte-identical, every module gets a key, each tier-broadening rule
selects its class (and only it), and reviewed override rows are applied or
refused when stale. Only pytest and the standard library are used.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

BUILDER_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_affected_test_map.py"

FILES: dict[str, str] = {
    "pyproject.toml": "",
    "torchlens/__init__.py": (
        "def __getattr__(name):\n    from ._utils.facade import resolve\n    return resolve(name)\n"
    ),
    "torchlens/_utils/__init__.py": "",
    "torchlens/_utils/facade.py": "from .._errors import Oops\n",
    "torchlens/_errors.py": "class Oops(Exception):\n    pass\n",
    "torchlens/errors/__init__.py": "from .._errors import Oops\n",
    "torchlens/_io/__init__.py": "",
    "torchlens/_io/format_contract.py": "VERSION = 1\n",
    "torchlens/_io/field_policy.py": "POLICY = {}\n",
    "torchlens/_io/reader.py": "from .format_contract import VERSION\n",
    "torchlens/schemas/__init__.py": "",
    "torchlens/core/__init__.py": "",
    "torchlens/core/widget_registry.py": "TABLE = {}\n",
    "torchlens/core/hub.py": "def register_thing(x):\n    return x\n",
    "torchlens/core/a.py": "from .hub import register_thing\n",
    "torchlens/core/b.py": "from .hub import register_thing\n",
    "torchlens/core/c.py": "from .hub import register_thing\n",
    "torchlens/core/plain.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import torchlens\n",
    "tests/conftest.py": 'pytest_plugins = ["_plugin"]\n',
    "tests/_plugin.py": "",
    "tests/sub/conftest.py": "",
    "tests/test_reader.py": "from torchlens._io.reader import VERSION\n",
    "tests/test_plain.py": "from torchlens.core import plain\n",
    "tests/test_plain_extra.py": "import torchlens.core.plain\n",
    "tests/test_a.py": "from torchlens.core import a\n",
}


def _load_builder() -> ModuleType:
    """Import the builder by path.

    Returns
    -------
    ModuleType
        The builder module.
    """
    spec = importlib.util.spec_from_file_location("_affected_map_builder_unit", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture(name="repo")
def _repo(tmp_path: Path) -> Path:
    """Write the synthetic repository and return its root."""
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    tiers = {"version": 1, "inner": {"seconds": 300, "map": {}}}
    (tmp_path / "test-tiers.json").write_text(json.dumps(tiers), encoding="utf-8")
    return tmp_path


def _render(builder: ModuleType, repo: Path) -> str:
    result = builder.build(repo)
    return builder.render(repo, result.mapping, result.tier)


def test_two_builds_are_byte_identical(repo: Path) -> None:
    """Building twice from the same tree renders the same bytes."""
    builder = _load_builder()
    assert _render(builder, repo) == _render(builder, repo)


def test_every_module_has_a_key(repo: Path) -> None:
    """Every synthetic module matches an ``inner.map`` key."""
    import fnmatch

    mapping = _load_builder().build(repo).mapping
    modules = [rel for rel in FILES if rel.startswith("torchlens/")]
    assert [m for m in modules if not any(fnmatch.fnmatchcase(m, k) for k in mapping)] == []


def test_broaden_rules_select_exactly_their_classes(repo: Path) -> None:
    """Each tier rule selects its class; plain modules keep a file list only."""
    tier = _load_builder().build(repo).tier
    assert tier == {
        "pyproject.toml": "pytest-config",
        "tests/_plugin.py": "pytest-plugin",
        "tests/conftest.py": "conftest",
        "tests/sub/conftest.py": "conftest",
        "torchlens/__init__.py": "import-wiring",
        "torchlens/_errors.py": "refusal-registry",
        "torchlens/_io/field_policy.py": "schema-name",
        "torchlens/_io/format_contract.py": "io-contract",
        "torchlens/_utils/facade.py": "import-wiring",
        "torchlens/core/hub.py": "registry-callsites",
        "torchlens/core/widget_registry.py": "registry-name",
        "torchlens/errors/__init__.py": "refusal-registry",
        "torchlens/schemas/*": "schema-json",
    }


def test_rendered_tier_keeps_committed_settings(repo: Path) -> None:
    """``render`` regenerates ``paths`` but keeps a committed tier setting."""
    builder = _load_builder()
    data = json.loads((repo / "test-tiers.json").read_text(encoding="utf-8"))
    data["inner"]["tier"] = {"seconds": 1200}
    (repo / "test-tiers.json").write_text(json.dumps(data), encoding="utf-8")
    rendered = json.loads(_render(builder, repo))["inner"]["tier"]
    assert rendered["seconds"] == 1200
    assert rendered["marker"] == "smoke"
    assert rendered["paths"]["tests/conftest.py"] == "conftest"


def test_override_rows_rank_first(repo: Path) -> None:
    """A reviewed override test leads its module's list."""
    builder = _load_builder()
    rows = {"torchlens/_io/reader.py": {"tests": ["tests/test_a.py"], "why": "covers a branch"}}
    (repo / "scripts").mkdir()
    (repo / builder.OVERRIDES_FILE).write_text(json.dumps({"modules": rows}), encoding="utf-8")
    result = builder.build(repo)
    assert result.overridden == ["torchlens/_io/reader.py"]
    assert result.mapping["torchlens/_io/reader.py"] == ["tests/test_a.py", "tests/test_reader.py"]


def test_marked_override_rows_keep_their_place(repo: Path) -> None:
    """A reviewed override file that applies a ``rare`` mark still leads its list."""
    builder = _load_builder()
    (repo / "tests/test_a.py").write_text(
        "import pytest\nfrom torchlens.core import a\n\n\n@pytest.mark.rare\ndef test_x():\n    pass\n",
        encoding="utf-8",
    )
    rows = {"torchlens/_io/reader.py": {"tests": ["tests/test_a.py"], "why": "covers a branch"}}
    (repo / "scripts").mkdir()
    (repo / builder.OVERRIDES_FILE).write_text(json.dumps({"modules": rows}), encoding="utf-8")
    result = builder.build(repo)
    assert result.mapping["torchlens/_io/reader.py"] == ["tests/test_a.py", "tests/test_reader.py"]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"torchlens/gone.py": {"tests": ["tests/test_a.py"], "why": "x"}}, "unknown module"),
        ({"torchlens/core/a.py": {"tests": ["tests/test_gone.py"], "why": "x"}}, "unknown test"),
        ({"torchlens/core/a.py": {"tests": ["tests/test_a.py"], "why": " "}}, "no reason"),
    ],
)
def test_stale_override_rows_are_refused(
    repo: Path, row: dict[str, dict[str, object]], message: str
) -> None:
    """A stale or unexplained override row stops the build."""
    builder = _load_builder()
    (repo / "scripts").mkdir()
    (repo / builder.OVERRIDES_FILE).write_text(json.dumps({"modules": row}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        builder.build(repo)
