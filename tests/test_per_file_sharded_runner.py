"""Argument handling of scripts/run_pytest_per_file_sharded.py (weekly slow tier)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_pytest_per_file_sharded.py"


def _load_runner() -> ModuleType:
    """Import the runner script as a module."""

    spec = importlib.util.spec_from_file_location("run_pytest_per_file_sharded", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_file_batch_size_overrides_parse_per_file() -> None:
    runner = _load_runner()
    parsed = runner.parse_file_batch_sizes(["tests/test_a.py=1", "tests/test_b.py=5"])
    assert parsed == {"tests/test_a.py": 1, "tests/test_b.py": 5}


@pytest.mark.parametrize("spec", ["tests/test_a.py", "=3", "tests/test_a.py=0"])
def test_malformed_file_batch_size_refuses(spec: str) -> None:
    runner = _load_runner()
    with pytest.raises(ValueError):
        runner.parse_file_batch_sizes([spec])
