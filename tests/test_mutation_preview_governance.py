"""Governance for the preview-backend validation mutant rosters.

The rosters in ``tests/support/mutation_preview.py`` are hand-listed source
snippets inside named functions of the JAX, tinygrad and MLX validation
paths. A snippet that drifts out of the product code would make its mutant
unplantable, and the camp would refuse at roster assembly instead of
measuring a margin. This test plants every roster entry on a temporary copy
of its file, proves the plant changed the file and still compiles, and proves
the restore is byte-exact, so a drifted roster fails here, on every tier,
without any preview framework installed.
"""

from __future__ import annotations

import importlib.util
import py_compile
import shutil
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAX_PER_BACKEND = 8


def _load_preview_module():
    """Import the preview camp as a module (its import has no side effects)."""

    path = _REPO_ROOT / "tests" / "support" / "mutation_preview.py"
    spec = importlib.util.spec_from_file_location("_mutation_preview_gov", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _roster_entries() -> list[tuple[str, str]]:
    module = _load_preview_module()
    return [(backend, mid) for backend, roster in module.ROSTERS.items() for mid in roster]


def test_each_preview_backend_roster_is_bounded_and_nonempty() -> None:
    """Every preview backend carries between one and eight planted mutants."""

    module = _load_preview_module()
    assert set(module.ROSTERS) == set(module.BACKENDS)
    for backend, roster in module.ROSTERS.items():
        assert 1 <= len(roster) <= _MAX_PER_BACKEND, backend
        assert all(mid.startswith(("J", "T", "X")) for mid in roster), backend


@pytest.mark.parametrize(("backend", "mid"), _roster_entries())
def test_preview_mutant_plants_compiles_and_restores(
    backend: str, mid: str, tmp_path: Path
) -> None:
    """Each roster entry targets live source: the plant changes it, the restore is exact."""

    module = _load_preview_module()
    mutant = module.ROSTERS[backend][mid]
    source = _REPO_ROOT / mutant.rel
    assert source.exists(), f"{mid}: {mutant.rel} is gone"
    sandbox = tmp_path / "repo"
    (sandbox / mutant.rel).parent.mkdir(parents=True)
    shutil.copy(source, sandbox / mutant.rel)
    original = module.apply_mutant(sandbox, mutant)
    mutated = (sandbox / mutant.rel).read_text(encoding="utf-8")
    assert mutated != original, f"{mid}: plant left the file unchanged"
    py_compile.compile(str(sandbox / mutant.rel), doraise=True)
    (sandbox / mutant.rel).write_text(original, encoding="utf-8")
    assert (sandbox / mutant.rel).read_bytes() == source.read_bytes(), f"{mid}: restore drifted"


def test_preview_suite_derivation_reaches_each_validation_file() -> None:
    """The marker scan enrolls each backend's validation test module."""

    module = _load_preview_module()
    expected = {
        "jax": "tests/test_jax_backend_capture.py",
        "tinygrad": "tests/test_tinygrad_backend_validation.py",
        "mlx": "tests/test_mlx_backend_validation.py",
    }
    for backend, (marker, _env) in module.BACKENDS.items():
        suite = module.derive_suite(_REPO_ROOT, marker)
        assert expected[backend] in suite, (backend, suite)
        assert all(Path(path).name.startswith("test_") for path in suite), backend
