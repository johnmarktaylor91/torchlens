"""Clean-environment install tests for the built wheel and sdist.

The wheel and sdist are built once per module. For every extras set of
``tools/packaging/matrix.toml`` a fresh ``python -m venv`` receives the artifact
with the set's extras under ``tools/packaging/constraints/<set>.txt``; a probe
subprocess then imports torchlens, runs the health call, one trace and one save
and load round trip, and exercises every dependency gate: absent extras refuse
with the typed ``MissingDependencyError`` and its code, present extras resolve.
Incompatible sets must fail to install, or install and refuse typed.

Selection: ``TORCHLENS_PACKAGING_SETS`` is a comma list of set names, ``pr`` or
``extended`` (a group), or ``all``; unset means the ``pr`` group. Every venv
install needs the package indexes named in ``matrix.toml``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.packaging]

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = Path(__file__).resolve().parent / "_probe_script.py"
INSTALL_TIMEOUT_S = 1800
PROBE_TIMEOUT_S = 600


def _load_generator() -> ModuleType:
    path = REPO_ROOT / "tools" / "packaging" / "generate_constraints.py"
    spec = importlib.util.spec_from_file_location("_packaging_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()
MATRIX = GENERATOR.load_matrix()
SETS = {entry.name: entry for entry in GENERATOR.resolve_sets(MATRIX, GENERATOR.declared_extras())}


def _selected_set_names() -> list[str]:
    raw = os.environ.get("TORCHLENS_PACKAGING_SETS", "pr").strip()
    if raw == "all":
        return sorted(SETS)
    if raw in GENERATOR.GROUPS:
        return sorted(name for name, entry in SETS.items() if entry.group == raw)
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = sorted(set(names) - set(SETS))
    assert not unknown, f"TORCHLENS_PACKAGING_SETS names unknown sets {unknown}"
    return names


def _clean_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME", "PIP_CONSTRAINT"}
    }
    env["OMP_NUM_THREADS"] = "1"
    return env


def _interpreter(version: str) -> str:
    """Return an interpreter for ``version`` or skip: the host lacks it, CI does not."""

    if f"{sys.version_info.major}.{sys.version_info.minor}" == version:
        return sys.executable
    found = shutil.which(f"python{version}")
    if found is None and shutil.which("uv") is not None:
        proc = subprocess.run(
            ["uv", "python", "find", version], capture_output=True, text=True, check=False
        )
        found = proc.stdout.strip() if proc.returncode == 0 else None
    if not found:
        pytest.skip(f"python {version} is not installed on this host")
    return found


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build the wheel and the sdist once, from a neutral working directory."""

    pytest.importorskip("build")
    out = tmp_path_factory.mktemp("dist")
    # A neutral cwd keeps a repo-root ./build/ directory from shadowing the
    # `build` module (python -m build puts the cwd on sys.path).
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(out), str(REPO_ROOT)],
        cwd=out,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = sorted(out.glob("torchlens-*.whl"))
    sdists = sorted(out.glob("torchlens-*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1, sorted(p.name for p in out.iterdir())
    return {"wheel": wheels[0], "sdist": sdists[0]}


def _install(python: str, requirement: str, constraints: Path) -> subprocess.CompletedProcess[str]:
    defaults = MATRIX["defaults"]
    indexes = [
        "--index-url",
        defaults["index_url"],
        "--extra-index-url",
        defaults["extra_index_url"],
    ]
    if shutil.which("uv") is not None:
        command = ["uv", "pip", "install", "--python", python, *indexes]
        command += ["--index-strategy", "unsafe-best-match", "-c", str(constraints), requirement]
    else:
        command = [python, "-m", "pip", "install", *indexes, "-c", str(constraints), requirement]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=_clean_env(),
        timeout=INSTALL_TIMEOUT_S,
        check=False,
    )


def _gate_present(name: str, gate: dict[str, Any], entry: Any, version: str) -> bool:
    if name in entry.present_gates:
        return True
    if gate["extra"] not in _expanded_extras(entry.extras):
        return False
    minimum = gate.get("python_min")
    return minimum is None or _version_tuple(version) >= _version_tuple(minimum)


def _expanded_extras(extras: tuple[str, ...]) -> set[str]:
    aliases = MATRIX.get("aliases", {})
    return {aliases.get(extra, extra) for extra in extras}


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".")[:2])


def _probe_spec(entry: Any, version: str, save_path: Path) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for name, gate in MATRIX["gates"].items():
        present = _gate_present(name, gate, entry, version)
        spec = dict(gate)
        if present and gate.get("present") == "import":
            spec["check"] = "import"
        spec["expected"] = "resolved" if present else "refused"
        gates[name] = spec
    return {"save_path": str(save_path), "gates": gates, "import_fails": list(entry.import_fails)}


def _run_probe(python: str, spec: dict[str, Any], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [python, "-I", str(PROBE_SCRIPT), json.dumps(spec)],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=_clean_env(),
        timeout=PROBE_TIMEOUT_S,
        check=False,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = [line for line in proc.stdout.splitlines() if line.startswith("PROBE-RESULT ")]
    assert len(lines) == 1, proc.stdout[-3000:]
    return json.loads(lines[0].removeprefix("PROBE-RESULT "))


def _assert_gate(name: str, gate: dict[str, Any], outcome: dict[str, Any]) -> None:
    if gate["expected"] == "resolved":
        assert outcome == {"outcome": "resolved"}, f"gate {name} should resolve: {outcome}"
        return
    assert outcome["outcome"] == "refused", f"gate {name} should refuse typed: {outcome}"
    assert outcome["class"] == "MissingDependencyError", outcome
    assert outcome["code"] == gate["code"], outcome
    if "dependency" in gate:
        assert outcome["dependency"] == gate["dependency"], outcome


def _requirement(artifact: Path, extras: tuple[str, ...]) -> str:
    suffix = f"[{','.join(extras)}]" if extras else ""
    return f"torchlens{suffix} @ {artifact.resolve().as_uri()}"


@pytest.fixture
def venv_dir(tmp_path: Path) -> Iterator[Path]:
    """A venv location deleted after the test: each holds a full torch install."""

    path = tmp_path / "venv"
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("set_name", _selected_set_names())
def test_clean_install(
    set_name: str, kind: str, artifacts: dict[str, Path], venv_dir: Path, tmp_path: Path
) -> None:
    """Install one artifact with one extras set into a fresh venv and probe it."""

    entry = SETS[set_name]
    base_python = _interpreter(entry.python)
    venv = venv_dir
    subprocess.run([base_python, "-m", "venv", str(venv)], check=True, env=_clean_env())
    python = str(venv / "bin" / "python")
    constraints = GENERATOR.CONSTRAINTS_DIR / f"{set_name}.txt"
    installed = _install(python, _requirement(artifacts[kind], entry.extras), constraints)

    if entry.expect == "install_fails":
        assert installed.returncode != 0, f"{set_name} is declared incompatible but installed"
        return
    if entry.expect == "install_or_refuse" and installed.returncode != 0:
        return
    assert installed.returncode == 0, f"install failed:\n{installed.stderr[-4000:]}"

    spec = _probe_spec(entry, entry.python, tmp_path / "roundtrip.tlspec")
    if entry.expect == "install_or_refuse":
        spec["gates"] = {name: spec["gates"][name] for name in entry.refuses}
        for gate in spec["gates"].values():
            gate.pop("dependency", None)  # an import failure names the module path
            gate["expected"] = "refused"
    result = _run_probe(python, spec, tmp_path)

    assert Path(result["torchlens_file"]).resolve().is_relative_to(venv.resolve()), result
    assert result["doctor"] == "DoctorReport", result
    traced, loaded = result["num_ops"]
    assert traced > 0 and traced == loaded, result
    for name, gate in spec["gates"].items():
        _assert_gate(name, gate, result["gates"][name])
    for module_name in entry.import_fails:
        outcome = result["import_fails"][module_name]
        assert outcome["outcome"] == "import_error", (
            f"{module_name} should fail to import: {outcome}"
        )


def test_generated_constraints_are_current() -> None:
    """The committed constraints files equal a fresh generation from matrix.toml."""

    script = GENERATOR.HERE / "generate_constraints.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--check"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_generator_check_is_red_capable(tmp_path: Path) -> None:
    """--check fails on a stale file and on an unexpected file."""

    script = GENERATOR.HERE / "generate_constraints.py"
    out = tmp_path / "constraints"
    subprocess.run([sys.executable, str(script), "--out", str(out)], check=True)
    command = [sys.executable, str(script), "--check", "--out", str(out)]
    assert subprocess.run(command, capture_output=True, check=False).returncode == 0
    (out / "none.txt").write_text("torch==0.0\n")
    assert subprocess.run(command, capture_output=True, check=False).returncode == 1
    subprocess.run([sys.executable, str(script), "--out", str(out)], check=True)
    (out / "orphan.txt").write_text("")
    assert subprocess.run(command, capture_output=True, check=False).returncode == 1


def test_matrix_covers_every_declared_extra() -> None:
    """Dropping an extra's set (without an alias) is refused by the generator."""

    matrix = GENERATOR.load_matrix()
    del matrix["sets"]["tabular"]
    with pytest.raises(GENERATOR.MatrixError, match="tabular"):
        GENERATOR.resolve_sets(matrix, GENERATOR.declared_extras())


def test_matrix_torch_pin_matches_the_pinned_smoke_row() -> None:
    """The matrix's shared torch pin is the torch of pr.yml's pinned smoke row."""

    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "pr.yml").read_text())
    smoke_torches = {
        str(row["torch"]) for row in workflow["jobs"]["smoke"]["strategy"]["matrix"]["include"]
    }
    pins = [pin for pin in MATRIX["constraints"]["common"] if pin.startswith("torch==")]
    assert len(pins) == 1 and pins[0].removeprefix("torch==") in smoke_torches, (
        pins,
        smoke_torches,
    )
