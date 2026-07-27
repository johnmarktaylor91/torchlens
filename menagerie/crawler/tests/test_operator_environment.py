"""Targeted tests for the exact-lock environment operator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlsplit

import pytest
import yaml

from menagerie.crawler.driver_admission import CommandEnvironmentBackend
from menagerie.crawler.env_lifecycle import (
    installed_package_inventory_bytes,
    parse_exact_lock,
    parse_resolved_export,
)
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.operator_environment import (
    CommandObservation,
    EnvironmentOperator,
    PermanentEnvironmentOperatorError,
    TransientEnvironmentOperatorError,
)


class FakeToolRunner:
    """Inject deterministic conda-lock, conda-create, and probe observations."""

    def __init__(
        self,
        package_rows: Sequence[Mapping[str, Any]],
        *,
        solve_observations: Sequence[CommandObservation] = (),
    ) -> None:
        """Store solved rows and optional faults returned before solve success."""

        self.package_rows = tuple(package_rows)
        self.solve_observations = list(solve_observations)
        self.solve_calls = 0
        self.create_calls = 0
        self.probe_calls: list[tuple[str, ...]] = []
        self.commands: list[tuple[str, ...]] = []
        self.environments: list[Mapping[str, str]] = []

    def __call__(
        self,
        command: Sequence[str],
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> CommandObservation:
        """Return the configured observation and materialize requested fake outputs."""

        del timeout_seconds
        argv = tuple(command)
        self.commands.append(argv)
        self.environments.append(dict(environment))
        if "lock" in argv:
            self.solve_calls += 1
            if self.solve_observations:
                observation = self.solve_observations.pop(0)
                if observation.returncode != 0:
                    return observation
            lock_path = Path(argv[argv.index("--lockfile") + 1])
            lock_path.write_text(
                yaml.safe_dump({"package": list(self.package_rows)}),
                encoding="utf-8",
            )
            return CommandObservation(0, "solved", "")
        if "create" in argv:
            self.create_calls += 1
            prefix = Path(argv[argv.index("--prefix") + 1])
            metadata_root = prefix / "conda-meta"
            metadata_root.mkdir(parents=True)
            for index, row in enumerate(self.package_rows):
                filename = unquote(Path(urlsplit(str(row["url"])).path).name)
                stem = (
                    filename[: -len(".tar.bz2")]
                    if filename.endswith(".tar.bz2")
                    else filename[: -len(".conda")]
                )
                name = str(row["name"])
                version = str(row["version"])
                build = stem.removeprefix(f"{name}-{version}-")
                digest = str(row["hash"]["sha256"])
                (metadata_root / f"{index}.json").write_text(
                    json.dumps(
                        {
                            "name": name,
                            "version": version,
                            "build": build,
                            "url": row["url"],
                            "sha256": digest,
                        }
                    ),
                    encoding="utf-8",
                )
            return CommandObservation(0, "created", "")
        self.probe_calls.append(argv)
        if "missing.module" in argv:
            return CommandObservation(1, "", "ModuleNotFoundError: missing.module")
        return CommandObservation(0, "", "")


def _write_environment(path: Path, *, pip: bool = False) -> Path:
    """Write one minimal conda environment specification."""

    dependencies: list[Any] = ["python=3.11"]
    if pip:
        dependencies.append({"pip": ["example==1"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "test-environment",
                "channels": ["conda-forge", "nodefaults"],
                "dependencies": dependencies,
            }
        ),
        encoding="utf-8",
    )
    return path.resolve()


def _package_row(artifact: Path, content: bytes) -> dict[str, Any]:
    """Return one conda-lock row bound to an exact local artifact."""

    artifact.write_bytes(content)
    return {
        "name": "demo-package",
        "version": "1.2.3",
        "manager": "conda",
        "platform": "osx-arm64",
        "url": artifact.resolve().as_uri(),
        "hash": {"sha256": hash_bytes(content).removeprefix("sha256:")},
    }


def _operator(
    tmp_path: Path,
    runner: FakeToolRunner,
    *,
    sleeper: Any = lambda _seconds: None,
) -> EnvironmentOperator:
    """Build one operator with fake argv-only tool boundaries."""

    return EnvironmentOperator(
        state_root=(tmp_path / "operator-state").resolve(),
        conda_lock_command=("/fake/conda-lock",),
        conda_command=("/fake/conda",),
        runner=runner,
        sleeper=sleeper,
    )


def test_solve_synthesizes_sha256_lock_export_receipts_and_hits_cache(
    tmp_path: Path,
) -> None:
    """A second intent/target solve returns unchanged reverified cache artifacts."""

    content = b"exact conda package bytes"
    artifact = tmp_path / "demo-package-1.2.3-py311h123_0.conda"
    runner = FakeToolRunner((_package_row(artifact, content),))
    operator = _operator(tmp_path, runner)
    environment = _write_environment(tmp_path / "envs" / "core" / "environment.yml")

    first = operator.solve(environment, "osx-arm64")
    second = operator.solve(environment, "osx-arm64")

    assert runner.solve_calls == 1
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["artifact_bytes"] == 0
    assert first["lock_path"] == second["lock_path"]
    assert first["resolved_export_path"] == second["resolved_export_path"]
    lock_bytes = Path(first["lock_path"]).read_bytes()
    receipts = parse_exact_lock(lock_bytes)
    assert len(receipts) == 1
    assert len(receipts[0].sha256.removeprefix("sha256:")) == 64
    assert b"@EXPLICIT" in lock_bytes
    assert b"#sha256=" not in lock_bytes
    export_bytes = Path(first["resolved_export_path"]).read_bytes()
    assert parse_resolved_export(export_bytes) == export_bytes
    package = json.loads(export_bytes)["packages"][0]
    assert package["build"] == "py311h123_0"
    assert first["artifacts"] == second["artifacts"]
    receipt_path = Path(first["artifacts"][0]["path"])
    assert receipt_path.read_bytes() == content
    assert receipt_path.name == receipts[0].sha256.removeprefix("sha256:")
    assert all("render" not in command for command in runner.commands)


def test_force_resolve_is_the_only_way_to_replace_valid_solve_cache(
    tmp_path: Path,
) -> None:
    """Specification changes remain generation-stable until operator force is explicit."""

    content = b"stable artifact"
    artifact = tmp_path / "demo-package-1.2.3-build_0.conda"
    runner = FakeToolRunner((_package_row(artifact, content),))
    operator = _operator(tmp_path, runner)
    environment = _write_environment(tmp_path / "envs" / "core" / "environment.yml")

    first = operator.solve(environment, "osx-arm64")
    environment.write_text(
        environment.read_text(encoding="utf-8").replace("python=3.11", "python=3.12"),
        encoding="utf-8",
    )
    cached = operator.solve(environment, "osx-arm64")
    forced = operator.solve(environment, "osx-arm64", force_resolve=True)

    assert runner.solve_calls == 2
    assert cached["cache_hit"] is True
    assert forced["cache_hit"] is False
    assert Path(first["lock_path"]).read_bytes() == Path(forced["lock_path"]).read_bytes()


def test_driver_gate_a_accepts_operator_lock_and_artifact_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production adapter re-hashes and accepts the wrapper's exact solve payload."""

    content = b"driver-admitted artifact"
    artifact = tmp_path / "demo-package-1.2.3-build_0.conda"
    runner = FakeToolRunner((_package_row(artifact, content),))
    operator = _operator(tmp_path, runner)
    environment = _write_environment(tmp_path / "envs" / "core" / "environment.yml")
    payload = operator.solve(environment, "osx-arm64")
    backend = CommandEnvironmentBackend(("unused",))

    def return_payload(action: str, *arguments: str) -> dict[str, Any]:
        """Return the already-produced operator payload to the production verifier."""

        assert action == "solve"
        assert arguments == (str(environment), "osx-arm64")
        return payload

    monkeypatch.setattr(backend, "_json_action", return_payload)
    solved = backend.solve(environment, "osx-arm64")

    assert solved.lock_bytes == Path(payload["lock_path"]).read_bytes()
    assert solved.resolved_export_bytes == Path(payload["resolved_export_path"]).read_bytes()
    assert tuple(solved.artifact_receipts) == parse_exact_lock(solved.lock_bytes)


def test_transient_solve_fault_retries_without_becoming_permanent(
    tmp_path: Path,
) -> None:
    """An injected transient subprocess failure is retried and then succeeds."""

    content = b"eventual artifact"
    artifact = tmp_path / "demo-package-1.2.3-build_0.conda"
    runner = FakeToolRunner(
        (_package_row(artifact, content),),
        solve_observations=(
            CommandObservation(1, "", "HTTP 503: service temporarily unavailable"),
        ),
    )
    backoffs: list[float] = []
    operator = _operator(tmp_path, runner, sleeper=backoffs.append)
    environment = _write_environment(tmp_path / "envs" / "core" / "environment.yml")

    result = operator.solve(environment, "osx-arm64")

    assert result["cache_hit"] is False
    assert runner.solve_calls == 2
    assert backoffs == [1.0]


def test_permanent_and_exhausted_transient_faults_remain_distinct(
    tmp_path: Path,
) -> None:
    """Permanent solve rejection is not retried while exhausted transport is retryable."""

    content = b"unused artifact"
    artifact = tmp_path / "demo-package-1.2.3-build_0.conda"
    environment = _write_environment(tmp_path / "envs" / "core" / "environment.yml")
    permanent_runner = FakeToolRunner(
        (_package_row(artifact, content),),
        solve_observations=(
            CommandObservation(1, "", "package specifications are incompatible"),
        ),
    )
    with pytest.raises(PermanentEnvironmentOperatorError, match="failed permanently"):
        _operator(tmp_path, permanent_runner).solve(environment, "osx-arm64")
    assert permanent_runner.solve_calls == 1

    transient_runner = FakeToolRunner(
        (_package_row(artifact, content),),
        solve_observations=tuple(
            CommandObservation(75, "", "temporary failure") for _ in range(3)
        ),
    )
    backoffs: list[float] = []
    with pytest.raises(TransientEnvironmentOperatorError, match="3 attempts"):
        _operator(tmp_path, transient_runner, sleeper=backoffs.append).solve(
            environment, "osx-arm64"
        )
    assert transient_runner.solve_calls == 3
    assert backoffs == [1.0, 2.0]


def test_solve_refuses_pip_before_invoking_conda_lock(tmp_path: Path) -> None:
    """A pip section cannot silently bypass the conda-meta inventory contract."""

    runner = FakeToolRunner(())
    operator = _operator(tmp_path, runner)
    environment = _write_environment(
        tmp_path / "envs" / "core" / "environment.yml", pip=True
    )

    with pytest.raises(PermanentEnvironmentOperatorError, match="pip sections"):
        operator.solve(environment, "osx-arm64")
    assert runner.solve_calls == 0


def test_create_stages_verified_cas_offline_and_remove_retains_cas(
    tmp_path: Path,
) -> None:
    """Create uses a generation-local cache; exact removal leaves shared CAS bytes."""

    content = b"offline package"
    artifact = tmp_path / "demo-package-1.2.3-build_0.tar.bz2"
    runner = FakeToolRunner((_package_row(artifact, content),))
    operator = _operator(tmp_path, runner)
    environment = _write_environment(tmp_path / "envs" / "core" / "environment.yml")
    solved = operator.solve(environment, "osx-arm64")
    prefix = (tmp_path / "runtime" / "prefix").resolve()

    created = operator.create(Path(solved["lock_path"]), prefix)

    package_cache = Path(created["package_cache_path"])
    staged = package_cache / artifact.name
    assert staged.read_bytes() == content
    assert runner.create_calls == 1
    create_environment = runner.environments[-1]
    assert create_environment["CONDA_OFFLINE"] == "true"
    assert create_environment["CONDA_PKGS_DIRS"] == str(package_cache)
    assert "HTTP_PROXY" not in create_environment
    assert "--offline" in runner.commands[-1]
    assert installed_package_inventory_bytes(prefix) == parse_resolved_export(
        Path(solved["resolved_export_path"]).read_bytes()
    )
    cas_path = Path(solved["artifacts"][0]["path"])
    removed = operator.remove(prefix)
    assert removed["cas_retained"] is True
    assert not prefix.exists()
    assert not package_cache.exists()
    assert cas_path.read_bytes() == content
    with pytest.raises(PermanentEnvironmentOperatorError, match="not an exact registered"):
        operator.remove(prefix)


def test_probe_returns_exact_declared_order_and_bounded_failure_detail(
    tmp_path: Path,
) -> None:
    """Import, export, and source-build receipts preserve declaration order exactly."""

    runner = FakeToolRunner(())
    operator = _operator(tmp_path, runner)
    prefix = (tmp_path / "runtime" / "prefix").resolve()
    interpreter = prefix / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    probes = {
        "imports": ["present.module", "missing.module"],
        "export_checks": [{"module": "present.module", "attribute": "Model"}],
        "source_build": [
            {
                "name": "compiled-extension",
                "packages": ["present"],
                "command": ["python", "-c", "import present._C"],
                "max_attempts": 2,
            }
        ],
    }

    result = operator.probe(prefix, json.dumps(probes))

    assert [row["name"] for row in result["results"]] == [
        "import:present.module",
        "import:missing.module",
        "export:present.module:Model",
        "source-build:compiled-extension",
    ]
    assert [row["passed"] for row in result["results"]] == [True, False, True, True]
    assert result["results"][1]["detail"] == "ModuleNotFoundError: missing.module"
    assert all(len(row["detail"]) <= 1_000 for row in result["results"])
    assert runner.probe_calls[-1][0] == str(interpreter)
    assert "PYTHONPATH" not in runner.environments[-1]
    assert runner.environments[-1]["HF_HUB_OFFLINE"] == "1"
    assert runner.environments[-1]["PIP_NO_INDEX"] == "1"
