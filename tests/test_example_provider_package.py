"""The installable example provider package (ecosystem build item B3).

``examples/provider_package`` is a real distribution: this test builds its
wheel, installs the wheel into one fresh virtual environment and the source
tree editable into another, and then shows TorchLens discovers the provider
from installed metadata alone and activates it through the export door, in a
subprocess of each environment. Nothing is mocked: the environments come from
the standard ``venv`` module with pip from ``ensurepip``, every pip call runs
``--no-index``, and the fresh environments see the running TorchLens and torch
through one ``site.addsitedir`` line, so the provider's own install is the only
thing they add.
"""

from __future__ import annotations

import json
import shutil
import site
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from torchlens.ecosystem import plugins

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_DIR = _REPO_ROOT / "examples" / "provider_package"
_DISTRIBUTION = "torchlens-example-provider"
_MODULE = "torchlens_example_provider"
_TARGET = "example_provider_summary"
_GROUP = "torchlens.export_targets"

_PROBE = textwrap.dedent(
    f"""
    import json
    import sys

    import torch

    import torchlens as tl
    from torchlens.ecosystem import plugins
    from torchlens.export import export_target_info, resolve_export_target

    rows = [r for r in plugins.discover({_GROUP!r}) if r.distribution == {_DISTRIBUTION!r}]
    imported_by_discovery = {_MODULE!r} in sys.modules
    report = plugins.activate({_DISTRIBUTION!r})
    info = export_target_info({_TARGET!r})
    trace = tl.trace(
        torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU()), torch.randn(1, 3)
    )
    summary = resolve_export_target({_TARGET!r})(trace)
    module = sys.modules[{_MODULE!r}]
    print(json.dumps({{
        "rows": [[r.group, r.name, r.value, r.distribution, r.version] for r in rows],
        "imported_by_discovery": imported_by_discovery,
        "loaded": [[c.group, c.name] for c in report.loaded],
        "failed": [[c.group, c.name] for c in report.failed],
        "provider_distribution": info.provider.distribution,
        "tier": info.capabilities["tier"],
        "summary": summary,
        "trace_layer_labels": [str(label) for label in trace.layer_labels],
        "module_file": module.__file__,
        "torchlens_file": tl.__file__,
    }}))
    """
)


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one command, failing with its full output when it exits nonzero."""

    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {command}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _fresh_venv(root: Path) -> Path:
    """Create a fresh venv that also sees the running interpreter's site-packages.

    ``addsitedir`` (not a bare path line) so the running environment's own
    ``.pth`` files, including an editable TorchLens install, still apply.
    """

    _run([sys.executable, "-m", "venv", str(root)], cwd=root.parent)
    python = root / "bin" / "python"
    own_site = _run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        cwd=root.parent,
    ).stdout.strip()
    parent_sites = [*site.getsitepackages(), site.getusersitepackages()]
    lines = [f"import site; site.addsitedir({path!r})" for path in parent_sites]
    (Path(own_site) / "zz_torchlens_parent_env.pth").write_text("\n".join(lines) + "\n")
    return python


def _probe(python: Path, workdir: Path) -> dict[str, object]:
    """Run the discovery and activation probe in one environment."""

    script = workdir / "probe_provider.py"
    script.write_text(_PROBE, encoding="utf-8")
    out = _run([str(python), "-I", str(script)], cwd=workdir).stdout.strip().splitlines()[-1]
    return json.loads(out)


def _assert_discovered_and_activated(result: dict[str, object]) -> None:
    """The provider is found from metadata, imported only by activation, and works."""

    assert result["rows"] == [
        [_GROUP, "example_summary", f"{_MODULE}:export_target", _DISTRIBUTION, "0.1.0"]
    ]
    assert result["imported_by_discovery"] is False
    assert result["loaded"] == [[_GROUP, "example_summary"]]
    assert result["failed"] == []
    assert result["provider_distribution"] == _DISTRIBUTION
    assert result["tier"] == "present"
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert summary["layer_labels"] == result["trace_layer_labels"]
    assert summary["num_layers"] == len(result["trace_layer_labels"]) > 0
    assert (
        Path(str(result["torchlens_file"]))
        .resolve()
        .is_relative_to(Path(plugins.__file__).resolve().parents[2])
    ), "the probe environment must import the TorchLens under test"


@pytest.fixture(scope="module")
def package_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy the example package so builds leave nothing in the repository."""

    root = tmp_path_factory.mktemp("example_provider")
    copy = root / "provider_package"
    shutil.copytree(_PACKAGE_DIR, copy, ignore=shutil.ignore_patterns("build", "*.egg-info"))
    return copy


def test_provider_is_not_installed_in_the_test_environment() -> None:
    """Negative control: the running environment has no example provider."""

    assert not [r for r in plugins.discover(_GROUP) if r.distribution == _DISTRIBUTION]


def test_built_wheel_installs_and_is_discovered(package_copy: Path, tmp_path: Path) -> None:
    """Build the wheel, install it into a fresh venv, discover and activate it."""

    python = _fresh_venv(tmp_path / "venv_wheel")
    wheelhouse = tmp_path / "wheelhouse"
    _run(
        [
            str(python),
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-index",
            "--no-build-isolation",
            "-w",
            str(wheelhouse),
            str(package_copy),
        ],
        cwd=tmp_path,
    )
    wheels = sorted(wheelhouse.glob("torchlens_example_provider-0.1.0-*.whl"))
    assert len(wheels) == 1, sorted(p.name for p in wheelhouse.iterdir())
    _run(
        [str(python), "-m", "pip", "install", "--no-deps", "--no-index", str(wheels[0])],
        cwd=tmp_path,
    )
    result = _probe(python, tmp_path)
    _assert_discovered_and_activated(result)
    module_file = Path(str(result["module_file"])).resolve()
    assert (tmp_path / "venv_wheel").resolve() in module_file.parents, module_file


def test_editable_install_is_discovered(package_copy: Path, tmp_path: Path) -> None:
    """Install the source tree editable into a fresh venv, discover and activate it."""

    python = _fresh_venv(tmp_path / "venv_editable")
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--no-build-isolation",
            "-e",
            str(package_copy),
        ],
        cwd=tmp_path,
    )
    result = _probe(python, tmp_path)
    _assert_discovered_and_activated(result)
    module_file = Path(str(result["module_file"])).resolve()
    assert package_copy.resolve() in module_file.parents, module_file
