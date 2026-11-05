"""The [neuro,brainscore] conflict refuses truthfully (``neuro_dependency_conflict``).

``torchlens[neuro,brainscore]`` installs rsatoolbox 0.3 beside the
scikit-learn<1.6 pin brainscore-vision brings, and ``import rsatoolbox``
then fails. These tests simulate that environment and pin:

- the metadata-only diagnosis (``torchlens.neuro._conflicts``) fires exactly
  for rsatoolbox >= 0.3 with scikit-learn < 1.6 and names the pinning
  package when it is installed;
- the facade gate stops advertising ``rdms``/``datasets`` (``hasattr``
  answers ``False``) and access raises the typed refusal with the conflict
  and the fix on ``fields``, without importing rsatoolbox;
- first use through ``require_rsatoolbox`` raises the same typed refusal
  when the real import fails, keeps the failed import on ``__context__``,
  and says "installed but failed to import" for an undiagnosed failure;
- a fresh interpreter with planted dist-info (rsatoolbox 0.3.2,
  scikit-learn 1.5.2, brainscore-vision 2.3.22) and an rsatoolbox package
  whose import fails reproduces the whole story end to end.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

import torchlens.neuro as neuro
from torchlens._errors import MissingDependencyError
from torchlens.neuro import _conflicts, _handoff

_CONFLICT_VERSIONS = {
    "rsatoolbox": "0.3.2",
    "scikit-learn": "1.5.2",
    "brainscore-vision": "2.3.22",
}


def _fake_installed(versions: dict[str, str]) -> object:
    """Return a stand-in for ``_conflicts._installed`` over a version table."""

    def installed(distribution: str) -> str | None:
        return versions.get(distribution)

    return installed


@pytest.fixture
def rsatoolbox_present(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make ``find_spec("rsatoolbox")`` answer present and its import fail."""

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "rsatoolbox":
            return importlib.machinery.ModuleSpec("rsatoolbox", None)
        return real_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(_conflicts, "find_spec", fake_find_spec)
    # A None entry makes ``import rsatoolbox`` raise ImportError: the
    # simulated failing import.
    monkeypatch.setitem(sys.modules, "rsatoolbox", None)
    for name in ("rdms", "datasets"):
        monkeypatch.delitem(vars(neuro), name, raising=False)
    yield


@pytest.mark.parametrize(
    ("versions", "expected"),
    [
        (_CONFLICT_VERSIONS, "brainscore-vision 2.3.22"),
        ({"rsatoolbox": "0.3.0", "scikit-learn": "1.5"}, "no pinning package"),
        ({"rsatoolbox": "0.3.2", "scikit-learn": "1.6.1"}, "no conflict"),
        ({"rsatoolbox": "0.2.0", "scikit-learn": "1.5.2"}, "no conflict"),
        ({"rsatoolbox": "0.1.5", "scikit-learn": "1.2.0"}, "no conflict"),
        ({"scikit-learn": "1.5.2", "brainscore-vision": "2.3.22"}, "no conflict"),
        ({"rsatoolbox": "0.3.2"}, "no conflict"),
        ({"rsatoolbox": "unknown", "scikit-learn": "1.5.2"}, "no conflict"),
    ],
)
def test_diagnosis_fires_only_for_the_measured_conflict(
    monkeypatch: pytest.MonkeyPatch, versions: dict[str, str], expected: str
) -> None:
    """rsatoolbox >= 0.3 with scikit-learn < 1.6 is the only diagnosed case."""

    monkeypatch.setattr(_conflicts, "_installed", _fake_installed(versions))
    conflict = _conflicts.rsatoolbox_conflict()
    if expected == "no conflict":
        assert conflict is None
        return
    assert conflict is not None
    assert conflict.rsatoolbox == versions["rsatoolbox"]
    assert conflict.scikit_learn == versions["scikit-learn"]
    assert conflict.pinned_by == (None if expected == "no pinning package" else expected)


def test_gate_refuses_and_stops_advertising(
    monkeypatch: pytest.MonkeyPatch, rsatoolbox_present: None
) -> None:
    """Touching ``neuro.rdms`` under the conflict raises the typed refusal."""

    monkeypatch.setattr(_conflicts, "_installed", _fake_installed(_CONFLICT_VERSIONS))
    for name in ("rdms", "datasets"):
        assert hasattr(neuro, name) is False
        with pytest.raises(MissingDependencyError) as info:
            getattr(neuro, name)
        err = info.value
        assert isinstance(err, AttributeError)
        assert err.fields["code"] == "neuro_dependency_conflict"
        assert err.fields["attribute"] == name
        assert err.fields["dependency"] == "rsatoolbox"
        assert err.fields["dependency_version"] == "0.3.2"
        assert err.fields["conflicting"] == "scikit-learn"
        assert err.fields["conflicting_version"] == "1.5.2"
        assert err.fields["pinned_by"] == "brainscore-vision 2.3.22"
        message = str(err)
        assert "installed but cannot import" in message
        assert "scikit-learn>=1.6" in message
        assert "separate environments" in message
        assert "requires rsatoolbox: install it" not in message


def test_gate_without_pinning_package_teaches_the_upgrade(
    monkeypatch: pytest.MonkeyPatch, rsatoolbox_present: None
) -> None:
    """No brainscore-vision installed: the fix is the scikit-learn upgrade."""

    versions = {"rsatoolbox": "0.3.2", "scikit-learn": "1.5.2"}
    monkeypatch.setattr(_conflicts, "_installed", _fake_installed(versions))
    with pytest.raises(MissingDependencyError) as info:
        neuro.rdms  # noqa: B018 - the access IS the provocation.
    assert info.value.fields["code"] == "neuro_dependency_conflict"
    assert info.value.fields["pinned_by"] is None
    assert 'pip install "scikit-learn>=1.6"' in str(info.value)


def test_first_use_raises_the_typed_refusal_over_the_failed_import(
    monkeypatch: pytest.MonkeyPatch, rsatoolbox_present: None
) -> None:
    """``require_rsatoolbox`` names the conflict, not the installed extra."""

    monkeypatch.setattr(_conflicts, "_installed", _fake_installed(_CONFLICT_VERSIONS))
    with pytest.raises(MissingDependencyError) as info:
        _handoff.require_rsatoolbox(attribute="datasets")
    err = info.value
    assert err.fields["code"] == "neuro_dependency_conflict"
    assert err.fields["owner"] == "torchlens.neuro"
    assert err.fields["attribute"] == "datasets"
    assert isinstance(err.__context__, ImportError)


def test_undiagnosed_import_failure_says_installed(
    monkeypatch: pytest.MonkeyPatch, rsatoolbox_present: None
) -> None:
    """An installed rsatoolbox failing for another reason is not 'install it'."""

    versions = {"rsatoolbox": "0.3.2", "scikit-learn": "1.6.1"}
    monkeypatch.setattr(_conflicts, "_installed", _fake_installed(versions))
    with pytest.raises(ImportError, match="installed but failed to import") as info:
        _handoff.require_rsatoolbox()
    assert not isinstance(info.value, MissingDependencyError)
    assert isinstance(info.value.__cause__, ImportError)


def test_absent_rsatoolbox_still_teaches_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """With rsatoolbox absent the refusal is unchanged: install the extra."""

    monkeypatch.setattr(_conflicts, "find_spec", lambda name, *a, **k: None)
    monkeypatch.setitem(sys.modules, "rsatoolbox", None)
    with pytest.raises(ImportError, match=r'pip install "torchlens\[neuro\]"'):
        _handoff.require_rsatoolbox()


_PROBE = textwrap.dedent(
    """
    import json, sys
    import torchlens.neuro as neuro
    from torchlens._errors import MissingDependencyError
    from torchlens.neuro._handoff import require_rsatoolbox

    out = {"all": sorted(neuro.__all__), "dir_has_rdms": "rdms" in dir(neuro)}
    out["hasattr_rdms"] = hasattr(neuro, "rdms")
    out["imported_by_probes"] = bool(getattr(sys, "_tl_fake_rsatoolbox_ran", False))
    try:
        neuro.rdms
        out["access"] = "resolved"
    except MissingDependencyError as exc:
        out["access"] = exc.fields["code"]
        out["pinned_by"] = exc.fields["pinned_by"]
    try:
        require_rsatoolbox()
        out["first_use"] = "resolved"
    except MissingDependencyError as exc:
        out["first_use"] = exc.fields["code"]
        out["first_use_context"] = str(exc.__context__)
    out["imported_by_first_use"] = bool(getattr(sys, "_tl_fake_rsatoolbox_ran", False))
    print(json.dumps(out))
    """
)


def _plant_distribution(root: Path, name: str, version: str) -> None:
    """Write a minimal dist-info so ``importlib.metadata`` reports it."""

    dist_info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
    )


@pytest.mark.heavy
def test_planted_conflict_environment_end_to_end(tmp_path: Path) -> None:
    """A fresh interpreter in a planted [neuro,brainscore] layout refuses truthfully."""

    for name, version in _CONFLICT_VERSIONS.items():
        _plant_distribution(tmp_path, name, version)
    package = tmp_path / "rsatoolbox"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import sys\n"
        "sys._tl_fake_rsatoolbox_ran = True\n"
        "raise ImportError(\"cannot import name 'validate_data' from "
        "'sklearn.utils.validation'\")\n",
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parent.parent
    env_path = f"{tmp_path}{':' + str(repo_root)}"
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env={"PYTHONPATH": env_path, "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert "rdms" not in result["all"] and "datasets" not in result["all"]
    assert result["dir_has_rdms"] is False
    assert result["hasattr_rdms"] is False
    assert result["imported_by_probes"] is False
    assert result["access"] == "neuro_dependency_conflict"
    assert result["pinned_by"] == "brainscore-vision 2.3.22"
    assert result["first_use"] == "neuro_dependency_conflict"
    assert "validate_data" in result["first_use_context"]
    assert result["imported_by_first_use"] is True
