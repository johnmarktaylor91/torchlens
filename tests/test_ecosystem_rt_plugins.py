"""Provider mechanics (ecosystem MEMO section 5, build item B3).

Consent law v1, executed: DISCOVERY IS METADATA-ONLY (the provider sentinel
module never imports before explicit activation), activation is explicit and
distribution-scoped, ``TORCHLENS_PLUGINS=none`` beats everything, and every
refusal carries its stable code. The fake provider distributions live in an
on-disk site directory with genuine dist-info metadata, so discovery and
activation run through real ``importlib.metadata`` mechanics rather than
mocks.
"""

from __future__ import annotations

import hashlib
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from torchlens.ecosystem import plugins
from torchlens.ecosystem.plugins import (
    ENTRY_POINT_GROUPS,
    PluginLoadWarning,
    activate,
    activate_configured,
    discover,
    plugin_status,
)
from torchlens.errors import ConfigurationError

pytestmark = pytest.mark.smoke

_PROVIDER_MODULES = (
    "fakeprov_eco_rt",
    "fakeprov_eco_rt_appl",
    "fakeprov_eco_rt_broken",
)


@pytest.fixture(scope="module")
def provider_site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one site dir holding the test provider distributions."""

    site = tmp_path_factory.mktemp("eco_rt_site")
    (site / "fakeprov_eco_rt.py").write_text(
        textwrap.dedent(
            '''
            """Well-behaved test provider module (lane F32 plugin tests)."""


            def _export_fn(trace, path=None):
                """Trivial export function for the registered target."""

                return {"ok": True}


            def export_factory():
                """Zero-arg factory returning the frozen export-target row."""

                return {"name": "eco_rt_demo_target", "fn": _export_fn, "tier": "present"}


            def bad_return_factory():
                """Factory returning the WRONG shape for the export door."""

                return ["not", "a", "row"]


            def bad_backend_factory():
                """Factory returning a non-BackendSpec for the backend door."""

                return "not-a-backend-spec"


            NOT_CALLABLE = 42
            '''
        ),
        encoding="utf-8",
    )
    (site / "fakeprov_eco_rt_broken.py").write_text(
        'raise ImportError("broken provider module")\n', encoding="utf-8"
    )
    (site / "fakeprov_eco_rt_appl.py").write_text("SHOULD_NEVER_IMPORT = True\n", encoding="utf-8")

    def _dist(name: str, group: str, ep_name: str, target: str) -> None:
        """Write one minimal genuine dist-info declaring one entry point."""

        info = site / f"{name.replace('-', '_')}-0.1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.1.0\n", encoding="utf-8"
        )
        (info / "entry_points.txt").write_text(
            f"[{group}]\n{ep_name} = {target}\n", encoding="utf-8"
        )

    _dist("eco-rt-good", "torchlens.export_targets", "demo", "fakeprov_eco_rt:export_factory")
    _dist(
        "eco-rt-badret",
        "torchlens.export_targets",
        "badret",
        "fakeprov_eco_rt:bad_return_factory",
    )
    _dist("eco-rt-notcall", "torchlens.export_targets", "notcall", "fakeprov_eco_rt:NOT_CALLABLE")
    _dist("eco-rt-appl", "torchlens.appliances", "appl", "fakeprov_eco_rt_appl:factory")
    _dist("eco-rt-broken", "torchlens.export_targets", "broken", "fakeprov_eco_rt_broken:factory")
    _dist("eco-rt-backend", "torchlens.backends", "fakeback", "fakeprov_eco_rt:bad_backend_factory")
    _dist("eco-rt-recipes", "torchlens.recipes", "fakerecipe", "fakeprov_eco_rt:export_factory")
    return site


@pytest.fixture(autouse=True)
def _clean_plugin_state() -> Iterator[None]:
    """Snapshot and restore the session ledger; drop sentinel imports."""

    before = dict(plugins._RECORDS)
    try:
        yield
    finally:
        plugins._RECORDS.clear()
        plugins._RECORDS.update(before)
        for name in _PROVIDER_MODULES:
            sys.modules.pop(name, None)


@pytest.fixture()
def on_path(provider_site: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put the provider site on sys.path and guarantee a clean kill switch."""

    monkeypatch.syspath_prepend(str(provider_site))
    monkeypatch.delenv(plugins.PLUGINS_ENV_VAR, raising=False)
    return provider_site


# ---------------------------------------------------------------------------
# Discovery: metadata-only, sentinel never imports.
# ---------------------------------------------------------------------------


def test_discover_is_metadata_only_and_never_imports(on_path: Path) -> None:
    """discover() lists every candidate without importing any provider."""

    rows = discover()
    distributions = {row.distribution for row in rows}
    assert {"eco-rt-good", "eco-rt-appl", "eco-rt-broken"} <= distributions
    good = next(row for row in rows if row.distribution == "eco-rt-good")
    assert good.group == "torchlens.export_targets"
    assert good.value == "fakeprov_eco_rt:export_factory"
    assert good.version == "0.1.0"
    for module in _PROVIDER_MODULES:
        assert module not in sys.modules, module


def test_discover_unknown_group_refuses_typed(on_path: Path) -> None:
    """A group outside the approved five refuses plugin_group_unknown."""

    with pytest.raises(ConfigurationError) as excinfo:
        discover("torchlens.bogus")
    assert excinfo.value.fields["code"] == "plugin_group_unknown"
    assert "approved groups" in str(excinfo.value)


def test_plugin_status_reads_without_importing(on_path: Path) -> None:
    """The four-state listing starts every candidate at 'installed'."""

    listing = plugin_status()
    states = {
        record.candidate.distribution: record.state
        for record in listing
        if record.candidate.distribution.startswith("eco-rt-")
    }
    assert states and set(states.values()) == {"installed"}
    for module in _PROVIDER_MODULES:
        assert module not in sys.modules, module


def test_metadata_parse_failure_warns_never_raises(
    on_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Broken dist-info metadata is disclosed as a coded warning, not fatal."""

    def _boom() -> None:
        raise ValueError("corrupt dist-info")

    monkeypatch.setattr(plugins.metadata, "entry_points", _boom)
    with pytest.warns(PluginLoadWarning) as caught:
        rows = discover()
    assert rows == ()
    assert all(w.message.fields["code"] == "plugin_metadata_unreadable" for w in caught)


# ---------------------------------------------------------------------------
# Activation: explicit, distribution-scoped, doors enforced.
# ---------------------------------------------------------------------------


def test_activation_imports_commits_and_registers(on_path: Path) -> None:
    """Explicit activation loads the factory and commits through the door."""

    from torchlens.export import _registry as export_registry

    report = activate("eco-rt-good")
    try:
        assert [candidate.name for candidate in report.loaded] == ["demo"]
        assert report.failed == ()
        assert report.declaration_digest is None
        assert "fakeprov_eco_rt" in sys.modules
        assert "eco_rt_demo_target" in export_registry.export_targets()
        states = {record.candidate.distribution: record.state for record in plugin_status()}
        assert states["eco-rt-good"] == "loaded"
    finally:
        export_registry.unregister_export_target("eco_rt_demo_target")


def test_activation_not_installed_refuses_with_signpost(on_path: Path) -> None:
    """An unknown distribution refuses and NAMES the installed providers."""

    with pytest.raises(ConfigurationError) as excinfo:
        activate("eco-rt-missing")
    assert excinfo.value.fields["code"] == "plugin_distribution_not_installed"
    assert "eco-rt-good" in str(excinfo.value)


def test_activation_unknown_group_refuses(on_path: Path) -> None:
    """activate(groups=...) polices the approved-group vocabulary."""

    with pytest.raises(ConfigurationError) as excinfo:
        activate("eco-rt-good", groups=("torchlens.bogus",))
    assert excinfo.value.fields["code"] == "plugin_group_unknown"


def test_broken_provider_strict_raises_and_records_failed(on_path: Path) -> None:
    """Strict mode: a provider import failure raises typed, state='failed'."""

    with pytest.raises(ConfigurationError) as excinfo:
        activate("eco-rt-broken")
    assert excinfo.value.fields["code"] == "plugin_strict_load_failed"
    assert isinstance(excinfo.value.__cause__, ImportError)
    states = {record.candidate.distribution: record.state for record in plugin_status()}
    assert states["eco-rt-broken"] == "failed"


def test_broken_provider_nonstrict_warns_and_continues(on_path: Path) -> None:
    """strict=False: the batch continues past a broken provider, warned."""

    with pytest.warns(PluginLoadWarning) as caught:
        report = activate("eco-rt-broken", strict=False)
    assert [candidate.name for candidate in report.failed] == ["broken"]
    assert report.loaded == ()
    assert any(w.message.fields["code"] == "plugin_provider_load_failed" for w in caught)


def test_noncallable_target_refuses_typed(on_path: Path) -> None:
    """An entry point resolving to a non-callable refuses (strict chains it)."""

    with pytest.raises(ConfigurationError) as excinfo:
        activate("eco-rt-notcall")
    assert excinfo.value.fields["code"] == "plugin_strict_load_failed"
    cause = excinfo.value.__cause__
    assert isinstance(cause, ConfigurationError)
    assert cause.fields["code"] == "plugin_activation_invalid"


def test_wrong_export_row_shape_refuses(on_path: Path) -> None:
    """A factory returning the wrong shape for its door refuses typed."""

    with pytest.raises(ConfigurationError) as excinfo:
        activate("eco-rt-badret")
    cause = excinfo.value.__cause__
    assert isinstance(cause, ConfigurationError)
    assert cause.fields["code"] == "plugin_activation_invalid"


def test_wrong_backend_result_refuses(on_path: Path) -> None:
    """The backend door rejects a non-BackendSpec factory result."""

    with pytest.raises(ConfigurationError) as excinfo:
        activate("eco-rt-backend")
    cause = excinfo.value.__cause__
    assert isinstance(cause, ConfigurationError)
    assert cause.fields["code"] == "plugin_activation_invalid"


def test_appliance_door_refuses_before_import(on_path: Path) -> None:
    """The unlanded appliance door refuses BEFORE importing the provider."""

    with pytest.raises(ConfigurationError) as excinfo:
        activate("eco-rt-appl")
    cause = excinfo.value.__cause__
    assert isinstance(cause, ConfigurationError)
    assert cause.fields["code"] == "plugin_group_door_unavailable"
    assert "fakeprov_eco_rt_appl" not in sys.modules


def test_recipes_group_dispatches_to_the_recipes_door(
    on_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """torchlens.recipes activation routes through the G2 recipes door."""

    import torchlens.semantic.recipes as recipes_module

    calls: list[list[str]] = []
    monkeypatch.setattr(
        recipes_module, "activate_entrypoint_recipes", lambda names: calls.append(list(names))
    )
    report = activate("eco-rt-recipes")
    assert calls == [["fakerecipe"]]
    assert [candidate.name for candidate in report.loaded] == ["fakerecipe"]
    assert "fakeprov_eco_rt" not in sys.modules  # the door owns the load


# ---------------------------------------------------------------------------
# Kill switch and configured activation (declaration digest).
# ---------------------------------------------------------------------------


def test_env_kill_switch_beats_everything(on_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TORCHLENS_PLUGINS=none refuses activation but leaves discovery inert-ok."""

    monkeypatch.setenv(plugins.PLUGINS_ENV_VAR, "none")
    assert any(row.distribution == "eco-rt-good" for row in discover())
    with pytest.raises(ConfigurationError) as excinfo:
        activate("eco-rt-good")
    assert excinfo.value.fields["code"] == "plugin_activation_disabled"


def test_activate_configured_reports_declaration_digest(on_path: Path) -> None:
    """The activation notice reports the SHA-256 of the consumed declaration."""

    from torchlens.export import _registry as export_registry

    reports = activate_configured(["eco-rt-good"], source="test-config")
    try:
        expected = hashlib.sha256(b"eco-rt-good").hexdigest()
        assert [report.declaration_digest for report in reports] == [expected]
        assert [candidate.name for report in reports for candidate in report.loaded] == ["demo"]
    finally:
        export_registry.unregister_export_target("eco_rt_demo_target")


def test_activate_configured_consumes_env_declaration(
    on_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """names=None with source='env' consumes the TORCHLENS_PLUGINS list."""

    from torchlens.export import _registry as export_registry

    monkeypatch.setenv(plugins.PLUGINS_ENV_VAR, "eco-rt-good")
    reports = activate_configured()
    try:
        assert len(reports) == 1
        assert reports[0].declaration_digest == hashlib.sha256(b"eco-rt-good").hexdigest()
    finally:
        export_registry.unregister_export_target("eco_rt_demo_target")


def test_activate_configured_none_declaration_refuses_with_digest(
    on_path: Path,
) -> None:
    """A literal 'none' declaration refuses typed and still reports its digest."""

    with pytest.raises(ConfigurationError) as excinfo:
        activate_configured(["none"], source="test-config")
    assert excinfo.value.fields["code"] == "plugin_activation_disabled"
    assert excinfo.value.fields["declaration_digest"] == hashlib.sha256(b"none").hexdigest()


def test_cloned_repo_declaration_activates_nothing(on_path: Path) -> None:
    """Composition row 9: installed declarations alone never import code.

    The dist-info rows sit on sys.path for this whole module and NOTHING
    imports until the explicit call; this test pins the negative half.
    """

    discover()
    plugin_status()
    for module in _PROVIDER_MODULES:
        assert module not in sys.modules, module


def test_approved_groups_are_exactly_the_architecture_five() -> None:
    """The group vocabulary is closed at the five architecture-approved rows."""

    assert ENTRY_POINT_GROUPS == (
        "torchlens.backends",
        "torchlens.transforms",
        "torchlens.export_targets",
        "torchlens.appliances",
        "torchlens.recipes",
    )
