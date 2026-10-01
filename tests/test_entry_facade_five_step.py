"""Five-step facade order: the testable spec, pinned (megasprint A10).

Architecture memo 5.4 / neuro memo D15: (1) underscore -> plain
``AttributeError`` immediately; (2) redirect table -> typed teaching
``AttributeError``, no dependency check; (3) refusal table -> typed
``AttributeError``; (4) real active name -> per-name dependency gate naming
the exact package and install command; (5) otherwise plain
``AttributeError``. Every typed error subclasses ``AttributeError`` so
``hasattr`` can never explode, and resolving a facade imports nothing
foreign to answer a negative (pickle-safety).
"""

from __future__ import annotations

import pickle
import sys

import pytest

import torchlens as tl
from torchlens._errors import FacadeTeachingError, MissingDependencyError
from torchlens.utils.facade import DependencyGate, facade_dir, resolve_facade_attr

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# The mechanism, exercised through a synthetic namespace (every step).
# ---------------------------------------------------------------------------


def _resolve(name: str, **overrides: object) -> object:
    """Resolve ``name`` through a fully populated synthetic facade."""

    tables: dict[str, object] = {
        "owner": "torchlens._synthetic",
        "name": name,
        "module_globals": {},
        "lazy_attrs": {"real": ("torchlens.utils.facade", "facade_dir")},
        "redirects": {"moved": "use tl.new_home.moved"},
        "refusals": {"never": "deliberately unsupported; use other_tool"},
        "dependencies": {},
    }
    tables.update(overrides)
    return resolve_facade_attr(**tables)  # type: ignore[arg-type]


def test_step1_underscore_short_circuits_plain() -> None:
    """Underscore names raise PLAIN AttributeError even when tabled."""

    with pytest.raises(AttributeError) as excinfo:
        _resolve(
            "_moved",
            redirects={"_moved": "never consulted"},
            lazy_attrs={"_moved": ("torchlens.utils.facade", "facade_dir")},
        )
    assert type(excinfo.value) is AttributeError


def test_step2_redirect_is_typed_teaching_attributeerror() -> None:
    """Redirect rows raise FacadeTeachingError naming the canonical spelling."""

    with pytest.raises(FacadeTeachingError) as excinfo:
        _resolve("moved")
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert "tl.new_home.moved" in str(excinfo.value)
    assert isinstance(excinfo.value, AttributeError)


def test_step2_redirect_beats_real_name_and_needs_no_dependency() -> None:
    """A redirect row wins over a same-named real row, without any dep check."""

    with pytest.raises(FacadeTeachingError):
        _resolve(
            "moved",
            lazy_attrs={"moved": ("torchlens.utils.facade", "facade_dir")},
            dependencies={"moved": DependencyGate("not_a_real_pkg_xyz", "pip install nothing")},
        )


def test_step3_refusal_is_typed_attributeerror() -> None:
    """Refusal rows raise FacadeTeachingError with the refusal code."""

    with pytest.raises(FacadeTeachingError) as excinfo:
        _resolve("never")
    assert excinfo.value.fields["code"] == "facade_refusal"
    assert isinstance(excinfo.value, AttributeError)


def test_step4_real_name_resolves_and_caches() -> None:
    """Real active names resolve and cache into the owning globals."""

    cache: dict[str, object] = {}
    value = _resolve("real", module_globals=cache)
    assert value is facade_dir
    assert cache["real"] is facade_dir


def test_step4_missing_dependency_names_package_and_install() -> None:
    """The per-name gate names the exact package and install command."""

    with pytest.raises(MissingDependencyError) as excinfo:
        _resolve(
            "real",
            dependencies={
                "real": DependencyGate("not_a_real_pkg_xyz", 'pip install "torchlens[extra]"')
            },
        )
    err = excinfo.value
    assert err.fields["code"] == "facade_dependency_missing"
    assert err.fields["dependency"] == "not_a_real_pkg_xyz"
    assert "not_a_real_pkg_xyz" in str(err)
    assert 'pip install "torchlens[extra]"' in str(err)
    assert isinstance(err, AttributeError)


def test_step4_broken_resolution_import_stays_hasattr_safe() -> None:
    """A real row whose module import fails raises the typed error, chained."""

    with pytest.raises(MissingDependencyError) as excinfo:
        _resolve("real", lazy_attrs={"real": ("torchlens_no_such_module_xyz", None)})
    assert isinstance(excinfo.value, AttributeError)
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_step5_unknown_name_is_plain_attributeerror() -> None:
    """Unknown names raise PLAIN AttributeError."""

    with pytest.raises(AttributeError) as excinfo:
        _resolve("unknown_name")
    assert type(excinfo.value) is AttributeError


def test_typed_errors_pickle_with_fields() -> None:
    """The typed facade errors survive pickling with their structured fields."""

    redirect = FacadeTeachingError("x moved", code="facade_redirect", remedy="use y")
    missing = MissingDependencyError(
        "z needs w", code="facade_dependency_missing", remedy="pip install w", dependency="w"
    )
    assert pickle.loads(pickle.dumps(redirect)).fields["code"] == "facade_redirect"
    assert pickle.loads(pickle.dumps(missing)).fields["dependency"] == "w"


# ---------------------------------------------------------------------------
# The live namespaces: root, appliances, integrations.
# ---------------------------------------------------------------------------


def test_root_reachability_rows_resolve() -> None:
    """tl.bridge / tl.callbacks / tl.neuro / tl.notebook / tl.load_extraction resolve."""

    assert tl.bridge.__name__ == "torchlens.bridge"
    assert tl.callbacks.__name__ == "torchlens.callbacks"
    assert tl.neuro.__name__ == "torchlens.neuro"
    assert tl.notebook.__name__ == "torchlens.notebook"
    assert callable(tl.load_extraction)
    assert "load_extraction" in tl.__all__


def test_root_underscore_rows_are_gone() -> None:
    """No underscore name resolves through the root facade (``_trace`` removed)."""

    assert not any(name.startswith("_") for name in tl._LAZY_ATTRS)
    with pytest.raises(AttributeError):
        tl._trace  # noqa: B018 - attribute access IS the assertion.


def test_hasattr_never_explodes_across_namespaces() -> None:
    """hasattr answers False (never raises) on every facade namespace."""

    for namespace in (tl, tl.neuro, tl.notebook, tl.bridge, tl.callbacks):
        assert hasattr(namespace, "definitely_not_a_real_name_xyz") is False
        assert hasattr(namespace, "_ipython_canary_method_should_not_exist_") is False


def test_appliance_probes_import_nothing_foreign() -> None:
    """Probing appliance namespaces never imports the extras' foreign packages.

    Delta-based: other tests in a shared session may legitimately import the
    foreign packages, so the oracle is that THE PROBES add none of them (the
    absolute cold-process assertion lives in the cold-import matrix).
    """

    foreign = ("rsatoolbox", "brainscore_core", "IPython", "jupyter_client")
    before = {name for name in foreign if name in sys.modules}
    for name in ("anything", "rdms", "datasets", "__wrapped__"):
        assert hasattr(tl.neuro, name) is False
        assert hasattr(tl.notebook, name) is False
    after = {name for name in foreign if name in sys.modules}
    assert after == before, f"appliance probes imported foreign packages: {after - before}"


def test_dir_lists_real_public_names_and_nothing_else() -> None:
    """dir() stops advertising implementation imports on every facade."""

    for namespace in (tl, tl.neuro, tl.notebook, tl.bridge, tl.callbacks):
        listed = dir(namespace)
        assert "importlib" not in listed
        assert "annotations" not in listed
        assert "Any" not in listed
        assert "ModuleType" not in listed
        assert not any(
            name.startswith("_") and not (name.startswith("__") and name.endswith("__"))
            for name in listed
        ), (namespace.__name__, listed)
    assert "trace" in dir(tl)
    assert "lightning" in dir(tl.callbacks)
    assert "captum" in dir(tl.bridge)


def test_bridge_and_callbacks_unknown_names_stay_plain() -> None:
    """Unknown bridge/callback names raise plain AttributeError (step 5)."""

    with pytest.raises(AttributeError) as excinfo:
        tl.bridge.not_a_bridge_module  # noqa: B018
    assert type(excinfo.value) is AttributeError
    with pytest.raises(AttributeError) as excinfo:
        tl.callbacks.not_a_callback  # noqa: B018
    assert type(excinfo.value) is AttributeError


def test_facade_dir_helper_contract() -> None:
    """facade_dir keeps dunders, drops private names, merges name sets."""

    listed = facade_dir(
        {"__name__": "x", "_private": 1, "public": 2, "__all__": []},
        {"lazy_name", "_hidden_lazy"},
    )
    assert "__name__" in listed
    assert "__all__" in listed
    assert "public" in listed
    assert "lazy_name" in listed
    assert "_private" not in listed
    assert "_hidden_lazy" not in listed
