"""Check preview roster data without importing frameworks or building models."""

import numpy as np
import pytest

from tests.conformance._rosters import jax, tf

_BACKENDS = {"tf": tf, "jax": jax}


@pytest.mark.parametrize("backend_name", ("tf", "jax"))
def test_roster_group_sizes_and_family_indexes(backend_name: str) -> None:
    """Keep config-built coverage and ecosystem denominators explicit."""
    backend = _BACKENDS[backend_name]
    config = tuple(spec["family"] for spec in backend.ROW_SPECS if spec["group"] == "config_built")
    ecosystem = tuple(spec["family"] for spec in backend.ROW_SPECS if spec["group"] == "ecosystem")
    assert len(config) >= 8
    assert len(ecosystem) >= 2
    assert config == backend.CONFIG_BUILT_FAMILIES
    assert ecosystem == backend.ECOSYSTEM_FAMILIES
    families = [spec["family"] for spec in backend.ROW_SPECS]
    assert len(families) == len(set(families))


@pytest.mark.parametrize("backend_name", ("tf", "jax"))
def test_config_built_op_inventories(backend_name: str) -> None:
    """Require ordered nonempty inventories beginning with an input operation."""
    for spec in _BACKENDS[backend_name].ROW_SPECS:
        if spec["group"] == "config_built":
            ops = spec["expected_op_kinds"]
            assert isinstance(ops, tuple) and ops
            assert ops[0] == "input"
            assert all(isinstance(kind, str) and kind for kind in ops)


@pytest.mark.parametrize("backend_name", ("tf", "jax"))
def test_intermediates_are_finite_float_arrays(backend_name: str) -> None:
    """Require finite NumPy oracles, including every listed repeated occurrence."""
    for spec in _BACKENDS[backend_name].ROW_SPECS:
        assert isinstance(spec["expected_intermediates"], dict)
        for kind, value in spec["expected_intermediates"].items():
            assert isinstance(kind, str)
            values = value if isinstance(value, list) else [value]
            assert values
            for array in values:
                assert isinstance(array, np.ndarray)
                assert array.dtype in (np.dtype("float32"), np.dtype("float64"))
                assert np.isfinite(array).all()


@pytest.mark.parametrize("backend_name", ("tf", "jax"))
def test_roster_preserves_spec_order_and_model_type(backend_name: str) -> None:
    """Construct only roster records, preserving every spec family in order."""
    backend = _BACKENDS[backend_name]
    rows = backend.roster()
    assert isinstance(rows, tuple)
    assert all(isinstance(row, backend.RosterModel) for row in rows)
    assert tuple(row.family for row in rows) == tuple(
        spec["family"] for spec in backend.ROW_SPECS if spec["group"] == "config_built"
    )


@pytest.mark.parametrize("backend_name", ("tf", "jax"))
def test_ecosystem_rows_join_only_behind_the_switch(
    backend_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ecosystem rows are scored only when switched on; parked families never run."""
    from tests.conformance._rosters import ECOSYSTEM_ENV

    backend = _BACKENDS[backend_name]
    monkeypatch.delenv(ECOSYSTEM_ENV, raising=False)
    assert all(row.group == "config_built" for row in backend.roster())
    monkeypatch.setenv(ECOSYSTEM_ENV, "1")
    parked = getattr(backend, "PARKED_ECOSYSTEM_FAMILIES", ())
    expected = tuple(
        spec["family"]
        for spec in backend.ROW_SPECS
        if spec["group"] == "config_built" or spec["family"] not in parked
    )
    assert tuple(row.family for row in backend.roster()) == expected


@pytest.mark.parametrize("backend_name", ("tf", "jax"))
def test_preview_adapter_scores_the_backend_roster_module(
    backend_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preview adapter's default roster is the backend's ``_rosters`` module."""
    from tests.conformance._preview_adapters import PreviewBackendAdapter
    from tests.conformance._rosters import ECOSYSTEM_ENV

    monkeypatch.delenv(ECOSYSTEM_ENV, raising=False)
    adapter_rows = PreviewBackendAdapter(backend_name).default_roster()
    assert tuple(row.family for row in adapter_rows) == tuple(
        row.family for row in _BACKENDS[backend_name].roster()
    )


@pytest.mark.parametrize("backend_name", ("tf", "jax"))
def test_specs_carry_complete_fields_and_boolean_flags(backend_name: str) -> None:
    """Keep extended row metadata available even with the older RosterModel."""
    required = {
        "family",
        "realism",
        "builder",
        "checkpoint_evidence",
        "expected_op_kinds",
        "expected_intermediates",
        "has_constants",
        "random_init",
        "group",
    }
    for spec in _BACKENDS[backend_name].ROW_SPECS:
        assert required <= spec.keys()
        assert isinstance(spec["has_constants"], bool)
        assert isinstance(spec["random_init"], bool)
        assert callable(spec["builder"])


def test_tf_ecosystem_has_checkpoint_evidence() -> None:
    """Keep Keras checkpoint evidence nonempty without downloading it."""
    for spec in tf.ROW_SPECS:
        if spec["group"] == "ecosystem":
            assert spec["checkpoint_evidence"]


def test_jax_ecosystem_has_checkpoint_evidence() -> None:
    """Keep the pinned Flax checkpoint evidence nonempty without downloading it."""
    ecosystem = [spec for spec in jax.ROW_SPECS if spec["group"] == "ecosystem"]
    assert len(ecosystem) == 2
    assert all(spec["checkpoint_evidence"] for spec in ecosystem)
    assert jax.PARKED_ECOSYSTEM_FAMILIES == ()
