"""Receptive-field edge neutrality on a parent with no single shape.

A preview backend's multi-output equation (a JAX ``split``) materializes as one op
whose ``shape`` is ``None``. The geometry-neutral edge test must answer for such a
parent (not neutral) instead of raising ``TypeError`` from ``tuple(None)``, which
aborted every metadata-invariant pass over a trace that held one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from torchlens.receptive_field._engine import _edge_is_geometry_neutral


def _edge(backend: str, parent_shape: object, edge_use: str = "data") -> tuple[object, ...]:
    """Return ``(op, parent, records_by_parent_label)`` for one recorded edge."""

    parent = SimpleNamespace(
        label="split_1_1", layer_label="split_1_1", _layer_label_raw="split_1_1", shape=parent_shape
    )
    op = SimpleNamespace(
        func_name="reshape", source_trace=SimpleNamespace(_provenance_backend=backend)
    )
    record = SimpleNamespace(edge_use=edge_use, arg_kind="positional", arg_path=(0,))
    return op, parent, {"split_1_1": (record,)}


@pytest.mark.parametrize("backend", ["jax", "tf", "paddle", "torch"])
def test_shapeless_parent_edge_is_not_neutral(backend: str) -> None:
    """A data edge from a parent with ``shape=None`` is data-bearing on every backend."""

    op, parent, records = _edge(backend, None)

    assert _edge_is_geometry_neutral(op, parent, records) is False


def test_control_edge_from_shapeless_parent_stays_neutral() -> None:
    """A control-only edge is neutral before any shape is read."""

    op, parent, records = _edge("jax", None, edge_use="control")

    assert _edge_is_geometry_neutral(op, parent, records) is True


def test_preview_scalar_parent_edge_is_not_neutral() -> None:
    """Preview edges carry no schema facts, so even a scalar parent is not neutral."""

    op, parent, records = _edge("jax", ())

    assert _edge_is_geometry_neutral(op, parent, records) is False
