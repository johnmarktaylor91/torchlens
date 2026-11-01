"""A backend's ``maturity`` label can never outrun its evidence.

``maturity`` is a recorded maintainer decision (only JMT flips it), stored
once on the backend's protocol object and mirrored by its ``BackendSpec``.
These tests pin the evidence side: a backend registered ``"stable"`` ships a
scorecard whose six graduation items are all met, so the label cannot claim
more than the conformance suite measured. Torch, the reference, is the one
stable backend today and must pass.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources
from typing import Any

import pytest

from torchlens.backends.registry import get_backend_spec, registered_backend_specs

#: The graduation item ids (SPEC section 5), in report order.
GRADUATION_ITEM_IDS: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6")


def _registered_specs() -> tuple[Any, ...]:
    return tuple(spec for spec in registered_backend_specs() if spec.protocol is not None)


def _shipped_scorecard(backend: str) -> Mapping[str, Any] | None:
    """The scorecard record shipped in the package for ``backend``, if any."""

    record_path = resources.files("torchlens.backends").joinpath(backend, "scorecard.json")
    if not record_path.is_file():
        return None
    record: Mapping[str, Any] = json.loads(record_path.read_text(encoding="utf-8"))
    return record


def stable_label_problems(backend: str, record: Mapping[str, Any] | None) -> list[str]:
    """Why a ``stable`` label is not backed by ``record`` (empty when it is).

    Parameters
    ----------
    backend:
        Registered backend name.
    record:
        The backend's shipped scorecard record, or ``None`` when none ships.

    Returns
    -------
    list[str]
        One sentence per unmet obligation.
    """

    if record is None:
        return [f"{backend} is stable but ships no scorecard.json"]
    problems: list[str] = []
    if record.get("backend") != backend:
        problems.append(f"scorecard names backend {record.get('backend')!r}, not {backend!r}")
    if record.get("maturity") != "stable":
        problems.append(f"scorecard records maturity {record.get('maturity')!r}, not 'stable'")
    items = {str(item.get("id")): item for item in record.get("graduation") or ()}
    for item_id in GRADUATION_ITEM_IDS:
        item = items.get(item_id)
        if item is None:
            problems.append(f"graduation item {item_id} is missing")
        elif item.get("met") is not True:
            problems.append(f"graduation item {item_id} is not met: {item.get('detail', '')}")
    return problems


def test_spec_and_protocol_carry_one_maturity() -> None:
    """The spec's label and its protocol object's label are the same value."""

    specs = _registered_specs()
    assert specs, "no backend registers a protocol"
    for spec in specs:
        assert spec.maturity == spec.protocol().maturity, spec.name


def test_torch_is_the_registered_stable_reference() -> None:
    """Torch is registered stable and every preview is registered preview."""

    labels = {spec.name: spec.maturity for spec in _registered_specs()}
    assert labels["torch"] == "stable"
    previews = {name for name in ("tf", "jax", "mlx", "paddle", "tinygrad") if name in labels}
    assert {labels[name] for name in previews} <= {"preview"}


def test_every_stable_backend_has_all_graduation_items_met() -> None:
    """A ``stable`` backend's shipped scorecard meets G1 to G6 (torch included)."""

    stable = [spec.name for spec in _registered_specs() if spec.maturity == "stable"]
    assert "torch" in stable
    problems = {name: stable_label_problems(name, _shipped_scorecard(name)) for name in stable}
    assert not any(problems.values()), problems


def _record(**graduation_met: bool) -> dict[str, Any]:
    return {
        "backend": "torch",
        "maturity": "stable",
        "graduation": [
            {"id": item_id, "met": graduation_met.get(item_id, True), "detail": "planted"}
            for item_id in GRADUATION_ITEM_IDS
        ],
    }


@pytest.mark.parametrize("item_id", GRADUATION_ITEM_IDS)
def test_one_unmet_item_breaks_the_stable_label(item_id: str) -> None:
    """Anti-vacuity: each planted unmet item is caught, by id."""

    problems = stable_label_problems("torch", _record(**{item_id: False}))
    assert len(problems) == 1 and item_id in problems[0]


def test_missing_items_and_missing_records_break_the_stable_label() -> None:
    """A record without items, or no record at all, never backs ``stable``."""

    record = _record()
    record["graduation"] = record["graduation"][:-1]
    assert stable_label_problems("torch", record) == ["graduation item G6 is missing"]
    assert stable_label_problems("torch", None) == ["torch is stable but ships no scorecard.json"]
    preview_record = _record()
    preview_record["maturity"] = "preview"
    assert len(stable_label_problems("torch", preview_record)) == 1


def test_a_well_formed_stable_record_passes() -> None:
    """The checker accepts a record with every item met."""

    assert stable_label_problems("torch", _record()) == []
    assert get_backend_spec("torch").protocol().maturity == "stable"
