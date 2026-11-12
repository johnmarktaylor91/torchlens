"""No backend carries a maturity label; the graduation evidence stays.

The ``preview``/``stable`` label (``BackendSpec.maturity``) was retired on
2026-11-12 (maintainer ruling). Each backend's shipped scorecard keeps its six
graduation items as evidence of how far the conformance suite has measured it.
These tests pin both halves: the label is gone from every spec, protocol
object and shipped record, every shipped record still embeds the six items,
and torch, the reference, must meet all six.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from tests.conformance._graduation import GraduationItem
from tests.conformance._scorecard import PreviewScorecard, shipped_scorecard
from torchlens.backends._scorecard_rows import shipped_scorecard_path
from torchlens.backends.registry import registered_backend_specs

#: The graduation item ids (SPEC section 5), in report order.
GRADUATION_ITEM_IDS: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6")


def _registered_specs() -> tuple[Any, ...]:
    return tuple(spec for spec in registered_backend_specs() if spec.protocol is not None)


def graduation_problems(backend: str, card: PreviewScorecard | None) -> list[str]:
    """Why ``card`` does not show ``backend`` meeting every graduation item (empty when it does).

    Parameters
    ----------
    backend:
        Registered backend name.
    card:
        The backend's shipped scorecard, or ``None`` when none ships.

    Returns
    -------
    list[str]
        One sentence per unmet obligation.
    """

    if card is None:
        return [f"{backend} ships no scorecard.json"]
    problems: list[str] = []
    if card.backend != backend:
        problems.append(f"scorecard names backend {card.backend!r}, not {backend!r}")
    items = {item.id: item for item in card.graduation}
    for item_id in GRADUATION_ITEM_IDS:
        item = items.get(item_id)
        if item is None:
            problems.append(f"graduation item {item_id} is missing")
        elif item.met is not True:
            problems.append(f"graduation item {item_id} is not met: {item.detail}")
    return problems


def test_no_spec_protocol_or_shipped_record_carries_a_maturity_label() -> None:
    """The retired label is absent from every spec, protocol object and shipped record."""

    specs = _registered_specs()
    assert {"torch", "jax", "tf", "mlx", "paddle", "tinygrad"} <= {spec.name for spec in specs}
    for spec in specs:
        assert not hasattr(spec, "maturity"), spec.name
        assert not hasattr(spec.protocol(), "maturity"), spec.name
        path = shipped_scorecard_path(spec.name)
        if path.is_file():
            assert "maturity" not in json.loads(path.read_text(encoding="utf-8")), spec.name


def test_every_shipped_scorecard_embeds_the_six_graduation_items() -> None:
    """Each backend's shipped record keeps G1 to G6, in order, as evidence."""

    for spec in _registered_specs():
        card = shipped_scorecard(spec.name)
        assert card is not None, f"{spec.name} ships no scorecard.json"
        assert card.backend == spec.name
        assert [item.id for item in card.graduation] == list(GRADUATION_ITEM_IDS), spec.name


@pytest.mark.xfail(
    strict=True,
    reason=(
        "torch graduation waits on the random_forward roster row (reference parity, "
        "validation, and validation after load, AGT-2500); the reason-name cells passed "
        "once the spec reason names landed, and the shipped torch scorecard still fails "
        "the random_forward cells"
    ),
)
def test_the_torch_reference_meets_every_graduation_item() -> None:
    """Torch's shipped scorecard meets G1 to G6."""

    problems = graduation_problems("torch", shipped_scorecard("torch"))
    assert not problems, problems


def _record(**graduation_met: bool) -> PreviewScorecard:
    return PreviewScorecard(
        scorecard_version="planted",
        backend="torch",
        framework_version="planted",
        suite_version="planted",
        torchlens_version="planted",
        cells=(),
        counts={},
        highest_tier="C2",
        graduation=tuple(
            GraduationItem(id=item_id, met=graduation_met.get(item_id, True), detail="planted")
            for item_id in GRADUATION_ITEM_IDS
        ),
    )


@pytest.mark.parametrize("item_id", GRADUATION_ITEM_IDS)
def test_one_unmet_item_is_reported_by_id(item_id: str) -> None:
    """Anti-vacuity: each planted unmet item is caught, by id."""

    problems = graduation_problems("torch", _record(**{item_id: False}))
    assert len(problems) == 1 and item_id in problems[0]


def test_missing_items_missing_records_and_wrong_backend_are_reported() -> None:
    """A record without items, no record at all, or another backend's record is a problem."""

    record = _record()
    record = replace(record, graduation=record.graduation[:-1])
    assert graduation_problems("torch", record) == ["graduation item G6 is missing"]
    assert graduation_problems("torch", None) == ["torch ships no scorecard.json"]
    assert graduation_problems("torch", replace(_record(), backend="jax")) == [
        "scorecard names backend 'jax', not 'torch'"
    ]
    assert graduation_problems("torch", _record()) == []
