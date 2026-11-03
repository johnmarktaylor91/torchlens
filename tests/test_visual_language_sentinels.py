"""The sentinel strip of the visual language deck is fixed: fourteen ids, and every active row draws.

The strip (``scripts/visual_language/sentinels.py``) is the set of renders every visual change
is checked against, before and after. Its ids never change; a row whose subject does not
exist yet is listed as pending with its reason rather than invented. The render test draws
every panel of every active row with the call the row names, through Graphviz.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.visual_language.sentinels import SENTINELS, SKINS, sentinel_ids, sentinel_slides

FOURTEEN = tuple(f"SN{n:02d}" for n in range(1, 15))


def test_strip_has_exactly_the_fourteen_ids() -> None:
    """The strip is the fixed fourteen rows, in order, each id once."""

    assert sentinel_ids() == FOURTEEN


def test_every_row_says_why_and_pending_rows_say_what_is_missing() -> None:
    """Each row carries its reason for being in the set; a pending row or part names its gap."""

    for row in SENTINELS:
        assert row.why and row.models, row.id
        assert row.status in ("active", "pending"), row.id
        if row.status == "pending":
            assert row.reason and not row.panels, row.id
        else:
            assert row.panels, row.id
        assert all(reason for reason in row.pending_parts.values()), row.id


def test_first_picture_row_names_the_facade_it_stands_in_for() -> None:
    """Row 13 draws the one-call facade under today's spelling and says the name is pending."""

    row = {r.id: r for r in SENTINELS}["SN13"]
    assert [p.call for p in row.panels] == ["quickstart_render"]
    assert "pending name: tl.render" in row.note


def test_row_one_is_drawn_once_per_skin_and_the_rest_once() -> None:
    """The deck section has one page per skin for row 1 and one page per other active row."""

    ids = [slide.id for slide in sentinel_slides()]
    assert ids[: len(SKINS)] == [f"sentinel-01-{skin}" for skin in SKINS]
    active = [row for row in SENTINELS if row.status == "active" and row.id != "SN01"]
    assert len(ids) == len(SKINS) + len(active)
    skinned = sentinel_slides()[: len(SKINS)]
    for slide, skin in zip(skinned, SKINS, strict=True):
        assert {p.kwargs["theme"] for p in slide.panels} == {skin}


_ACTIVE = [
    (row.id, slide)
    for slide in sentinel_slides()
    for row in SENTINELS
    if slide.id.startswith(f"sentinel-{row.id[2:]}-")
]


@pytest.mark.heavy
@pytest.mark.parametrize("row_id, slide", _ACTIVE, ids=[s.id for _r, s in _ACTIVE])
def test_every_active_row_renders(row_id: str, slide, tmp_path: Path) -> None:
    """Every panel of an active row draws a non-empty DOT graph and an SVG picture."""

    from scripts.visual_language import capture

    for panel in slide.panels:
        record = capture.render_panel(slide, panel, tmp_path)
        stem = record["stem"]
        dot = (tmp_path / f"{stem}.dot").read_text()
        assert "digraph" in dot or "graph" in dot, (row_id, stem)
        assert (tmp_path / f"{stem}.svg").stat().st_size > 0, (row_id, stem)
        assert (tmp_path / f"{stem}.json").stat().st_size > 0, (row_id, stem)
