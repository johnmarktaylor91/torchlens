"""Trusted-intake variant designation over real crawl roster rows.

The roster carries sibling relationships in its ``family`` column, not in its
``variant`` column: 96% of rows have ``variant: ""`` and the populated minority
holds harvest provenance notes rather than variant designations. The record
schema nevertheless requires ``identity.variant`` to be a mandatory non-empty
string. These tests pin the derivation that reconciles the two without inventing
data, using rows lifted verbatim from ``menagerie/data/crawl_roster.jsonl``.

Identity and dedup questions -- whether the same architecture harvested from
three zoos is one model or three -- are deliberately NOT decided here. They are
deferred to the post-tracing stage, where a TorchLens trace of the real graph
settles sameness decisively.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from menagerie.crawler.intake import (
    IntakeError,
    create_intake_snapshot,
    load_intake_snapshot,
    trusted_identity_fields,
)
from menagerie.crawler.schema import SCHEMA_DIRECTORY

ROSTER_PATH = Path(__file__).resolve().parents[3] / "menagerie" / "data" / "crawl_roster.jsonl"

# Verbatim roster rows. Three multi-member families; ``centernet`` deliberately
# spans three independent zoos.
BEIT_ROWS: tuple[dict[str, Any], ...] = (
    {
        "domain": "vision/segmentation",
        "era": "2021",
        "family": "beit",
        "name": "mmseg_beit",
        "variant": "",
        "zoo": "mmsegmentation-configs",
    },
    {
        "domain": "vision/image",
        "era": "2021",
        "family": "beit",
        "name": "beit_base_patch16_224",
        "variant": "",
        "zoo": "timm",
    },
    {
        "domain": "vision/image",
        "era": "2021",
        "family": "beit",
        "name": "beit_base_patch16_384",
        "variant": "",
        "zoo": "timm",
    },
    {
        "domain": "vision/image",
        "era": "2021",
        "family": "beit",
        "name": "beit_large_patch16_224",
        "variant": "",
        "zoo": "timm",
    },
)
SEW_RESNET_ROWS: tuple[dict[str, Any], ...] = tuple(
    {
        "domain": "spiking/neuromorphic",
        "era": "2021-2026",
        "family": "sew_resnet",
        "name": name,
        "variant": "",
        "zoo": "spikingjelly",
    }
    for name in (
        "spikingjelly_sew_resnet34",
        "spikingjelly_sew_resnet50",
        "spikingjelly_sew_resnet101",
        "spikingjelly_sew_resnet152",
    )
)
CENTERNET_ROWS: tuple[dict[str, Any], ...] = (
    {
        "domain": "vision/detection-tracking",
        "era": "2019",
        "family": "centernet",
        "name": "centernet_dla34",
        "variant": "",
        "zoo": "unregistered-classics-pytorch",
    },
    {
        "domain": "vision/detection-tracking",
        "era": "2019",
        "family": "centernet",
        "name": "centernet_dla34_ctdet",
        "variant": "",
        "zoo": "xingyizhou/CenterNet",
    },
    {
        "domain": "vision/detection-tracking",
        "era": "unknown",
        "family": "centernet",
        "name": "mmdet_centernet_centernet_r18_dcnv2_8xb16_crop512_140e_coco",
        "variant": "",
        "zoo": "open-mmlab/mmdetection",
    },
)
SINGLETON_ROW: dict[str, Any] = {
    "domain": "graph/geometric",
    "era": "2026",
    "family": "grapnet",
    "name": "GrapNet",
    "source_url": "arXiv:2606.18923",
    "variant": "",
    "zoo": "unregistered-classics-pytorch",
}
# The populated 4% of the ``variant`` column is harvest provenance, not a
# designation. This row is verbatim from the roster.
PROVENANCE_NOTE_ROW: dict[str, Any] = {
    "aliases": ["combined_queue:AnimeGAN"],
    "name": "AnimeGAN",
    "source_url": "https://github.com/TachibanaYoshino/AnimeGAN",
    "triage_class": "SOURCE_AVAILABLE",
    "variant": "animegan2-pytorch (community)",
    "zoo": "discovered-pytorch",
}


def _write_jsonl(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    """Write compact fixture JSONL.

    Parameters
    ----------
    path:
        Fixture path.
    rows:
        JSON object rows.
    """

    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _snapshot(tmp_path: Path, rows: tuple[dict[str, Any], ...]) -> Any:
    """Build one intake snapshot from verbatim roster rows.

    Parameters
    ----------
    tmp_path:
        Test-local directory.
    rows:
        Roster rows to snapshot.

    Returns
    -------
    IntakeSnapshot
        Materialized snapshot.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    _write_jsonl(master, rows)
    _write_jsonl(deferred, ())
    return create_intake_snapshot(master, deferred, tmp_path / "snapshots")


@pytest.fixture(scope="module")
def identity_validator() -> Draft202012Validator:
    """Return a validator for the shipped record identity block.

    Returns
    -------
    Draft202012Validator
        Validator bound to ``model-common.schema.json#/$defs/identity``.
    """

    common = json.loads((SCHEMA_DIRECTORY / "model-common.schema.json").read_text(encoding="utf-8"))
    registry: Registry[Any] = Registry[Any]().with_resource(
        common["$id"], Resource.from_contents(common)
    )
    return Draft202012Validator(
        {"$ref": f"{common['$id']}#/$defs/identity"},
        format_checker=FormatChecker(),
        registry=registry,
    )


def _identity_block(item: Any) -> dict[str, Any]:
    """Return the full record identity block for one intake item.

    Parameters
    ----------
    item:
        Trusted intake roster member.

    Returns
    -------
    dict[str, Any]
        Identity block shaped exactly as the driver writes it.
    """

    return {
        "canonical_name": item.name,
        "aliases": [],
        "acronym": None,
        **trusted_identity_fields(item),
        "duplicate_of": None,
        "alias_of": None,
    }


@pytest.mark.parametrize(
    ("label", "rows"),
    [("beit", BEIT_ROWS), ("sew_resnet", SEW_RESNET_ROWS), ("centernet", CENTERNET_ROWS)],
)
def test_multi_member_family_siblings_stay_distinct_and_schema_valid(
    tmp_path: Path,
    identity_validator: Draft202012Validator,
    label: str,
    rows: tuple[dict[str, Any], ...],
) -> None:
    """Every sibling of a real multi-member family gets its own valid identity."""

    snapshot = _snapshot(tmp_path, rows)
    identities = {item.stable_id: _identity_block(item) for item in snapshot.items}

    assert len(identities) == len(rows)
    for identity in identities.values():
        identity_validator.validate(identity)
        assert identity["variant"]

    designations = [identity["variant"] for identity in identities.values()]
    assert len(set(designations)) == len(rows), f"{label} siblings collapsed onto one designation"
    assert set(designations) == {row["name"] for row in rows}


@pytest.mark.parametrize(
    ("label", "rows"),
    [("beit", BEIT_ROWS), ("sew_resnet", SEW_RESNET_ROWS), ("centernet", CENTERNET_ROWS)],
)
def test_multi_member_family_siblings_stay_recognisably_grouped(
    tmp_path: Path, label: str, rows: tuple[dict[str, Any], ...]
) -> None:
    """Siblings keep one groupable family hint so description reuse can key on it."""

    snapshot = _snapshot(tmp_path, rows)

    assert {item.family for item in snapshot.items} == {label}
    # Grouping is a hint, never a merge: each sibling remains its own record and
    # its own family representative until the post-tracing stage decides sameness.
    assert len({item.stable_id for item in snapshot.items}) == len(rows)
    for item in snapshot.items:
        assert item.family_representative_id == item.stable_id
        assert item.declares_family_variant is False


def test_same_family_from_three_zoos_stays_three_records(tmp_path: Path) -> None:
    """Cross-zoo arrivals are intentionally separate records at this stage.

    ``centernet`` reaches the roster from three independent zoos. Nothing here
    merges, aliases, or deduplicates them: whether they are the same model is
    decided after TorchLens tracing, not from names and metadata.
    """

    snapshot = _snapshot(tmp_path, CENTERNET_ROWS)
    identities = [_identity_block(item) for item in snapshot.items]

    assert {item.zoo for item in snapshot.items} == {
        "unregistered-classics-pytorch",
        "xingyizhou/CenterNet",
        "open-mmlab/mmdetection",
    }
    assert len({item.stable_id for item in snapshot.items}) == 3
    assert len({identity["variant"] for identity in identities}) == 3
    assert all(identity["duplicate_of"] is None for identity in identities)
    assert all(identity["alias_of"] is None for identity in identities)


def test_single_member_family_validates(
    tmp_path: Path, identity_validator: Draft202012Validator
) -> None:
    """A sole family member has a well-defined designation and scope."""

    snapshot = _snapshot(tmp_path, (SINGLETON_ROW,))
    item = snapshot.items[0]
    identity = _identity_block(item)

    identity_validator.validate(identity)
    assert identity["variant"] == "GrapNet"
    assert identity["variant_scope"] == "family"
    assert identity["family_representative_id"] == item.stable_id


def test_empty_variant_column_never_reaches_the_record(
    tmp_path: Path, identity_validator: Draft202012Validator
) -> None:
    """The original crash input now yields a valid, non-empty designation."""

    snapshot = _snapshot(tmp_path, BEIT_ROWS)
    for item in snapshot.items:
        assert item.variant == ""
        identity_validator.validate(_identity_block(item))


def test_provenance_note_is_not_promoted_into_a_designation(tmp_path: Path) -> None:
    """A populated provenance note stays in the natural key, out of the record."""

    snapshot = _snapshot(tmp_path, (PROVENANCE_NOTE_ROW,))
    item = snapshot.items[0]

    assert item.natural_key == (
        "AnimeGAN",
        "discovered-pytorch",
        "animegan2-pytorch (community)",
    )
    assert trusted_identity_fields(item)["variant"] == "AnimeGAN"


def test_declared_size_variant_uses_its_trusted_selector_token(tmp_path: Path) -> None:
    """A declared size variant is designated by the token that specializes the recipe."""

    representative = {"name": "resnet", "zoo": "timm", "variant": "", "family": "resnet"}
    snapshot = _snapshot(tmp_path, (representative,))
    representative_id = snapshot.items[0].stable_id

    variant_row = {
        "name": "resnet50",
        "zoo": "timm",
        "variant": "resnet50",
        "family": "resnet",
        "variant_scope": "family",
        "family_representative_id": representative_id,
    }
    bound = _snapshot(tmp_path / "bound", (representative, variant_row))
    by_name = {item.name: item for item in bound.items}

    assert by_name["resnet50"].declares_family_variant is True
    assert trusted_identity_fields(by_name["resnet50"])["variant"] == "resnet50"
    assert by_name["resnet"].declares_family_variant is False
    assert trusted_identity_fields(by_name["resnet"])["variant"] == "resnet"


def test_declared_size_variant_without_a_token_is_rejected(tmp_path: Path) -> None:
    """An unselectable family binding fails intake instead of being defaulted."""

    representative = {"name": "resnet", "zoo": "timm", "variant": "", "family": "resnet"}
    snapshot = _snapshot(tmp_path, (representative,))
    representative_id = snapshot.items[0].stable_id
    broken = {
        "name": "resnet50",
        "zoo": "timm",
        "variant": "",
        "family": "resnet",
        "variant_scope": "family",
        "family_representative_id": representative_id,
    }

    with pytest.raises(IntakeError, match="no trusted variant token"):
        _snapshot(tmp_path / "broken", (representative, broken))


def test_family_hint_survives_snapshot_reload(tmp_path: Path) -> None:
    """The grouping hint is part of the immutable snapshot, not a build-time value."""

    snapshot = _snapshot(tmp_path, BEIT_ROWS)
    loaded = load_intake_snapshot(snapshot.root)

    assert loaded.items == snapshot.items
    assert {item.family for item in loaded.items} == {"beit"}


def test_fixture_rows_are_verbatim_roster_rows() -> None:
    """Every fixture row above is present byte-for-byte in the shipped roster."""

    roster = {
        json.dumps(json.loads(line), sort_keys=True)
        for line in ROSTER_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    fixtures = (
        *BEIT_ROWS,
        *SEW_RESNET_ROWS,
        *CENTERNET_ROWS,
        SINGLETON_ROW,
        PROVENANCE_NOTE_ROW,
    )
    missing = [row["name"] for row in fixtures if json.dumps(row, sort_keys=True) not in roster]
    assert not missing, f"fixture rows are not verbatim roster rows: {missing}"
