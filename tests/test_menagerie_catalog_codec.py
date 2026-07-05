"""Tests for menagerie catalog row worker JSON codec."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, fields
from typing import Any

from menagerie.catalog import CatalogRow, catalog_row_from_payload


class FieldTrackingPayload(Mapping[str, Any]):
    """Mapping that records every membership check."""

    def __init__(self, values: dict[str, Any]) -> None:
        """Store payload values.

        Parameters
        ----------
        values:
            Backing payload values.
        """

        self._values = values
        self.seen_names: set[str] = set()

    def __contains__(self, key: object) -> bool:
        """Record field membership checks.

        Parameters
        ----------
        key:
            Candidate mapping key.

        Returns
        -------
        bool
            Whether the key exists in the payload.
        """

        if isinstance(key, str):
            self.seen_names.add(key)
        return key in self._values

    def __getitem__(self, key: str) -> Any:
        """Return a payload value.

        Parameters
        ----------
        key:
            Payload key.

        Returns
        -------
        Any
            Payload value.
        """

        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate payload keys.

        Returns
        -------
        Iterator[str]
            Payload key iterator.
        """

        return iter(self._values)

    def __len__(self) -> int:
        """Return payload size.

        Returns
        -------
        int
            Number of payload values.
        """

        return len(self._values)


def _row() -> CatalogRow:
    """Build a catalog row fixture.

    Returns
    -------
    CatalogRow
        Catalog row with non-default worker-boundary fields.
    """

    return CatalogRow(
        model_id=1,
        display_index=1,
        stable_id="m-codec",
        name="CodecNet",
        variant="v1",
        family="unit",
        family_normalized="unit",
        domain="unit",
        zoo="unit-zoo",
        constructor_call="torch.nn.Identity()",
        input_shape="(1,)",
        input_dtype="float32",
        era="2026",
        verified=True,
        notes="codec fixture",
        source="catalog",
        recipe_revision_sha256="recipe-codec",
        input_is_real=False,
        verification_expectation="deferred",
        quarantine=True,
    )


def test_catalog_row_worker_json_codec_round_trips_all_current_fields() -> None:
    """Worker row JSON codec preserves fields that previously dropped."""

    encoded = json.dumps(asdict(_row()))
    decoded = catalog_row_from_payload(json.loads(encoded))

    assert decoded.quarantine is True
    assert decoded.input_is_real is False
    assert decoded.verification_expectation == "deferred"
    assert decoded == _row()


def test_catalog_row_worker_json_codec_field_census_is_complete() -> None:
    """Every ``CatalogRow`` dataclass field is visited by the codec."""

    field_names = {field.name for field in fields(CatalogRow)}
    payload = FieldTrackingPayload(asdict(_row()))

    decoded = catalog_row_from_payload(payload)

    assert set(asdict(decoded)) == field_names
    assert payload.seen_names == field_names
