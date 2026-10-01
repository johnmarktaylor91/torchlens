"""Counterfactual-harvest schema (leverage B17 / D-13): fields land NOW.

The manifest/resume schema for intervened dataset extraction lands at launch
unconditionally — adding a resume-compared policy field to a RELEASED
artifact is exactly the migration this schema exists to prevent. Execution
ships per ADDRESS CLASS only when that class's receipt lane, save path, and
the real 64-image resnet18 workload are green; until an owner certifies a
class, requesting execution refuses typed and finer addresses refuse typed —
never silently coarsen.

Everything here is DOCUMENTED-UNSTABLE pending naming-session ratification.
The manifest block key is ``tl_counterfactual_harvest_v1``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .._errors import InvalidArgumentError

__all__ = [
    "ADDRESS_CLASSES",
    "EXECUTABLE_ADDRESS_CLASSES",
    "HARVEST_BLOCK_KEY",
    "HarvestSpec",
    "harvest_block",
    "require_executable",
    "validate_harvest_block",
]

HARVEST_BLOCK_KEY = "tl_counterfactual_harvest_v1"

#: Closed intervention-policy vocabulary. ``"none"`` is EXPLICIT: a clean
#: harvest declares it, so a reader can distinguish "no intervention" from
#: "written before the schema existed".
_POLICIES = ("none", "intervened")

#: Closed engine vocabulary (mirrors the sweep engine lanes).
_ENGINES = ("live_hook", "replay")

#: Closed verification vocabulary. The cone oracle is the universal one:
#: movers are a subset of the cone; equality only ever as a per-fixture
#: declared expectation (leverage D-12).
_VERIFICATIONS = ("none", "movers_subset_of_cone")

#: Closed address-class vocabulary, coarsest first (memo D-13 gate order).
ADDRESS_CLASSES = ("module", "type", "save_predicate", "label", "op", "selection")

#: Address classes whose EXECUTION gates an owner has certified (receipt
#: lane + save path + the real 64-image resnet18 workload green). Flipping a
#: class in is the execution owner's one-line change, made only beside the
#: measured gate evidence — the schema itself never adjudicates a gate.
EXECUTABLE_ADDRESS_CLASSES: frozenset[str] = frozenset()


@dataclass(frozen=True)
class HarvestSpec:
    """One harvest's declared intervention identity (persisted, resume-compared).

    ``policy="none"`` requires every other field at its clean default;
    ``policy="intervened"`` requires the engine, address class, target
    resolution (the resolved structural site keys), and receipt digest —
    a counterfactual harvest without fire evidence is not a record.
    """

    policy: str = "none"
    engine: str | None = None
    address_class: str | None = None
    target_site_keys: tuple[str, ...] = ()
    receipt_digest: str | None = None
    verification: str = "none"

    def __post_init__(self) -> None:
        """Validate the closed vocabularies and the policy coherence matrix."""

        if self.policy not in _POLICIES:
            raise InvalidArgumentError(
                f"harvest policy {self.policy!r} is outside the closed "
                f"vocabulary {list(_POLICIES)}",
                code="harvest_schema_invalid",
                remedy="declare policy='none' or policy='intervened'",
            )
        if self.verification not in _VERIFICATIONS:
            raise InvalidArgumentError(
                f"harvest verification {self.verification!r} is outside the "
                f"closed vocabulary {list(_VERIFICATIONS)}",
                code="harvest_schema_invalid",
                remedy="declare a documented verification",
            )
        if self.policy == "none":
            if (
                self.engine is not None
                or self.address_class is not None
                or self.target_site_keys
                or self.receipt_digest is not None
            ):
                raise InvalidArgumentError(
                    "policy='none' declares a CLEAN harvest; engine, address "
                    "class, targets, and receipt must be absent (a clean "
                    "record never carries intervention fields).",
                    code="harvest_schema_invalid",
                    remedy="drop the intervention fields, or declare policy='intervened'",
                )
            return
        if self.engine not in _ENGINES:
            raise InvalidArgumentError(
                f"harvest engine {self.engine!r} is outside the closed vocabulary {list(_ENGINES)}",
                code="harvest_schema_invalid",
                remedy="declare engine='live_hook' or engine='replay'",
            )
        if self.address_class not in ADDRESS_CLASSES:
            raise InvalidArgumentError(
                f"harvest address class {self.address_class!r} is outside the "
                f"closed vocabulary {list(ADDRESS_CLASSES)}",
                code="harvest_schema_invalid",
                remedy="declare a documented address class",
            )
        if not self.target_site_keys or not all(
            isinstance(key, str) and key for key in self.target_site_keys
        ):
            raise InvalidArgumentError(
                "an intervened harvest must resolve its targets to structural "
                "site keys before writing a record (labels renumber; site "
                "keys are the persisted identity).",
                code="harvest_schema_invalid",
                remedy="resolve targets to op.site_key values",
            )
        if not self.receipt_digest:
            raise InvalidArgumentError(
                "an intervened harvest must carry its fire receipt digest — "
                "fire evidence IS the receipt (leverage D-6), and a "
                "counterfactual record without one is not a record.",
                code="harvest_schema_invalid",
                remedy="record the intervention receipt digest",
            )


def require_executable(spec: HarvestSpec) -> None:
    """Refuse execution for any address class whose gates are uncertified.

    Raises
    ------
    InvalidArgumentError
        ``harvest_address_class_ungated`` naming the class and the gate
        evidence its owner must certify. Finer addresses refuse; nothing is
        ever silently coarsened to a gated class.
    """

    if spec.policy == "none":
        return
    if spec.address_class not in EXECUTABLE_ADDRESS_CLASSES:
        raise InvalidArgumentError(
            f"counterfactual harvest execution for address class "
            f"{spec.address_class!r} is GATED: its receipt lane, save path, "
            "and the real 64-image resnet18 workload must be certified green "
            "by the execution owner first (leverage D-13). The schema and "
            "manifest fields are live; execution per class flips in beside "
            "its measured gate evidence. Requests are never silently "
            "coarsened to another class.",
            code="harvest_address_class_ungated",
            remedy="run the clean (policy='none') harvest, or wait for the class gate",
        )


def harvest_block(spec: HarvestSpec) -> dict[str, Any]:
    """Serialize one spec as the manifest block (string-only payload)."""

    payload = asdict(spec)
    payload["target_site_keys"] = list(spec.target_site_keys)
    return {HARVEST_BLOCK_KEY: payload}


def validate_harvest_block(block: Any) -> HarvestSpec:
    """Parse and validate one persisted manifest block, fail-closed.

    Raises
    ------
    InvalidArgumentError
        ``harvest_schema_invalid`` on any unknown field, missing field, or
        vocabulary violation — a foreign or torn block never half-loads.
    """

    if not isinstance(block, dict) or set(block) != {
        "policy",
        "engine",
        "address_class",
        "target_site_keys",
        "receipt_digest",
        "verification",
    }:
        raise InvalidArgumentError(
            "malformed tl_counterfactual_harvest_v1 block: expected exactly "
            "the six declared fields",
            code="harvest_schema_invalid",
            remedy="write the block with harvest_block(spec)",
        )
    keys = block["target_site_keys"]
    if not isinstance(keys, (list, tuple)):
        raise InvalidArgumentError(
            "tl_counterfactual_harvest_v1 target_site_keys must be a list",
            code="harvest_schema_invalid",
            remedy="write the block with harvest_block(spec)",
        )
    return HarvestSpec(
        policy=block["policy"],
        engine=block["engine"],
        address_class=block["address_class"],
        target_site_keys=tuple(keys),
        receipt_digest=block["receipt_digest"],
        verification=block["verification"],
    )
