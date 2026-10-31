"""Structure-only -> runnable bridge: entry-dark L7b wave-1 machinery.

The declared late-bind posture (design memo sec 5/8): a structure-only
capture's declared state slots carry GEOMETRY digests at capture time and
receive their VALUES at bind time, where real byte digests are computed so a
late-bound run is attestable against what was ACTUALLY bound. The
closed-vocabulary ``StateSource`` member this posture needs is an S2
AMENDMENT REQUEST (R-L7B-1) authored by the S2 author; this module therefore
ships ENTRY-DARK — pure functions with no public wiring, no new vocabulary,
no trace state — behind :data:`STRUCTURE_ONLY_BRIDGE_ENTRY_OPEN`. The
capability rows ``save_runnable`` / ``live_replay`` /
``runnable_ready_composition`` keep refusing until the amendment lands and
the L7b implementation PR flips them with evidence.

S1 CONSUMER DISCIPLINE: late-bound values validate through the published
seam surface ``validate_nonpersistent_buffer_mapping_for_descriptor``
(``torchlens/_runnable_state.py`` extension point E3); this module never
reads runnable Trace state and never adds a binding/validation layer of its
own.

DIGESTS NEVER SKIPPED (the memo's mechanical meaning, both postures):

- capture-time declared-slot digests are GEOMETRY digests under the meta
  domain tag — a real digest of everything that exists, under a tag that can
  never masquerade as content; genuine failure degrades to the manifest
  sentinel grammar ``unavailable:<ExceptionName>``, never ``None``, never
  absent;
- bind-time digests are real SHA-256 BYTE digests, mandatory: a value with
  no byte content (meta/storageless) refuses typed, and a digest failure
  refuses typed with the cause chained — never a silent skip.

G2 DOMAIN DISCIPLINE: bound values digest under the REAL tensor tag,
declared slots keep the geometry (meta-domain) tag, and the two can never
collide by construction (pinned in tests/test_structure_only_bridge.py).

NAMING: every spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification; the refusal reuses the existing closed
``state_metadata_mismatch`` vocabulary (a dedicated code, if any, is the S2
amendment's to assign — this module mints none).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "STRUCTURE_ONLY_BRIDGE_ENTRY_OPEN",
    "BoundBufferBinding",
    "compute_bound_state_digests",
    "declared_slot_geometry_digest",
    "validate_and_digest_bound_buffers",
]


STRUCTURE_ONLY_BRIDGE_ENTRY_OPEN: Final[bool] = False
"""Entry gate for the structure-only -> runnable bridge.

``False`` until the S2 StateSource amendment (request R-L7B-1) lands; it is
flipped ONLY by the L7b amendment implementation PR, in the same change that
wires the bridge into the S1 coordinator verbs and flips the three L7b
capability rows with evidence. While ``False`` the bridge is test-reachable
pure machinery and the capability-row refusals are the only public behavior.
"""


def declared_slot_geometry_digest(value: Any) -> str:
    """Return the capture-time GEOMETRY digest for one declared state slot.

    The digest is computed under the meta domain tag (``b"tensor\\0meta\\0"``
    framing) by routing a meta twin of ``value`` through the one content
    digest authority (:func:`torchlens.fingerprint.content`) — zero framing
    duplication, byte-identical to the in-tree meta digest, and identical for
    a real tensor and its meta twin because only dtype/shape/layout enter the
    digest. It can never equal a byte digest of the same tensor (G2 domain
    separation).

    Parameters
    ----------
    value:
        Declared slot tensor; real or meta substrate.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 geometry digest, or the manifest
        sentinel ``unavailable:<ExceptionName>`` when the digest genuinely
        cannot be computed — never ``None``, never absent.
    """

    import torch

    from .. import fingerprint as tl_hash

    try:
        twin = value if value.device.type == "meta" else torch.empty_like(value, device="meta")
        return tl_hash.content(twin)
    except Exception as exc:  # noqa: BLE001 — sentinel grammar, never absent
        return f"unavailable:{type(exc).__name__}"


def compute_bound_state_digests(values: Mapping[str, Any]) -> dict[str, str]:
    """Compute the MANDATORY bind-time byte digests for bound state values.

    Every entry receives a real SHA-256 byte digest through the one content
    digest authority (:func:`torchlens.fingerprint.content`), so a late-bound run is
    attestable against what was ACTUALLY bound. There is no skip path: a
    value with no byte content (meta/storageless — digesting its geometry
    would cross the G2 domain boundary and let a valueless slot masquerade as
    bound bytes) refuses typed, and a digest failure refuses typed with the
    cause chained.

    Parameters
    ----------
    values:
        Canonical state names mapped to bound (real-substrate) tensors.

    Returns
    -------
    dict[str, str]
        One lowercase hexadecimal SHA-256 byte digest per entry; the key set
        equals ``values``' key set exactly.

    Raises
    ------
    StateBindingError
        ``state_metadata_mismatch`` when a value carries no byte content or
        its byte digest cannot be computed.
    """

    from .. import fingerprint as tl_hash
    from ..errors.runnable import StateBindingError
    from ..runnable import RunnableErrorCode

    digests: dict[str, str] = {}
    for name, value in values.items():
        device_type = getattr(getattr(value, "device", None), "type", None)
        if device_type == "meta":
            raise StateBindingError(
                f"Bind-time digest refused for state slot {name!r}: the bound "
                "value is a meta tensor and carries no byte content. A "
                "late-bound run is attestable only against real bound bytes; "
                "digesting geometry here would let a valueless slot "
                "masquerade as bound state (G2 domain discipline).",
                code=RunnableErrorCode.STATE_METADATA_MISMATCH.value,
                remedy=(
                    "bind a real-substrate tensor for this slot (materialize "
                    "the value before binding); geometry-only digests belong "
                    "to the declared slot, not the bound value"
                ),
                state_dict_name=name,
            )
        try:
            digests[name] = tl_hash.content(value)
        except Exception as exc:
            raise StateBindingError(
                f"Bind-time digest FAILED for state slot {name!r} "
                f"({type(exc).__name__}: {exc}). Bind digests are mandatory "
                "— a late-bound run without them could never be attested "
                "against what was bound, so the bind refuses rather than "
                "silently skipping the digest.",
                code=RunnableErrorCode.STATE_METADATA_MISMATCH.value,
                remedy=(
                    "bind a dense, byte-readable tensor for this slot; if the "
                    "value is correct, the chained cause names what made its "
                    "bytes unreadable"
                ),
                state_dict_name=name,
            ) from exc
    return digests


@dataclass(frozen=True)
class BoundBufferBinding:
    """Frozen result of validating and digesting late-bound buffer values.

    ``staged`` is the detached, slot-keyed mapping returned by the S1
    validator; ``bind_digests`` are the mandatory bind-time byte digests of
    exactly the staged values; ``declared_digests`` are the geometry digests
    of the same slots (the capture-time posture), kept alongside so the two
    domains are visibly distinct and never crossed.
    """

    staged: Mapping[str, Any]
    bind_digests: Mapping[str, str]
    declared_digests: Mapping[str, str]


def validate_and_digest_bound_buffers(
    descriptor: Any,
    values: Mapping[str, Any],
) -> BoundBufferBinding:
    """Validate late-bound non-persistent buffer values and digest them.

    The one binding path of the declared late-bind posture: values validate
    through the EXISTING S1 surface
    :func:`torchlens._runnable_state.validate_nonpersistent_buffer_mapping_for_descriptor`
    (no new validation layer — the memo's resized delta), then every staged
    value receives its mandatory bind-time byte digest and every slot its
    declared geometry digest.

    Parameters
    ----------
    descriptor:
        Parsed ``SparseRunDescriptor`` supplying the non-persistent buffer
        slot contracts.
    values:
        Registered buffer names mapped to late-bound tensor values.

    Returns
    -------
    BoundBufferBinding
        Staged values plus both digest families; digests are present for
        every staged slot, never skipped.

    Raises
    ------
    StateBindingError
        If names, shapes, dtypes, roles, or aliases violate the slot
        contracts (S1 validator, unchanged), or if any mandatory bind digest
        cannot be computed.
    """

    from .._runnable_state import validate_nonpersistent_buffer_mapping_for_descriptor

    staged = validate_nonpersistent_buffer_mapping_for_descriptor(descriptor, values)
    bind_digests = compute_bound_state_digests(staged)
    declared_digests = {
        name: declared_slot_geometry_digest(value) for name, value in staged.items()
    }
    return BoundBufferBinding(
        staged=staged,
        bind_digests=bind_digests,
        declared_digests=declared_digests,
    )
