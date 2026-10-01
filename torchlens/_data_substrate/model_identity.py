"""Model identity for extraction artifacts (extract memo D6 + item 6).

``model_identity: "measured" (DEFAULT) | "asserted" | "none"``:

* ``measured`` — a CRYPTOGRAPHIC digest (blake2b default, sha256 selectable)
  over the COMPLETE ORDERED MODEL STATE: every parameter AND persistent
  buffer, including integer, bool, and 0-dim entries, each leaf folding
  name+shape+dtype+layout+bytes, folded in canonical ``state_dict()`` order
  as a THREADED MERKLE FOLD (deterministic across thread counts). The
  algorithm id + version ride the manifest so any consumer in any language
  can recompute it forever.
* ``asserted`` — an explicit caller claim (immutable checkpoint + revision)
  for cases where measurement is IMPOSSIBLE (meta-device models,
  disk-offloaded shards). Absent an assertion the record is
  ``"unavailable"`` and cross-process resume refuses typed.
* ``none`` — explicit recorded opt-out; resume refuses to compare.

ALWAYS AND INDEPENDENTLY: when ``config._commit_hash`` is readable (a private
attribute — read defensively) it is recorded as ``hub_identity`` with its
source. It is PROVENANCE, never the check: absent for torchvision and
``save_pretrained`` reloads, and unchanged across in-place weight edits —
exactly wrong for LoRA-merge / pruning / lesion workflows. ``"sampled"`` is
not offered. Resume compares at the recorded level and refuses cross-level
comparisons.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from .._errors import InvalidArgumentError
from .digests import merkle_digest

__tl_layer__ = "L3"

__all__ = ["MODEL_IDENTITY_LEVELS", "compute_model_identity"]

#: Closed identity-level vocabulary (extract D6; "sampled" is retired).
MODEL_IDENTITY_LEVELS: tuple[str, ...] = ("measured", "asserted", "none")

#: Pinned id of the state fold: complete ordered ``state_dict()`` (parameters
#: plus persistent buffers, int/bool/0-dim included) through the threaded
#: Merkle digest.
MODEL_STATE_DIGEST_ID = "tl_model_state_merkle"

#: Pinned version of the state-fold encoding.
MODEL_STATE_DIGEST_VERSION = 1


def _read_hub_identity(model: nn.Module) -> tuple[str | None, str | None]:
    """Defensively read HF hub provenance off the model config.

    Parameters
    ----------
    model:
        The model whose ``config._commit_hash`` may exist.

    Returns
    -------
    tuple[str | None, str | None]
        ``(hub_identity, hub_identity_source)``; ``(None, None)`` when
        unreadable. Provenance only — never the identity check.
    """

    try:
        commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    except Exception:  # noqa: BLE001 - hostile config property must not break the provenance read
        return None, None
    if isinstance(commit, str) and commit:
        return commit, "config._commit_hash"
    return None, None


def _state_is_measurable(model: nn.Module) -> tuple[bool, str | None]:
    """Report whether the complete model state is byte-readable on this host.

    Parameters
    ----------
    model:
        The model to inspect.

    Returns
    -------
    tuple[bool, str | None]
        ``(measurable, reason)``; the reason names the first meta-device
        entry when measurement is impossible.
    """

    for name, tensor in model.state_dict().items():
        if tensor.is_meta:
            return False, f"state entry {name!r} lives on the meta device"
    return True, None


def compute_model_identity(
    model: nn.Module,
    *,
    level: str = "measured",
    hash_name: str = "blake2b",
    assertion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the model-identity record for an extraction signature (D6).

    Parameters
    ----------
    model:
        The model being harvested.
    level:
        ``"measured"`` (default) | ``"asserted"`` | ``"none"``.
    hash_name:
        Crypto hash for the measured digest (``"blake2b"`` or ``"sha256"``).
    assertion:
        Caller claim (e.g. ``{"checkpoint": ..., "revision": ...}``) for the
        asserted level.

    Returns
    -------
    dict[str, Any]
        JSON-portable identity record: ``level`` (which degrades to
        ``"unavailable"`` when measurement is impossible and no assertion
        was supplied), digest facts where measured, the assertion where
        asserted, and the independent hub provenance rows.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_model_identity_invalid`` on a level outside the closed
        vocabulary, or ``asserted`` without an assertion.
    """

    if level not in MODEL_IDENTITY_LEVELS:
        raise InvalidArgumentError(
            f"model_identity level {level!r} is not in the closed vocabulary "
            f"{MODEL_IDENTITY_LEVELS} ('sampled' is not offered: a sampled "
            "fingerprint admits exact algebraic collisions).",
            code="extraction_model_identity_invalid",
            remedy=(
                "pass 'measured' (default), 'none', or an explicit assertion "
                "mapping such as {'checkpoint': ..., 'revision': ...}"
            ),
            level=level,
        )
    hub_identity, hub_source = _read_hub_identity(model)
    record: dict[str, Any] = {
        "level": level,
        "hub_identity": hub_identity,
        "hub_identity_source": hub_source,
    }
    if level == "none":
        return record
    if level == "asserted":
        if not assertion:
            raise InvalidArgumentError(
                "model_identity='asserted' needs an explicit assertion "
                "(an immutable checkpoint + revision claim); an assertion-free "
                "'asserted' level would be an identity claim with no content.",
                code="extraction_model_identity_invalid",
                remedy=(
                    "pass the assertion mapping itself, e.g. "
                    "model_identity={'checkpoint': ..., 'revision': ...}"
                ),
                level=level,
            )
        record["assertion"] = dict(assertion)
        return record
    measurable, reason = _state_is_measurable(model)
    if not measurable:
        if assertion:
            record["level"] = "asserted"
            record["assertion"] = dict(assertion)
            record["measurement_unavailable_reason"] = reason
            return record
        record["level"] = "unavailable"
        record["measurement_unavailable_reason"] = reason
        return record
    digest = merkle_digest(iter(model.state_dict().items()), hash_name=hash_name)
    record["digest"] = digest.digest
    record["algorithm_id"] = MODEL_STATE_DIGEST_ID
    record["algorithm_version"] = MODEL_STATE_DIGEST_VERSION
    record["hash"] = digest.hash_name
    record["n_state_entries"] = digest.n_leaves
    return record
