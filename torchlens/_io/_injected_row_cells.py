"""Cell checks for injected-op rows: the identity grammar and the codec envelope.

These checks read only decoded table cells, so the format 3 validator and the
load split (:mod:`torchlens._io.injection_codec`) share them without importing
the codec's capture and replay machinery. Every check is fail-closed: a forged
row refuses with a typed code, never degrades.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, NoReturn

from .format_contract import ArtifactError, BlobRef

_ENVELOPE_KEY = "injection_codec_v1"
"""The op-row annotations key carrying the codec extras."""

_ENVELOPE_FIELDS = frozenset({"version", "host_label", "callable", "fire_device", "args", "kwargs"})
"""Closed key set of one codec envelope."""

_LITERAL_TYPES = (bool, int, float, str, bytes)
"""Closed literal arg-snapshot types (plus ``None``)."""

_CALLABLE_KEY_FIELDS = frozenset(
    {"namespace", "qualname", "dispatch_kind", "version", "import_path"}
)
"""Closed field set of one persisted callable registry key."""


def _is_int(value: Any) -> bool:
    """Whether ``value`` is an int and not a bool."""

    return isinstance(value, int) and not isinstance(value, bool)


def _is_sequence(value: Any) -> bool:
    """Whether ``value`` is a sequence other than a string or bytes."""

    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def refuse_stamped_model_rows(rows: Iterable[tuple[int, Mapping[str, Any]]]) -> None:
    """Refuse model (main-key) op rows, given as ``(op table index, row)``, that carry
    ``injection_provenance``.

    A stamped model op is a forged injected row. Each refuses with the codes the load
    split uses, in its order: the identity grammar, then the codec envelope. A row that
    passes both is still refused by the caller's coverage invariant.
    """

    for index, row in rows:
        label = str(row.get("layer_label") or row.get("label") or f"op[{index}]")
        check_injected_row_cells(row["injection_provenance"], row.get("annotations"), label)


_INJECTION_PROVENANCE_FIELDS = frozenset(
    {
        "host_site_key",
        "spec_rule_id",
        "host_pass",
        "firing_index",
        "nesting_path",
        "local_op_ordinal",
        "output_slot",
    }
)


def _injection_invalid(message: str, reason: str) -> NoReturn:
    """Raise a typed injected-op identity load refusal.

    Parameters
    ----------
    message:
        Structural violation detail.
    reason:
        Machine-readable failure class.

    Raises
    ------
    ArtifactError
        Always.
    """

    field = "Op.injection_provenance"
    remedy = (
        "re-capture with log_injections and re-save with one current "
        "TorchLens version; do not hand-edit injected-op identity records"
    )
    raise ArtifactError(
        f"{field} is invalid in the loaded artifact: {message}. Remedy: {remedy}.",
        code="artifact_injection_provenance_invalid",
        field=field,
        reason=reason,
        remedy=remedy,
    )


def check_injection_provenance_row(record: Any, label: str) -> None:
    """Validate ONE injected-op identity record fail-closed (C07 grammar).

    Shared by the layer_list surface validator in
    :mod:`torchlens._io.forgery_validation` (defense in depth --
    the load split removes genuine injected rows before it runs) and by
    the injected-op codec's split validator
    (:mod:`torchlens._io.injection_codec`), so a forged record refuses with
    the same typed code from either door.
    """

    if not isinstance(record, Mapping) or set(record) != _INJECTION_PROVENANCE_FIELDS:
        _injection_invalid(
            f"op {label!r} carries a malformed injected-op identity record: {record!r}",
            "record_schema",
        )
    for field_name in ("host_site_key", "spec_rule_id"):
        if not isinstance(record[field_name], str) or not record[field_name]:
            _injection_invalid(
                f"op {label!r} {field_name} must be a non-empty string",
                "record_strings",
            )
    if not _is_int(record["host_pass"]) or record["host_pass"] < 1:
        _injection_invalid(
            f"op {label!r} host_pass must be a positive int (1-based pass index)",
            "record_host_pass",
        )
    for field_name in ("firing_index", "local_op_ordinal", "output_slot"):
        if not _is_int(record[field_name]) or record[field_name] < 0:
            _injection_invalid(
                f"op {label!r} {field_name} must be a non-negative int",
                "record_ordinals",
            )
    nesting = record["nesting_path"]
    if not _is_sequence(nesting) or any(not _is_int(step) or step < 0 for step in nesting):
        _injection_invalid(
            f"op {label!r} nesting_path must be a sequence of non-negative ints",
            "record_nesting",
        )


def check_injected_row_cells(provenance: Any, annotations: Any, label: str) -> None:
    """Check one decoded injected-row candidate before any record is built.

    Runs the load split's row checks in their order on table cells: the
    provenance grammar, then the ``injection_codec_v1`` envelope.

    Parameters
    ----------
    provenance, annotations:
        The row's ``injection_provenance`` and ``annotations`` cells.
    label:
        Row label used in the refusal message.
    """

    check_injection_provenance_row(provenance, label)
    _validate_envelope(SimpleNamespace(annotations=annotations), label)


def _refuse_codec(message: str, reason: str) -> NoReturn:
    """Raise one typed codec-envelope load refusal."""

    raise ArtifactError(
        f"{message} -- the artifact's injected-op codec envelope is forged or hand-edited",
        code="artifact_injection_codec_invalid",
        field="Op.annotations.injection_codec_v1",
        reason=reason,
        remedy=(
            "re-capture with log_injections and re-save with one current "
            "TorchLens version; do not hand-edit injected-op rows"
        ),
    )


def _validate_envelope(row: Any, label: str) -> None:
    """Validate one row's codec envelope shape fail-closed."""

    annotations = getattr(row, "annotations", None)
    if not isinstance(annotations, dict) or _ENVELOPE_KEY not in annotations:
        _refuse_codec(
            f"injected-op row {label!r} carries no {_ENVELOPE_KEY} envelope",
            "codec_missing",
        )
    envelope = annotations[_ENVELOPE_KEY]
    if not isinstance(envelope, dict) or set(envelope) != _ENVELOPE_FIELDS:
        _refuse_codec(
            f"injected-op row {label!r} envelope keys are not the closed codec set",
            "codec_schema",
        )
    if envelope["version"] != 1:
        _refuse_codec(
            f"injected-op row {label!r} envelope version {envelope['version']!r} is unknown",
            "codec_version",
        )
    host_label = envelope["host_label"]
    if not isinstance(host_label, str) or not host_label:
        _refuse_codec(f"injected-op row {label!r} host_label is not a string", "codec_host")
    if not label.startswith(f"{host_label}/inj_"):
        _refuse_codec(
            f"injected-op row label {label!r} does not extend its host label "
            f"{host_label!r} with the /inj_ suffix",
            "codec_label",
        )
    _validate_envelope_callable(envelope["callable"], label)
    fire_device = envelope["fire_device"]
    if fire_device is not None and not isinstance(fire_device, str):
        _refuse_codec(f"injected-op row {label!r} fire_device is not a string", "codec_device")
    _validate_envelope_args(envelope, label)


def _validate_envelope_callable(callable_ref: Any, label: str) -> None:
    """Validate one persisted callable registry key fail-closed."""

    if callable_ref is None:
        return
    if not isinstance(callable_ref, dict) or set(callable_ref) != _CALLABLE_KEY_FIELDS:
        _refuse_codec(
            f"injected-op row {label!r} callable reference keys are not the "
            "closed registry-key set",
            "codec_callable",
        )
    for key_field in ("namespace", "qualname", "dispatch_kind"):
        if not isinstance(callable_ref[key_field], str) or not callable_ref[key_field]:
            _refuse_codec(
                f"injected-op row {label!r} callable {key_field} is not a non-empty string",
                "codec_callable",
            )
    if not isinstance(callable_ref["version"], int) or isinstance(callable_ref["version"], bool):
        _refuse_codec(
            f"injected-op row {label!r} callable version is not an int",
            "codec_callable",
        )
    if callable_ref["import_path"] is not None and not isinstance(callable_ref["import_path"], str):
        _refuse_codec(
            f"injected-op row {label!r} callable import_path is not a string",
            "codec_callable",
        )


def _validate_envelope_args(envelope: dict[str, Any], label: str) -> None:
    """Validate the envelope's encoded args/kwargs payloads fail-closed."""

    for slot_name in ("args", "kwargs"):
        payload = envelope[slot_name]
        if payload is None:
            continue
        if not isinstance(payload, list):
            _refuse_codec(
                f"injected-op row {label!r} envelope {slot_name} is not a list",
                "codec_args",
            )
        for item in payload:
            if slot_name == "kwargs":
                _validate_encoded_pair(item, label, "envelope kwargs entry")
            else:
                _validate_encoded_value(item, label)


def _validate_encoded_pair(item: Any, label: str, what: str) -> None:
    """Validate one ``[str_key, encoded_value]`` pair fail-closed."""

    if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
        _refuse_codec(f"injected-op row {label!r} {what} is malformed", "codec_args")
    _validate_encoded_value(item[1], label)


def _validate_encoded_value(encoded: Any, label: str) -> None:
    """Validate one encoded arg value against the closed grammar."""

    if not isinstance(encoded, dict) or set(encoded) != {"kind", "value"}:
        _refuse_codec(
            f"injected-op row {label!r} carries a malformed encoded replay arg",
            "codec_args",
        )
    kind = encoded["kind"]
    value = encoded["value"]
    if kind in ("tensor", "literal", "dtype", "device"):
        _validate_encoded_leaf(kind, value, label)
        return
    if kind in ("list", "tuple", "dict"):
        if not isinstance(value, list):
            _refuse_codec(
                f"injected-op row {label!r} {kind} arg is not an item list",
                "codec_args",
            )
        for item in value:
            if kind == "dict":
                _validate_encoded_pair(item, label, "dict arg entry")
            else:
                _validate_encoded_value(item, label)
        return
    _refuse_codec(
        f"injected-op row {label!r} encoded arg kind {kind!r} is not in the closed grammar",
        "codec_args",
    )


def _validate_encoded_leaf(kind: str, value: Any, label: str) -> None:
    """Validate one leaf-kind encoded arg value fail-closed."""

    if kind == "tensor" and not isinstance(value, BlobRef):
        _refuse_codec(
            f"injected-op row {label!r} tensor arg does not reference a payload blob",
            "codec_args",
        )
    if kind == "literal" and value is not None and not isinstance(value, _LITERAL_TYPES):
        _refuse_codec(
            f"injected-op row {label!r} literal arg has non-literal type {type(value).__name__}",
            "codec_args",
        )
    if kind in ("dtype", "device") and not isinstance(value, str):
        _refuse_codec(
            f"injected-op row {label!r} {kind} arg is not a string",
            "codec_args",
        )
