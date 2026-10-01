"""Injected-op persistence codec (``log_injections`` stage 2, lane F44).

Persistence rides the C07-declared tlspec v9 entry-dark slot
``Op.injection_provenance`` -- NO schema bump: each recorded
:class:`~torchlens.intervention.injection.InjectedOp` is written into the
artifact as one synthesized op ROW carrying the slot verbatim, appended to
the scrubbed ``layer_list`` AFTER every 1:1 live/scrubbed consumer has run.
Codec extras (the ``module:qualname`` callable reference resolved through
the trusted-callable resolver, fire-time arg snapshots for validation
replay, the fire device) ride the row's ``annotations`` under the
``injection_codec_v1`` envelope; tensor payloads ride ordinary safetensors
blobs.

At load the rows are SPLIT back out of ``layer_list`` inside
``Trace.__setstate__`` BEFORE any persisted-claim validator or accessor can
observe them (the site-key totality validator would otherwise correctly
refuse the partial family), validated fail-closed -- forged provenance
refuses, never degrades -- and finalized after the blob machinery is ready
(the ``rehydrate_trace`` seam) into ordinary injected-op records with
``attestation="unattested"``: loading NEVER attests a claim
(degrade-to-unattested); the replay door is
:func:`torchlens.intervention.injection.attest_injected_ops`.

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

__tl_layer__ = "L3"

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NoReturn

import torch

from .format_contract import BlobRef, TorchLensIOError
from .scrub import BlobSpec

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

__all__ = [
    "append_injected_op_rows",
    "finalize_loaded_injected_ops",
    "split_restored_injected_rows",
]

_ENVELOPE_KEY = "injection_codec_v1"
"""The op-row annotations key carrying the codec extras."""

_ENVELOPE_FIELDS = frozenset({"version", "host_label", "callable", "fire_device", "args", "kwargs"})
"""Closed key set of one codec envelope."""

_LITERAL_TYPES = (bool, int, float, str, bytes)
"""Closed literal arg-snapshot types (plus ``None``)."""


# ---------------------------------------------------------------------------
# Save side
# ---------------------------------------------------------------------------


def append_injected_op_rows(
    trace: Trace,
    scrubbed_state: dict[str, Any],
    blob_specs: list[BlobSpec],
    *,
    include_outs: bool,
    backend_name: str,
) -> None:
    """Append one synthesized scrubbed op row per recorded injected op.

    Runs at the save seam AFTER every consumer that pairs the live and
    scrubbed layer lists 1:1 (visualization policy, fast-copy specs) and
    BEFORE the blob-write loop and metadata pickling, so the minted payload
    specs are written like any other blob.

    Parameters
    ----------
    trace:
        Live trace being saved (its injected records are read, never
        mutated).
    scrubbed_state:
        Scrubbed metadata dict about to be pickled.
    blob_specs:
        Accumulated blob specs; payload specs for injected outs and arg
        snapshots are appended here.
    include_outs:
        Resolved activation-persistence knob. When ``False`` the injected
        rows persist identity-only (no payloads, no replay args), exactly
        like model op outs.
    backend_name:
        Logical backend for the minted payload specs.
    """

    from ..intervention.injection import injected_ops

    records = injected_ops(trace)
    if not records:
        return
    rows = scrubbed_state.get("layer_list")
    if not isinstance(rows, list):
        raise TorchLensIOError(
            "Scrubbed state has no layer_list to carry injected-op rows.",
            code="injection_persist_state_invalid",
        )
    counter = [_max_numeric_blob_id(blob_specs)]
    for record in records:
        host_site_key = record.provenance.host_site_key
        if not isinstance(host_site_key, str) or not host_site_key:
            raise TorchLensIOError(
                f"injected op {record.label!r} has no resolved host site key; the "
                "persisted injected-op identity anchors on the host's structural "
                "position and cannot be written without one",
                code="injection_persist_unanchored",
            )
        rows.append(
            _synthesize_row(
                record,
                blob_specs,
                counter,
                include_outs=include_outs,
                backend_name=backend_name,
            )
        )


def _max_numeric_blob_id(blob_specs: list[BlobSpec]) -> int:
    """Return the highest numeric blob ordinal already minted."""

    highest = 0
    for spec in blob_specs:
        try:
            highest = max(highest, int(spec.blob_id))
        except ValueError:
            continue
    return highest


def _mint_blob(
    value: torch.Tensor,
    label: str,
    blob_specs: list[BlobSpec],
    counter: list[int],
    backend_name: str,
) -> BlobRef:
    """Mint one payload blob spec and return its reference."""

    counter[0] += 1
    blob_id = f"{counter[0]:010d}"
    blob_specs.append(
        BlobSpec(
            blob_id=blob_id,
            value=value.detach(),
            kind="out",
            label=label,
            logical_backend=backend_name,
        )
    )
    return BlobRef(blob_id=blob_id, kind="out")


def _synthesize_row(
    record: Any,
    blob_specs: list[BlobSpec],
    counter: list[int],
    *,
    include_outs: bool,
    backend_name: str,
) -> Any:
    """Build one already-scrubbed op row for one injected record."""

    from ..constants import LAYER_PASS_LOG_FIELD_ORDER
    from ..data_classes.op import Op

    out_value: Any = None
    args_encoded: Any = None
    kwargs_encoded: Any = None
    if include_outs:
        if isinstance(record.out, torch.Tensor):
            out_value = _mint_blob(record.out, record.label, blob_specs, counter, backend_name)
        if record.saved_args is not None:
            args_encoded = [
                _encode_value(value, record.label, blob_specs, counter, backend_name)
                for value in record.saved_args
            ]
            kwargs_encoded = [
                [name, _encode_value(value, record.label, blob_specs, counter, backend_name)]
                for name, value in (record.saved_kwargs or ())
            ]
    envelope = {
        "version": 1,
        "host_label": record.host_label,
        "callable": _encode_registry_key(record.callable_ref),
        "fire_device": record.fire_device,
        "args": args_encoded,
        "kwargs": kwargs_encoded,
    }
    provenance = record.provenance
    fields: dict[str, Any] = dict.fromkeys(LAYER_PASS_LOG_FIELD_ORDER)
    fields.update(
        {
            "layer_label": record.label,
            "func_name": record.func_name,
            "type": record.layer_type,
            "out": out_value,
            "injection_provenance": {
                "host_site_key": provenance.host_site_key,
                "spec_rule_id": provenance.spec_rule_id,
                "host_pass": provenance.host_pass,
                "firing_index": provenance.firing_index,
                "nesting_path": tuple(provenance.nesting_path),
                "local_op_ordinal": provenance.local_op_ordinal,
                "output_slot": provenance.output_slot,
            },
            "annotations": {_ENVELOPE_KEY: envelope},
        }
    )
    return Op(fields)


_CALLABLE_KEY_FIELDS = frozenset(
    {"namespace", "qualname", "dispatch_kind", "version", "import_path"}
)
"""Closed field set of one persisted callable registry key."""


def _encode_registry_key(callable_ref: Any) -> dict[str, Any] | None:
    """Encode one FunctionRegistryKey as its plain persisted field dict."""

    if callable_ref is None:
        return None
    return {
        "namespace": callable_ref.namespace,
        "qualname": callable_ref.qualname,
        "dispatch_kind": callable_ref.dispatch_kind,
        "version": callable_ref.version,
        "import_path": callable_ref.import_path,
    }


def _encode_value(
    value: Any,
    label: str,
    blob_specs: list[BlobSpec],
    counter: list[int],
    backend_name: str,
) -> dict[str, Any]:
    """Encode one snapshotted arg value into the closed envelope grammar."""

    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "value": _mint_blob(value, label, blob_specs, counter, backend_name),
        }
    if value is None or isinstance(value, _LITERAL_TYPES):
        return {"kind": "literal", "value": value}
    if isinstance(value, torch.dtype):
        return {"kind": "dtype", "value": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        return {"kind": "device", "value": str(value)}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "value": [
                _encode_value(item, label, blob_specs, counter, backend_name) for item in value
            ],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "value": [
                [key, _encode_value(item, label, blob_specs, counter, backend_name)]
                for key, item in value.items()
            ],
        }
    # _snapshot_call_values admits nothing else; a foreign object here means
    # the record was tampered with in-session -- refuse rather than launder.
    raise TorchLensIOError(
        f"injected op {label!r} carries an unencodable replay arg of type {type(value).__name__}",
        code="injection_persist_arg_unencodable",
    )


# ---------------------------------------------------------------------------
# Load side: split (Trace.__setstate__ seam, BEFORE persisted-claim validators)
# ---------------------------------------------------------------------------


def split_restored_injected_rows(trace: Any) -> None:
    """Split provenance-carrying rows out of a restored ``layer_list``.

    Runs inside ``Trace.__setstate__`` before
    ``validate_persisted_forgery_surfaces`` and before any accessor rebuild:
    injected rows are outside the model op family by construction, so no
    loaded consumer (site-key totality, selectors, lookup, validation
    oracles) may ever observe them in ``layer_list``. Every structural
    check here is fail-closed: forged provenance refuses, never degrades.
    """

    rows = trace.__dict__.get("layer_list")
    if not isinstance(rows, list):
        return
    injected = [op for op in rows if getattr(op, "injection_provenance", None) is not None]
    if not injected:
        return
    remaining = [op for op in rows if getattr(op, "injection_provenance", None) is None]
    _validate_split_rows(injected, remaining)
    rows[:] = remaining
    from ..intervention.injection import injection_state

    state = injection_state(trace)
    state["pending_loaded_rows"] = injected


def _refuse_codec(message: str, reason: str) -> NoReturn:
    """Raise one typed codec-envelope load refusal."""

    raise TorchLensIOError(
        f"{message} -- the artifact's injected-op codec envelope is forged or hand-edited",
        code="artifact_injection_codec_invalid",
        field="Op.annotations.injection_codec_v1",
        reason=reason,
        remedy=(
            "re-capture with log_injections and re-save with one current "
            "TorchLens version; do not hand-edit injected-op rows"
        ),
    )


def _validate_split_rows(injected: list[Any], remaining: list[Any]) -> None:
    """Validate the split-out injected rows fail-closed (forgery refuses)."""

    remaining_site_keys = {
        key for op in remaining if isinstance((key := getattr(op, "site_key", None)), str)
    }
    injected_labels: set[str] = set()
    durable_keys: set[tuple[Any, ...]] = set()
    for row in injected:
        injected_labels.add(_validate_one_split_row(row, remaining_site_keys, durable_keys))
    for op in remaining:
        for relation in ("parents", "children"):
            for related in tuple(getattr(op, relation, ()) or ()):
                if related in injected_labels:
                    _refuse_codec(
                        f"model op {getattr(op, 'layer_label', '<unknown>')!r} references "
                        f"injected-op row {related!r} in its {relation}",
                        "record_referenced",
                    )


def _validate_one_split_row(
    row: Any,
    remaining_site_keys: set[str],
    durable_keys: set[tuple[Any, ...]],
) -> str:
    """Validate one injected row's identity and anchoring; return its label."""

    from .forgery_validation import check_injection_provenance_row

    label = getattr(row, "layer_label", None)
    if not isinstance(label, str) or not label:
        _refuse_codec("an injected-op row carries no layer_label", "record_label")
    record = row.injection_provenance
    check_injection_provenance_row(record, label)
    durable = (
        record["host_site_key"],
        record["spec_rule_id"],
        record["host_pass"],
        record["firing_index"],
        tuple(record["nesting_path"]),
        record["local_op_ordinal"],
        record["output_slot"],
    )
    if durable in durable_keys:
        _refuse_codec(
            f"injected-op row {label!r} duplicates another row's durable identity key",
            "record_duplicate",
        )
    durable_keys.add(durable)
    if record["host_site_key"] not in remaining_site_keys:
        _refuse_codec(
            f"injected-op row {label!r} anchors to host site key "
            f"{record['host_site_key']!r}, which no retained model op carries",
            "host_missing",
        )
    _validate_row_op_identity(row, label)
    _validate_envelope(row, label)
    return label


def _validate_row_op_identity(row: Any, label: str) -> None:
    """Refuse rows wearing placeholder identity or model dataflow claims.

    An injected row may never wear the intervention_replacement placeholder
    identity: that would steal the metadata-invariant carve-out scoped to
    GENUINE user interventions (the 2026-06-02 incident law).
    """

    for field_name in ("func_name", "type"):
        value = getattr(row, field_name, None)
        if not isinstance(value, str) or not value or "intervention" in value:
            _refuse_codec(
                f"injected-op row {label!r} carries illegal {field_name} {value!r}",
                "record_func",
            )
    if getattr(row, "intervention_replaced", None):
        _refuse_codec(
            f"injected-op row {label!r} claims intervention_replaced",
            "record_func",
        )
    for relation in ("parents", "children"):
        if tuple(getattr(row, relation, ()) or ()):
            _refuse_codec(
                f"injected-op row {label!r} claims model dataflow {relation}; "
                "injected ops are outside the dataflow graph by construction",
                "record_graph_entangled",
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


# ---------------------------------------------------------------------------
# Load side: finalize (rehydrate_trace seam, AFTER the blob machinery is ready)
# ---------------------------------------------------------------------------


def finalize_loaded_injected_ops(
    trace: Any,
    *,
    materialize: Callable[[BlobRef], Any],
) -> None:
    """Rebuild injected-op records from the split rows (degrade-to-unattested).

    Every payload reference materializes eagerly here (injected families are
    small by construction); the rebuilt records carry
    ``attestation="unattested"`` -- loading never attests -- with the
    callable's trust classification disclosed as the reason when resolution
    already failed without importing anything foreign.
    """

    state = trace.__dict__.get("_tl_injection_state")
    if not isinstance(state, dict):
        return
    pending = state.pop("pending_loaded_rows", None)
    if not pending:
        return
    from ..intervention.injection import InjectedOp, InjectionProvenance

    records: list[InjectedOp] = []
    for row in pending:
        envelope = row.annotations[_ENVELOPE_KEY]
        provenance_record = row.injection_provenance
        out = row.out
        if isinstance(out, BlobRef):
            out = materialize(out)
        saved_args: tuple[Any, ...] | None = None
        saved_kwargs: tuple[tuple[str, Any], ...] | None = None
        if envelope["args"] is not None:
            saved_args = tuple(_decode_value(encoded, materialize) for encoded in envelope["args"])
            saved_kwargs = tuple(
                (name, _decode_value(encoded, materialize))
                for name, encoded in (envelope["kwargs"] or ())
            )
        callable_ref = _decode_registry_key(envelope["callable"])
        records.append(
            InjectedOp(
                label=row.layer_label,
                host_label=envelope["host_label"],
                func_name=row.func_name,
                layer_type=row.type,
                out=out,
                provenance=InjectionProvenance(
                    host_site_key=provenance_record["host_site_key"],
                    spec_rule_id=provenance_record["spec_rule_id"],
                    host_pass=provenance_record["host_pass"],
                    firing_index=provenance_record["firing_index"],
                    nesting_path=tuple(provenance_record["nesting_path"]),
                    local_op_ordinal=provenance_record["local_op_ordinal"],
                    output_slot=provenance_record["output_slot"],
                ),
                callable_ref=callable_ref,
                saved_args=saved_args,
                saved_kwargs=saved_kwargs,
                attestation="unattested",
                attestation_reason=_classify_callable(callable_ref),
                fire_device=envelope["fire_device"],
            )
        )
    state["records"] = records
    state["resolved"] = True
    state["armed"] = False


def _decode_value(encoded: dict[str, Any], materialize: Callable[[BlobRef], Any]) -> Any:
    """Decode one envelope-encoded replay arg (grammar already validated)."""

    kind = encoded["kind"]
    value = encoded["value"]
    if kind == "list":
        return [_decode_value(item, materialize) for item in value]
    if kind == "tuple":
        return tuple(_decode_value(item, materialize) for item in value)
    if kind == "dict":
        return {key: _decode_value(item, materialize) for key, item in value}
    return _decode_leaf(kind, value, materialize)


def _decode_leaf(kind: str, value: Any, materialize: Callable[[BlobRef], Any]) -> Any:
    """Decode one leaf-kind replay arg (tensor/dtype/device/literal)."""

    if kind == "tensor":
        return materialize(value)
    if kind == "dtype":
        resolved = getattr(torch, value, None)
        if not isinstance(resolved, torch.dtype):
            _refuse_codec(f"encoded dtype {value!r} does not resolve on torch", "codec_args")
        return resolved
    if kind == "device":
        return torch.device(value)
    return value


def _decode_registry_key(encoded: dict[str, Any] | None) -> Any:
    """Rebuild one FunctionRegistryKey from its persisted field dict."""

    if encoded is None:
        return None
    from ..intervention.types import FunctionRegistryKey

    return FunctionRegistryKey(
        namespace=encoded["namespace"],
        qualname=encoded["qualname"],
        dispatch_kind=encoded["dispatch_kind"],
        version=encoded["version"],
        import_path=encoded["import_path"],
    )


def _classify_callable(callable_ref: Any) -> str | None:
    """Classify a callable reference's load-time trust WITHOUT importing.

    ``None`` means the reference resolves through the fixed trusted
    namespaces and awaits explicit replay
    (:func:`torchlens.intervention.injection.attest_injected_ops`);
    foreign references classify ``callable_untrusted`` (the resolver
    default-denies their import) and unresolvable trusted-namespace
    references ``callable_missing`` -- the degrade-to-unattested arms.
    """

    if callable_ref is None:
        return "callable_ref_unavailable"
    from ..intervention.errors import ReplayPreconditionError, UntrustedCallableError
    from ..intervention.resolver import resolve_function_registry_key

    try:
        resolve_function_registry_key(callable_ref)
    except UntrustedCallableError:
        return "callable_untrusted"
    except (ReplayPreconditionError, AttributeError, ImportError, KeyError, TypeError, ValueError):
        return "callable_missing"
    return None
