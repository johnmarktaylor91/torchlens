"""compare: two-artifact structural + value diff with coverage honesty.

Direction is always ``subject - reference`` and the result says so. Manifests
and fingerprints are compared FIRST: a fingerprint mismatch is loud
relationship evidence, never a silent numeric diff of two different models --
the structural tier is still returned (evidence, not just a verdict), but
value matching never loosens to labels. Coverage lives in the HEADER: "0
mismatches" must never read as "no differences" when the truth is "nothing
comparable was saved".
"""

from __future__ import annotations

from typing import Any

#: Result schema id.
COMPARE_SCHEMA = "torchlens.agent.compare.v1"

#: Closed per-site row states.
ROW_STATES = (
    "comparable",
    "only_reference",
    "only_subject",
    "shape_changed",
    "dtype_changed",
    "payload_unavailable",
)


def _site_index(log: Any) -> tuple[dict[str, Any], str]:
    """Index one trace's ops by structural site key (legacy fallback disclosed).

    Parameters
    ----------
    log:
        Loaded ``Trace``.

    Returns
    -------
    tuple[dict, str]
        Key -> op map and the match basis (``site_key`` or
        ``legacy_address_shape``).
    """

    ops = list(getattr(log, "layer_list", []) or [])
    keys = [getattr(op, "site_key", None) for op in ops]
    if ops and all(key is not None for key in keys):
        index: dict[str, Any] = {}
        for op, key in zip(ops, keys, strict=True):
            index[f"{key}|{int(getattr(op, 'pass_index', 1) or 1)}"] = op
        return index, "site_key"
    index = {}
    for op in ops:
        shape = getattr(op, "shape", None)
        address = str(getattr(op, "layer_label", "unknown"))
        index[
            f"{address}|{int(getattr(op, 'pass_index', 1) or 1)}|{tuple(shape) if shape else None}"
        ] = op
    return index, "legacy_address_shape"


def _value_tier(ref_op: Any, sub_op: Any, rtol: float, atol: float) -> dict[str, Any]:
    """Compute one comparable pair's value metrics (float64, CPU).

    Parameters
    ----------
    ref_op:
        Matched reference op record with a readable payload.
    sub_op:
        Matched subject op record with a readable payload.
    rtol:
        Echoed allclose relative tolerance.
    atol:
        Echoed allclose absolute tolerance.

    Returns
    -------
    dict[str, Any]
        Value metrics, or a ``payload_unavailable`` state row.
    """

    from .._io.payload_reader import read_op_payload

    try:
        reference = read_op_payload(ref_op)
        subject = read_op_payload(sub_op)
    except Exception:  # noqa: BLE001 - any read failure is the honest payload_unavailable row state
        return {"state": "payload_unavailable"}
    import torch

    if not isinstance(reference, torch.Tensor) or not isinstance(subject, torch.Tensor):
        return {"state": "payload_unavailable"}
    ref64 = reference.detach().to("cpu", dtype=torch.float64).flatten()
    sub64 = subject.detach().to("cpu", dtype=torch.float64).flatten()
    delta = sub64 - ref64
    denominator = ref64.norm() * sub64.norm()
    cosine = float((ref64 @ sub64 / denominator).item()) if float(denominator.item()) > 0 else None
    return {
        "state": "comparable",
        "max_abs_delta": delta.abs().max().item(),
        "mean_abs_delta": delta.abs().mean().item(),
        "l2_delta": delta.norm().item(),
        "relative_l2": (delta.norm() / ref64.norm()).item()
        if float(ref64.norm().item()) > 0
        else None,
        "cosine": cosine,
        "allclose": bool(torch.allclose(sub64, ref64, rtol=rtol, atol=atol, equal_nan=True)),
    }


def _payload_readable(op: Any) -> bool:
    """Whether a non-attaching payload read can serve this op."""

    slot = getattr(op, "_slot", None)
    if callable(slot):
        return slot("out") is not None or slot("out_ref") is not None
    return getattr(op, "out_ref", None) is not None or getattr(op, "out", None) is not None


def _matched_pair_row(ref_op: Any, sub_op: Any, rtol: float, atol: float) -> dict[str, Any]:
    """Build one matched-site row: shape/dtype tier, then the value tier."""

    row: dict[str, Any] = {"label": str(getattr(sub_op, "label", ""))}
    ref_shape = getattr(ref_op, "shape", None)
    sub_shape = getattr(sub_op, "shape", None)
    if ref_shape != sub_shape:
        row["state"] = "shape_changed"
        row["reference_shape"] = list(ref_shape) if ref_shape else None
        row["subject_shape"] = list(sub_shape) if sub_shape else None
        return row
    if str(getattr(ref_op, "dtype", None)) != str(getattr(sub_op, "dtype", None)):
        row["state"] = "dtype_changed"
        return row
    if _payload_readable(ref_op) and _payload_readable(sub_op):
        row.update(_value_tier(ref_op, sub_op, rtol, atol))
        return row
    row["state"] = "comparable"
    row["value_tier"] = "skipped_payload_not_saved_in_both"
    return row


def compare_traces(  # noqa: PLR0913 - the registry-declared request record, keyword-threaded
    reference: Any,
    subject: Any,
    *,
    ref_block: dict[str, Any],
    sub_block: dict[str, Any],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_rows: int = 200,
) -> dict[str, Any]:
    """Compare two loaded traces: structural tier always, value tier where honest.

    Parameters
    ----------
    reference:
        Loaded reference ``Trace`` (direction: subject - reference).
    subject:
        Loaded subject ``Trace``.
    ref_block:
        Reference envelope artifact block (fingerprint evidence).
    sub_block:
        Subject envelope artifact block (fingerprint evidence).
    rtol:
        allclose relative tolerance, echoed in the result.
    atol:
        allclose absolute tolerance, echoed in the result.
    max_rows:
        Cap on emitted per-site rows (changed rows rank first).

    Returns
    -------
    dict[str, Any]
        Compare data block with the non-droppable coverage header.
    """

    fingerprint_match = ref_block.get("model_fingerprint") is not None and ref_block.get(
        "model_fingerprint"
    ) == sub_block.get("model_fingerprint")
    ref_index, ref_basis = _site_index(reference)
    sub_index, sub_basis = _site_index(subject)
    match_basis = "site_key" if ref_basis == sub_basis == "site_key" else "legacy_address_shape"
    if match_basis == "legacy_address_shape":
        ref_index, _ = _site_index(reference)
        sub_index, _ = _site_index(subject)

    rows: list[dict[str, Any]] = []
    comparable = 0
    value_compared = 0
    changed = 0
    for key in sorted(set(ref_index) | set(sub_index)):
        ref_op = ref_index.get(key)
        sub_op = sub_index.get(key)
        row: dict[str, Any] = {"key": key}
        if ref_op is None:
            row["state"] = "only_subject"
            row["label"] = str(getattr(sub_op, "label", ""))
        elif sub_op is None:
            row["state"] = "only_reference"
            row["label"] = str(getattr(ref_op, "label", ""))
        else:
            row.update(_matched_pair_row(ref_op, sub_op, rtol, atol))
            if row["state"] in ("comparable",):
                comparable += 1
                if "allclose" in row:
                    value_compared += 1
                    if not row["allclose"]:
                        changed += 1
        rows.append(row)

    def _row_rank(row: dict[str, Any]) -> tuple[int, str]:
        """Changed/asymmetric rows rank first; ties break on the site key."""

        interesting = row["state"] != "comparable" or row.get("allclose") is False
        return (0 if interesting else 1, row["key"])

    ranked = sorted(rows, key=_row_rank)
    emitted = ranked[:max_rows]
    return {
        "direction": "subject_minus_reference",
        "fingerprint_match": fingerprint_match,
        "fingerprint_note": (
            None
            if fingerprint_match
            else (
                "model fingerprints differ: these are different models or "
                "different weights; value rows are relationship evidence, "
                "not a regression verdict"
            )
        ),
        "match_basis": match_basis,
        "rtol": rtol,
        "atol": atol,
        # Non-droppable coverage header (memo 3.7).
        "coverage": {
            "sites_reference": len(ref_index),
            "sites_subject": len(sub_index),
            "sites_matched": sum(
                1 for row in rows if row["state"] not in ("only_reference", "only_subject")
            ),
            "sites_comparable": comparable,
            "sites_value_compared": value_compared,
            "sites_changed": changed,
            "note": (
                "0 changed rows means nothing among the VALUE-COMPARED sites "
                "moved; sites without saved payloads in both artifacts were "
                "never numerically compared"
            ),
        },
        "rows": emitted,
        "rows_total": len(rows),
    }
