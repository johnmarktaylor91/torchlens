"""Audited expected-opaque boundaries of the aten-dispatch completeness witness.

The reviewable table of wrapper scopes whose aten dispatches are opaque by design
(metadata and export conversions that TorchLens never records as ops, the scalar
extraction family, the in-forward ``torch.autograd.grad`` engine pass, and the
foreign-kernel ``autograd.Function`` boundary), its hard budget, and the derived
wrapper-wide set. ``completeness_witness`` re-exports every name; split out along the
table seam for the file-size ratchet, logic unchanged. A new row needs a regression
test and a reason.
"""

from __future__ import annotations

from ._completeness_types import AuditedCompletenessBoundary

__all__ = (
    "AUDITED_COMPLETENESS_BOUNDARIES",
    "MAX_AUDITED_COMPLETENESS_BOUNDARIES",
    "_EXPECTED_OPAQUE_WRAPPERS",
)

MAX_AUDITED_COMPLETENESS_BOUNDARIES = 11
"""Hard budget preventing expected-opaque wrapper scopes from growing unchecked."""

AUDITED_COMPLETENESS_BOUNDARIES: tuple[AuditedCompletenessBoundary, ...] = (
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:numpy:not_logged",
        operator=None,
        reason="existing metadata/export conversion boundary; TorchLens never records it as an op",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:__array__:not_logged",
        operator=None,
        reason="existing NumPy protocol conversion boundary; TorchLens never records it as an op",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:size:not_logged",
        operator=None,
        reason="existing tensor shape metadata boundary; TorchLens never records it as an op",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:dim:not_logged",
        operator=None,
        reason="existing tensor rank metadata boundary; TorchLens never records it as an op",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:item:logged",
        operator="aten._local_scalar_dense.default",
        reason="item extracts a Python scalar, and TorchLens intentionally records no scalar-output op",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:__bool__:logged",
        operator="aten._local_scalar_dense.default",
        reason="tensor truth testing extracts a Python bool, which is intentionally not an op output",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:__float__:logged",
        operator="aten._local_scalar_dense.default",
        reason="float(tensor) extracts a Python float, which is intentionally not an op output",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:__int__:logged",
        operator="aten._local_scalar_dense.default",
        reason="int(tensor) extracts a Python int, which is intentionally not an op output",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:__index__:logged",
        operator="aten._local_scalar_dense.default",
        reason=(
            "operator.index(tensor) extracts a Python int, which is intentionally not an op output"
        ),
    ),
    AuditedCompletenessBoundary(
        wrapper_name="autograd:grad",
        operator=None,
        reason=(
            "torch.autograd.grad executes a separately captured backward pass; its engine "
            "dispatches are not forward operations"
        ),
    ),
    AuditedCompletenessBoundary(
        wrapper_name="autograd_function:foreign",
        operator=None,
        reason=(
            "an autograd.Function whose forward left torch (numpy / dlpack round trip) is "
            "recorded as one autogradfunction boundary op that replays through apply; every "
            "dispatch inside it belongs to that op"
        ),
    ),
)
"""Reviewable exact expected-opaque rows; additions require a regression test and reason."""

if len(AUDITED_COMPLETENESS_BOUNDARIES) > MAX_AUDITED_COMPLETENESS_BOUNDARIES:
    raise RuntimeError("TorchLens completeness boundary budget exceeded.")

_EXPECTED_OPAQUE_WRAPPERS = frozenset(
    row.wrapper_name for row in AUDITED_COMPLETENESS_BOUNDARIES if row.operator is None
)
