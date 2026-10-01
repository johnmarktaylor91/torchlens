"""The shared first-bad-thing result vocabulary for TorchLens diagnostics.

Observe memo item 5: ``find_nan`` / ``bisect_nan`` / ``bisect_precision`` /
the backward NaN bisector / ``check_determinism`` each answer "what was the
first bad thing this run did" -- one shared result shape means the fourth and
fifth tools join ONE vocabulary instead of adding new ones. Every spelling
here is DOCUMENTED-UNSTABLE pending naming-session ratification.

Also home to the shared AMP/GradScaler disclosure text (item 6, the R20
precondition): gradients captured from a backward run under
``torch.amp.GradScaler`` carry the loss scale (~2**16 at the default
``init_scale``), so magnitude-based verdicts must accept ``grad_scale=`` and
non-finite verdicts must name fp16 scale overflow as a possibility instead of
false-positive certainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["FirstBadThing", "amp_scaled_gradients_hint"]

#: Closed coverage vocabulary: "complete" (every relevant payload checked),
#: "found_first_among_checked" (earlier unchecked payloads exist -- the claim
#: is first-among-checked, never absolute first), "partial" (some payloads
#: unavailable but no finding precedence at stake), "none" (nothing checked).
COVERAGE_VALUES = ("complete", "found_first_among_checked", "partial", "none")


@dataclass(frozen=True)
class FirstBadThing:
    """One first-bad-thing finding in the shared diagnostic vocabulary.

    Parameters
    ----------
    found:
        Whether the producing tool found a bad thing at all.
    tool:
        Producing diagnostic (``"find_nan"``, ``"bisect_nan"``,
        ``"bisect_precision"``, ``"bisect_nan_backward"``,
        ``"check_determinism"``).
    kind:
        What the bad thing is: ``"nan"`` / ``"inf"`` / ``"nan+inf"`` /
        ``"precision_divergence"`` / ``"nondeterminism"`` / ``"none"``.
    label:
        Canonical public label of the implicated site (pass-qualified for
        multi-pass layers), or ``None`` with ``label_status`` explaining why.
    label_status:
        ``"final"`` / ``"pruned"`` / ``"unavailable"`` / ``"none"`` -- the
        item-1 resolver vocabulary.
    module:
        Containing module address, when known.
    source_line:
        ``"file:line"`` of the implicated forward call site, when known.
    backward_pass:
        One-based backward pass for backward findings, else ``None``.
    coverage:
        Claim strength from :data:`COVERAGE_VALUES`: a finding over
        incomplete evidence is ``"found_first_among_checked"``, never
        absolute first.
    uncertainty_zone:
        Labels whose payloads could not be checked and may hide the actual
        first bad thing.
    detection_basis:
        How the finding was measured (``"saved_activations"``,
        ``"saved_gradients"``, ``"live_capture"``, ``"double_run"``).
        Reserved so the future live/anomaly tier joins the same shape.
    message:
        Human-readable summary.
    """

    found: bool
    tool: str
    kind: str
    label: str | None = None
    label_status: str = "none"
    module: str | None = None
    source_line: str | None = None
    backward_pass: int | None = None
    coverage: str = "complete"
    uncertainty_zone: tuple[str, ...] = field(default_factory=tuple)
    detection_basis: str = "saved_activations"
    message: str = ""


def amp_scaled_gradients_hint(*, all_nonfinite: bool = False) -> str:
    """Return the shared AMP/GradScaler disclosure text (item 6 / R20).

    Parameters
    ----------
    all_nonfinite:
        Whether the caller's finding is non-finite gradients (fp16 scale
        OVERFLOW wording) rather than merely large norms (scale wording).

    Returns
    -------
    str
        Disclosure naming the GradScaler possibility and the remedy; it never
        edits the caller's rows or verdicts.
    """

    if all_nonfinite:
        return (
            "if this backward ran under torch.amp.GradScaler, fp16 intermediate "
            "gradients carry the loss scale (~2**16 at the default init_scale) "
            "and can overflow to inf before the scaler skips the step -- an inf "
            "BIRTH under autocast fp16 may be scale overflow, not a model "
            "defect; pass grad_scale=scaler.get_scale() to disclose the scale "
            "alongside the ledger"
        )
    return (
        "if this backward ran under torch.amp.GradScaler the captured gradients "
        "carry the loss scale (~2**16 at the default init_scale) -- pass "
        "grad_scale=scaler.get_scale() to report in unscaled units"
    )
