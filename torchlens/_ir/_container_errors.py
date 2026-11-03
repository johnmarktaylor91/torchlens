"""The typed default-deny refusal shared by output-container reconstruction.

``container.py`` (the container codec) and ``distribution_container.py`` (the
``torch.distributions`` layout it delegates to) both raise this refusal. It lives
below both so neither module imports the other back to reach it.
"""

from __future__ import annotations

from typing import Any, cast

from .._errors import _actionable_message, _ActionableErrorMixin
from ..errors._base import TorchLensError


class ContainerReconstructionError(_ActionableErrorMixin, TorchLensError, ValueError):
    """Raised when an output-container spec names a type that is not admissible.

    The output ``ContainerSpec`` is portable, attacker-influenceable data. Its
    ``type_module`` / ``type_qualname`` name the concrete container class to rebuild.
    This error is the default-deny tripwire: a spec that names a type outside the
    benign container allowlist for its ``kind`` is refused BEFORE any construction,
    mirroring the safe-unpickler's global denial. Subclasses ``ValueError`` so the
    runnable run wrapper reports it as a typed ``RunPreconditionError`` denial.

    r3 b1-opus R64-3: the class also escapes RAW from the PUBLIC documented
    ``Op.multi_output_type`` property, so it is a user-catchable typed refusal,
    not only an internal codec error: it resolves from ``torchlens.errors``,
    carries ``fields["code"] == "container_spec_inadmissible"`` plus a remedy,
    and rows in ``docs/reference/error_refusal_contract.md``.
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str = "container_spec_inadmissible",
        remedy: str = (
            "Re-save the artifact from a trusted capture; the recorded "
            "output-container spec is corrupt, tampered, or names an "
            "inadmissible type."
        ),
        **context: object,
    ) -> None:
        """Initialize the typed default-deny container-spec refusal.

        Parameters
        ----------
        problem:
            Description of the inadmissible spec fact.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast("dict[str, Any]", context),
        )


__all__ = ("ContainerReconstructionError",)
