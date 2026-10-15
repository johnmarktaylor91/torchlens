"""The ``"mi"`` appliance behind ``trace.using("mi")``.

A thin trace-bound view over kit functions that already exist: every method
calls the module-level function with this trace, so
``trace.using("mi").head_scores(...)`` returns exactly what
``torchlens.mechinterp.head_scores(trace, ...)`` returns. Registered with
the appliance registry under the name ``"mi"`` through a lazy
``"module:attribute"`` factory, so ``import torchlens`` does not import the
kit.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from typing import Any

from ._factored import FactoredMatrix
from ._params import ov_circuit, qk_circuit
from ._scores import HeadScoreResult, head_scores

__all__ = ["MechInterpAppliance"]


class MechInterpAppliance:
    """Mech-interp kit calls bound to one trace (``trace.using("mi")``)."""

    def __init__(self, trace: Any) -> None:
        """Bind the appliance to ``trace``.

        Parameters
        ----------
        trace:
            A finished torchlens trace.
        """

        self._trace = trace

    @property
    def trace(self) -> Any:
        """The trace this appliance reads."""

        return self._trace

    def head_scores(
        self,
        *,
        kind: str = "induction",
        measure: str = "mul",
        layers: Any = None,
        pattern_source: Any = None,
        period: int | None = None,
    ) -> HeadScoreResult:
        """Score every head against one taxonomy mask (see :func:`head_scores`)."""

        return head_scores(
            self._trace,
            kind=kind,
            measure=measure,
            layers=layers,
            pattern_source=pattern_source,
            period=period,
        )

    def qk_circuit(self, attention: Any) -> FactoredMatrix:
        """Return one attention module's per-head QK circuit (see :func:`qk_circuit`)."""

        return qk_circuit(self._trace, attention)

    def ov_circuit(self, attention: Any) -> FactoredMatrix:
        """Return one attention module's per-head OV circuit (see :func:`ov_circuit`)."""

        return ov_circuit(self._trace, attention)

    def __repr__(self) -> str:
        """Name the bound methods."""

        return "MechInterpAppliance(head_scores, qk_circuit, ov_circuit)"
