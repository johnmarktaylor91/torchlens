"""The ``"mechinterp"`` appliance behind ``trace.using("mechinterp")``.

A thin trace-bound view over kit functions that already exist: every method
calls the module-level function with this trace, so
``trace.using("mechinterp").head_scores(...)`` returns exactly what
``torchlens.mechinterp.head_scores(trace, ...)`` returns. Registered with
the appliance registry under the name ``"mechinterp"`` through a lazy
``"module:attribute"`` factory, so ``import torchlens`` does not import the
kit.

The appliance name ``"mechinterp"`` and ``logit_lens`` on it are ruled spellings
(H.28, names.5); the other method spellings stay DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .._semantic.logit_lens import LensFunc, LogitLens, logit_lens as _logit_lens
from ._factored import FactoredMatrix
from ._params import ov_circuit, qk_circuit
from ._scores import FixedPeriodInduction, HeadScores, head_scores

__all__ = ["MechInterpAppliance"]


class MechInterpAppliance:
    """Mech-interp kit calls bound to one trace (``trace.using("mechinterp")``)."""

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
        kind: str | FixedPeriodInduction = "induction",
        measure: str = "mul",
        layers: Any = None,
        pattern_source: Any = None,
    ) -> HeadScores:
        """Score every head against one taxonomy mask (see :func:`head_scores`)."""

        return head_scores(
            self._trace,
            kind=kind,
            measure=measure,
            layers=layers,
            pattern_source=pattern_source,
        )

    def qk_circuit(self, attention: Any) -> FactoredMatrix:
        """Return one attention module's per-head QK circuit (see :func:`qk_circuit`)."""

        return qk_circuit(self._trace, attention)

    def ov_circuit(self, attention: Any) -> FactoredMatrix:
        """Return one attention module's per-head OV circuit (see :func:`ov_circuit`)."""

        return ov_circuit(self._trace, attention)

    def logit_lens(
        self,
        *,
        facet: str = "resid_post",
        modules: Sequence[str] | None = None,
        lens: LensFunc | Mapping[str, LensFunc] | None = None,
        verify: bool = True,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> LogitLens:
        """Project per-block hidden states through the model's output head.

        The public logit lens (names.5): ``trace.using("mechinterp").logit_lens()``
        returns what the private ``torchlens._semantic.logit_lens.logit_lens``
        returns for this trace, including its fail-closed verification
        refusals (``logit_lens_head_unavailable``, ``logit_lens_identity_mismatch``,
        ``logit_lens_identity_unrecorded``, ...).

        Parameters
        ----------
        facet:
            Facet name read per block (default ``"resid_post"``).
        modules:
            Optional block module addresses to project; default every block.
        lens:
            Optional lens callable or per-module mapping replacing the model's head.
        verify:
            Whether the final block's projection is checked against the model's
            own logits before any prediction is returned.
        rtol, atol:
            Tolerances for that check.

        Returns
        -------
        LogitLens
            Per-block projections through the head.
        """

        return _logit_lens(
            self._trace,
            facet=facet,
            layers=modules,
            lens=lens,
            verify=verify,
            rtol=rtol,
            atol=atol,
        )

    def __repr__(self) -> str:
        """Name the bound methods."""

        return "MechInterpAppliance(head_scores, qk_circuit, ov_circuit, logit_lens)"
