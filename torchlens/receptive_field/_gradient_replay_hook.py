"""Install point for the gradient probes on traces without backward capture.

The gradient gates in :mod:`._gradient` (receptive) and :mod:`._gradient_forward`
(projective) route a trace whose backend has no backward capture to the cone-replay
probes in :mod:`._gradient_provider`. That module builds its results with the gates'
own builders, so a gate importing it would close an import cycle. Instead the
provider installs its two probes here when it is imported, and the package
``__init__`` imports the provider, so both probes are installed before any trace
can reach a gate. This module imports nothing from the gates or the provider.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Final

from ._errors import ReceptiveFieldUnavailableError

if TYPE_CHECKING:
    from ._types import GradReceptiveField

    ReplayProbe = Callable[..., GradReceptiveField | Mapping[str, GradReceptiveField]]

#: Refusal code of every gradient probe that cannot replay its cone.
REPLAY_REFUSAL_CODE: Final = "gradient_replay_unavailable"

_REPLAY_PROBES: dict[str, Callable[..., Any]] = {}


def install_replay_probes(*, receptive: ReplayProbe, projective: ReplayProbe) -> None:
    """Install the receptive and projective cone-replay probes (once, at provider import)."""

    _REPLAY_PROBES.update(receptive=receptive, projective=projective)


def replay_probe(direction: str) -> ReplayProbe:
    """Return the installed cone-replay probe for ``direction``, refusing when none is installed."""

    try:
        return _REPLAY_PROBES[direction]
    except KeyError:
        raise ReceptiveFieldUnavailableError(
            f"The {direction} gradient probe for traces without backward capture is not "
            "installed. Import torchlens.receptive_field and probe again.",
            code=REPLAY_REFUSAL_CODE,
        ) from None
