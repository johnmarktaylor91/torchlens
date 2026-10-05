"""Paddle parameter-write guard for the preview capture.

The coverage oracle credits an unlabeled op input that IS a registered
parameter of the captured tree as a known pre-forward source. That credit is
sound only while the parameter still holds its pre-forward value. A Paddle
kernel can write a parameter in place without passing through any wrapped
op (training-mode ``batch_norm`` updates ``_mean``/``_variance`` from the
batch), and a later read of the written value depends on the forward's
inputs through an edge the graph does not carry; replay cannot see it
either, because it reads the post-write value the op saw. The guard keeps
an exact host copy of every registered parameter taken before the forward.
A parameter read that sees a changed value is not credited, and a forward
that changed any parameter records a trace-level marker, so validation
fails closed on a hidden write whether or not anything reads it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["PARAMETER_WRITE_MARKER", "ParameterWriteGuard"]

PARAMETER_WRITE_MARKER = "parameter written during forward"


def _host_copy(param: Any, trace: Any | None) -> np.ndarray:
    """Return an exact host copy of one parameter, invisible to the capture.

    ``Tensor.numpy`` runs inside the capture wrapper's emit path, so the
    capture depth is raised for the read: a nested call passes straight to
    the original and is never recorded (or denied) as a user op.
    """

    depth = int(getattr(trace, "_paddle_capture_depth", 0)) if trace is not None else 0
    if trace is not None:
        trace._paddle_capture_depth = depth + 1
    try:
        return np.array(param.numpy(), copy=True)
    finally:
        if trace is not None:
            trace._paddle_capture_depth = depth


def _same_value(left: np.ndarray, right: np.ndarray) -> bool:
    """Return whether two host copies are bit-identical (NaN and -0.0 exact)."""

    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


@dataclass
class ParameterWriteGuard:
    """Pre-forward fingerprints of the captured tree's registered parameters.

    Attributes
    ----------
    snapshots
        ``id(param) -> (address, param, host copy)``; the guard holds each
        parameter so its identity stays valid for the capture.
    """

    snapshots: dict[int, tuple[str, Any, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def capture(cls, root: object, param_address_by_id: Mapping[int, str]) -> ParameterWriteGuard:
        """Fingerprint every registered parameter of ``root`` before the forward."""

        snapshots: dict[int, tuple[str, Any, np.ndarray]] = {}
        named_parameters = getattr(root, "named_parameters", None)
        if not param_address_by_id or not callable(named_parameters):
            return cls(snapshots)
        for _name, param in named_parameters():
            address = param_address_by_id.get(id(param))
            if address is not None and id(param) not in snapshots:
                snapshots[id(param)] = (address, param, _host_copy(param, None))
        return cls(snapshots)

    def was_written(self, param: Any, trace: Any) -> bool:
        """Return whether ``param`` differs from its pre-forward value.

        A parameter the guard never fingerprinted counts as written: the
        credit is only ever given against a recorded pre-forward value.
        """

        entry = self.snapshots.get(id(param))
        if entry is None or entry[1] is not param:
            return True
        return not _same_value(entry[2], _host_copy(param, trace))

    def written_addresses(self, trace: Any) -> tuple[str, ...]:
        """Return the addresses of every parameter the forward changed."""

        return tuple(
            address
            for address, param, before in self.snapshots.values()
            if not _same_value(before, _host_copy(param, trace))
        )
