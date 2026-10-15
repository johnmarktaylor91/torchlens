"""steering-vectors converter.

Turns a TorchLens :class:`~torchlens.intervention.Direction` (built by
``torchlens.intervention.direction_from_pairs``) into a real
``steering_vectors.SteeringVector``. Nothing is recomputed here: the vector is
the direction's own, keyed by the decoder layer the library patches, so
``SteeringVector.apply(model, multiplier=m)`` adds ``m * direction.vector`` to
that block's output, as ``direction.steer(m)`` does.
"""

from __future__ import annotations

from typing import Any


def from_direction(
    direction: Any, *, layer: int | None = None, layer_type: str = "decoder_block"
) -> Any:
    """Return a ``steering_vectors.SteeringVector`` holding ``direction.vector``.

    Parameters
    ----------
    direction:
        A TorchLens ``Direction``.
    layer:
        Decoder layer the library patches; defaults to the trailing index of
        the direction's site label (see ``Direction.library_layer``).
    layer_type:
        steering-vectors layer type (default ``"decoder_block"``).

    Returns
    -------
    Any
        A real ``steering_vectors.SteeringVector``.

    Raises
    ------
    ImportError
        If steering-vectors is unavailable.
    """

    try:
        import steering_vectors
    except ImportError as exc:
        raise ImportError(
            "steering-vectors bridge requires the `steering` extra: install torchlens[steering]."
        ) from exc
    index = direction.library_layer(layer)
    return steering_vectors.SteeringVector({index: direction.vector.detach().clone()}, layer_type)


__all__ = ["from_direction"]
