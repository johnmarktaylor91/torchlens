"""dialz converter.

Turns a TorchLens :class:`~torchlens.intervention.Direction` (built by
``torchlens.intervention.fit_direction``) into a real
``dialz.SteeringVector``. Nothing is recomputed here: the vector is the
direction's own, as float32, keyed by the decoder layer ``SteeringModel``
wraps, so ``set_control(vector, scalar=m)`` adds ``m * direction.vector`` to
that block's output, as ``direction.steer(m)`` does.
"""

from __future__ import annotations

from typing import Any

from .._errors import InvalidArgumentError


def from_direction(
    direction: Any, *, layer: int | None = None, model_type: str | None = None
) -> Any:
    """Return a ``dialz.SteeringVector`` holding ``direction.vector``.

    Parameters
    ----------
    direction:
        A TorchLens ``Direction``.
    layer:
        Decoder layer ``SteeringModel`` applies it to; defaults to the trailing
        index of the direction's site label (see ``Direction.library_layer``).
    model_type:
        ``SteeringVector.model_type``; defaults to ``direction.model_type``.

    Returns
    -------
    Any
        A real ``dialz.SteeringVector``.

    Raises
    ------
    ImportError
        If dialz is unavailable.
    InvalidArgumentError
        ``direction_model_type_unknown`` when no model type is given or recorded.
    """

    try:
        import dialz
    except ImportError as exc:
        raise ImportError(
            "dialz bridge requires the `dialz` extra: install torchlens[dialz]."
        ) from exc
    index = direction.library_layer(layer)
    found = model_type if model_type is not None else direction.model_type
    if not isinstance(found, str):
        raise InvalidArgumentError(
            "The direction records no model type and none was given",
            code="direction_model_type_unknown",
            remedy="pass model_type= (for example 'llama')",
        )
    values = direction.vector.detach().cpu().float().numpy()
    return dialz.SteeringVector(model_type=found, directions={index: values})


__all__ = ["from_direction"]
