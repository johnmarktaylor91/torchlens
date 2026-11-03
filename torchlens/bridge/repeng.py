"""repeng converter.

Turns a TorchLens :class:`~torchlens.intervention.Direction` (built by
``torchlens.intervention.fit_direction``) into a real
``repeng.ControlVector``. Nothing is recomputed here: the vector is the
direction's own, as float32, keyed by the decoder layer ``ControlModel``
patches, so ``set_control(vector, coeff=m)`` adds ``m * direction.vector`` to
that block's output, as ``direction.steer(m)`` does.
"""

from __future__ import annotations

from typing import Any

from .._errors import InvalidArgumentError


def from_direction(
    direction: Any, *, layer: int | None = None, model_type: str | None = None
) -> Any:
    """Return a ``repeng.ControlVector`` holding ``direction.vector``.

    Parameters
    ----------
    direction:
        A TorchLens ``Direction``.
    layer:
        Decoder layer ``ControlModel`` applies it to; defaults to the trailing
        index of the direction's site label (see ``Direction.library_layer``).
    model_type:
        ``ControlVector.model_type``; defaults to ``direction.model_type``.

    Returns
    -------
    Any
        A real ``repeng.ControlVector``.

    Raises
    ------
    ImportError
        If repeng is unavailable.
    InvalidArgumentError
        ``direction_model_type_unknown`` when no model type is given or recorded.

    Examples
    --------
    >>> # requires: repeng
    >>> net = nn.Sequential(nn.Linear(4, 4)).eval()
    >>> pos = tl.trace(net, torch.randn(3, 4))
    >>> neg = tl.trace(net, torch.randn(3, 4))
    >>> direction = tl.intervention.fit_direction(pos, neg, "linear", read_token_index=None)
    >>> vector = tl.bridge.repeng.from_direction(direction, layer=1, model_type="tiny")
    >>> list(vector.directions)
    [1]
    """

    try:
        import repeng
    except ImportError as exc:
        raise ImportError(
            "repeng bridge requires the `repeng` extra: install torchlens[repeng]."
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
    return repeng.ControlVector(model_type=found, directions={index: values})


__all__ = ["from_direction"]
