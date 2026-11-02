"""Layer receptive-field functions reached from the records through ``_data_classes._doors.receptive_field``.

The ``Layer.receptive_field`` and ``Layer.projective_field`` properties
forward here; each function rejects a multi-pass layer with the package's
``AmbiguousPassError`` and otherwise reads the only pass's field view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._errors import AmbiguousPassError

if TYPE_CHECKING:
    from .._data_classes.layer import Layer
    from ._view import ReceptiveFieldView

__all__: list[str] = []


def layer_receptive_field(layer: Layer) -> ReceptiveFieldView:
    """Return the only pass's receptive field or reject a recurrent layer.

    Raises
    ------
    AmbiguousPassError
        If this logical layer contains multiple executed passes.
    """

    if layer.num_passes != 1 or len(layer.ops) != 1:
        passes = ", ".join(f"layer.ops[{index}]" for index in layer.ops)
        raise AmbiguousPassError(
            f"Layer {layer.layer_label!r} has {layer.num_passes} passes: {passes}."
        )
    return layer.ops[0].receptive_field


def layer_projective_field(layer: Layer) -> ReceptiveFieldView:
    """Return the only pass's projective field or reject pass ambiguity."""

    if layer.num_passes != 1 or len(layer.ops) != 1:
        passes = ", ".join(f"layer.ops[{index}]" for index in layer.ops)
        raise AmbiguousPassError(
            f"Layer {layer.layer_label!r} has {layer.num_passes} passes: {passes}."
        )
    return layer.ops[0].projective_field
