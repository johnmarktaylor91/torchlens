"""Step 6 buffer-source wiring and buffer value comparison for ``control_flow``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._utils.display import identity
from .._utils.tensor_utils import safe_copy, tensor_nanequal

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def _buffer_values_identical(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two buffer payloads hold exactly the same values.

    Buffer provenance (whether a buffer version replays as ``identity`` of its
    source op, and whether two reads of one buffer version merge) is decided by
    this comparison, so it must not depend on the values the forward happens to
    reach. IEEE ``torch.equal`` reads a NaN-bearing tensor as unequal to an exact
    copy of itself, so a model whose buffer reaches NaN from one entry state but
    not another recorded two different graphs. This is the exact (no tolerance)
    ``tensor_nanequal`` that validation replay checks the ``identity`` op against:
    identical NaN and Inf positions, identical finite values, identical zero signs.
    Meta payloads carry no values, so they never compare identical (W1-FAB).

    Parameters
    ----------
    left:
        First captured buffer-side payload.
    right:
        Second captured buffer-side payload.

    Returns
    -------
    bool
        True when the payloads are exactly equal under NaN-equal semantics.
    """

    if left.is_meta or right.is_meta:
        return False
    return tensor_nanequal(left, right)


def _buffer_source_value_matches(
    source: Op, buffer_layer: Op, *, structure_only: bool = False
) -> bool:
    """Return whether a buffer-version source op output equals the full buffer value.

    Structure-only captures retain no payloads, so the value comparison
    cannot run; the verdict falls back to the DECLARED geometry (shape +
    dtype agreement between the producer op and the buffer's write record) —
    a hypothesis claim exactly like the shapes it rides on, corroborated or
    refuted by discharge against a real capture (weightsfree memo D7: the
    parity gate is the acceptance authority, the graph must not diverge from
    the real capture's purely because values were unavailable).
    """

    if structure_only and (source.out is None or buffer_layer.out is None):
        source_shape = getattr(source, "shape", None)
        buffer_shape = getattr(buffer_layer, "shape", None)
        return (
            source_shape is not None
            and buffer_shape is not None
            and tuple(source_shape) == tuple(buffer_shape)
            and str(getattr(source, "dtype", None)) == str(getattr(buffer_layer, "dtype", None))
        )
    if source.out is None or buffer_layer.out is None:
        return False
    if not isinstance(source.out, torch.Tensor) or not isinstance(buffer_layer.out, torch.Tensor):
        return False
    if tuple(source.out.shape) != tuple(buffer_layer.out.shape):
        return False
    try:
        return _buffer_values_identical(source.out, buffer_layer.out)
    except Exception:
        return False


def _wire_buffer_to_source(self: Trace, layer: Op, layer_label: str) -> None:
    """Connect one buffer op to its ``buffer_source`` op (Step 6, part 1).

    Links parent and child, replays the buffer as ``identity`` of its source
    when the values match (else marks replay unvalidated), inherits the
    source's ancestry, points the first argument at the source and saves a
    copy of the source output as the buffer's argument.

    Parameters
    ----------
    self:
        Trace being postprocessed.
    layer:
        Buffer op whose ``graph.buffer_source`` is set.
    layer_label:
        Label of ``layer`` in ``self.buffer_layers``.
    """

    if layer.graph.buffer_source not in layer.parents:
        layer.parents.append(layer.graph.buffer_source)
    if layer_label not in self[layer.graph.buffer_source].children:
        self[layer.graph.buffer_source].children.append(layer_label)
    self[layer.graph.buffer_source]._graph_has_children = True
    source_matches_buffer = _buffer_source_value_matches(
        self[layer.graph.buffer_source],
        layer,
        structure_only=bool(self._options_structure_only),
    )
    if source_matches_buffer:
        layer.func = identity
        layer.func_name = "identity"
    else:
        layer.buffer_replay_validated = False
    layer._graph_has_input_ancestor = bool(self[layer.graph.buffer_source].graph.has_input_ancestor)
    layer.input_ancestors.update(self[layer.graph.buffer_source].input_ancestors)
    layer.graph.root_ancestors.discard(layer._label_raw)
    layer.graph.root_ancestors.update(self[layer.graph.buffer_source].graph.root_ancestors)
    layer.call.parent_arg_positions["args"][0] = layer.graph.buffer_source
    if (self[layer.graph.buffer_source].out is not None) and (layer.call.saved_args is not None):
        layer.call.saved_args.append(
            safe_copy(self[layer.graph.buffer_source].out, detach_tensor=True)
        )
