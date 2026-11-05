"""Annotation-key resolution for repgeom node visuals.

``tl.viz.nodes.mds_scatter`` / ``rdm`` / ``scree`` look up stored blobs under
``op:<pass label>`` (a pass-qualified site) or ``layer:<layer label>`` (a
single-pass layer). The callback argument is not always the record that owns
the key:

- an unrolled node is handed to ``node_spec_fn`` as its aggregate Layer, whose
  per-pass ``label`` is undefined for a multi-pass layer; the per-pass Op the
  node draws rides ``_SPEC_SLOT_RENDERED_NODE`` while the callback runs;
- a collapsed module box is handed to ``collapsed_node_spec_fn`` as its
  aggregate Module; the ops the box stands for ride ``_SPEC_SLOT_BOX_OPS``, and
  the box's annotation is the one stored on its call's output op.
"""

from __future__ import annotations

from typing import Any

from .._utils._multipass_access import get_multipass_attr
from .._utils._render_slots import _SPEC_SLOT_BOX_OPS, _SPEC_SLOT_RENDERED_NODE

__tl_layer__ = "L6"

#: Where the Module record class is defined. Identified by name and module
#: along the MRO (as ``_utils._record_group_table`` does for Op and Layer), so
#: this L6 module never imports the record layer, which reaches the renderer.
_MODULE_RECORD = ("Module", "torchlens._data_classes.module")


def _is_module_record(node: Any) -> bool:
    """Return whether ``node`` is a ``Module`` record (or a subclass of one)."""

    return any((klass.__name__, klass.__module__) == _MODULE_RECORD for klass in type(node).__mro__)


def node_source_trace(node: Any) -> Any | None:
    """Return the Trace owning a rendered Layer, Op or collapsed Module.

    Parameters
    ----------
    node:
        Callback argument handed over by ``Trace.draw``.

    Returns
    -------
    Any | None
        Owning Trace, or ``None`` when it is gone or unknown.
    """

    if _is_module_record(node):
        return node.trace
    return getattr(node, "source_trace", None)


def node_caption(node: Any, key: str) -> str:
    """Return the caption line for a decorated node.

    Parameters
    ----------
    node:
        Callback argument handed over by ``Trace.draw``.
    key:
        Annotation key that matched, used when the node has no label.

    Returns
    -------
    str
        Module address for a collapsed box, else the layer (or op) label.
    """

    if _is_module_record(node):
        return str(node.address)
    layer_label = get_multipass_attr(node, "layer_label", None, multipass=None)
    if layer_label is not None:
        return str(layer_label)
    return str(get_multipass_attr(node, "label", None, multipass=None) or key)


def node_shape(node: Any, spec: Any) -> str:
    """Return the node shape for a decorated node.

    Parameters
    ----------
    node:
        Callback argument handed over by ``Trace.draw``.
    spec:
        Default ``NodeSpec`` for the node.

    Returns
    -------
    str
        The collapsed box's own shape for a Module (it stays a module box),
        else ``"box"``.
    """

    if _is_module_record(node):
        return str(spec.shape or "box3d")
    return "box"


def annotation_key_candidates(node: Any) -> list[str]:
    """Return annotation keys to try for a rendered node, most specific first.

    Parameters
    ----------
    node:
        Callback argument handed over by ``Trace.draw`` (Layer, Op or Module).

    Returns
    -------
    list[str]
        Unique ``op:`` / ``layer:`` keys without the blob-kind prefix.
    """

    if _is_module_record(node):
        return _unique(_module_box_candidates(node))
    candidates: list[str] = []
    rendered = _SPEC_SLOT_RENDERED_NODE.get()
    if rendered is not None and rendered is not node and _same_layer(rendered, node):
        candidates.extend(_record_candidates(rendered))
    candidates.extend(_record_candidates(node))
    return _unique(candidates)


def _record_candidates(record: Any) -> list[str]:
    """Return the ``op:`` and ``layer:`` keys a Layer or Op answers to."""

    candidates = []
    # A rolled multi-pass Layer has no single per-pass label; plain getattr
    # would leak the multi-pass ValueError tripwire and kill the whole draw.
    label = get_multipass_attr(record, "label", None, multipass=None)
    if label is not None:
        candidates.append(f"op:{label}")
    layer_label = get_multipass_attr(record, "layer_label", None, multipass=None)
    if layer_label is not None:
        candidates.append(f"layer:{layer_label}")
    return candidates


def _same_layer(rendered: Any, node: Any) -> bool:
    """Return whether the slot's rendered record is a pass of ``node``."""

    rendered_layer = get_multipass_attr(rendered, "layer_label", None, multipass=None)
    node_layer = get_multipass_attr(node, "layer_label", None, multipass=None)
    return rendered_layer is not None and rendered_layer == node_layer


def _module_box_candidates(module: Any) -> list[str]:
    """Return the keys stored on the output ops of the one call a box draws.

    A box drawing several calls (a rolled multi-call module) has no single
    output, exactly like a rolled multi-pass layer, so it gets no candidates.
    Output ops surfaced as their own nodes are not in the box's op set and are
    skipped, so the box never repeats a neighbouring node's picture.
    """

    box_ops = _SPEC_SLOT_BOX_OPS.get()
    calls = [module.calls[index] for index in range(len(module.calls))]
    if box_ops is not None:
        members = set(box_ops)
        calls = [call for call in calls if members.intersection(call.ops)]
    if len(calls) != 1:
        return []
    output_ops = [str(label) for label in calls[0].output_ops]
    if box_ops is not None:
        output_ops = [label for label in output_ops if label in members]
    candidates = []
    for label in output_ops:
        candidates.append(f"op:{label}")
        # A single-pass layer's annotation is stored under its layer label;
        # the key only exists for single-pass layers, so the bare layer of a
        # pass label can never pick up another pass's annotation.
        candidates.append(f"layer:{label.rsplit(':', 1)[0]}")
    return candidates


def _unique(candidates: list[str]) -> list[str]:
    """Return ``candidates`` without repeats, first occurrence kept."""

    return list(dict.fromkeys(candidates))
