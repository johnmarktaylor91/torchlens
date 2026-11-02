"""The one label vocabulary: raw, pass-qualified op and pass-free layer labels.

TorchLens mints three label shapes for one operation. The raw label is minted
during the forward pass and carries the ``_raw`` sentinel suffix
(``relu_1_2_raw``). The op label is the final, pass-qualified spelling
(``conv2d_1_5:2``: layer type, type index, step index, colon, pass number).
The layer label is the final pass-free spelling (``conv2d_1_5``). Input,
output and buffer labels carry no step index (``input_1:1``, ``input_1``).
The short forms drop the step index (``conv2d_1:2``, ``conv2d_1``). This
module is where every internal site spells those conversions; the public
properties (``Op.label``, ``Op.layer_label``, ``Op.short_label``,
``Op.short_layer_label``, ``Op.raw_label``, ``Layer.layer_label``) keep their
names and values and read through it as the record lanes move their bodies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._constants import _RAW_LABEL_SUFFIX, _is_suffixed_raw_label

if TYPE_CHECKING:
    from .op import Op
    from .trace import Trace

__all__: list[str] = []

#: Separator between a layer label and its pass number in an op label.
PASS_SEPARATOR = ":"


def is_raw(label: object) -> bool:
    """Return whether ``label`` is a raw label (carries the ``_raw`` sentinel suffix)."""

    return _is_suffixed_raw_label(label)


def is_pass_qualified(label: str) -> bool:
    """Return whether ``label`` is an op label (ends in a colon and a pass number)."""

    _layer_label, separator, pass_text = label.rpartition(PASS_SEPARATOR)
    return bool(separator) and pass_text.isdigit()


def split_pass(op_label: str) -> tuple[str, int | None]:
    """Return ``(layer_label, pass_number)`` for ``op_label``.

    A pass-free label (a layer label) returns ``(label, None)``.
    """

    if not is_pass_qualified(op_label):
        return op_label, None
    layer_label, _separator, pass_text = op_label.rpartition(PASS_SEPARATOR)
    return layer_label, int(pass_text)


def join_pass(layer_label: str, pass_number: int) -> str:
    """Return the op label for ``layer_label`` on pass ``pass_number``."""

    return f"{layer_label}{PASS_SEPARATOR}{pass_number}"


def strip_step_index(layer_label: str) -> str:
    """Return the short layer label: ``layer_label`` with its step index dropped.

    A layer label ends in two numeric segments (type index, step index) or, for
    input, output and buffer layers, in one (type index); only the step index
    is dropped, so a label that already has the short shape is returned as is.
    """

    segments = layer_label.split("_")
    trailing_numeric = 0
    for segment in reversed(segments):
        if not segment.isdigit():
            break
        trailing_numeric += 1
    if trailing_numeric >= 2:
        return "_".join(segments[:-1])
    return layer_label


def short_op_label(op_label: str) -> str:
    """Return the short op label: the step index dropped, the pass number kept."""

    layer_label, pass_number = split_pass(op_label)
    short_layer = strip_step_index(layer_label)
    if pass_number is None:
        return short_layer
    return join_pass(short_layer, pass_number)


def raw_to_final_op(trace: Trace, raw_label: str) -> str | None:
    """Return the final op label minted for ``raw_label``, or ``None`` when unknown."""

    return trace._raw_to_final_op_labels.get(raw_label)


def raw_to_final_layer(trace: Trace, raw_label: str) -> str | None:
    """Return the final lookup label for ``raw_label``, or ``None`` when unknown.

    The lookup label is the layer label for a single-pass layer and the
    pass-qualified op label for a multi-pass one, exactly as postprocess
    records it.
    """

    return trace._raw_to_final_layer_labels.get(raw_label)


def final_to_raw_layer(trace: Trace, layer_label: str) -> str | None:
    """Return the raw label behind the final lookup label ``layer_label``, or ``None``."""

    return trace._final_to_raw_layer_labels.get(layer_label)


def fx_label(fx_qualpath: str | None, fx_call_index: int | None) -> str | None:
    """Return a torch.fx-style label for an op when module context is available."""

    if fx_qualpath is None:
        return None
    return f"{fx_qualpath.replace('.', '_')}_{fx_call_index}"


def public_raw_label(label_raw: object) -> str:
    """Return the public raw ordinal label without the raw-label suffix.

    Returns
    -------
    str
        Realtime ordinal identity for an op before equivalent-op grouping.
    """

    label = label_raw
    if isinstance(label, str) and label.endswith(_RAW_LABEL_SUFFIX):
        return label[: -len(_RAW_LABEL_SUFFIX)]
    return str(label)


def own_label_spellings(op: Op) -> set[str]:
    """Return every spelling a relation list may use for THIS op.

    On a finished trace a parent's ``children`` (or a child's ``parents``)
    may record this op under either its bare ``layer_label`` or -- on a
    multi-pass layer -- its pass-qualified ``layer_label:pass`` spelling,
    so self-exclusion in ``siblings``/``co_parents`` must cover BOTH
    (excluding only the bare spelling appended the op itself on every
    multi-pass layer). Unfinished traces still use raw labels.
    """

    _finished_trace = op._source_trace_or_none()
    _finished = op._tracing_finished or (
        _finished_trace is not None and _finished_trace._tracing_finished
    )
    if not _finished:
        return {op._label_raw}
    spellings = {op.layer_label}
    pass_index = op.pass_number
    if isinstance(pass_index, int):
        spellings.add(f"{op.layer_label}:{pass_index}")
    return spellings


def streaming_label(op: Op) -> str:
    """Best available label for sink/writer callbacks during or after postprocess.

    Returns
    -------
    str
        Pass-qualified label when available, otherwise the current layer label.
    """

    for candidate in (
        op.label,
        op.layer_label,
        op._layer_label_raw,
        op._label_raw,
    ):
        if candidate is not None:
            return str(candidate)
    return "<unknown>"
