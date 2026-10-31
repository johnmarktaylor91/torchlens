"""The metadata line formatter for live narration (snoop D2).

One fixed-order single line per event on the shared tensor-core grammar:
common fields are POSITIONAL columns that line up down the page (at 300
repetitions ``key=value`` labels spend 30-40% of the width restating the
schema); rare fields (``pass=``, ``out=``, ``late=``) appear labeled, only
when informative. Pinned echo constraints (snoop 3.2): bounded predictable
per-field width, byte-stable transcripts (no addresses, wall-clock, or
memory figures on the default line), no escape codes off-tty, missing facts
OMITTED never zeroed, and the stats grammar must render with the finiteness
fields absent. Glyphs are ASCII (``>``/``<``/``!!``/``#``); the ``|`` stats
delimiter's EXISTENCE is pinned, its glyph is the naming sprint's.
"""

from __future__ import annotations

from typing import Any

from ._event import NarrationEvent, NarrationStats

__tl_layer__ = "L5"

#: Two spaces of indent per module depth (module depth, never Python call
#: depth -- snooper indented by call depth, which in an HF model is a tour of
#: ``torch/nn/modules/module.py``).
INDENT_UNIT = "  "

#: Minimum padded widths for the aligned positional columns.
_ORDINAL_WIDTH = 5
_LABEL_WIDTH = 24
_TENSOR_WIDTH = 22

#: Compact dtype tokens on the shared grammar (lovely panel vocabulary).
_DTYPE_TOKENS = {
    "float32": "f32",
    "float64": "f64",
    "float16": "f16",
    "bfloat16": "bf16",
    "float8_e4m3fn": "f8e4m3",
    "float8_e5m2": "f8e5m2",
    "int64": "i64",
    "int32": "i32",
    "int16": "i16",
    "int8": "i8",
    "uint8": "u8",
    "bool": "bool",
    "complex64": "c64",
    "complex128": "c128",
}


def compact_dtype(dtype: Any) -> str | None:
    """Return the compact dtype token for a dtype-like value.

    Parameters
    ----------
    dtype:
        ``DtypeRef``, backend dtype object, or canonical dtype string.

    Returns
    -------
    str | None
        Compact token (``f32``/``bf16``/...) or ``None`` when unavailable.
    """

    if dtype is None:
        return None
    name = getattr(dtype, "name", None) or str(dtype)
    bare = name.rsplit(".", 1)[-1]
    return _DTYPE_TOKENS.get(bare, bare)


def compact_device(device: Any) -> str | None:
    """Return the canonical device token for a device-like value.

    Parameters
    ----------
    device:
        ``DeviceRef``, backend device object, or device string.

    Returns
    -------
    str | None
        Canonical device string (``cpu``/``cuda:0``) or ``None``.
    """

    if device is None:
        return None
    return getattr(device, "name", None) or str(device)


def format_shape(shape: tuple[int, ...] | None) -> str | None:
    """Return the bracketed shape token, e.g. ``[1,9,768]``.

    Parameters
    ----------
    shape:
        Output tensor shape, or ``None`` when unavailable.

    Returns
    -------
    str | None
        Compact shape token; ``[]`` is a scalar; ``None`` means omitted.
    """

    if shape is None:
        return None
    return "[" + ",".join(str(dim) for dim in shape) + "]"


def _format_value(value: float) -> str:
    """Format one stats value with bounded width (4 significant digits)."""

    if value != value:  # NaN compares unequal to itself.
        return "nan"
    if value in (float("inf"), float("-inf")):
        return "inf" if value > 0 else "-inf"
    if value == 0:
        return "0"
    if abs(value) >= 1e5 or abs(value) < 1e-4:
        return f"{value:.3e}"
    return f"{value:.4g}"


def render_stats_segment(stats: NarrationStats) -> str:
    """Render the stats segment on the shared grammar (snoop D4).

    The sampled policy ``~``-marks every moment and carries the mandatory,
    non-suppressible ``sampled=k/N`` evidence token; it NEVER renders a
    finiteness field (the hard gate: the grammar renders with the finiteness
    family absent). Exact prints the census only when nonzero, plus the
    ``finite`` clearance token when exactly zero. Reuse prints only what
    another armed feature already paid for.

    Parameters
    ----------
    stats:
        Structured stats record for one narrated tensor.

    Returns
    -------
    str
        Space-joined stats tokens; empty when nothing is claimable.
    """

    mark = "~=" if stats.policy == "sampled" else "="
    moments = (
        ("mean", stats.mean),
        ("sd", stats.sd),
        ("min", stats.minimum),
        ("max", stats.maximum),
    )
    tokens = [f"{name}{mark}{_format_value(value)}" for name, value in moments if value is not None]
    if stats.policy == "sampled":
        tokens.append(f"sampled={stats.sample_size}/{stats.population}")
        return " ".join(tokens)
    tokens.extend(_finiteness_tokens(stats))
    return " ".join(tokens)


def _finiteness_tokens(stats: NarrationStats) -> list[str]:
    """Return the finiteness tokens for an exact/reuse census (never sampled).

    A complete census prints the nonzero counts, or the ``finite`` clearance
    token when exactly zero everywhere; an incomplete census falls back to
    the boolean claim when one exists, else claims nothing.
    """

    census = (
        ("nan=", stats.nan_count),
        ("+inf=", stats.posinf_count),
        ("-inf=", stats.neginf_count),
    )
    if all(count is not None for _, count in census):
        tokens = [f"{label}{count}" for label, count in census if count]
        return tokens or ["finite"]
    if stats.has_nonfinite is not None:
        return ["nonfinite" if stats.has_nonfinite else "finite"]
    return []


def render_line(event: NarrationEvent) -> str:
    """Render one ``NarrationEvent`` as its narration line.

    Parameters
    ----------
    event:
        Frozen narration record.

    Returns
    -------
    str
        The rendered line, without a trailing newline.
    """

    indent = INDENT_UNIT * max(event.module_depth, 0)
    structural = _structural_line(event, indent)
    if structural is not None:
        return structural
    line = f"{indent}{'  '.join(_op_columns(event))}".rstrip()
    if event.stats is not None:
        segment = render_stats_segment(event.stats)
        if segment:
            line = f"{line} | {segment}"
    return line


def _structural_line(event: NarrationEvent, indent: str) -> str | None:
    """Render a non-op structural event, or ``None`` for op/source rows."""

    if event.kind == "module_enter":
        return f"{indent}> {event.address or '?'} ({event.module_type or '?'})"
    if event.kind == "module_exit":
        ops = event.text or "0 ops"
        return f"{indent}< {event.address or '?'}  ({ops})"
    if event.kind == "attempted":
        return f"!! {event.text or ''}"
    if event.kind == "note":
        return event.text or ""
    return None


def _op_columns(event: NarrationEvent) -> list[str]:
    """Build the aligned positional + labeled columns for one op/source row."""

    columns: list[str] = [f"#{event.ordinal:<{_ORDINAL_WIDTH}}", f"{event.label:<{_LABEL_WIDTH}}"]
    tensor_tokens = [
        token
        for token in (
            format_shape(event.shape),
            event.dtype,
            event.device,
        )
        if token
    ]
    if tensor_tokens:
        columns.append(f"{' '.join(tensor_tokens):<{_TENSOR_WIDTH}}")
    if event.address:
        columns.append(f"@{event.address}")
    if event.source_location:
        columns.append(event.source_location)
    if event.pass_index > 1:
        columns.append(f"pass={event.pass_index}")
    if event.output_index:
        columns.append(f"out={event.output_index}")
    if event.late is not None:
        columns.append(f"late={event.late}")
    for key, value in event.extra_labels:
        columns.append(f"{key}={value}")
    if event.intervened:
        columns.append("[intervened]")
    return columns


__all__ = [
    "INDENT_UNIT",
    "compact_device",
    "compact_dtype",
    "format_shape",
    "render_line",
    "render_stats_segment",
]
