"""Stats-line renderers: the core grammar over TensorStats (C02; lovely 4).

The core (fixed semantic order, memo 4.1):

``<dtype><shape>@<device>  <n= if rank>=2>  <IEC bytes if >=1KiB>
<distribution>  <health>  <evidence deviations>``

ASCII is CANONICAL; the unicode upgrade differs ONLY through the declared
glyph table (the 9-glyph ramp -- D3/D4), and CI asserts
``ascii == degrade(unicode)`` byte-for-byte. The precision law (D7/D22):
4 significant digits with trailing zeros KEPT, scientific outside
[1e-4, 1e4), exact integers never gain a decimal point, and a sampled
scalar prints only the digits its own standard error supports.

Pure formatters: nothing here touches a tensor.
"""

from __future__ import annotations

import math

from ._tensor_stats import TensorStats

#: The package glyph table IS the ramp and nothing else (D4): nine ASCII
#: glyphs and their unicode upgrades, plus nothing.
GLYPH_RAMP_ASCII = " .:-=+*#%"
GLYPH_RAMP_UNICODE = " ▁▂▃▄▅▆▇█"
#: unicode -> ascii degrade map (CI-checked byte-for-byte).
DEGRADE_TABLE = {
    unicode_glyph: ascii_glyph
    for unicode_glyph, ascii_glyph in zip(GLYPH_RAMP_UNICODE, GLYPH_RAMP_ASCII, strict=True)
    if unicode_glyph != ascii_glyph
}


def degrade(text: str) -> str:
    """Degrade a unicode render to its ASCII form through the glyph table."""

    return "".join(DEGRADE_TABLE.get(char, char) for char in text)


def format_sig(value: float, digits: int = 4) -> str:
    """Precision-law scalar formatter (D7).

    4 significant digits with trailing zeros KEPT (``%.4g`` strips them, so
    ``0.99997`` printed ``1`` and no column aligned -- the live bug this
    replaces); scientific notation outside ``[1e-4, 1e4)``; integral floats
    keep the digit budget rather than gaining a spurious wide tail.
    """

    if value != value or value in (float("inf"), float("-inf")):
        return str(value)
    if value == 0:
        return "0.000"[: digits + 1] if digits > 1 else "0"
    magnitude = abs(value)
    if magnitude < 1e-4 or magnitude >= 1e4:
        text = f"{value:.{max(digits - 1, 0)}e}"
        mantissa, exponent = text.split("e")
        return f"{mantissa}e{int(exponent):+03d}"
    decimals = digits - 1 - int(math.floor(math.log10(magnitude)))
    decimals = max(decimals, 0)
    return f"{value:.{decimals}f}"


def format_sampled(value: float, se: float | None, digits: int = 4) -> str:
    """Sampled-scalar precision law (D22).

    Bound form below ``2*se``; otherwise ``clamp(1, 4,
    floor(log10(|v|/se)))`` supported digits. The ``~`` method mark is the
    caller's (it discloses METHOD; this law bounds MAGNITUDE).
    """

    if se is None or se <= 0:
        return format_sig(value, digits)
    if abs(value) < 2 * se:
        return f"~0 (+-{format_sig(2 * se, 1)})"
    supported = int(math.floor(math.log10(abs(value) / se)))
    return format_sig(value, max(1, min(digits, supported)))


def format_count(count: int) -> str:
    """Compact human count (n=803k form; exact below 1000)."""

    if count < 1000:
        return str(count)
    for suffix, divisor in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if count >= divisor:
            scaled = count / divisor
            if scaled >= 100:
                return f"{scaled:.0f}{suffix}"
            return f"{scaled:.3g}{suffix}"
    return str(count)


def format_iec_bytes(nbytes: int) -> str:
    """IEC byte string (KiB/MiB/GiB), 3 significant digits."""

    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.3g} {unit}"
        value /= 1024
    return f"{int(nbytes)} B"


def sparkline(counts: tuple[int, ...], *, style: str = "ascii") -> str:
    """Render histogram counts through the declared ramp.

    An occupied bin never renders as the empty glyph: the lowest nonzero
    rank is the first ramp step (the end-bin repair upstream keeps
    extreme-proven bins nonzero).
    """

    ramp = GLYPH_RAMP_ASCII if style == "ascii" else GLYPH_RAMP_UNICODE
    peak = max(counts) if counts else 0
    if peak == 0:
        return ramp[0] * len(counts)
    glyphs = []
    for count in counts:
        if count == 0:
            glyphs.append(ramp[0])
            continue
        rank = 1 + int((len(ramp) - 2) * (count / peak))
        rank = min(rank, len(ramp) - 1)
        glyphs.append(ramp[rank])
    return "".join(glyphs)


def _zero_field(stats: TensorStats) -> str | None:
    """The D10 zero field: FORM is the control, never a threshold."""

    if stats.zero_count is None or stats.zero_count == 0 or stats.true_count is not None:
        return None
    if stats.all_zero:
        return None
    if stats.numel >= 1000 and stats.zero_count / stats.numel >= 0.01:
        return f"zero={100.0 * stats.zero_count / stats.numel:.3g}%"
    return f"zero={stats.zero_count}/{format_count(stats.numel)}"


def _health_fields(stats: TensorStats) -> list[str]:
    """NaN / +Inf / -Inf, separate, printed whenever nonzero (D11)."""

    fields = []
    for name, count in (
        ("nan", stats.nan_count),
        ("+inf", stats.posinf_count),
        ("-inf", stats.neginf_count),
    ):
        if count == 0:
            continue
        fraction = count / stats.numel
        if fraction >= 0.001:
            fields.append(f"{name}={100.0 * fraction:.3g}%!")
        else:
            fields.append(f"{name}={count}/{format_count(stats.numel)}!")
    return fields


def _distribution_zone(stats: TensorStats, style: str, *, spark: bool = True) -> list[str]:
    """The distribution zone: special cases, sparkline bracket, moments.

    ``spark=False`` is the width-degradation form: the bracket keeps the
    EXTREMA (never droppable, D14) and loses only the ramp glyphs.
    """

    integer_family = stats.dtype.startswith(("i8", "i16", "i32", "i64", "u8"))

    def value_token(value: float) -> str:
        """Format one value under the integer-family exactness rule."""

        # Exact integers never gain a decimal point (D7): "i64 constant=
        # 1.0000 is a number that cannot exist".
        if integer_family and float(value).is_integer():
            return str(int(value))
        return format_sig(value)

    special: str | None = None
    constant = stats.constant_value
    if stats.is_empty:
        special = "empty"
    elif stats.numel == 1 and stats.finite_min is not None:
        token = (
            str(int(stats.finite_min))
            if integer_family and float(stats.finite_min).is_integer()
            else format_sig(stats.finite_min, 5)
        )
        special = f"= {token}"
    elif stats.all_true:
        special = "all_true"
    elif stats.all_false:
        special = "all_false"
    elif stats.true_count is not None:
        special = f"true={stats.true_count}/{format_count(stats.numel)}"
    elif stats.no_finite_values:
        special = "no_finite_values"
    elif stats.all_zero:
        special = "all_zero"
    elif constant is not None and stats.sd == 0:
        special = f"constant={value_token(constant)}"
    if special is not None:
        return [special]
    parts: list[str] = []
    # Small-n raw values inline when they fit the ~40-column distribution
    # budget (D13): at that size the values ARE the summary; integer
    # families show ONLY the values (moments of 8 ints are noise dressed
    # as a distribution, but exact integer moments still print per D8/D12
    # when the values do not fit).
    if stats.small_values is not None:
        inline = "[" + " ".join(value_token(value) for value in stats.small_values) + "]"
        if len(inline) <= 40:
            parts.append(inline)
    if not parts:
        if (
            spark
            and stats.histogram_counts is not None
            and stats.finite_min is not None
            and stats.finite_max is not None
        ):
            mark = "~" if stats.histogram_evidence.sampled else ""
            parts.append(
                f"[{format_sig(stats.finite_min)} {mark}"
                f"{sparkline(stats.histogram_counts, style=style)} "
                f"{format_sig(stats.finite_max)}]"
            )
        elif stats.finite_min is not None and stats.finite_max is not None:
            parts.append(f"[{format_sig(stats.finite_min)} .. {format_sig(stats.finite_max)}]")
    label = "|mean|" if stats.magnitude_basis else "mean"
    sd_label = "|sd|" if stats.magnitude_basis else "sd"
    if stats.mean is not None:
        if stats.mean_evidence.sampled:
            parts.append(f"{label}=~{format_sampled(stats.mean, stats.mean_se)}")
        else:
            parts.append(f"{label}={format_sig(stats.mean)}")
    elif stats.mean_evidence.policy == "unavailable" and stats.mean_evidence.reason:
        parts.append(f"{label}=unavailable({stats.mean_evidence.reason})")
    if stats.sd is not None:
        if stats.sd_evidence.sampled:
            parts.append(f"{sd_label}=~{format_sampled(stats.sd, stats.sd_se)}")
        else:
            parts.append(f"{sd_label}={format_sig(stats.sd)}")
    return parts


def render_core_line(
    stats: TensorStats,
    *,
    style: str = "ascii",
    max_width: int | None = None,
) -> str:
    """Render the core stats line (memo 4.1 grammar).

    Width degradation (D14) drops bytes, then ``n=``, then the sparkline --
    never shape/dtype/extrema/health.
    """

    def assemble(include_bytes: bool, include_n: bool, include_spark: bool) -> str:
        """Assemble one candidate line at the given degradation rung."""

        head = f"{stats.dtype}[{','.join(str(dim) for dim in stats.shape)}]@{stats.device}"
        parts = [head]
        if include_n and len(stats.shape) >= 2:
            parts.append(f"n={format_count(stats.numel)}")
        if include_bytes and stats.nbytes is not None and stats.nbytes >= 1024:
            parts.append(f"({format_iec_bytes(stats.nbytes)})")
        parts.extend(_distribution_zone(stats, style, spark=include_spark))
        zero_field = _zero_field(stats)
        if zero_field:
            parts.append(zero_field)
        parts.extend(_health_fields(stats))
        if stats.sd_evidence.sampled and stats.sd_evidence.sample_size:
            parts.append(
                f"(sampled {format_count(stats.sd_evidence.sample_size)}/"
                f"{format_count(stats.numel)})"
            )
        if stats.role is not None:
            parts.append(f"role={stats.role}")
        return " ".join(parts)

    line = assemble(True, True, True)
    if max_width is None or len(line) <= max_width:
        return line
    for include_bytes, include_n, include_spark in (
        (False, True, True),
        (False, False, True),
        (False, False, False),
    ):
        line = assemble(include_bytes, include_n, include_spark)
        if len(line) <= max_width:
            return line
    return line
