"""sRGB hex <-> OKLab conversion for the colour ramps (Ottosson 2020).

OKLab is a perceptual colour space: equal distances look like roughly equal
colour differences, and its ``L`` axis tracks perceived lightness. The ramp
interpolator (:func:`torchlens._visualization._encoding.interpolate_hex`)
blends stops here instead of in gamma-encoded sRGB, whose midpoints are
muddy and whose lightness steps are uneven.
"""

from __future__ import annotations

import math
from functools import lru_cache

OKLab = tuple[float, float, float]


def _to_linear(channel: int) -> float:
    """Decode one 0-255 sRGB channel to linear light in [0, 1]."""

    value = channel / 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _to_srgb(value: float) -> int:
    """Encode one linear-light channel to a clamped 0-255 sRGB integer."""

    value = max(0.0, min(1.0, value))
    encoded = 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055
    return round(max(0.0, min(1.0, encoded)) * 255)


def _cbrt(value: float) -> float:
    """Return the real cube root, sign kept."""

    return math.copysign(abs(value) ** (1 / 3), value)


@lru_cache(maxsize=256)
def hex_to_oklab(color: str) -> OKLab:
    """Convert ``#RRGGBB`` to OKLab ``(L, a, b)``.

    Parameters
    ----------
    color:
        Hex colour with a leading ``#``.

    Returns
    -------
    tuple[float, float, float]
        Lightness in [0, 1] and the two opponent axes.

    Examples
    --------
    >>> from torchlens._visualization._oklab import hex_to_oklab
    >>> round(hex_to_oklab("#FFFFFF")[0], 3)
    1.0
    """

    stripped = color.lstrip("#")
    red, green, blue = (_to_linear(int(stripped[i : i + 2], 16)) for i in (0, 2, 4))
    long_ = _cbrt(0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue)
    medium = _cbrt(0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue)
    short = _cbrt(0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue)
    return (
        0.2104542553 * long_ + 0.7936177850 * medium - 0.0040720468 * short,
        1.9779984951 * long_ - 2.4285922050 * medium + 0.4505937099 * short,
        0.0259040371 * long_ + 0.7827717662 * medium - 0.8086757660 * short,
    )


def oklab_to_hex(lab: OKLab) -> str:
    """Convert OKLab ``(L, a, b)`` to ``#RRGGBB``, clamping out-of-gamut channels.

    Parameters
    ----------
    lab:
        Lightness and the two opponent axes.

    Returns
    -------
    str
        Upper-case hex colour.

    Examples
    --------
    >>> from torchlens._visualization._oklab import hex_to_oklab, oklab_to_hex
    >>> oklab_to_hex(hex_to_oklab("#993404"))
    '#993404'
    """

    lightness, axis_a, axis_b = lab
    long_ = (lightness + 0.3963377774 * axis_a + 0.2158037573 * axis_b) ** 3
    medium = (lightness - 0.1055613458 * axis_a - 0.0638541728 * axis_b) ** 3
    short = (lightness - 0.0894841775 * axis_a - 1.2914855480 * axis_b) ** 3
    red = 4.0767416621 * long_ - 3.3077115913 * medium + 0.2309699292 * short
    green = -1.2684380046 * long_ + 2.6097574011 * medium - 0.3413193965 * short
    blue = -0.0041960863 * long_ - 0.7034186147 * medium + 1.7076147010 * short
    return f"#{_to_srgb(red):02X}{_to_srgb(green):02X}{_to_srgb(blue):02X}"


@lru_cache(maxsize=64)
def stop_positions(stops: tuple[str, ...]) -> tuple[float, ...]:
    """Place each stop at its share of the path's OKLab length.

    Placing stops by cumulative perceptual distance (not at even index
    steps) makes equal value steps read as equal colour steps along the
    whole ramp, even where the source list's own steps are uneven.

    Parameters
    ----------
    stops:
        Two or more hex colours, low value first.

    Returns
    -------
    tuple[float, ...]
        One position in [0, 1] per stop: 0 first, 1 last, non-decreasing.

    Examples
    --------
    >>> from torchlens._visualization._oklab import stop_positions
    >>> stop_positions(("#000000", "#FFFFFF"))
    (0.0, 1.0)
    """

    labs = [hex_to_oklab(stop) for stop in stops]
    lengths = [math.dist(low, high) for low, high in zip(labs, labs[1:], strict=False)]
    total = sum(lengths)
    if total == 0.0:
        return tuple(index / (len(stops) - 1) for index in range(len(stops)))
    positions = [0.0]
    for length in lengths:
        positions.append(positions[-1] + length / total)
    positions[-1] = 1.0
    return tuple(positions)
