"""Sequential colour-ramp stop lists for the encoding channels.

A ramp is an ordered stop list, low value first; the colour channel
interpolates through it in OKLab at even perceptual speed
(:func:`torchlens._visualization._encoding.interpolate_hex`). Each family
carries one stop list per page: on a light page a high value is dark, on the
dark page a high value is bright, so the highest value is always the colour
farthest from the page.

The shipped family is :data:`DEFAULT_RAMP_FAMILY`; changing that one
constant switches every skin that uses the default ramp. Blue is allowed in a
data-encoded ramp; only the grey parameter nodes are kept free of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

#: ColorBrewer YlOrBr, 9 classes, light to dark.
YLORBR_9: tuple[str, ...] = (
    "#FFFFE5",
    "#FFF7BC",
    "#FEE391",
    "#FEC44F",
    "#FE9929",
    "#EC7014",
    "#CC4C02",
    "#993404",
    "#662506",
)

#: Viridis sampled at nine even steps, dark to light.
VIRIDIS_9: tuple[str, ...] = (
    "#440154",
    "#472D7B",
    "#3B528B",
    "#2C728E",
    "#21918C",
    "#28AE80",
    "#5EC962",
    "#ADDC30",
    "#FDE725",
)

#: Family -> (light-page stops, dark-page stops), each low value first. The
#: light YlOrBr list drops the 9-class list's first class: ``#FFFFE5`` sits
#: 0.035 OKLab from a white node, so the lowest encoded node would look
#: unencoded (the white-on-white absence defect). Viridis runs light to dark
#: on the light page, so a slow op is dark.
RAMP_FAMILIES: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = MappingProxyType(
    {
        "ylorbr": (YLORBR_9[1:], tuple(reversed(YLORBR_9))),
        "viridis": (tuple(reversed(VIRIDIS_9)), VIRIDIS_9),
    }
)

#: The shipped default family: ``"ylorbr"`` or ``"viridis"``.
DEFAULT_RAMP_FAMILY = "ylorbr"

LIGHT_SEQUENTIAL_RAMP, DARK_SEQUENTIAL_RAMP = RAMP_FAMILIES[DEFAULT_RAMP_FAMILY]
