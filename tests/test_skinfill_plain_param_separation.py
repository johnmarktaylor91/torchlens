"""Plain against parameterised node fills in the shipped skins.

Readers of the dark and colorblind skins could not tell a plain operation
from a node that uses parameters: the plain fill sat 11.0 dE (dark) and
8.1 dE (colorblind, trainable) from the parameter fills, under the 12 dE
floor the colour checks use. Both skins now fill plain nodes with the page
colour. The separation is measured here with a stdlib implementation of the
same check: full-severity Machado (2009) colour-vision simulation, then
dE76 in CIELAB, minimised over normal, deutan, protan and tritan vision.

The other skins (default, paper, high_contrast) are pinned to their values
before the change so a later edit cannot move them unnoticed.
"""

from __future__ import annotations

import math

import pytest

from torchlens.visualization.themes import (
    LEGACY_SEMANTIC_PALETTE,
    OKABE_ITO_SEMANTIC_PALETTE,
    THEME_PRESETS,
)

#: Full-severity Machado (2009) simulation matrices, applied in linear sRGB.
_MACHADO = {
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}
_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_D65_WHITE = (0.95047, 1.0, 1.08883)
_FLOOR_DE = 12.0
_PARAM_ROLES = ("params_generic", "params_trainable", "params_frozen", "params_gradient")
_NAMED = {"white": "#FFFFFF", "black": "#000000"}


def _channels(color: str) -> tuple[float, float, float]:
    text = _NAMED.get(color.lower(), color).lstrip("#")
    red, green, blue = (int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4))
    return red, green, blue


def _to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _to_encoded(value: float) -> float:
    return 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055


def _lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    linear = [_to_linear(value) for value in rgb]
    xyz = [
        sum(row[col] * linear[col] for col in range(3)) / white
        for row, white in zip(_SRGB_TO_XYZ, _D65_WHITE, strict=True)
    ]
    f_x, f_y, f_z = (
        value ** (1 / 3) if value > 0.008856 else 7.787 * value + 16 / 116 for value in xyz
    )
    return 116 * f_y - 16, 500 * (f_x - f_y), 200 * (f_y - f_z)


def _simulate(
    rgb: tuple[float, float, float], matrix: tuple[tuple[float, ...], ...]
) -> tuple[float, float, float]:
    linear = [_to_linear(value) for value in rgb]
    mixed = [min(1.0, max(0.0, sum(row[col] * linear[col] for col in range(3)))) for row in matrix]
    red, green, blue = (_to_encoded(value) for value in mixed)
    return red, green, blue


def _min_delta_e(color_a: str, color_b: str) -> float:
    """Smallest dE76 between two colours over normal vision and the three simulations."""

    rgb_a, rgb_b = _channels(color_a), _channels(color_b)
    views = [(rgb_a, rgb_b)]
    views += [(_simulate(rgb_a, matrix), _simulate(rgb_b, matrix)) for matrix in _MACHADO.values()]
    return min(math.dist(_lab(view_a), _lab(view_b)) for view_a, view_b in views)


def _parameter_fills(skin: str, roles: tuple[str, ...]) -> list[tuple[str, str]]:
    palette = THEME_PRESETS[skin].semantic_palette
    return [(role, segment) for role in roles for segment in palette[role].split(":")]


def test_min_delta_e_reproduces_the_measured_base_pairs() -> None:
    """The implementation reproduces the base measurements the fix answers."""

    assert round(_min_delta_e("#1F2937", "#374151"), 1) == 11.0
    assert round(_min_delta_e("#F0F0F0", "#D9D9D9"), 1) == 8.1
    assert round(_min_delta_e("#F0F0F0", "#B0B0B0"), 1) == 23.0


@pytest.mark.parametrize(
    ("skin", "roles"),
    [
        ("dark", _PARAM_ROLES),
        # The colorblind generic-parameter fill is checked separately below.
        ("colorblind", ("params_trainable", "params_frozen", "params_gradient")),
    ],
)
def test_plain_fill_separates_from_parameter_fills(skin: str, roles: tuple[str, ...]) -> None:
    """Every plain-against-parameter pair (each gradient segment too) reaches 12 dE."""

    plain = THEME_PRESETS[skin].default_fill
    short = {
        f"{role}:{segment}": round(_min_delta_e(plain, segment), 1)
        for role, segment in _parameter_fills(skin, roles)
        if _min_delta_e(plain, segment) < _FLOOR_DE
    }
    assert not short, f"{skin}: plain {plain} under {_FLOOR_DE} dE against {short}"


def test_colorblind_generic_parameter_fill_keeps_its_measured_separation() -> None:
    """The generic-parameter grey stays at least 8.7 dE from the plain fill.

    It does not reach the 12 dE floor: no light neutral grey reaches 12 dE
    against a white plain fill without moving closer to the trainable or
    frozen grey. Pinned at its measured value so it cannot regress.
    """

    plain = THEME_PRESETS["colorblind"].default_fill
    generic = THEME_PRESETS["colorblind"].semantic_palette["params_generic"]
    assert round(_min_delta_e(plain, generic), 1) >= 8.7


def test_unchanged_skins_keep_their_palettes_and_plain_fills() -> None:
    """Default, paper and high_contrast keep their palettes and plain fills."""

    assert dict(LEGACY_SEMANTIC_PALETTE) == {
        "input": "#98FB98",
        "output": "#ff9999",
        "params_generic": "#E6E6E6",
        "params_trainable": "#D9D9D9",
        "params_frozen": "#B0B0B0",
        "params_gradient": "#D9D9D9:#B0B0B0",
        "boolean": "#F7D460",
        "buffer": "#E69F00",
        "intervention": "#CC79A7",
    }
    assert dict(OKABE_ITO_SEMANTIC_PALETTE) == {
        "input": "#009E73",
        "output": "#D55E00",
        "params_generic": "#CFE5F5",
        "params_trainable": "#56B4E9",
        "params_frozen": "#2E86C1",
        "params_gradient": "#56B4E9:#2E86C1",
        "boolean": "#F0E442",
        "buffer": "#E69F00",
        "intervention": "#CC79A7",
    }
    expected = {
        "torchlens": (OKABE_ITO_SEMANTIC_PALETTE, "white", "black", "black"),
        "paper": (LEGACY_SEMANTIC_PALETTE, "#F7F7F7", "#222222", "#111111"),
        "high_contrast": (LEGACY_SEMANTIC_PALETTE, "white", "black", "black"),
    }
    for skin, (palette, fill, border, font) in expected.items():
        theme = THEME_PRESETS[skin]
        assert theme.semantic_palette is palette, skin
        inks = (theme.default_fill, theme.default_border, theme.default_font)
        assert inks == (fill, border, font), skin


def test_changed_skins_keep_their_role_colours() -> None:
    """Only the plain fill moved: the dark and colorblind role colours are as before."""

    assert THEME_PRESETS["colorblind"].semantic_palette is LEGACY_SEMANTIC_PALETTE
    dark = THEME_PRESETS["dark"]
    assert {role: dark.semantic_palette[role] for role in _PARAM_ROLES} == {
        "params_generic": "#374151",
        "params_trainable": "#374151",
        "params_frozen": "#374151",
        "params_gradient": "#374151:#4B5563",
    }
    assert dark.default_fill == dark.graph["bgcolor"]
    colorblind = THEME_PRESETS["colorblind"]
    assert colorblind.default_fill == colorblind.graph["bgcolor"]
