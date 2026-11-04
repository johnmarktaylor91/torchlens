"""Shipped skin palettes and plain fills (kept from the plain-parameter separation file).

Kept: shrink round 4 removed this file; the stack's Okabe-Ito default-skin work changed
the two palette pins below, so they stay. The separation measurements were removed.
"""

from __future__ import annotations

from torchlens._visualization.themes import (
    LEGACY_SEMANTIC_PALETTE,
    OKABE_ITO_SEMANTIC_PALETTE,
    THEME_PRESETS,
)

_PARAM_ROLES = ("params_generic", "params_trainable", "params_frozen", "params_gradient")


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
        "params_generic": "#DADADA",
        "params_trainable": "#C0C0C0",
        "params_frozen": "#808080",
        "params_gradient": "#C0C0C0:#808080",
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
    """Role colours as shipped: colorblind renders Okabe-Ito; dark weights are grey, frozen darker."""

    assert THEME_PRESETS["colorblind"].semantic_palette is OKABE_ITO_SEMANTIC_PALETTE
    dark = THEME_PRESETS["dark"]
    assert {role: dark.semantic_palette[role] for role in _PARAM_ROLES} == {
        "params_generic": "#DADADA",
        "params_trainable": "#DADADA",
        "params_frozen": "#808080",
        "params_gradient": "#DADADA:#808080",
    }
    assert dark.default_fill == dark.graph["bgcolor"]
    colorblind = THEME_PRESETS["colorblind"]
    assert colorblind.default_fill == colorblind.graph["bgcolor"]
