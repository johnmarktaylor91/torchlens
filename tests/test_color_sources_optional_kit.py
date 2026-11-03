"""``color_by=`` source detection when the attribution graph kit is absent."""

from __future__ import annotations

import importlib

from torchlens._visualization._color_sources import _is_attribution_source


class _Opaque:
    """Neither a string, a mapping, a callable nor a container."""


def test_present_attribution_graph_kit_still_rejects_opaque_sources() -> None:
    """The same opaque value is not a source when the kit imports."""

    importlib.import_module("torchlens.attribution.graph")
    assert _is_attribution_source(_Opaque()) is False
