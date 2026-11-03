"""``color_by=`` source detection when the attribution graph kit is absent."""

from __future__ import annotations

import importlib
import sys

from torchlens._visualization._color_sources import _is_attribution_source


class _Opaque:
    """Neither a string, a mapping, a callable nor a container."""


def test_missing_attribution_graph_kit_is_not_an_attribution_source(monkeypatch) -> None:
    """With ``torchlens.attribution.graph`` unimportable the probe answers False, not ImportError.

    The attribution package is imported first so that only the graph kit is
    missing: a ``None`` entry in ``sys.modules`` makes the guarded import raise
    ``ImportError`` while the package itself stays importable.
    """

    importlib.import_module("torchlens.attribution")
    monkeypatch.setitem(sys.modules, "torchlens.attribution.graph", None)
    assert _is_attribution_source(_Opaque()) is False


def test_present_attribution_graph_kit_still_rejects_opaque_sources() -> None:
    """The same opaque value is not a source when the kit imports."""

    importlib.import_module("torchlens.attribution.graph")
    assert _is_attribution_source(_Opaque()) is False
