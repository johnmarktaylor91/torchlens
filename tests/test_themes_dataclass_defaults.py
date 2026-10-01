"""Regression pins for dataclass field defaults that Python 3.11 rejects.

Python 3.11 refuses any dataclass field default whose class has no ``__hash__``
(``ValueError: mutable default ... use default_factory``) at class creation.
A ``MappingProxyType`` default passes on 3.10 (only list/dict/set are checked)
and on 3.12+ (mappingproxy gained ``__hash__``), so the defect was invisible
outside 3.11, where it broke collection of every module importing the themes.
"""

from __future__ import annotations

import dataclasses
import sys
from types import MappingProxyType

import pytest

import torchlens  # noqa: F401  (populates sys.modules with the eager spine)
from torchlens.visualization import themes

pytestmark = pytest.mark.smoke


def _fails_py311_default_rule(default: object) -> bool:
    """Return whether Python 3.11 would reject ``default`` as a dataclass default.

    3.11 rejects any default whose class has no ``__hash__``. ``MappingProxyType``
    gained ``__hash__`` only in 3.12, so it is named explicitly: on 3.12+ the
    generic check alone would not see it.
    """

    return type(default).__hash__ is None or isinstance(default, MappingProxyType)


def test_semantic_palette_default_is_the_shared_read_only_mapping() -> None:
    """The default keeps the legacy palette's value, identity, and immutability."""

    theme = themes.VisualizationTheme(
        name="probe",
        graph={},
        node={},
        edge={},
        default_fill="white",
        default_border="black",
        default_font="black",
    )
    assert theme.semantic_palette is themes.LEGACY_SEMANTIC_PALETTE
    assert isinstance(theme.semantic_palette, MappingProxyType)
    with pytest.raises(TypeError):
        theme.semantic_palette["op"] = "#000000"  # type: ignore[index]


def test_loaded_torchlens_dataclass_defaults_pass_the_py311_hash_rule() -> None:
    """Every loaded TorchLens dataclass default satisfies Python 3.11's rule.

    The rule is applied explicitly so the check is red on every interpreter,
    not only on 3.11 where class creation itself would fail.
    """

    offenders: list[str] = []
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("torchlens") or module is None:
            continue
        for obj in list(vars(module).values()):
            if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
                continue
            if obj.__module__ != module_name:
                continue
            for item in dataclasses.fields(obj):
                default = item.default
                if default is dataclasses.MISSING:
                    continue
                if _fails_py311_default_rule(default):
                    offenders.append(f"{module_name}.{obj.__qualname__}.{item.name}")
    assert not offenders, (
        "dataclass defaults with an unhashable class fail at class creation on "
        f"Python 3.11; use field(default_factory=...): {sorted(offenders)}"
    )


def test_py311_default_rule_detects_planted_defaults() -> None:
    """The rule flags the defaults 3.11 refuses and passes hashable ones."""

    assert _fails_py311_default_rule(MappingProxyType({"op": "#000000"}))
    assert _fails_py311_default_rule([])
    assert _fails_py311_default_rule({})
    assert not _fails_py311_default_rule("#000000")
    assert not _fails_py311_default_rule(None)
    assert not _fails_py311_default_rule(("a", "b"))
