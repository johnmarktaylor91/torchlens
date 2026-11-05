"""Accessors read torch submodules that are loaded and never import new ones.

On torch 2.1, importing ``torch._higher_order_ops.cond`` imports
``torch._dynamo``, and ``torch._subclasses.fake_tensor`` can be loaded while
``functional_tensor`` is not. TorchLens registered its witness HOP rules at
import through an importing lookup (so a plain ``import torchlens`` imported
Dynamo), and its tracing-tensor probe imported every candidate once one was
loaded (an import inside the first capture's RNG window), and its Dynamo
"is compiling" probe imported ``torch._dynamo`` mid-capture. These tests pin
the read-only behaviour and the later pickup of a module imported afterwards.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types
import warnings
from collections.abc import Sequence
from typing import Any

import pytest
import torch
from torch._ops import HigherOrderOperator

import torchlens._utils._torch_compat as tc
from torchlens.backends.torch import _completeness_hop as hop
from torchlens.backends.torch._modes import _TorchLensDispatchMode


class _ImportRecorder(importlib.abc.MetaPathFinder):
    """Meta-path finder that records lookups of the watched module names."""

    def __init__(self, watched: set[str]) -> None:
        """Record lookups of any name in ``watched``."""

        self.watched = watched
        self.seen: list[str] = []

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: types.ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        """Record a watched lookup and defer to the normal finders."""

        if fullname in self.watched:
            self.seen.append(fullname)
        return None


def _record_imports(monkeypatch: pytest.MonkeyPatch, *names: str) -> _ImportRecorder:
    """Install an import recorder for ``names`` for the rest of the test."""

    recorder = _ImportRecorder(set(names))
    monkeypatch.setattr(sys, "meta_path", [recorder, *sys.meta_path])
    return recorder


def _fake_module(monkeypatch: pytest.MonkeyPatch, name: str, attribute: str) -> type[Any]:
    """Load a module ``name`` holding a fresh class under ``attribute``."""

    module = types.ModuleType(name)
    cls = type(attribute, (), {})
    setattr(module, attribute, cls)
    monkeypatch.setitem(sys.modules, name, module)
    return cls


def test_tracing_types_never_import_a_candidate_that_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One loaded candidate is read; the other is neither imported nor latched out."""

    loaded, later = "_tl_test_tracing_loaded", "_tl_test_tracing_later"
    monkeypatch.setattr(tc, "_TRACING_TENSOR_CANDIDATES", ((loaded, "Fake"), (later, "Functional")))
    monkeypatch.setattr(tc, "_TRACING_TENSOR_TYPES", ())
    monkeypatch.setattr(tc, "_TRACING_TENSOR_TYPES_PROBED", False)
    monkeypatch.setattr(tc, "HAS_TRACING_TENSOR_TYPES", False)
    monkeypatch.delitem(sys.modules, later, raising=False)
    fake = _fake_module(monkeypatch, loaded, "Fake")
    recorder = _record_imports(monkeypatch, later)

    assert tc.get_tracing_tensor_types() == (fake,)
    assert recorder.seen == []
    assert tc._TRACING_TENSOR_TYPES_PROBED is False

    functional = _fake_module(monkeypatch, later, "Functional")
    assert tc.get_tracing_tensor_types() == (fake, functional)
    assert tc._TRACING_TENSOR_TYPES_PROBED is True


def test_higher_order_operator_lookup_never_imports_its_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator whose module is not loaded resolves to ``None`` without an import."""

    name = "torch._higher_order_ops.cond"
    monkeypatch.delitem(sys.modules, name, raising=False)
    recorder = _record_imports(monkeypatch, name)

    assert tc.get_higher_order_operator("cond", "cond_op") is None
    assert recorder.seen == []


class _ProbeOperator(HigherOrderOperator):
    """A higher-order operator private to this test, with an identity eager kernel."""

    def __init__(self) -> None:
        """Name the operator uniquely so it never collides with torch's own."""

        super().__init__("torchlens_test_lazy_probe")

    def __call__(self, *args: Any) -> Any:
        """Dispatch like any higher-order operator."""

        return super().__call__(*args)


def test_a_hop_rule_whose_module_loads_later_registers_on_mode_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration skips an unloaded operator and picks it up when the mode is entered."""

    operator = _ProbeOperator()
    if not hasattr(operator, "python_key_table"):
        # torch 2.1 spells the per-mode table python_key_mode_table and no witness
        # rule resolves there; give this private operator the later spelling.
        operator.python_key_table = operator.python_key_mode_table  # type: ignore[attr-defined]
    operator.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)(lambda x: x)
    module = types.ModuleType("torch._higher_order_ops._torchlens_test_probe")
    module.probe_op = operator  # type: ignore[attr-defined]
    monkeypatch.setattr(hop, "_DENSE_UNDER_MODE", (("_torchlens_test_probe", "probe_op"),))
    monkeypatch.delitem(sys.modules, module.__name__, raising=False)
    probe_mode = type("_ProbeMode", (_TorchLensDispatchMode,), {})

    assert "torchlens_test_lazy_probe" not in hop.register_witness_hop_rules(probe_mode)
    assert probe_mode not in operator.python_key_table
    assert probe_mode._hop_rules_pending is hop.register_witness_hop_rules

    monkeypatch.setitem(sys.modules, module.__name__, module)
    with probe_mode():
        pass
    assert probe_mode in operator.python_key_table


def test_dynamo_probe_never_imports_while_dynamo_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With Dynamo unloaded nothing compiles: ``False``, no import, no warning, no latch."""

    first, second = "_tl_test_compiler_probe", "_tl_test_dynamo_probe"
    monkeypatch.setattr(
        tc, "_DYNAMO_IS_COMPILING_CANDIDATES", ((first, "is_compiling"), (second, "is_compiling"))
    )
    monkeypatch.setattr(tc, "_DYNAMO_IS_COMPILING_FN", None)
    monkeypatch.setattr(tc, "_DYNAMO_IS_COMPILING_PROBED", False)
    monkeypatch.setattr(tc, "HAS_DYNAMO_IS_COMPILING", False)
    for name in (first, second, "torch._dynamo"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    recorder = _record_imports(monkeypatch, first, second)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert tc.dynamo_is_compiling() is False
    assert recorder.seen == []
    assert tc._DYNAMO_IS_COMPILING_PROBED is False

    module = types.ModuleType(first)
    module.is_compiling = lambda: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, first, module)
    assert tc.dynamo_is_compiling() is False
    assert tc.HAS_DYNAMO_IS_COMPILING is True and tc._DYNAMO_IS_COMPILING_PROBED is True
