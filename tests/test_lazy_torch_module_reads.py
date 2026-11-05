"""Accessors read torch submodules that are loaded and never import new ones.

On torch 2.1, importing ``torch._higher_order_ops.cond`` imports
``torch._dynamo``, and ``torch._subclasses.fake_tensor`` can be loaded while
``functional_tensor`` is not. TorchLens registered its witness HOP rules at
import through an importing lookup (so a plain ``import torchlens`` imported
Dynamo), and its tracing-tensor probe imported every candidate once one was
loaded (an import inside the first capture's RNG window). These tests pin the
read-only behaviour and the later pickup of a module imported afterwards.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
from collections.abc import Sequence
from typing import Any

import pytest

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


def _resolvable_rule_module() -> str:
    """Return the module of a witness HOP rule that resolves once imported."""

    candidates = [("cond", "cond_op"), ("scan", "scan_op"), *hop._DENSE_UNDER_MODE]
    for module, attribute in candidates:
        try:
            importlib.import_module(f"torch._higher_order_ops.{module}")
        except ImportError:
            continue
        op = tc.get_higher_order_operator(module, attribute)
        if op is not None and any(rule_op is op for rule_op, _ in hop._witness_hop_rules()):
            return module
    raise AssertionError("no witness HOP rule resolves on this torch")


def test_a_hop_rule_whose_module_loads_later_registers_on_mode_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration skips an unloaded operator and picks it up when the mode is entered."""

    module = f"torch._higher_order_ops.{_resolvable_rule_module()}"
    loaded = sys.modules[module]
    probe_mode = type("_ProbeMode", (_TorchLensDispatchMode,), {})
    monkeypatch.delitem(sys.modules, module)
    hidden = {str(op.name()) for op, _ in hop._witness_hop_rules()}
    registered = set(hop.register_witness_hop_rules(probe_mode))

    assert registered == hidden
    assert probe_mode._hop_rules_pending is hop.register_witness_hop_rules

    monkeypatch.setitem(sys.modules, module, loaded)
    ops = [op for op, _ in hop._witness_hop_rules()]
    assert any(probe_mode not in op.python_key_table for op in ops)
    with probe_mode():
        pass
    assert all(probe_mode in op.python_key_table for op in ops)
