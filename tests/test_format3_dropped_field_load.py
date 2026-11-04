"""A 3.0 load answers the dropped and flattened fields as the live trace does.

A 3.0 row stores no column for a ``DROP`` field, and the writer stores an empty
derived-gradient accessor as its member list ``[]``. The record state hooks fill
what the row leaves out, so ``module.cls`` reads ``None``, ``param.grads`` is an empty
gradient accessor, and ``trace._autograd_derived_grads`` is an empty derived-gradient
accessor, each matching the live trace (the surface oracle's tlspec stage guards the
same values across six models).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _format3_envelope import resave, save_tiny

import torchlens as tl
from torchlens._data_classes.derived_grad import DerivedGradAccessor
from torchlens._data_classes.op import GradientRecordAccessor

pytestmark = pytest.mark.heavy


def test_dropped_and_flattened_fields_load_as_live(tmp_path: Path) -> None:
    live = save_tiny(tmp_path / "first", "portable")
    loaded = tl.load(tmp_path / "first")

    assert isinstance(loaded._autograd_derived_grads, DerivedGradAccessor)
    assert len(loaded._autograd_derived_grads) == len(live._autograd_derived_grads) == 0

    live_modules = {module.address: module for module in live.modules}
    assert {module.address for module in loaded.modules} == set(live_modules)
    for module in loaded.modules:
        assert module.cls is None
        assert module.cls == live_modules[module.address].cls

    live_params = {param.address: param for param in live.params}
    assert live_params
    for param in loaded.params:
        assert isinstance(param.grads, GradientRecordAccessor)
        assert len(param.grads) == len(live_params[param.address].grads) == 0


def test_filled_fields_resave_the_same_cells(tmp_path: Path) -> None:
    save_tiny(tmp_path / "first", "portable")
    loaded = tl.load(tmp_path / "first")
    resave(loaded, tmp_path / "second", "portable")
    again = tl.load(tmp_path / "second")
    assert isinstance(again._autograd_derived_grads, DerivedGradAccessor)
    assert all(module.cls is None for module in again.modules)
    assert all(len(param.grads) == 0 for param in again.params)
