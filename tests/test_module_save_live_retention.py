"""``save=tl.module(...)`` retains module outputs at module exit, not by per-op escrow.

A deferred selector made only of ``tl.module`` terms joined by ``|`` resolves to
module-output ops and nothing else. Escrowing every op's output for it (and
spilling the escrow to temporary files past the 64 MiB RAM budget) cost a
sparse trace more than saving everything. These tests pin three things:

1. the module-exit path copies exactly the selected module outputs and writes
   nothing to disk;
2. it saves the same ops, values and metadata as the per-op escrow path it
   replaces, and as an independent save-everything trace;
3. selectors that cannot settle at module exit keep the per-op escrow.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.capture.session as session_mod
import torchlens.ir.selector_eval as selector_eval


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = _Inner()
        self.lin2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.inner(x)) + x


class _TupleOut(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x * 2, x + 1


class _DictOut(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"a": torch.tanh(x), "b": x.sum(-1)}


class _Model(nn.Module):
    """Nested module, a module called twice, tuple and dict outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()
        self.shared = nn.Linear(4, 4)
        self.tup = _TupleOut()
        self.dct = _DictOut()
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        h = self.shared(h)
        h = self.shared(torch.sigmoid(h))
        a, b = self.tup(h)
        d = self.dct(a * b)
        return self.head(d["a"] + d["b"].unsqueeze(-1))


def _model_and_input() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return _Model().eval(), torch.randn(3, 4)


@contextmanager
def _counting(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
    """Count escrow copies and escrow spill writes during one capture."""

    counts = {"copies": 0, "spills": 0}
    real_copy = session_mod.safe_copy
    real_save = torch.save

    def counting_copy(*args: Any, **kwargs: Any) -> Any:
        counts["copies"] += 1
        return real_copy(*args, **kwargs)

    def counting_save(*args: Any, **kwargs: Any) -> Any:
        counts["spills"] += 1
        return real_save(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(session_mod, "safe_copy", counting_copy)
        patch.setattr(torch, "save", counting_save)
        yield counts


def _saved(log: Any) -> dict[str, Any]:
    return {op.layer_label: op for op in log.layer_list if op.has_saved_activation}


def _snapshot(log: Any) -> list[tuple[Any, ...]]:
    """Per-op metadata that a selective save must not change."""

    return [
        (
            op.layer_label,
            op.has_saved_activation,
            tuple(op.shape or ()),
            str(op.dtype),
            tuple(op.output_of_module_calls or ()),
            tuple(op.parents),
            tuple(op.children),
        )
        for op in log.layer_list
    ]


SELECTORS = {
    "nested": lambda: tl.module("block.inner"),
    "outer_with_nested_output": lambda: tl.module("block"),
    "called_twice": lambda: tl.module("shared"),
    "second_pass_only": lambda: tl.module("shared:2"),
    "tuple_output": lambda: tl.module("tup"),
    "dict_output": lambda: tl.module("dct"),
    "union_with_model_output_parent": lambda: tl.module("block.inner") | tl.module("head"),
    "union_three": lambda: tl.module("shared") | tl.module("dct") | tl.module("tup"),
}


@pytest.mark.parametrize("name", sorted(SELECTORS))
def test_module_save_copies_only_selected_outputs(name: str, monkeypatch: pytest.MonkeyPatch):
    """The selected module outputs are the only escrow copies; nothing spills."""

    model, x = _model_and_input()
    selector = SELECTORS[name]()
    rows = list(tl.trace(model, x).find_sites(selector))  # one row per layer pass
    expected = {site.layer_label for site in rows}
    assert expected, f"{name}: selector matched nothing on the full trace"
    # The model-output op is served from the live output tensor, never escrowed.
    model_output_rows = sum(site.layer_label.startswith("output") for site in rows)

    with _counting(monkeypatch) as counts:
        log = tl.trace(model, x, save=selector)

    assert set(_saved(log)) == expected
    assert len(rows) - model_output_rows <= counts["copies"] <= len(rows), (
        f"{name}: {counts['copies']} escrow copies for {len(rows)} selected layer passes "
        f"({len(log.layer_list)} ops in the trace)"
    )
    assert counts["spills"] == 0


@pytest.mark.parametrize("name", sorted(SELECTORS))
def test_module_save_matches_full_trace_values(name: str):
    """Independent oracle: every saved payload equals the save-everything trace's."""

    model, x = _model_and_input()
    selector = SELECTORS[name]()
    full = _saved(tl.trace(model, x))
    sparse = _saved(tl.trace(model, x, save=selector))
    for label, op in sparse.items():
        assert torch.equal(op.out, full[label].out), f"{name}: {label} payload differs"


@pytest.mark.parametrize("name", sorted(SELECTORS))
def test_module_exit_path_equals_per_op_escrow_path(name: str, monkeypatch: pytest.MonkeyPatch):
    """The module-exit path is a pure cost change over the per-op escrow it replaces."""

    model, x = _model_and_input()
    selector = SELECTORS[name]()
    live = tl.trace(model, x, save=selector)
    with monkeypatch.context() as patch:
        patch.setattr(selector_eval, "module_union_addresses", lambda _selector: None)
        escrowed = tl.trace(model, x, save=selector)

    assert _snapshot(live) == _snapshot(escrowed)
    assert getattr(live, "_tl_save_selector_fire_count", None) == getattr(
        escrowed, "_tl_save_selector_fire_count", None
    )
    escrowed_saved = _saved(escrowed)
    for label, op in _saved(live).items():
        assert torch.equal(op.out, escrowed_saved[label].out), f"{name}: {label} differs"


@pytest.mark.smoke
def test_module_save_with_module_intervention_keeps_replacement(monkeypatch):
    """The elicit spelling: save and steer the same module; the replacement is saved."""

    model, x = _model_and_input()
    site = tl.module("block")
    spec = tl.when(site, tl.scale(2.0))
    full = tl.trace(model, x, intervene=spec)
    expected = {s.layer_label: s for s in full.find_sites(site)}

    with _counting(monkeypatch) as counts:
        log = tl.trace(model, x, save=site | tl.module("head"), intervene=spec)

    saved = _saved(log)
    assert set(expected) <= set(saved)
    for label in expected:
        assert torch.equal(saved[label].out, full[label].out)
        assert saved[label].intervention_replaced
    assert counts["spills"] == 0
    assert counts["copies"] <= len(saved)


def test_forced_spill_writes_only_selected_outputs(monkeypatch):
    """Past the RAM budget only the selected module outputs reach temporary files."""

    from torchlens.capture.plan import CapturePlan

    real_compile = CapturePlan.compile.__func__

    def tiny_budget_compile(cls: Any, **kwargs: Any) -> Any:
        import dataclasses

        plan = real_compile(cls, **kwargs)
        return dataclasses.replace(
            plan,
            retention_profile=dataclasses.replace(
                plan.retention_profile, activation_ram_budget_bytes=1
            ),
        )

    monkeypatch.setattr(CapturePlan, "compile", classmethod(tiny_budget_compile))
    model, x = _model_and_input()
    selector = tl.module("shared")
    with _counting(monkeypatch) as counts:
        log = tl.trace(model, x, save=selector)
    assert counts["spills"] == 2, counts
    full = _saved(tl.trace(model, x))
    for label, op in _saved(log).items():
        assert torch.equal(op.out, full[label].out)


@pytest.mark.parametrize(
    "selector_factory",
    [
        pytest.param(lambda: tl.module("block") | tl.func("tanh"), id="module_or_func"),
        pytest.param(lambda: tl.module("block") & tl.func("add"), id="module_and_func"),
    ],
)
def test_mixed_module_selectors_keep_per_op_escrow(selector_factory, monkeypatch):
    """Selectors mixing tl.module with other terms still escrow every op, and stay exact."""

    model, x = _model_and_input()
    selector = selector_factory()
    full_log = tl.trace(model, x)
    expected = {site.layer_label for site in full_log.find_sites(selector)}
    with _counting(monkeypatch) as counts:
        log = tl.trace(model, x, save=selector)
    saved = _saved(log)
    assert set(saved) == expected
    assert counts["copies"] >= len(log.layer_list) - 2
    full = _saved(full_log)
    for label, op in saved.items():
        assert torch.equal(op.out, full[label].out)


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (tl.module("a"), ("a",)),
        (tl.module("a") | tl.module("b:2"), ("a", "b:2")),
        (tl.module("a") | tl.func("relu"), None),
        (tl.module("a") & tl.module("b"), None),
        (~tl.module("a"), None),
        (tl.func("relu"), None),
        (None, None),
    ],
)
def test_module_union_addresses_classification(selector, expected):
    """Only pure ``tl.module`` unions settle at module exit."""

    assert selector_eval.module_union_addresses(selector) == expected
