"""Signature pins for the intervention and rerun doors and the backend ``save=`` keyword.

``do()`` takes its site and edit positionally on every door; ``Trace.run`` takes
``model`` and ``x`` positionally and keeps the transactional keywords off its
signature, which ``Trace.run_transaction`` owns; the non-torch backends'
``capture_trace`` spell the selection ``save=`` as ``tl.trace`` does.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import torchlens as tl
from torchlens._data_classes.layer import Layer
from torchlens._data_classes.op import Op
from torchlens._data_classes.trace import Trace
from torchlens._postprocess._selective_save import pop_static_label_save_predicate
from torchlens.errors import BackendUnsupportedError
from torchlens.recording._types import Recording

BACKENDS = Path(__file__).resolve().parents[1] / "torchlens" / "backends"
TRANSACTION_KEYWORDS = ("inputs", "seed", "fast", "carry_state", "until", "save", "on_divergence")


def _kinds(function: object) -> dict[str, inspect._ParameterKind]:
    return {
        name: parameter.kind
        for name, parameter in inspect.signature(function).parameters.items()
        if name != "self"
    }


@pytest.mark.parametrize(
    ("function", "positional"),
    [
        (Trace.do, ("site", "edit")),
        (Recording.do, ("site", "edit")),
        (Layer.do, ("edit",)),
        (Op.do, ("edit",)),
        (Trace.run, ("model", "x")),
    ],
)
def test_door_inputs_are_positional_only(function: object, positional: tuple[str, ...]) -> None:
    """The site, edit, model and input are positional-only; every other keyword is named."""

    kinds = _kinds(function)
    for name in positional:
        assert kinds[name] is inspect.Parameter.POSITIONAL_ONLY, name
    rest = [name for name in kinds if name not in positional]
    assert all(kinds[name] is inspect.Parameter.KEYWORD_ONLY for name in rest), kinds


def test_no_door_takes_the_old_edit_spellings() -> None:
    """``value_or_hook`` and ``input_transform`` are gone from the do() doors."""

    for function in (Trace.do, Recording.do, Layer.do, Op.do):
        assert not {"value_or_hook", "input_transform"} & set(_kinds(function))


def test_transactional_keywords_live_on_run_transaction_only() -> None:
    """``Trace.run`` takes none of the transactional keywords; ``run_transaction`` takes all."""

    run, transaction = _kinds(Trace.run), _kinds(Trace.run_transaction)
    assert not set(TRANSACTION_KEYWORDS) & set(run)
    assert set(TRANSACTION_KEYWORDS) <= set(transaction)
    assert all(kind is inspect.Parameter.KEYWORD_ONLY for kind in transaction.values())


def _capture_trace_parameters(backend: str) -> list[str]:
    tree = ast.parse((BACKENDS / backend / "backend.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "capture_trace":
            arguments = node.args
            return [arg.arg for arg in [*arguments.args, *arguments.kwonlyargs]]
    raise AssertionError(f"{backend}: no capture_trace")


@pytest.mark.parametrize("backend", ["jax", "mlx", "paddle", "tf", "tinygrad"])
def test_backend_capture_trace_spells_the_selection_save(backend: str) -> None:
    """Every non-torch ``capture_trace`` takes ``save=`` and no ``layers_to_save=``."""

    parameters = _capture_trace_parameters(backend)
    assert "save" in parameters
    assert "layers_to_save" not in parameters


@pytest.mark.parametrize("value", [None, "all"])
def test_static_save_predicate_keeps_everything_for_all(value: object) -> None:
    """Omitted or ``"all"`` means save everything: no predicate, and the key is consumed."""

    kwargs = {} if value is None else {"save": value}
    assert pop_static_label_save_predicate(kwargs, backend_name="test") is None
    assert "save" not in kwargs


def test_save_predicate_passes_selectors_and_callables_and_refuses_other_values() -> None:
    """Selectors and predicate callables pass through; any other value refuses typed."""

    selector = tl.func("relu")
    assert pop_static_label_save_predicate({"save": selector}, backend_name="test") is selector
    predicate = lambda op: True  # noqa: E731
    assert pop_static_label_save_predicate({"save": predicate}, backend_name="test") is predicate
    with pytest.raises(BackendUnsupportedError):
        pop_static_label_save_predicate({"save": 3}, backend_name="test")
