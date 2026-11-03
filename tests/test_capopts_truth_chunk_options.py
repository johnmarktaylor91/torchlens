"""A06 capture-options truth: the chunked path drops NO capture option.

WALKTHROUGH list-A row 26 (second clause): ``chunk_size=`` silently dropped
``save_budget`` -- the chunk path rebuilds a recursive ``CaptureOptions``
from a hand-enumerated field list, so session knobs absent from that list
reset to their defaults on every chunked capture (the same include-list
disease the cache key had). The recursive constructor now covers every
``CaptureOptions`` field, and the AST tripwire here turns any future field
that is not consciously routed through the chunk path into a red test.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens._capture.prepare.recursive_options as prepare_recursive_options
from torchlens._save_budget import SaveBudgetExceededError
from torchlens.options import CaptureOptions


class SmallNet(nn.Module):
    """fc -> relu."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


_FIELD_GROUPS = ("_copied_capture_fields", "_routed_capture_fields")


def _literal_dict_keys(function: ast.FunctionDef) -> list[str]:
    """Return the string keys of the ``return {...}`` literal in ``function``, in order."""

    for statement in ast.walk(function):
        if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Dict):
            keys = statement.value.keys
            assert all(
                isinstance(key, ast.Constant) and isinstance(key.value, str) for key in keys
            ), f"{function.name} must name every field as a literal string key (no ** entries)"
            return [key.value for key in keys if isinstance(key, ast.Constant)]
    raise AssertionError(f"{function.name} has no `return {{...}}` literal")


def _recursive_capture_options_keywords() -> set[str]:
    """Extract the field names of the chunk path's CaptureOptions rebuild.

    The rebuild in ``_recursive_capture_options`` is exactly
    ``CaptureOptions(**_copied_capture_fields(...), **_routed_capture_fields(...))``;
    each group returns a dict literal naming its fields. A field named in both
    groups, or any other argument in the rebuild call, fails here.
    """

    tree = ast.parse(inspect.getsource(prepare_recursive_options))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    rebuild_calls = [
        node
        for node in ast.walk(functions["_recursive_capture_options"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CaptureOptions"
    ]
    assert len(rebuild_calls) == 1, (
        "the chunk path's CaptureOptions(...) rebuild was not found (once) in "
        "torchlens._capture.prepare.recursive_options._recursive_capture_options; if the chunk "
        "path was restructured, re-point this oracle at the new rebuild site"
    )
    (rebuild,) = rebuild_calls
    splatted = [
        keyword.value.func.id
        for keyword in rebuild.keywords
        if keyword.arg is None
        and isinstance(keyword.value, ast.Call)
        and isinstance(keyword.value.func, ast.Name)
    ]
    assert not rebuild.args and len(rebuild.keywords) == len(_FIELD_GROUPS), (
        "the rebuild call must hold exactly the two field-group splats and nothing else"
    )
    assert tuple(splatted) == _FIELD_GROUPS, (
        f"the rebuild call splats {splatted}, expected exactly {list(_FIELD_GROUPS)}"
    )
    groups = [_literal_dict_keys(functions[name]) for name in _FIELD_GROUPS]
    named = [field for group in groups for field in group]
    repeated = sorted({field for field in named if named.count(field) > 1})
    assert not repeated, f"fields named more than once across the field groups: {repeated}"
    return set(named)


def test_chunk_path_rebuild_covers_every_capture_option_field() -> None:
    """Set-difference oracle: a new CaptureOptions field cannot silently drop.

    The chunk path rebuilds the options object, so every field must be
    consciously routed (forwarded, resolved, or reset with a reason visible
    in the construction itself). A field missing from the rebuild resets to
    its default on chunked captures only -- exactly how ``save_budget`` was
    lost. Adding a CaptureOptions field makes this red until the chunk path
    names it.
    """

    field_names = {f.name for f in dataclasses.fields(CaptureOptions) if not f.name.startswith("_")}
    rebuilt = _recursive_capture_options_keywords()
    dropped = field_names - rebuilt
    assert not dropped, (
        f"CaptureOptions fields silently dropped by the chunked rebuild: "
        f"{sorted(dropped)}; route each through recursive_capture_options in "
        "torchlens/_capture/prepare/chunked.py (forward the resolved value, or reset it "
        "explicitly with a comment saying why)"
    )
    unknown = rebuilt - field_names
    assert not unknown, f"chunked rebuild names non-fields: {sorted(unknown)}"


def test_chunked_capture_honors_save_budget() -> None:
    """A 1-byte budget must trip on the chunked path exactly as unchunked."""

    x = torch.randn(8, 4)
    with pytest.raises(SaveBudgetExceededError):
        tl.trace(SmallNet(), x, capture=CaptureOptions(save_budget=1))
    with pytest.raises(SaveBudgetExceededError):
        tl.trace(SmallNet(), x, capture=CaptureOptions(save_budget=1, chunk_size=4))


@pytest.mark.smoke
def test_chunked_capture_carries_peak_memory_knob() -> None:
    """measure_python_peak_memory survives the chunked rebuild."""

    x = torch.randn(8, 4)
    log = tl.trace(
        SmallNet(),
        x,
        capture=CaptureOptions(
            measure_python_peak_memory=True, measure_host_peak_memory=True, chunk_size=4
        ),
    )
    peak = log.cost.forward_peak_memory
    assert peak is not None and peak > 0, (
        "measure_python_peak_memory=True on a chunked capture reported no "
        "tracemalloc peak; the knob was dropped by the recursive rebuild"
    )


def test_chunked_capture_carries_host_peak_memory_knob() -> None:
    """measure_host_peak_memory survives the chunked rebuild (measure.2)."""

    from torchlens._capture.peak_memory import CPU_PEAK_NOT_REQUESTED_REASON

    log = tl.trace(
        SmallNet(),
        torch.randn(8, 4),
        capture=CaptureOptions(measure_host_peak_memory=True, chunk_size=4),
    )
    assert log.cost.forward_peak_memory_reason != CPU_PEAK_NOT_REQUESTED_REASON, (
        "measure_host_peak_memory=True on a chunked capture read as not requested; "
        "the knob was dropped by the recursive rebuild"
    )
