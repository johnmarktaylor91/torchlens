"""Saved callable keys: the load-time verifier admits exactly what the replay resolver admits.

A trace saved with an intervention recipe lists every captured op's function registry key
in ``intervention.json``; loading it resolves each key through
``resolve_function_registry_key``. The sparse-run resolver in ``torchlens._io.runnable_load``
admits keys on the same allowlisted roots and enumerated torch namespaces
(``torchlens._utils._torch_symbols``). These tests pin that both read one table:

* a trace holding a ``torch.linalg`` op round-trips through the saved-format writer, loads and
  reruns equal to the live intervened trace;
* every key the two resolvers see agrees: admitted by both or refused by both;
* keys outside the purity-gated set still refuse with the existing typed errors.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from _format3_envelope import redigest
from torch import nn

import torchlens as tl
from torchlens._io import runnable_load
from torchlens._utils._callable_safety import is_pure_forward_callable
from torchlens._utils._torch_symbols import (
    ALLOWED_EXACT_ROOTS,
    ENUMERATED_TORCH_NAMESPACES,
    torch_namespace,
)
from torchlens.intervention._resolver_registry import resolve_function_registry_key
from torchlens.intervention._types import FunctionRegistryKey
from torchlens.intervention.errors import ReplayPreconditionError, UntrustedCallableError

_RECIPE_FILE = "intervention.json"


class _NormModel(nn.Module):
    """Linear map, relu, then a ``torch.linalg`` norm: the op the loader used to refuse."""

    def __init__(self) -> None:
        """Build the layer with fixed weights."""

        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the relu activation divided by its row norm."""

        h = torch.relu(self.lin(x)) + 1.0
        return h / torch.linalg.vector_norm(h, dim=-1, keepdim=True)


def _saved_trace(tmp_path: Path) -> tuple[Path, nn.Module, torch.Tensor]:
    """Save a scale-edited trace of ``_NormModel`` through the saved-format writer."""

    model = _NormModel().eval()
    value = torch.randn(2, 4, generator=torch.Generator().manual_seed(1))
    trace = tl.trace(model, value, intervene=tl.when(tl.func("relu"), tl.scale(0.5)))
    path = tmp_path / "norm"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trace.save(path, level="portable")
    return path, model, value


def _recipe_keys(path: Path) -> list[dict[str, Any]]:
    """Return the saved recipe's function registry key entries."""

    recipe = json.loads((path / _RECIPE_FILE).read_text(encoding="utf-8"))
    return list(recipe["function_registry_keys"])


def _with_extra_key(path: Path, key: dict[str, Any]) -> None:
    """Append one crafted key entry to the saved recipe and re-list it in the manifest."""

    recipe_path = path / _RECIPE_FILE
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    entries = recipe["function_registry_keys"]
    entries.append({**entries[-1], "key": key})
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    redigest(path, _RECIPE_FILE)


def test_linalg_recipe_round_trip_loads_and_reruns_equal(tmp_path: Path) -> None:
    """A recipe naming ``torch.linalg.vector_norm`` loads and reruns to the live output."""

    path, model, value = _saved_trace(tmp_path)
    namespaces = {entry["key"]["namespace"] for entry in _recipe_keys(path)}
    assert "torch.linalg" in namespaces, sorted(namespaces)

    loaded = tl.load(path)
    live = tl.trace(model, value, intervene=tl.when(tl.func("relu"), tl.scale(0.5)))
    rerun = loaded.run(model, value)
    got = [op.out for op in rerun.output_ops]
    want = [op.out for op in live.output_ops]
    assert len(got) == len(want)
    for left, right in zip(got, want):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def _replay_admits(key: FunctionRegistryKey) -> bool:
    """Return whether the sparse-run resolver's exact rung plus purity gate admits ``key``."""

    hit = runnable_load._resolve_exact_key(key, runnable_load._stock_path_from_key(key))
    if hit is None:
        return False
    return is_pure_forward_callable(runnable_load._unwrap_decorated(hit[0]))


def _verifier_admits(key: FunctionRegistryKey) -> bool:
    """Return whether the load-time resolver admits ``key``."""

    try:
        resolve_function_registry_key(key)
    except (ReplayPreconditionError, UntrustedCallableError):
        return False
    return True


def _parity_keys() -> list[FunctionRegistryKey]:
    """Keys on every enumerated namespace, plus crafted names that must refuse."""

    keys = []
    for namespace in sorted(ENUMERATED_TORCH_NAMESPACES - set(ALLOWED_EXACT_ROOTS)):
        root = torch_namespace(namespace)
        names = sorted(name for name in dir(root) if not name.startswith("__"))[:60]
        for name in [*names, "no_such_callable", "__class__", "_add_docstr"]:
            keys.append(FunctionRegistryKey(namespace, name, "function"))  # type: ignore[arg-type]
    for qualname in ("linalg.vector_norm", "fft.fft", "load", "save", "from_file", "onnx"):
        keys.append(FunctionRegistryKey("torch", qualname, "function"))
    return keys


def test_verifier_and_replay_resolver_admit_the_same_keys() -> None:
    """For every probed key, load-time verification and replay resolution agree."""

    keys = _parity_keys()
    disagreements = [
        (f"{key.namespace}.{key.qualname}", _verifier_admits(key), _replay_admits(key))
        for key in keys
        if _verifier_admits(key) != _replay_admits(key)
    ]
    assert not disagreements, disagreements[:10]
    linalg = FunctionRegistryKey("torch.linalg", "vector_norm", "function")
    assert _verifier_admits(linalg)


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(FunctionRegistryKey("os", "system", "function"), id="os-namespace"),  # type: ignore[arg-type]
        pytest.param(FunctionRegistryKey("builtins", "eval", "function"), id="builtins-namespace"),  # type: ignore[arg-type]
        pytest.param(
            FunctionRegistryKey("torch.linalg", "no_such_norm", "function"), id="linalg-missing"
        ),
        pytest.param(
            FunctionRegistryKey("torch.linalg", "os.system", "function"), id="linalg-dotted-walk"
        ),
        pytest.param(
            FunctionRegistryKey("torch.os", "system", "function"), id="torch-unlisted-namespace"
        ),  # type: ignore[arg-type]
    ],
)
def test_crafted_unresolvable_key_refuses_load_typed(
    tmp_path: Path, key: FunctionRegistryKey
) -> None:
    """An artifact naming a key outside the shared roots refuses at load, typed."""

    path, _model, _value = _saved_trace(tmp_path)
    _with_extra_key(
        path,
        {
            "namespace": key.namespace,
            "qualname": key.qualname,
            "dispatch_kind": "function",
            "version": 1,
            "import_path": None,
        },
    )
    with pytest.raises(ReplayPreconditionError):
        tl.load(path)


@pytest.mark.parametrize(
    ("key", "code"),
    [
        pytest.param(
            FunctionRegistryKey("custom", "system", "function", import_path="os:system"),
            "custom_callable_module_denied",
            id="os-system",
        ),
        pytest.param(
            FunctionRegistryKey("custom", "eval", "function", import_path="builtins:eval"),
            "custom_callable_module_denied",
            id="builtins-eval",
        ),
        pytest.param(
            FunctionRegistryKey("torch", "load", "function"),
            "custom_callable_not_pure",
            id="torch-load",
        ),
        pytest.param(
            FunctionRegistryKey("torch", "from_file", "function"),
            "custom_callable_not_pure",
            id="torch-from-file",
        ),
    ],
)
def test_gate_refused_key_still_refuses_resolution_typed(
    key: FunctionRegistryKey, code: str
) -> None:
    """Keys the purity or module gate refuses keep their existing typed refusal."""

    with pytest.raises(UntrustedCallableError) as info:
        resolve_function_registry_key(key)
    assert info.value.fields.get("code") == code
