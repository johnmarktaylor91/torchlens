"""Round-5 security regression: load-time / run-time ``torch.from_file`` file-I/O.

``torch.from_file`` is a ``_VariableFunctionsClass`` factory that CREATES/TRUNCATES
an arbitrary file (``shared=True``) or READS an arbitrary file into a tensor
(``shared=False``). Its real ``__module__`` is plainly ``"torch"``, so the
module-granular callable-safety policy admitted it, and it was reachable through
BOTH untrusted surfaces:

* VECTOR 1 (unpickler): ``_safe_getattr`` admitted
  ``getattr(torch._C._VariableFunctionsClass, "from_file")`` for a following pickle
  ``REDUCE`` -- arbitrary file creation/read DURING ``tl.load()`` with no trust
  opt-in.
* VECTOR 2 (resolver): ``is_pure_forward_callable(torch.from_file)`` returned
  ``True``, so a bundle op keyed ``("torch", "from_file")`` resolved and was CALLED
  at ``Trace.run()`` with recorded literal args.

The fix closes the CLASS: a qualname-level guard (audited denylist + structural
file/import/serialization name guard) layered on the module gate, consulted by the
unpickler ``_safe_getattr`` AND the shared ``is_pure_forward_callable`` policy the
resolvers use. These tests pin both vectors shut, prove the audited siblings are
denied, prove genuine pure forward ops still resolve/run/VERIFY, and prove all five
save levels still load.
"""

from __future__ import annotations

import os
import pickle

import pytest
import torch
from torch import nn

from torchlens._io._safe_unpickle import _safe_getattr
from torchlens._utils._callable_safety import is_pure_forward_callable
from torchlens.intervention._resolver_registry import resolve_function_registry_key
from torchlens.intervention._types import FunctionRegistryKey
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    replayable=True,
    capture_container_structure=True,
    cache=False,
)


# --------------------------------------------------------------------------- #
# Hand-written pickle opcodes reproducing the exact exploit stream.
# --------------------------------------------------------------------------- #


def test_safe_getattr_denies_from_file() -> None:
    """The getattr wrapper refuses the file-I/O builtin but keeps pure refs."""

    with pytest.raises(pickle.UnpicklingError):
        _safe_getattr(torch._C._VariableFunctionsClass, "from_file")
    # Pure factory / method references off the same holder still resolve.
    assert callable(_safe_getattr(torch._C._VariableFunctionsClass, "from_numpy"))
    assert callable(_safe_getattr(torch._C._VariableFunctionsClass, "frombuffer"))
    assert callable(_safe_getattr(torch._C.TensorBase, "sum"))


# --------------------------------------------------------------------------- #
# VECTOR 2: resolver refuses from_file at run() time (both key shapes).
# --------------------------------------------------------------------------- #


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "key",
    [
        FunctionRegistryKey(
            "custom",
            "from_file",
            "function",
            import_path="torch._C._VariableFunctionsClass:from_file",
        ),
    ],
    ids=["internal_builtin"],
)
def test_resolver_denies_from_file(key: FunctionRegistryKey) -> None:
    """torch.from_file is refused with a typed security error at both key shapes."""

    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key)


# --------------------------------------------------------------------------- #
# Close the CLASS: the audited side-effecting siblings are all denied.
# --------------------------------------------------------------------------- #


def _audited_dangerous_callables() -> list:
    """Return the audited side-effecting callables reachable in the allowed surface."""

    candidates = [
        torch.from_file,
        torch.import_ir_module,
        torch.import_ir_module_from_buffer,
        torch.PyTorchFileReader,
        torch.PyTorchFileWriter,
        torch.FileCheck,
        torch._load_global_deps,
        torch._preload_cuda_deps,
        torch._import_device_backends,
        torch._utils._import_dotted_name,
        torch.get_file_path,
        # Prior denials that must STAY closed.
        torch.load,
        torch.save,
        torch.jit.load,
        torch.jit.save,
        os.system,
        eval,
        pickle.loads,
        pickle.load,
    ]
    return [c for c in candidates if c is not None]


def test_audited_side_effecting_callables_denied() -> None:
    """Every audited file-I/O / serialization / import / process callable is denied."""

    for func in _audited_dangerous_callables():
        assert is_pure_forward_callable(func) is False, func


# --------------------------------------------------------------------------- #
# End-to-end: repointing a runnable op to torch.from_file must not touch a file.
# --------------------------------------------------------------------------- #


class _TinyLinear(nn.Module):
    """Minimal parameterized graph whose forward has a relu op to repoint."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(value))


def _relu_registry_index(run: dict) -> int:
    for index, entry in enumerate(run["callable_registry"]):
        if entry["key"]["qualname"] == "relu":
            return index
    raise AssertionError("expected a relu op in the tiny model")


# --------------------------------------------------------------------------- #
# Regression: every save level still loads; genuine models still VERIFY.
# --------------------------------------------------------------------------- #


class _ControlFlow(nn.Module):
    """Control-flow graph; a portable save embeds method refs via getattr."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.sum() > 0:
            return torch.relu(value) * 2
        return torch.sigmoid(value) - 1


class _StatefulLinear(nn.Module):
    """Runnable-eligible parameterized graph with a persistent buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(value)) * self.scale


class _TinyCNN(nn.Module):
    """Small convolutional network exercising pure forward ops end to end."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8, 10)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.relu(self.c1(value))
        value = torch.nn.functional.adaptive_avg_pool2d(value, 1).flatten(1)
        return self.fc(value)
