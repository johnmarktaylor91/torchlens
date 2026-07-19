"""Round-6 security regression: state-mutator + storage-unsafe callable denial.

Round-5 closed the FILE-I/O sub-class of side-effecting torch callables reachable
from an untrusted ``.tlspec`` bundle. This round closes the PROCESS-STATE-MUTATION
+ STORAGE-UNSAFE sub-class. Their real ``__module__`` is the allowlisted torch
namespace (``torch`` / ``torch.random`` / ``torch.autograd.grad_mode`` /
``torch._tensor_str``) or ``None`` (C tensor methods), so the module-granular
callable-safety policy admitted them, and they were reachable through BOTH
untrusted surfaces:

* ``is_pure_forward_callable(torch.set_default_dtype)`` returned ``True``, so a
  bundle op keyed ``("torch", "set_default_dtype")`` resolved and was CALLED at
  ``Trace.run()`` with recorded literal args -- a PERSISTENT global-state flip
  that outlives ``run()``.
* ``Tensor.resize_`` exposes UNINITIALIZED heap memory as a trace-output tensor
  (info-leak); ``Tensor.set_`` / ``set_source_*`` repoint storage.

Impact ceiling is MEDIUM: host-process state corruption + memory disclosure, NOT
RCE/file/exec. The fix extends the qualname-level guard (exact state-mutator +
storage-unsafe denylist + a leading-``set_``/``_set_`` structural prefix guard)
consulted by the unpickler ``_safe_getattr`` AND the shared
``is_pure_forward_callable`` policy the resolvers use.

These tests pin the mutator + storage-unsafe family shut at BOTH key shapes,
prove NO global state is flipped, prove ordinary in-place ELEMENTWISE forward ops
still resolve/run/VERIFY, and prove the round-5 file-I/O denials + all prior
denials stay closed.
"""

from __future__ import annotations

import operator
import os
import pickle
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io._safe_unpickle import _safe_getattr
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.resolver import resolve_function_registry_key
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness
from torchlens.utils._callable_safety import is_pure_forward_callable

_CAP = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


# --------------------------------------------------------------------------- #
# The confirmed state-mutator + storage-unsafe family is UNRESOLVABLE.
# --------------------------------------------------------------------------- #


def _state_mutator_callables() -> list:
    """Return the audited process-global-state mutators reachable in torch."""

    candidates = [
        torch.manual_seed,
        torch.seed,
        torch.set_rng_state,
        torch.set_default_dtype,
        torch.set_default_device,
        torch.set_default_tensor_type,
        torch.set_num_threads,
        torch.set_num_interop_threads,
        torch.set_grad_enabled,
        torch.set_deterministic_debug_mode,
        torch.use_deterministic_algorithms,
        torch.set_flush_denormal,
        torch.set_anomaly_enabled,
        torch.set_printoptions,
        torch.set_float32_matmul_precision,
        torch.set_warn_always,
        # Autocast setters (structural set_* prefix; not individually enumerated).
        torch.set_autocast_enabled,
        torch.set_autocast_cache_enabled,
        # Private torch._C setters (structural _set_* prefix).
        torch._C._set_grad_enabled,
        torch._C._set_cudnn_enabled,
    ]
    return [c for c in candidates if c is not None]


def _storage_unsafe_callables() -> list:
    """Return the audited storage-unsafe in-place tensor ops."""

    candidates = [
        torch.Tensor.set_,
        torch.Tensor.resize_,
        torch.Tensor.resize_as_,
        getattr(torch.Tensor, "resize_as_sparse_", None),
        getattr(torch.Tensor, "sparse_resize_", None),
        getattr(torch.Tensor, "sparse_resize_and_clear_", None),
    ]
    return [c for c in candidates if c is not None]


def test_state_mutators_denied_by_policy() -> None:
    """Every audited process-global-state mutator is refused by the shared policy."""

    for func in _state_mutator_callables():
        assert is_pure_forward_callable(func) is False, func


def test_storage_unsafe_ops_denied_by_policy() -> None:
    """Every audited storage-unsafe in-place op is refused by the shared policy."""

    for func in _storage_unsafe_callables():
        assert is_pure_forward_callable(func) is False, func


@pytest.mark.smoke
@pytest.mark.parametrize(
    "key",
    [
        FunctionRegistryKey("torch", "set_default_dtype", "function"),
        FunctionRegistryKey("torch", "manual_seed", "function"),
        FunctionRegistryKey("torch", "set_num_threads", "function"),
        FunctionRegistryKey("torch", "use_deterministic_algorithms", "function"),
        FunctionRegistryKey("torch.Tensor", "resize_", "method"),
        FunctionRegistryKey("torch.Tensor", "set_", "method"),
    ],
    ids=[
        "set_default_dtype",
        "manual_seed",
        "set_num_threads",
        "use_deterministic_algorithms",
        "resize_",
        "set_",
    ],
)
def test_resolver_denies_state_mutators(key: FunctionRegistryKey) -> None:
    """The state-mutator + storage-unsafe family is refused via the fixed-root gate.

    These fixed-root (``torch`` / ``torch.Tensor``) keys are the real run()-time
    exploit vector: they resolve by ``getattr`` and are gated ONLY by
    ``is_pure_forward_callable`` -- so a raised ``UntrustedCallableError`` proves
    the tightened policy, not some unrelated foreign-import gate.
    """

    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(key)


@pytest.mark.parametrize("qualname", ["add_", "mul_", "clamp_", "relu_", "copy_"])
def test_resolver_admits_inplace_elementwise_ops(qualname: str) -> None:
    """The SAME fixed-root Tensor path still resolves ordinary in-place ops."""

    resolved = resolve_function_registry_key(
        FunctionRegistryKey("torch.Tensor", qualname, "method")
    )
    assert callable(resolved)


def test_unpickler_denies_state_mutators_and_storage_unsafe() -> None:
    """VECTOR (unpickler): _safe_getattr refuses storage-unsafe ops, keeps pure ops.

    ``torch._C.TensorBase`` is an admitted getattr holder, so a pickle stream can
    reach ``resize_`` / ``set_`` off it; the shared policy must fail them closed
    while ordinary in-place elementwise refs still resolve.
    """

    tensor_base = torch._C.TensorBase
    for name in ("resize_", "resize_as_", "set_"):
        with pytest.raises(pickle.UnpicklingError):
            _safe_getattr(tensor_base, name)
    for name in ("add_", "mul_", "clamp_", "relu_", "copy_", "sum"):
        assert callable(_safe_getattr(tensor_base, name))


def test_resolving_set_default_dtype_does_not_flip_global_state() -> None:
    """A denied resolve must NOT persistently flip the process-global default dtype."""

    before = torch.get_default_dtype()
    with pytest.raises(UntrustedCallableError):
        resolve_function_registry_key(FunctionRegistryKey("torch", "set_default_dtype", "function"))
    assert torch.get_default_dtype() is before


# --------------------------------------------------------------------------- #
# Legitimate in-place ELEMENTWISE forward ops (and getters) STILL resolve.
# --------------------------------------------------------------------------- #


def test_inplace_elementwise_ops_still_admitted() -> None:
    """Ordinary in-place elementwise ops -- trailing underscore -- stay resolvable."""

    for func in (
        torch.Tensor.add_,
        torch.Tensor.mul_,
        torch.Tensor.sub_,
        torch.Tensor.div_,
        torch.Tensor.clamp_,
        torch.Tensor.relu_,
        torch.Tensor.sigmoid_,
        torch.Tensor.tanh_,
        torch.Tensor.copy_,
        torch.Tensor.zero_,
        torch.Tensor.fill_,
        torch.Tensor.normal_,
        torch.Tensor.uniform_,
        torch.Tensor.addcmul_,
        torch.Tensor.neg_,
        torch.Tensor.abs_,
    ):
        assert is_pure_forward_callable(func) is True, func


def test_pure_getters_and_factories_still_admitted() -> None:
    """Recognized reads / factories / ops stay resolvable.

    ``initial_seed`` / ``get_rng_state`` carry an aten operator schema, the factories and
    forward ops are overridable/aten, and ``operator.add`` is on the operator allowlist --
    all recognized operators, so the r43 structural inversion keeps them admitted.
    """

    for func in (
        torch.initial_seed,
        torch.get_rng_state,
        torch.from_numpy,
        torch.frombuffer,
        torch.matmul,
        torch.conv2d,
        torch.relu,
        operator.add,
    ):
        assert is_pure_forward_callable(func) is True, func


def test_nonoperator_config_getters_denied_under_r43() -> None:
    """r43: pure CONFIG getters that are NOT recognized operators are now DEFAULT-DENIED.

    These ``torch``-root reads (``get_default_dtype`` / ``get_num_threads`` /
    ``get_default_device`` / ``are_deterministic_algorithms_enabled`` /
    ``get_num_interop_threads``) are neither torch-overridable nor aten-schema ops and
    never appear as nodes in a captured forward DAG, so the r43 structural
    recognized-operator inversion refuses them by default (harmless tightening: they are
    pure reads, but default-deny is the point). Pre-r43 they slipped the module-prefix
    admission because their names dodged the mutator verb guard.
    """

    for func in (
        torch.get_default_dtype,
        torch.get_num_threads,
        torch.get_default_device,
        torch.are_deterministic_algorithms_enabled,
        torch.get_num_interop_threads,
    ):
        assert is_pure_forward_callable(func) is False, func


@pytest.mark.parametrize("qualname", ["from_numpy", "frombuffer", "matmul", "conv2d"])
def test_resolver_admits_pure_torch_ops(qualname: str) -> None:
    """Pure torch factories/ops still resolve through the intervention resolver."""

    resolved = resolve_function_registry_key(FunctionRegistryKey("torch", qualname, "function"))
    assert callable(resolved)


# --------------------------------------------------------------------------- #
# Prior denials (round-5 file-I/O + serialization + process) stay closed.
# --------------------------------------------------------------------------- #


def test_prior_denials_stay_closed() -> None:
    """The round-5 file-I/O and earlier serialization/process denials are intact."""

    for func in (
        torch.from_file,
        torch.load,
        torch.save,
        torch.jit.load,
        torch.jit.save,
        os.system,
        eval,
        pickle.loads,
        pickle.load,
    ):
        assert is_pure_forward_callable(func) is False, func


# --------------------------------------------------------------------------- #
# End-to-end: a real model built from in-place ops still round-trips + VERIFIES.
# --------------------------------------------------------------------------- #


class _InplaceNet(nn.Module):
    """Parameterized graph whose forward uses in-place elementwise ops."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.lin(value)
        value = value.relu_()
        value = value.mul_(2.0)
        value = value.clamp_(min=0.0, max=5.0)
        value = value.add_(1.0)
        return value


@pytest.mark.smoke
def test_inplace_model_runnable_still_verified(tmp_path: Path) -> None:
    """A real model built from in-place ops round-trips runnable and VERIFIES."""

    inputs = torch.randn(2, 4)
    model = _InplaceNet().eval()  # retain a live ref: include_weights needs the source model
    log = tl.trace(model, inputs, capture=_CAP)
    path = tmp_path / "inplace.tlspec"
    tl.save(log, path, level="runnable", include_weights=True)
    assert model is not None  # keep alive across save (defeats weakref GC)
    result = tl.load(path).run(inputs=inputs)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
