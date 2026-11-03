"""r35 stage-3 regressions: exact-semantics relaxations under the fail-closed
invariants -- corr2_2 (byte-exact alias coherence), corr2_3 (initializer
totality), corr2_4 (seeded-RNG isolation), hon1_4 (repr-independent structseq
fields), plus the mechanical float-``torch.equal`` grep lock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import StateSource
from torchlens.options import CaptureOptions


def _capture(model: nn.Module, args: Any) -> Any:
    return tl.trace(
        model,
        args,
        capture=CaptureOptions(replayable=True, capture_container_structure=True, cache=False),
    )


# ---------------------------------------------------------------------------
# corr2_2 -- tied NaN state binds; conflicting bytes still refuse.
# ---------------------------------------------------------------------------


class _TiedModel(nn.Module):
    """Two linears sharing ONE weight parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 3, bias=False)
        self.second = nn.Linear(3, 3, bias=False)
        self.second.weight = self.first.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))


# Kept: sole branch coverage of _runnable/state.py; the catch of its regression fault.
def test_r35_tied_nan_state_binds_and_saves(tmp_path: Path) -> None:
    """corr2_2: byte-identical tied NaN payloads are coherent, not a conflict."""

    model = _TiedModel().eval()
    with torch.no_grad():
        model.first.weight.fill_(float("nan"))
    x = torch.randn(2, 3)
    trace = _capture(model, x)
    path = tmp_path / "tied_nan.tlspec"
    trace.save(str(path), level="runnable", include_weights=True)
    loaded = tl.load(str(path))
    # Staging the model's own state_dict must also bind (same tied storage).
    loaded.load_state_dict(model.state_dict())
    result = loaded.run(inputs=x)
    assert result.report.state_source is StateSource.USER_STATE_DICT
    assert torch.isnan(result.output).all()


# ---------------------------------------------------------------------------
# corr2_3 -- initializer totality (torchlens_role_init_v2).
# ---------------------------------------------------------------------------


# Kept: a sole mutation kill.
@pytest.mark.filterwarnings(
    # A08 random-role-init disclosure (runnable_random_init_run): this test
    # exercises a weight-free runnable run DELIBERATELY; the warning is the
    # feature under test elsewhere (tests/test_persist_honesty_runnable_random.py).
    "default:This runnable artifact carries no model weights"
)
def test_r35_zero_fan_in_linear_runs_random_init(tmp_path: Path) -> None:
    """corr2_3 repro: ``nn.Linear(0, 2)`` random fallback completes typed."""

    with pytest.warns(UserWarning, match="Initializing zero-element tensors is a no-op"):
        model = nn.Linear(0, 2).eval()
    x = torch.randn(3, 0)
    trace = _capture(model, x)
    path = tmp_path / "zerofan.tlspec"
    trace.save(str(path), level="runnable")
    result = tl.load(str(path)).run(inputs=x, seed=1)
    assert result.report.state_source is StateSource.RANDOM_INITIALIZATION
    # Weight is (2, 0): the output is exactly the bias broadcast over the batch.
    assert result.output.shape == (3, 2)

    with pytest.warns(UserWarning, match="Initializing zero-element tensors is a no-op"):
        fresh = nn.Linear(0, 2)
    with torch.no_grad():
        fresh.bias.zero_()  # N1-a bias policy is zeros
    assert torch.equal(result.output, fresh(x))


# Kept: shares a parametrize list that is not trimmable per id with kept params.
@pytest.mark.parametrize(
    "shape",
    [(2, 0), (2, 0, 3), (0, 3), (0,)],
)
def test_r35_empty_slot_initializes_without_rng(shape: tuple[int, ...]) -> None:
    """Every legal empty shape allocates with PROVABLY zero RNG consumption."""

    from torchlens._runnable import (
        StateSlotBinding,
        StateSlotRole,
        TensorSlotDescriptor,
        TensorSlotRole,
    )
    from torchlens._runnable.state import _initialize_slot

    slot = TensorSlotDescriptor(
        slot_id="slot:weight_test",
        role=TensorSlotRole.PARAMETER,
        use_sites=(),
        shape=shape,
        dtype="torch.float32",
        rank=len(shape),
        device_type="cpu",
        device_index=None,
        mutable=False,
        version_of=None,
        producer_slot_id=None,
        output_path=None,
        input_binding=None,
        state_binding=StateSlotBinding(
            module_path="self",
            state_dict_name="weight",
            semantic_role=StateSlotRole.WEIGHT,
            trainable=True,
            persistent=True,
            alias_group=None,
            captured_requires_grad=True,
            captured_grad_fn=False,
            host_escape_disposition=None,
        ),
        host_escape=False,
        inert_sink=False,
    )
    before = torch.get_rng_state()
    value = _initialize_slot(slot, None)
    after = torch.get_rng_state()
    assert tuple(value.shape) == shape and value.numel() == 0
    assert torch.equal(before, after), "empty initialization consumed generator state"


# ---------------------------------------------------------------------------
# corr2_4 -- seeded-RNG isolation totality.
# ---------------------------------------------------------------------------


class _CudaDrawModel(nn.Module):
    """CPU-input model whose only CUDA appearance is a produced RNG draw."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.rand(x.shape, device="cuda")
        return x + noise.cpu()


# Kept: id not measured per id (rare tier).
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_r35_seeded_run_restores_produced_only_cuda_rng(tmp_path: Path) -> None:
    """corr2_4 repro: a produced-only-CUDA seeded run leaks no CUDA RNG state."""

    model = _CudaDrawModel().eval()
    x = torch.randn(2, 3)
    trace = _capture(model, x)
    path = tmp_path / "cudadraw.tlspec"
    trace.save(str(path), level="runnable")
    loaded = tl.load(str(path))

    torch.cuda.init()
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all()
    first = loaded.run(inputs=x, seed=11)
    cpu_after = torch.get_rng_state()
    cuda_after = torch.cuda.get_rng_state_all()
    assert torch.equal(cpu_before, cpu_after)
    assert all(torch.equal(b, a) for b, a in zip(cuda_before, cuda_after))
    # Fixed-seed determinism across runs.
    second = loaded.run(inputs=x, seed=11)
    assert torch.equal(first.output, second.output)


# ---------------------------------------------------------------------------
# hon1_4 -- repr-independent structseq field names.
# ---------------------------------------------------------------------------


class _MaxModel(nn.Module):
    def forward(self, x: torch.Tensor) -> Any:
        return torch.max(x, dim=1)
