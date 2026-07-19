"""r35 stage-3 regressions: exact-semantics relaxations under the fail-closed
invariants -- corr2_2 (byte-exact alias coherence), corr2_3 (initializer
totality), corr2_4 (seeded-RNG isolation), hon1_4 (repr-independent structseq
fields), plus the mechanical float-``torch.equal`` grep lock.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, StateSource

pytestmark = pytest.mark.smoke


def _capture(model: nn.Module, args: Any) -> Any:
    return tl.trace(
        model,
        args,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
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


def test_r35_tied_nan_state_binds_and_saves(tmp_path: Path) -> None:
    """corr2_2: byte-identical tied NaN payloads are coherent, not a conflict."""

    model = _TiedModel().eval()
    with torch.no_grad():
        model.first.weight.fill_(float("nan"))
    x = torch.randn(2, 3)
    trace = _capture(model, x)
    path = tmp_path / "tied_nan.tlspec"
    tl.save(trace, str(path), level="runnable", include_weights=True)
    loaded = tl.load(str(path))
    # Staging the model's own state_dict must also bind (same tied storage).
    loaded.load_state_dict(model.state_dict())
    result = loaded.run(inputs=x)
    assert result.report.state_source is StateSource.USER_STATE_DICT
    assert torch.isnan(result.output).all()


def test_r35_distinct_nan_payload_bytes_still_conflict() -> None:
    """Tripwire intact: alias values with DIFFERENT NaN payload bytes refuse."""

    from torchlens._runnable_state import _alias_values_coherent

    quiet_nan = torch.tensor([float("nan")], dtype=torch.float32)
    # Same logical NaN-ness, different payload bits.
    other_nan = torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32)
    assert torch.isnan(other_nan).all()
    assert not _alias_values_coherent(quiet_nan, other_nan)
    # Signed zero: +0.0 and -0.0 are DIFFERENT bytes -> conflict.
    assert not _alias_values_coherent(torch.tensor([0.0]), torch.tensor([-0.0]))
    # Identical bytes bind; ordinary conflicting bytes refuse.
    assert _alias_values_coherent(torch.tensor([1.5, 2.5]), torch.tensor([1.5, 2.5]))
    assert not _alias_values_coherent(torch.tensor([1.5, 2.5]), torch.tensor([1.5, 3.5]))


def test_r35_no_float_torch_equal_in_runnable_modules() -> None:
    """Mechanical class lock: no float ``torch.equal(`` in the runnable core.

    The only allowed occurrences are explicit whole-storage uint8-view byte
    comparisons, annotated ``# byte-exact uint8 view``.
    """

    import torchlens

    package_root = Path(torchlens.__file__).parent
    runnable_modules = (
        "_io/runnable.py",
        "_runnable_execution.py",
        "_runnable_state.py",
        "backends/torch/completeness_witness.py",
        "backends/torch/backend.py",
        "backends/torch/ops.py",
    )
    offenders: list[str] = []
    for relative in runnable_modules:
        lines = (package_root / relative).read_text().splitlines()
        for index, line in enumerate(lines):
            if not re.search(r"\btorch\.equal\(", line):
                continue
            # The annotation may sit on a continuation line after formatting.
            statement = " ".join(lines[index : index + 3])
            if "# byte-exact uint8 view" not in statement:
                offenders.append(f"{relative}:{index + 1}: {line.strip()}")
    assert offenders == []


# ---------------------------------------------------------------------------
# corr2_3 -- initializer totality (torchlens_role_init_v2).
# ---------------------------------------------------------------------------


def test_r35_zero_fan_in_linear_runs_random_init(tmp_path: Path) -> None:
    """corr2_3 repro: ``nn.Linear(0, 2)`` random fallback completes typed."""

    model = nn.Linear(0, 2).eval()
    x = torch.randn(3, 0)
    trace = _capture(model, x)
    path = tmp_path / "zerofan.tlspec"
    tl.save(trace, str(path), level="runnable")
    result = tl.load(str(path)).run(inputs=x, seed=1)
    assert result.report.state_source is StateSource.RANDOM_INITIALIZATION
    # Weight is (2, 0): the output is exactly the bias broadcast over the batch.
    assert result.output.shape == (3, 2)

    fresh = nn.Linear(0, 2)
    with torch.no_grad():
        fresh.bias.zero_()  # N1-a bias policy is zeros
    assert torch.equal(result.output, fresh(x))


@pytest.mark.parametrize(
    "shape",
    [(2, 0), (2, 0, 3), (0, 3), (0,)],
)
def test_r35_empty_slot_initializes_without_rng(shape: tuple[int, ...]) -> None:
    """Every legal empty shape allocates with PROVABLY zero RNG consumption."""

    from torchlens._runnable_state import _initialize_slot
    from torchlens.runnable import (
        StateSlotBinding,
        StateSlotRole,
        TensorSlotDescriptor,
        TensorSlotRole,
    )

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
        ),
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_r35_seeded_run_restores_produced_only_cuda_rng(tmp_path: Path) -> None:
    """corr2_4 repro: a produced-only-CUDA seeded run leaks no CUDA RNG state."""

    model = _CudaDrawModel().eval()
    x = torch.randn(2, 3)
    trace = _capture(model, x)
    path = tmp_path / "cudadraw.tlspec"
    tl.save(trace, str(path), level="runnable")
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


def test_r35_cpu_only_seeded_run_never_touches_cuda_lazy_seed(tmp_path: Path) -> None:
    """A CPU-only descriptor's seeded run must not queue a lazy CUDA seed."""

    model = nn.Sequential(nn.Linear(3, 2)).eval()
    x = torch.randn(2, 3)
    trace = _capture(model, x)
    path = tmp_path / "cpuonly.tlspec"
    tl.save(trace, str(path), level="runnable")
    loaded = tl.load(str(path))
    from torchlens._runnable_execution import _seeded_fork_devices

    descriptor = loaded.__dict__["_runnable_descriptor"]
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        assert _seeded_fork_devices(descriptor, 5) == list(range(torch.cuda.device_count()))
    else:
        assert _seeded_fork_devices(descriptor, 5) == []
    cpu_before = torch.get_rng_state()
    result = loaded.run(inputs=x, seed=5)
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE or (
        result.report.path_faithfulness is PathFaithfulness.VERIFIED
    )


# ---------------------------------------------------------------------------
# hon1_4 -- repr-independent structseq field names.
# ---------------------------------------------------------------------------


class _MaxModel(nn.Module):
    def forward(self, x: torch.Tensor) -> Any:
        return torch.max(x, dim=1)


def test_r35_large_int32_structseq_replays_verified(tmp_path: Path) -> None:
    """hon1_4 repro: a 40x50 int32 max structseq no longer false-diverges.

    Previously the repr-wrapped ``dtype=torch.int32`` line injected a phantom
    field, flipping the verdict on tensor size alone.
    """

    model = _MaxModel().eval()
    x = torch.arange(40 * 50, dtype=torch.int32).reshape(40, 50)
    trace = _capture(model, x)
    path = tmp_path / "structseq.tlspec"
    tl.save(trace, str(path), level="runnable")
    result = tl.load(str(path)).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    live = model(x)
    assert torch.equal(result.output.values, live.values)
    assert torch.equal(result.output.indices, live.indices)


def test_r35_structseq_helper_is_repr_independent() -> None:
    """Unit rows: exact fields for every family; repr text cannot influence."""

    from torchlens.utils._torch_compat import torch_structseq_field_names

    x = torch.randn(6, 7).to(torch.float64)
    assert torch_structseq_field_names(torch.max(x, dim=1)) == ("values", "indices")
    assert torch_structseq_field_names(torch.min(x, dim=0)) == ("values", "indices")
    assert torch_structseq_field_names(torch.median(x, dim=1)) == ("values", "indices")
    assert torch_structseq_field_names(torch.topk(x, k=2, dim=1)) == ("values", "indices")
    assert torch_structseq_field_names(torch.sort(x, dim=1)) == ("values", "indices")
    assert torch_structseq_field_names(torch.linalg.slogdet(x[:6, :6])) == (
        "sign",
        "logabsdet",
    )
    # Plain tuples and namedtuples are not structseqs.
    assert torch_structseq_field_names((1, 2)) == ()
    # No repr involvement: the module source must not parse repr for fields.
    import inspect

    from torchlens.utils import _torch_compat

    helper_source = inspect.getsource(_torch_compat.torch_structseq_field_names)
    assert "repr(" not in helper_source


def test_r35_repr_parsers_are_deleted() -> None:
    """The two repr regex parsers are gone with no fallback-to-repr path."""

    import inspect

    import torchlens._runnable_execution as execution
    import torchlens.backends.torch.ops as ops

    for source in (
        inspect.getsource(ops._torch_return_type_fields),
        inspect.getsource(execution._torch_structseq_field_names),
    ):
        assert "re.findall" not in source
        assert "re.finditer" not in source
        assert "repr(value)" not in source
