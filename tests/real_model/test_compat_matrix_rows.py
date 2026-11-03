"""Compat matrix rows for the current releases: router logits, tensor parallelism, FlashAttention 4.

Each test is one row of ``docs/reference/compat_matrix.md``. The rows run on whatever torch and
transformers the environment holds; the page records the versions and hosts each row was
measured on.

* Router-logit outputs: a small mixture-of-experts causal LM called with
  ``output_router_logits=True`` must trace to the exact model return (logits AND every
  per-layer router-logit tensor), expose each router-logit tensor as an output op, and pass
  forward replay validation.
* DTensor tensor parallelism: a forward sharded over a real two-process CPU ``gloo`` mesh
  runs, and capture refuses with the typed ``DistributedCaptureUnsupportedError`` naming the
  ``dtensor`` and ``tensor_parallel`` kinds (the planned design is
  ``docs/internal/reference/tensor_parallel_design.md``). The sharding is torch-native
  ``parallelize_module``. Transformers' ``tp_plan="auto"`` loading is the other half of the
  row; it loads a pinned registry checkpoint, so it lives in the R1 venue
  (``tests/real_model/r1/test_compat_matrix_hf_tp.py``).
* ``flash_attention_4``: capture plus validation on a CUDA device that the FlashAttention 4
  package supports; skipped elsewhere with the reason. It is a known capture gap today (strict
  xfail), and a companion test pins that a witnessed trace is marked unverified, not silent.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

import torchlens as tl

transformers = pytest.importorskip("transformers")

pytestmark = [pytest.mark.real_model, pytest.mark.real_class]

_TINY = {
    "vocab_size": 128,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
}


def _moe_model(family: str) -> torch.nn.Module:
    """Build a seeded two-layer, four-expert, top-2 MoE causal LM of one family."""

    builders = {
        "mixtral": (
            "MixtralForCausalLM",
            "MixtralConfig",
            {"intermediate_size": 64, "num_local_experts": 4, "num_experts_per_tok": 2},
        ),
        "qwen2_moe": (
            "Qwen2MoeForCausalLM",
            "Qwen2MoeConfig",
            {
                "intermediate_size": 64,
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
            },
        ),
        "olmoe": (
            "OlmoeForCausalLM",
            "OlmoeConfig",
            {"intermediate_size": 64, "num_experts": 4, "num_experts_per_tok": 2},
        ),
    }
    model_name, config_name, extra = builders[family]
    if not hasattr(transformers, config_name):
        pytest.skip(f"transformers {transformers.__version__} has no {config_name}")
    config = getattr(transformers, config_name)(**_TINY, **extra)
    torch.manual_seed(0)
    return getattr(transformers, model_name)(config).eval()


def _router_logits(output: Any) -> list[torch.Tensor]:
    """Return the per-layer router-logit tensors of a MoE model output."""

    logits = getattr(output, "router_logits", None)
    if logits is None:
        return []
    if isinstance(logits, torch.Tensor):
        return [logits]
    return [tensor for tensor in logits if isinstance(tensor, torch.Tensor)]


@pytest.mark.heavy
@pytest.mark.parametrize("family", ["mixtral", "qwen2_moe", "olmoe"])
def test_router_logit_outputs_capture_exactly_and_validate(family: str) -> None:
    """``output_router_logits=True`` traces to the exact return and replays."""

    model = _moe_model(family)
    input_ids = torch.randint(0, 128, (1, 7), generator=torch.Generator().manual_seed(1))
    kwargs = {"input_ids": input_ids, "output_router_logits": True}
    with torch.no_grad():
        direct = model(**kwargs)
    direct_router = _router_logits(direct)
    assert len(direct_router) == 2, "precondition: one router-logit tensor per MoE layer"

    trace = tl.trace(model, (), input_kwargs=dict(kwargs))
    try:
        assert trace.outcome.status.name == "COMPLETE"
        assert torch.equal(trace.output.logits, direct.logits)
        traced_router = _router_logits(trace.output)
        assert len(traced_router) == len(direct_router)
        assert all(torch.equal(a, b) for a, b in zip(traced_router, direct_router, strict=True))
        output_tensors = [op.out for op in trace.output_ops if isinstance(op.out, torch.Tensor)]
        for router in direct_router:
            assert any(
                out.shape == router.shape and torch.equal(out, router) for out in output_tensors
            ), "a router-logit tensor is missing from the trace's output ops"
    finally:
        trace.cleanup()
    assert tl.validate(model, (), dict(kwargs), scope="forward") is True


class _ShardedMLP(torch.nn.Module):
    """Two linears for a column-then-row tensor-parallel plan."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``fc2(relu(fc1(x)))``."""

        return self.fc2(torch.relu(self.fc1(x)))


def _refusal_kinds(model: torch.nn.Module, args: Any, kwargs: dict[str, Any]) -> list[str]:
    """Trace ``model`` and return the refusal's finding kinds (``[]`` means it captured)."""

    try:
        tl.trace(model, args, input_kwargs=kwargs)
    except tl.errors.DistributedCaptureUnsupportedError as exc:
        return sorted({finding.kind for finding in exc.fields["findings"]})
    return []


def _tp_rank(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    """One rank of the two-process CPU mesh: shard, run, then try to capture."""

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import parallel

    # Each spawned child owns its environment; publish its rank layout there as well.
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    store = dist.FileStore(init_file, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    result: dict[str, Any] = {"rank": rank}
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        torch.manual_seed(0)
        mlp = _ShardedMLP().eval()
        parallel.parallelize_module(
            mlp, mesh, {"fc1": parallel.ColwiseParallel(), "fc2": parallel.RowwiseParallel()}
        )
        x = torch.randn(2, 8, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            result["torch_forward_shape"] = list(mlp(x).shape)
        result["torch_weight_type"] = type(mlp.fc1.weight).__name__
        result["torch_kinds"] = _refusal_kinds(mlp, x, {})
    finally:
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@pytest.mark.slow
def test_dtensor_tensor_parallel_on_a_two_process_cpu_mesh_refuses_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sharded forward runs on two ranks; capture refuses typed on each rank."""

    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo unavailable")
    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this process")
    pytest.importorskip("torch.distributed.tensor.parallel")
    import torch.multiprocessing as mp

    # The row is a CPU mesh: the spawned ranks (which inherit this environment) see no CUDA device.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    if sys.platform == "darwin" and "GLOO_SOCKET_IFNAME" not in os.environ:
        # gloo on macOS otherwise binds the hostname's interface, and the shard scatter in
        # parallelize_module waits out its 30-minute timeout; the mesh is local, so use loopback.
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    world_size = 2
    mp.spawn(
        _tp_rank,
        args=(world_size, str(tmp_path / "init_store"), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    for rank in range(world_size):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["torch_weight_type"] == "DTensor", "precondition: the plan sharded fc1"
        assert result["torch_forward_shape"] == [2, 8]
        assert result["torch_kinds"] == ["dtensor", "tensor_parallel"], result


def _flash_attention_4_ready() -> str:
    """Return ``""`` when FlashAttention 4 can run here, else the skip reason."""

    if not torch.cuda.is_available():
        return "flash_attention_4 needs a CUDA device"
    available = getattr(transformers.utils, "is_flash_attn_4_available", None)
    if available is None:
        return f"transformers {transformers.__version__} has no flash_attention_4"
    if not available():
        return "the FlashAttention 4 package is not installed or does not support this GPU"
    return ""


_FA4_GAP = (
    "capture gap: FlashAttention 4 allocates its output with torch.empty and fills it from a "
    "CuTe kernel below the torch function layer, so the trace records the allocation instead of "
    "the attention; validation fails (the tripwire) and the trace is marked unverified"
)


def _fa4_llama() -> tuple[torch.nn.Module, torch.Tensor]:
    """Build a bf16 two-layer Llama on ``flash_attention_4`` on CUDA, or skip with the reason."""

    reason = _flash_attention_4_ready()
    if reason:
        pytest.skip(reason)
    config = transformers.LlamaConfig(**(_TINY | {"hidden_size": 64}), intermediate_size=128)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM._from_config(
        config, attn_implementation="flash_attention_4", dtype=torch.bfloat16
    )
    model = model.to("cuda").eval()
    assert model.config._attn_implementation == "flash_attention_4"
    input_ids = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1))
    return model, input_ids.to("cuda")


@pytest.mark.cuda
@pytest.mark.heavy
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_FA4_GAP)
def test_flash_attention_4_capture_and_validate() -> None:
    """A Llama on ``attn_implementation="flash_attention_4"`` traces exactly and replays."""

    model, input_ids = _fa4_llama()
    with torch.no_grad():
        direct = model(input_ids=input_ids).logits
    trace = tl.trace(model, (), input_kwargs={"input_ids": input_ids})
    try:
        assert trace.outcome.status.name == "COMPLETE"
        assert torch.equal(trace.output.logits, direct)
    finally:
        trace.cleanup()
    assert tl.validate(model, (), {"input_ids": input_ids}, scope="forward") is True


@pytest.mark.cuda
@pytest.mark.heavy
def test_flash_attention_4_capture_gap_is_flagged_not_silent() -> None:
    """Until the gap is closed, a witnessed ``flash_attention_4`` trace says it is unverified.

    The kernel's writes are invisible to capture, so the armed completeness witness must account
    for them as unexplained dispatch events and mark the trace unverified (a plain trace leaves
    the witness off and ``verified`` unrecorded). If this starts failing because the gap was
    closed, drop the xfail on the test above and this test together.
    """

    from torchlens import _state
    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    model, input_ids = _fa4_llama()
    saved_escape_detector = _state._escape_detector_mode
    saved_completeness_witness = _state._completeness_witness_mode
    unwrap_torch()
    wrap_torch(completeness_witness=True)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trace = tl.trace(model, (), input_kwargs={"input_ids": input_ids})
    finally:
        unwrap_torch()
        wrap_torch(
            escape_detector=saved_escape_detector,
            completeness_witness=saved_completeness_witness,
        )
    try:
        provenance = trace.provenance
        facts = (
            provenance.verified,
            provenance.verification_reason,
            trace.completeness_diagnostics,
        )
        assert provenance.verified is False, facts
        assert provenance.verification_reason == "dispatch_witness_unaccounted_ops", facts
        assert trace.completeness_diagnostics, facts
    finally:
        trace.cleanup()
    gap_warnings = [w for w in caught if issubclass(w.category, tl.errors.CaptureGapWarning)]
    assert gap_warnings, [str(w.message)[:200] for w in caught]
