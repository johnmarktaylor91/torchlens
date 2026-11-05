"""Compat matrix row: DTensor tensor parallelism loaded by transformers (``tp_plan="auto"``).

One row of ``docs/reference/compat_matrix.md``; its torch-native half is
``tests/real_model/test_compat_matrix_rows.py``. A pinned hub checkpoint (registry row
``r2-tiny-random-llama``) is loaded on each rank of a two-process CPU ``gloo`` mesh with
``distributed_config=transformers.DistributedConfig(tp_plan="auto")``. The sharded forward
runs (the projection weights are ``DTensor``s, the logits match the unsharded load), and
capture refuses at entry with the typed ``DistributedCaptureUnsupportedError`` naming the
``dtensor`` and ``tensor_parallel`` kinds.

The parent resolves the row through ``r1_loader``; each spawned rank loads the same row
through the r1 loader path (``load_hub_model``) at its pinned revision, offline, adding only
the sharding arguments. No rank touches the network.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

import torchlens as tl

pytestmark = [pytest.mark.slow, pytest.mark.real_model, pytest.mark.real_checkpoint]

ARTIFACT_ID = "r2-tiny-random-llama"
WORLD_SIZE = 2


def _refusal_kinds(model: torch.nn.Module, kwargs: dict[str, Any]) -> list[str]:
    """Trace ``model`` and return the refusal's finding kinds (``[]`` means it captured)."""

    try:
        tl.trace(model, (), input_kwargs=kwargs)
    except tl.errors.DistributedCaptureUnsupportedError as exc:
        return sorted({finding.kind for finding in exc.fields["findings"]})
    return []


def _tp_rank(rank: int, init_file: str, out_dir: str, input_ids: torch.Tensor) -> None:
    """One rank: load the registry row with ``tp_plan="auto"``, run it, then try to capture."""

    import torch.distributed as dist
    import transformers
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    from tests.real_model.r1.conftest import _HEAD_CLASS_BY_ARTIFACT, load_hub_model
    from tests.real_model.registry import load_registry

    # transformers applies a tp_plan only when the rank layout is in the process environment;
    # without it the checkpoint loads unsharded. The spawned child owns its environment.
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE), LOCAL_RANK=str(rank))
    dist.init_process_group(
        "gloo", store=dist.FileStore(init_file, WORLD_SIZE), rank=rank, world_size=WORLD_SIZE
    )
    result: dict[str, Any] = {"rank": rank}
    try:
        mesh = init_device_mesh("cpu", (WORLD_SIZE,))
        row = load_registry().checkpoint_evidence(ARTIFACT_ID)
        model = load_hub_model(
            row,
            _HEAD_CLASS_BY_ARTIFACT[ARTIFACT_ID],
            dtype=torch.float32,
            distributed_config=transformers.DistributedConfig(tp_plan="auto"),
            device_mesh=mesh,
        )
        result["weight_type"] = type(model.model.layers[0].self_attn.q_proj.weight).__name__
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        torch.save(logits.detach().clone(), Path(out_dir, f"rank{rank}_logits.pt"))
        result["kinds"] = _refusal_kinds(model, {"input_ids": input_ids})
    finally:
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


def _tp_plan_unavailable_reason() -> str:
    """Why this host cannot load a ``tp_plan`` model, or ``""`` when it can."""

    import transformers

    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        return "torch.distributed gloo unavailable"
    if torch.distributed.is_initialized():
        return "a process group is already initialized in this process"
    if not hasattr(transformers, "DistributedConfig"):
        return f"transformers {transformers.__version__} has no DistributedConfig"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "transformers refuses a tp_plan on any machine with an MPS device"
    return ""


def test_transformers_tp_plan_load_on_a_two_process_cpu_mesh_refuses_typed(
    r1_loader: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tp_plan-loaded checkpoint shards and runs on two ranks; capture refuses typed on each."""

    reason = _tp_plan_unavailable_reason()
    if reason:
        pytest.skip(reason)
    pytest.importorskip("torch.distributed.tensor.parallel")
    pytest.importorskip("accelerate", reason="transformers needs accelerate to apply a tp_plan")
    import torch.multiprocessing as mp

    # The parent resolves the pinned snapshot; the ranks then read it offline.
    reference = r1_loader(ARTIFACT_ID)["model"]
    input_ids = torch.randint(
        0, reference.config.vocab_size, (1, 6), generator=torch.Generator().manual_seed(4)
    )
    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    # The row is a CPU mesh. On a CUDA host transformers would bind each rank to its own GPU,
    # so the spawned ranks (which inherit this environment) see no CUDA device.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    if sys.platform == "darwin" and "GLOO_SOCKET_IFNAME" not in os.environ:
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    mp.spawn(
        _tp_rank,
        args=(str(tmp_path / "init_store"), str(tmp_path), input_ids),
        nprocs=WORLD_SIZE,
        join=True,
    )
    for rank in range(WORLD_SIZE):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["weight_type"] == "DTensor", "precondition: tp_plan sharded q_proj"
        logits = torch.load(tmp_path / f"rank{rank}_logits.pt")
        torch.testing.assert_close(logits, expected)
        assert result["kinds"] == ["dtensor", "tensor_parallel"], result
