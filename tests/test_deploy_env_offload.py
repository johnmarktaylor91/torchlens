"""Deployment envelope (lane F37): accelerate CPU/disk offload capture.

The population TorchLens targets loads big models through accelerate; these
tests pin the CPU-verified slice of the envelope: offloaded models are
ADMITTED with weights_map evidence, capture is pollution-free (op parity with
the plain model), weights attribute to their prep-time Param records, logits
match the bare forward, and forward replay validation passes. The bare-meta
refusal stays fail-closed.
"""

from __future__ import annotations

import tempfile
import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._deploy_env import offload_backed_state_paths
from torchlens._robustness import UnsupportedTensorVariantError

transformers = pytest.importorskip("transformers")
accelerate = pytest.importorskip("accelerate")


def _tiny_gpt2() -> nn.Module:
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=128, n_positions=64)
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(config).eval()


def _offloaded_twin(kind: str) -> tuple[nn.Module, nn.Module]:
    """Return (plain reference, offloaded twin) sharing identical weights."""

    reference = _tiny_gpt2()
    state = {k: v.clone() for k, v in reference.state_dict().items()}
    twin = _tiny_gpt2()
    twin.load_state_dict(state)
    if kind == "disk":
        twin = accelerate.disk_offload(
            twin, tempfile.mkdtemp(), execution_device=torch.device("cpu")
        )
    else:
        twin = accelerate.cpu_offload(twin, execution_device=torch.device("cpu"))
    return reference, twin


@pytest.mark.heavy
@pytest.mark.parametrize("kind", ["disk", "cpu"])
def test_offloaded_capture_is_clean_and_value_exact(kind: str) -> None:
    """Offloaded capture: op parity, param attribution, logits parity, no warnings."""

    reference, offloaded = _offloaded_twin(kind)
    input_ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        reference_logits = reference(input_ids).logits
    reference_trace = tl.trace(reference, input_ids)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        offload_trace = tl.trace(offloaded, input_ids)

    provenance_warnings = [w for w in caught if "provenance" in str(w.message)]
    assert not provenance_warnings
    assert offload_trace.capture_verified is None
    # Zero hook-infrastructure pollution: exact op-label parity with the plain twin.
    assert [op.label for op in offload_trace.ops.values()] == [
        op.label for op in reference_trace.ops.values()
    ]
    # Materialized weights attribute to their prep-time Param records.
    addmm = offload_trace["addmm_1"]
    assert addmm.param_shapes and addmm.num_params > 0
    # Captured logits equal the bare offloaded forward's reference values.
    logits_ops = [
        op
        for op in offload_trace.ops.values()
        if getattr(op.out, "shape", None) is not None
        and tuple(op.out.shape) == tuple(reference_logits.shape)
    ]
    assert any(torch.allclose(op.out, reference_logits, atol=1e-5) for op in logits_ops)


@pytest.mark.heavy
def test_offloaded_capture_passes_forward_replay_validation() -> None:
    """The validation tripwire itself runs green on a disk-offloaded model."""

    _, offloaded = _offloaded_twin("disk")
    input_ids = torch.randint(0, 128, (1, 8))
    assert tl.validate(offloaded, input_ids, scope="forward") is True


@pytest.mark.heavy
def test_offload_shims_are_removed_after_capture() -> None:
    """Hook shims are session-scoped: uninstalled after capture, recapture clean."""

    _, offloaded = _offloaded_twin("disk")
    input_ids = torch.randint(0, 128, (1, 8))
    first = tl.trace(offloaded, input_ids)
    for module in offloaded.modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        assert "pre_forward" not in hook.__dict__
        assert "post_forward" not in hook.__dict__
        assert "_tl_offload_shim" not in hook.__dict__
    second = tl.trace(offloaded, input_ids)
    assert len(second.ops) == len(first.ops)


@pytest.mark.smoke
def test_bare_meta_model_still_refuses_fail_closed() -> None:
    """A meta param with NO offload hook keeps the typed entry refusal."""

    with torch.device("meta"):
        model = nn.Linear(4, 2)
    with pytest.raises(UnsupportedTensorVariantError):
        tl.trace(model, torch.randn(1, 4))


class _StubHook:
    """Structural stand-in for accelerate's AlignDevicesHook."""

    def __init__(
        self,
        *,
        offload: bool = False,
        offload_buffers: bool = False,
        weights_map: dict | None = None,
        place_submodules: bool = False,
    ) -> None:
        self.offload = offload
        self.offload_buffers = offload_buffers
        self.weights_map = weights_map
        self.place_submodules = place_submodules

    def pre_forward(self, module, *args, **kwargs):  # pragma: no cover - shape only
        return args, kwargs

    def post_forward(self, module, output):  # pragma: no cover - shape only
        return output


@pytest.mark.smoke
def test_offload_backed_paths_require_weights_map_membership() -> None:
    """Key-inventory evidence is honored: absent keys stay unbacked (fail closed)."""

    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    model[0]._hf_hook = _StubHook(offload=True, weights_map={"weight": None})
    backed = offload_backed_state_paths(model)
    assert "0.weight" in backed
    assert "0.bias" not in backed  # not in the inventory
    assert "1.weight" not in backed  # no hook on that module


@pytest.mark.smoke
def test_offload_backed_paths_accept_full_name_inventories() -> None:
    """accelerate's PrefixedDataset serves FULL state-dict names from keys()."""

    model = nn.Sequential(nn.Linear(4, 4))
    model[0]._hf_hook = _StubHook(offload=True, weights_map={"0.weight": None, "0.bias": None})
    assert offload_backed_state_paths(model) == frozenset({"0.weight", "0.bias"})


@pytest.mark.smoke
def test_offload_backed_paths_exclude_buffers_unless_declared() -> None:
    """offload=True alone never marks buffers; offload_buffers=True does."""

    class WithBuffer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self.register_buffer("running", torch.zeros(2))

    model = WithBuffer()
    model._hf_hook = _StubHook(offload=True, place_submodules=True, weights_map=None)
    backed = offload_backed_state_paths(model)
    assert "linear.weight" in backed and "linear.bias" in backed
    assert "running" not in backed

    model_buffers = WithBuffer()
    model_buffers._hf_hook = _StubHook(
        offload=True, offload_buffers=True, place_submodules=True, weights_map=None
    )
    assert "running" in offload_backed_state_paths(model_buffers)
