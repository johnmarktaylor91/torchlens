"""A05 FIX-A: the replay output-contract fix (leverage B1, listA row 2).

Root cause (review N4, triple-replicated): ``_run_replay`` sliced every replay
member's ``container_path`` into the replayed call's output. A synthesized
boundary output node records the MODEL-output container path there (honest
output-contract metadata) while replaying THROUGH its parent's function
call, so the recorded path was re-applied to an already-resolved member:
an ``HFKey`` applied to a bare logits tensor crashes (the HF Cache crash),
and an integer-shaped path silently slices a dimension instead of a
container slot. The fix derives each boundary node's slot from its parent's
path; internal tuple/dict slicing of genuine multi-output calls is
preserved.

Arms follow the leverage memo's B1 per-item gate: the distilgpt2 arms
(default structured, ``use_cache=False`` structured, bare-tensor wrapper as
positive control), the nested-cache arm (real distilgpt2: Cache leaves as
boundary nodes), the one-component container arm, and the resnet18
byte-identical negative control proving the fix targets the right variable.
The universal oracle is SLOT EQUALITY: every boundary output node's replayed
payload must equal its parent op's resolved output slot -- it catches the
silent integer-path mis-slice, not just the crash.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl

TORCH_HUB_CHECKPOINTS = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
HF_HUB_CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"


def _require_torchvision_checkpoint(filename: str) -> None:
    if not (TORCH_HUB_CHECKPOINTS / filename).exists():
        pytest.skip(
            f"GATE A05_REAL_WEIGHTS: torchvision checkpoint {filename} not cached;"
            " fetch it once online, then rerun offline."
        )


def _require_hf_snapshot(repo_dirname: str) -> None:
    if not (HF_HUB_CACHE / repo_dirname).exists():
        pytest.skip(
            f"GATE A05_REAL_WEIGHTS: HF snapshot {repo_dirname} not cached;"
            " fetch it once online, then rerun offline."
        )


def _output_nodes(trace) -> list:
    return [op for op in trace.layer_list if getattr(op, "is_output", False) and op.out is not None]


def _parent_slot_out(trace, node) -> torch.Tensor:
    """The parent op's resolved output slot -- the value a boundary node must carry."""

    assert len(node.parents) == 1, f"boundary node {node.label} has {len(node.parents)} parents"
    parent_label = node.parents[0]
    same_call = [
        op
        for op in trace.layer_list
        if op.layer_label == parent_label and op.func_call_id == node.func_call_id
    ]
    assert len(same_call) == 1, (
        f"parent {parent_label!r} of {node.label} resolves to {len(same_call)} same-call ops"
    )
    return same_call[0].out


def _assert_boundary_slots_consistent(fork, source) -> None:
    """Slot equality on every boundary node, and the edit must reach the output."""

    nodes = _output_nodes(fork)
    assert nodes, "trace has no boundary output nodes"
    for node in nodes:
        assert torch.equal(node.out, _parent_slot_out(fork, node)), (
            f"boundary node {node.label} does not equal its parent's resolved slot"
        )
    moved = [node for node in nodes if not torch.equal(node.out, source[node.label].out)]
    assert moved, "the edit never reached the model-output container"


def _first_module_op(trace, func_names: tuple[str, ...]):
    for op in trace.layer_list:
        if (
            op.func_name in func_names
            and op.output_of_modules
            and not getattr(op, "is_output", False)
        ):
            return op
    raise AssertionError(f"no module-attributed op with func in {func_names}")


def _zero_ablate_whole_site(fork, label: str) -> None:
    fork.do(fork[label].__selection__().resolve(fork), tl.zero_ablate())


class _LogitsOnly(nn.Module):
    """Bare-tensor wrapper: the spelling that replayed fine BEFORE the fix."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.inner(input_ids=input_ids).logits


class _OneComponentContainer(nn.Module):
    """Model returning a ONE-component container: ``{"logits": tensor}``."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.lin(torch.relu(x))}


@pytest.mark.heavy
@pytest.mark.real_model
def test_distilgpt2_structured_arms_and_bare_wrapper_positive_control():
    """The three distilgpt2 arms (B1 gate): default structured, use_cache=False
    structured, bare-tensor wrapper positive control -- with the wrapper's edited
    logits bit-equal to the structured arm's (same weights, same edit, same math).
    """

    pytest.importorskip("transformers")
    from tests.real_model.r0 import families

    # Both model variants are constructed BEFORE the first capture.
    torch.manual_seed(families.SEED)
    structured = families.build_distilgpt2("eager").eval()
    wrapper = _LogitsOnly(structured).eval()
    no_cache = structured  # same instance; use_cache toggles per-arm via config
    ids = families._token_ids()

    def edited_fork(model, args, kwargs):
        trace = tl.trace(
            model, args, kwargs, capture=tl.options.CaptureOptions(intervention_ready=True)
        )
        target = _first_module_op(trace, ("addmm", "linear", "conv1d"))
        fork = trace.fork()
        _zero_ablate_whole_site(fork, target.label)
        return trace, fork

    # Arm 1: default structured output (ModelOutput container).
    trace_default, fork_default = edited_fork(structured, (), {"input_ids": ids})
    _assert_boundary_slots_consistent(fork_default, trace_default)

    # Arm 2: use_cache=False structured output.
    no_cache.config.use_cache = False
    try:
        trace_nocache, fork_nocache = edited_fork(no_cache, (), {"input_ids": ids})
    finally:
        no_cache.config.use_cache = True
    _assert_boundary_slots_consistent(fork_nocache, trace_nocache)

    # Arm 3: bare-tensor wrapper -- the positive control that replayed fine
    # before the fix; its edited logits must equal the structured arm's.
    trace_bare, fork_bare = edited_fork(wrapper, (ids,), {})
    _assert_boundary_slots_consistent(fork_bare, trace_bare)
    logits_nodes = [
        node for node in _output_nodes(fork_default) if node.out.shape == ids.shape + (512,)
    ]
    assert logits_nodes, "structured fork exposes no logits-shaped boundary node"
    assert torch.equal(_output_nodes(fork_bare)[0].out, logits_nodes[0].out), (
        "bare-wrapper and structured arms disagree on the edited logits"
    )


def test_one_component_container_do_lands():
    """One-component-logits arm: a ``{"logits": t}`` return crashed pre-fix."""

    torch.manual_seed(0)
    model = _OneComponentContainer().eval()
    x = torch.randn(2, 8)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    target = _first_module_op(trace, ("linear",))
    fork = trace.fork()
    _zero_ablate_whole_site(fork, target.label)
    assert torch.all(fork[target.label].out == 0)
    _assert_boundary_slots_consistent(fork, trace)


@pytest.mark.real_model
def test_multi_leaf_container_slots_stay_exact():
    """R0 container-output fixture: integer-shaped paths are the SILENT arm.

    ``{"main": t, "parts": [t*2, t*3], "pair": (sum, mean)}`` mints boundary
    nodes with list-index paths; pre-fix those sliced tensor DIMENSIONS on
    replay instead of container slots. Slot equality plus an independent
    recomputation of every leaf pins the exact values.
    """

    from tests.real_model.r0 import families

    model, args, kwargs = families.build_structural("container-output")
    trace = tl.trace(
        model, args, kwargs, capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    target = _first_module_op(trace, ("linear",))
    fork = trace.fork()
    _zero_ablate_whole_site(fork, target.label)
    _assert_boundary_slots_consistent(fork, trace)
    # Independent oracle: recompute every leaf from the ablated linear output.
    zero_lin = torch.zeros_like(trace[target.label].out)
    expected = {
        (2, 16): {"main": zero_lin, "parts0": zero_lin * 2, "parts1": zero_lin * 3},
        (): {"sum": zero_lin.sum(), "mean": zero_lin.mean()},
    }
    for node in _output_nodes(fork):
        shape = tuple(node.out.shape)
        assert any(torch.equal(node.out, candidate) for candidate in expected[shape].values()), (
            f"boundary node {node.label} does not match any recomputed container leaf"
        )


@pytest.mark.heavy
@pytest.mark.real_model
def test_real_distilgpt2_nested_cache_do_lands():
    """Nested-cache arm: real distilgpt2, Cache leaves as boundary nodes.

    The default forward returns logits plus ``past_key_values`` whose per-layer
    key/value tensors surface as boundary nodes with nested paths
    (``HFKey -> NamedField -> TupleIndex -> NamedField``); pre-fix the replay
    re-applied those paths to resolved members and crashed.
    """

    _require_hf_snapshot("models--distilgpt2")
    transformers = pytest.importorskip("transformers")

    model = transformers.AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    ids = torch.randint(0, 1000, (1, 16), generator=torch.Generator().manual_seed(0))
    trace = tl.trace(
        model, (), {"input_ids": ids}, capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    nested = [node for node in _output_nodes(trace) if len(tuple(node.container_path or ())) >= 3]
    assert nested, "real distilgpt2 exposes no nested-cache boundary nodes"
    target = _first_module_op(trace, ("addmm", "linear", "conv1d"))
    fork = trace.fork()
    _zero_ablate_whole_site(fork, target.label)
    _assert_boundary_slots_consistent(fork, trace)


@pytest.mark.heavy
@pytest.mark.real_model
def test_resnet18_bare_output_negative_control_bit_identical():
    """B1's negative control: bare-tensor outputs replay byte-identically.

    The fix must be a no-op for empty container paths: an eager rerun of the
    model with the same module output zero-ablated reproduces the replayed
    fork's output bit-exactly.
    """

    _require_torchvision_checkpoint("resnet18-f37072fd.pth")
    import torchvision

    model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    x = torch.randn(4, 3, 224, 224, generator=torch.Generator().manual_seed(1))
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    target = trace["conv2d_5_15"]
    assert target.output_of_modules == ("layer1.1.conv2",)
    fork = trace.fork()
    _zero_ablate_whole_site(fork, "conv2d_5_15")

    def zero_hook(_module, _inputs, output):
        return torch.zeros_like(output)

    handle = dict(model.named_modules())["layer1.1.conv2"].register_forward_hook(zero_hook)
    try:
        with torch.no_grad():
            eager_ablated = model(x)
    finally:
        handle.remove()
    replayed = _output_nodes(fork)
    assert len(replayed) == 1
    assert torch.equal(replayed[0].out, eager_ablated), (
        "replayed bare-tensor output is not byte-identical to the eager ablated rerun"
    )
    _assert_boundary_slots_consistent(fork, trace)
