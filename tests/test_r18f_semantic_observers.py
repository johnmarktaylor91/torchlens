"""R18F regression tests: residual recipe, patching state safety, observer labels/spans.

Covers:
- H3 residual recipe only fires on genuine attn+mlp transformer blocks; resid_mid is
  the real post-attention add, never the degenerate first-add-equals-output fallback.
- H8 activation/attribution patching snapshots and restores model state + global RNG
  around every counterfactual run, so the caller's model is not mutated and each
  counterfactual starts from identical state.
- M7 forward TapRecord.site_label resolves to the PUBLIC trace label, not the raw
  internal ``*_raw`` label.
- M9 (obs-local) observer span direction is enforced: a forward record only carries
  forward/both spans; a backward record only carries backward/both spans.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl

# --------------------------------------------------------------------------------------
# H3 - residual recipe structural gate
# --------------------------------------------------------------------------------------


class _ScaleBlock(nn.Module):
    """A ``*Block*``-named module that is NOT a transformer block (no attn/mlp)."""

    def __init__(self) -> None:
        """Initialize a single scaling child."""

        super().__init__()
        self.scale = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a plain residual scaling update, not an attention/MLP block."""

        return x + self.scale(x)


class _GenuineBlock(nn.Module):
    """A genuine transformer block with attention and MLP residual updates."""

    def __init__(self) -> None:
        """Initialize attention and MLP children."""

        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.mlp = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention and MLP residual updates."""

        resid_mid = x + self.attn(x)
        return resid_mid + self.mlp(resid_mid)


def test_residual_recipe_rejects_non_transformer_named_block() -> None:
    """A ``*Block*``-named module without attn+mlp children exposes no residual facets."""

    torch.manual_seed(0)
    log = tl.trace(
        _ScaleBlock(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    facets = log.modules["self"].facets

    assert not facets.has("resid_pre")
    assert not facets.has("resid_mid")
    assert not facets.has("resid_post")


def test_residual_recipe_marks_genuine_transformer_block() -> None:
    """A genuine attn+mlp block exposes resid_pre/mid/post; mid is not degenerate."""

    torch.manual_seed(0)
    log = tl.trace(
        _GenuineBlock(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    facets = log.modules["self"].facets

    assert facets.has("resid_pre")
    assert facets.has("resid_mid")
    assert facets.has("resid_post")
    # resid_mid is the post-attention add, not the first-add fallback that collapses
    # onto the block output (resid_post).
    assert not torch.equal(facets.resid_mid.value, facets.resid_post.value)


# --------------------------------------------------------------------------------------
# H8 - patching state / RNG safety
# --------------------------------------------------------------------------------------


class _StatefulBlock(nn.Module):
    """A transformer block whose forward mutates a buffer that affects its output."""

    def __init__(self) -> None:
        """Initialize attention, MLP, and a per-forward counter buffer."""

        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.mlp = nn.Linear(4, 4)
        self.register_buffer("counter", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run an attention/MLP residual update scaled by the mutating counter."""

        self.counter += 1.0
        resid_mid = x + self.attn(x) * self.counter
        return resid_mid + self.mlp(resid_mid)


class _StatefulModel(nn.Module):
    """Wrapper exposing a stateful transformer block as a non-root module."""

    def __init__(self) -> None:
        """Initialize the stateful block."""

        super().__init__()
        self.block = _StatefulBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stateful block."""

        return self.block(x)


def _metric(log: object) -> torch.Tensor:
    """Return a scalar metric from a trace output."""

    return log[log.output_layers[0]].out.sum()  # type: ignore[index]


def _patch_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return clean and corrupted toy inputs."""

    torch.manual_seed(0)
    return torch.randn(1, 3, 4), torch.randn(1, 3, 4)


def test_activation_patch_preserves_caller_model_state_and_rng() -> None:
    """Residual-stream patching must leave the caller's model state and RNG untouched."""

    # A real session has already initialized torchlens; warm it up so the one-time
    # first-capture setup does not perturb the RNG snapshot under test.
    tl.trace(nn.Linear(3, 3), torch.randn(2, 3))
    model = _StatefulModel()
    clean, corrupted = _patch_inputs()

    counter_before = model.block.counter.clone()
    rng_before = torch.get_rng_state().clone()
    tl.facets.patching.activation_patch_residual_stream(model, clean, corrupted, _metric)

    assert torch.equal(model.block.counter, counter_before)
    assert torch.equal(rng_before, torch.get_rng_state())


def test_activation_patch_is_idempotent_across_calls() -> None:
    """Two identical patching calls give identical results (no cross-call state drift)."""

    model = _StatefulModel()
    clean, corrupted = _patch_inputs()

    first = tl.facets.patching.activation_patch_residual_stream(model, clean, corrupted, _metric)
    second = tl.facets.patching.activation_patch_residual_stream(model, clean, corrupted, _metric)

    assert torch.equal(first, second)


# --------------------------------------------------------------------------------------
# M9 (obs-local) - observer span direction enforcement
# --------------------------------------------------------------------------------------


class _LinearRelu(nn.Module):
    """Tiny linear + ReLU module with a backward-capable ReLU grad_fn."""

    def __init__(self) -> None:
        """Initialize a single linear layer."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer followed by ReLU."""

        return torch.relu(self.linear(x))


def test_backward_record_span_direction_enforced() -> None:
    """A backward tap record carries backward/both spans, never a forward-only span."""

    torch.manual_seed(12)
    model = _LinearRelu()
    x = torch.randn(2, 3)
    tap = tl.tap(tl.grad_fn(type="relu"), direction="backward")

    with tl.span("fwd_only", direction="forward"), tl.span("bwd_ok", direction="backward"):
        trace = tl.trace(
            model, x, capture=tl.options.CaptureOptions(intervention_ready=True, hooks=tap)
        )
        trace.log_backward(trace[trace.output_layers[0]].out.sum())

    assert tap.records
    record = tap.records[0]
    assert record.direction == "backward"
    assert "fwd_only" not in record.span_names
    assert "bwd_ok" in record.span_names


def test_forward_record_span_direction_enforced() -> None:
    """A forward tap record carries forward/both spans, never a backward-only span."""

    torch.manual_seed(12)
    model = _LinearRelu()
    x = torch.randn(2, 3)
    tap = tl.tap(tl.func("relu"), direction="forward")

    with tl.span("bwd_only", direction="backward"), tl.span("fwd_ok", direction="forward"):
        tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True, hooks=tap))

    assert tap.records
    record = tap.records[0]
    assert record.direction == "forward"
    assert "bwd_only" not in record.span_names
    assert "fwd_ok" in record.span_names


# --------------------------------------------------------------------------------------
# M7 - forward tap site_label resolves to the public trace label
# --------------------------------------------------------------------------------------


def test_forward_tap_site_label_is_public_not_raw() -> None:
    """A forward tap's site_label is the public label, indexing the public trace."""

    torch.manual_seed(0)
    model = _LinearRelu()
    x = torch.randn(2, 3)
    tap = tl.tap(tl.func("relu"))

    log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True, hooks=tap))

    assert tap.records
    site_label = tap.records[0].site_label
    assert site_label is not None
    assert not site_label.endswith("_raw")
    # The public label actually indexes the finalized trace.
    assert site_label in log.layer_labels
    assert log[site_label] is not None
