"""A02 refusal + tripwire rows: anchors refuse rather than guess.

Planted-wrong and refusal rows for the residual/lm-head dataflow anchors
(mikit gate 1 item 7 discipline: every wrong implementation must FAIL typed,
never return a plausible tensor):

- resid_pre refuses on ambiguity (two matching streams) and on no-match
  (shape-anchored), with teaching detail.
- The hop rule's value grammar fails closed on train-mode dropout, on
  advanced (tensor) indexing, and on dimension-dropping subscripts; eval-mode
  dropout hops and verifies.
- Corrupted endpoint payloads FAIL the walk's payload-identity verification
  (the tripwire is armed, not decorative).
- tl.trace on a transformer_lens TransformerBridge refuses typed with the
  pristine-copy remedy (mikit F10); a namesake outside the transformer_lens
  namespace still traces.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


class _Block(nn.Module):
    """Minimal block that classifies as a transformer block (attn + mlp)."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.attn = nn.Linear(d, d)
        self.mlp = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


# ---------------------------------------------------------------------------
# resid_pre refusals
# ---------------------------------------------------------------------------


def test_resid_pre_refuses_on_ambiguous_streams() -> None:
    """Two same-shaped floating inputs that both feed the output refuse:
    resid_pre is never guessed from input order."""

    class TwoStreamBlock(nn.Module):
        def __init__(self, d: int = 8) -> None:
            super().__init__()
            self.attn = nn.Linear(d, d)
            self.mlp = nn.Linear(d, d)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y + self.mlp(self.attn(x))

    class Host(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = TwoStreamBlock()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x * 1.0, x * 2.0)

    log = tl.trace(Host().eval(), torch.randn(2, 3, 8))
    block = log.modules["block"]
    item = block.facets.menu()["resid_pre"]
    assert item.status == "structurally_absent"
    assert "ambiguous" in (item.detail or "")
    with pytest.raises((KeyError, AttributeError)):
        _ = block.facets.resid_pre


def test_resid_pre_refuses_when_no_input_matches_by_shape() -> None:
    """A block that narrows its output has no residual stream input: typed
    structural absence, teaching the dataflow + shape rule."""

    class NarrowingBlock(nn.Module):
        def __init__(self, d: int = 8) -> None:
            super().__init__()
            self.attn = nn.Linear(d, d)
            self.mlp = nn.Linear(d, d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = x + self.attn(x)
            return (y + self.mlp(y))[:, :, : y.shape[-1] // 2]

    class Host(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = NarrowingBlock()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    log = tl.trace(Host().eval(), torch.randn(2, 3, 8))
    item = log.modules["block"].facets.menu()["resid_pre"]
    assert item.status == "structurally_absent"
    assert "dataflow + shape" in (item.detail or "")


# ---------------------------------------------------------------------------
# Hop-rule grammar + verification tripwires (final-norm walk)
# ---------------------------------------------------------------------------


class _NormHead(nn.Module):
    """LayerNorm -> dropout -> lm_head: one grammar-gated hop on the walk."""

    def __init__(self, d: int = 8, vocab: int = 11, p: float = 0.5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(p)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.drop(self.ln(x)))


def test_final_norm_walk_hops_inert_dropout_in_eval_mode() -> None:
    """Eval-mode dropout is a proposed AND payload-verified hop: the norm
    anchors and gamma reads bitwise-equal to the live parameter."""

    model = _NormHead().eval()
    log = tl.trace(model, torch.randn(2, 3, 8))
    facets = log.modules["self"].facets
    assert facets.final_norm_kind == "layer_norm"
    assert torch.equal(facets.final_norm_gamma.value, model.ln.weight)


def test_final_norm_walk_refuses_train_mode_dropout() -> None:
    """Train-mode dropout is NOT value-preserving: the value grammar fails
    closed and the norm facets are typed-absent, naming the hop rule."""

    model = _NormHead().train()
    log = tl.trace(model, torch.randn(2, 3, 8))
    menu = log.modules["self"].facets.menu()
    assert menu["final_norm_kind"].status == "structurally_absent"
    assert "hop" in (menu["final_norm_kind"].detail or "")


def test_final_norm_walk_refuses_tensor_indexing() -> None:
    """Advanced (tensor) indexing between norm and head is not a hop: a
    data-dependent gather can never pose as a position mapping."""

    class GatherHead(nn.Module):
        def __init__(self, d: int = 8, vocab: int = 11) -> None:
            super().__init__()
            self.ln = nn.LayerNorm(d)
            self.lm_head = nn.Linear(d, vocab, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            keep = torch.tensor([0, 2])
            return self.lm_head(self.ln(x)[:, keep, :])

    log = tl.trace(GatherHead().eval(), torch.randn(2, 3, 8))
    menu = log.modules["self"].facets.menu()
    assert menu["final_norm_kind"].status == "structurally_absent"


def test_final_norm_walk_refuses_dimension_dropping_subscript() -> None:
    """An int subscript drops the position dimension; the index map would not
    be dimension-aligned, so the walk refuses rather than discloses wrong."""

    class LastTokenHead(nn.Module):
        def __init__(self, d: int = 8, vocab: int = 11) -> None:
            super().__init__()
            self.ln = nn.LayerNorm(d)
            self.lm_head = nn.Linear(d, vocab, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lm_head(self.ln(x)[:, -1])

    log = tl.trace(LastTokenHead().eval(), torch.randn(2, 3, 8))
    menu = log.modules["self"].facets.menu()
    assert menu["final_norm_kind"].status == "structurally_absent"


def test_hop_walk_payload_identity_tripwire_fires_on_corruption() -> None:
    """A corrupted anchor payload FAILS the endpoint verification: the walk
    refuses instead of anchoring -- the tripwire is armed, not decorative."""

    from torchlens.semantic._hops import HopRefusal, HopWalk, walk_upstream

    model = _NormHead().eval()
    log = tl.trace(model, torch.randn(2, 3, 8))
    head_call = log.modules["lm_head"]._single_call_or_error()
    start = log.ops[list(head_call.input_ops)[0]]

    def _is_norm_output(op) -> bool:
        stack = tuple(op.modules or ())
        return bool(stack) and str(stack[-1]).startswith("ln")

    healthy = walk_upstream(log, start, _is_norm_output)
    assert isinstance(healthy, HopWalk)
    assert healthy.verification == "payload_identity"
    anchor = healthy.anchor
    anchor.out = anchor.out + 1.0
    corrupted = walk_upstream(log, start, _is_norm_output)
    assert isinstance(corrupted, HopRefusal)
    assert "payload identity FAILED" in corrupted.reason


# ---------------------------------------------------------------------------
# F10: assignment-redirecting wrappers refuse typed at capture entry
# ---------------------------------------------------------------------------


class _WrappedInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _make_fake_bridge_class() -> type:
    """Replicate the bridge's assignment redirection under its namespace."""

    class TransformerBridge(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            object.__setattr__(self, "_wrapped", _WrappedInner())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self._wrapped(x)

        def __setattr__(self, name: str, value) -> None:
            wrapped = self.__dict__.get("_wrapped")
            if wrapped is not None:
                setattr(wrapped, name, value)
                return
            super().__setattr__(name, value)

    TransformerBridge.__module__ = "transformer_lens.model_bridge.bridge"
    return TransformerBridge


def test_transformer_bridge_refuses_typed_with_pristine_copy_remedy() -> None:
    from torchlens._model_wrappers import UninstrumentableModelWrapperError
    from torchlens.errors import CompatibilityError

    bridge = _make_fake_bridge_class()()
    with pytest.raises(UninstrumentableModelWrapperError) as excinfo:
        tl.trace(bridge, torch.randn(2, 4))
    assert isinstance(excinfo.value, CompatibilityError)
    assert excinfo.value.fields["code"] == "model_wrapper_uninstrumentable"
    assert excinfo.value.fields["remedy"]
    message = str(excinfo.value)
    assert "PRISTINE" in message or "pristine" in message
    assert "HookedTransformer" in message


def test_transformer_bridge_namesake_outside_namespace_traces() -> None:
    """Refusal is an exact structural match: a class merely NAMED
    TransformerBridge outside the transformer_lens namespace captures."""

    class TransformerBridge(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    log = tl.trace(TransformerBridge().eval(), torch.randn(2, 4))
    assert log.outcome.status.name == "COMPLETE"
