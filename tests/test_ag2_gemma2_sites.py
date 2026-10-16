"""Attribution graphs on a tiny randomly initialised Hugging Face Gemma-2.

Gemma-2 has no module for its attention pattern or its RMSNorm scale: the
pattern is the ``softmax`` op inside each ``self_attn`` (eager attention) and
the scale is the ``rsqrt`` op inside each ``Gemma2RMSNorm``. ``build_graph``
freezes them through op labels in ``frozen=``. Its logits are seeded before
the final softcap (``logits_at=SiteTarget("lm_head")``), and its token nodes
read ``model.embed_tokens``, whose output Gemma-2 scales by sqrt(d_model)
afterwards (an edge is the gradient there times the value there, so the
scale cancels).

The answer key never calls TorchLens: it runs the same Hugging Face model
under plain autograd with circuit-tracer's linearisation spelled as
``detach()`` (every ``softmax`` and ``rsqrt`` output detached, every
post-feedforward norm output replaced by a fresh leaf, the embedding output
a leaf), then contracts each target's gradient with each source vector.

Tolerance, stated before the first run: the Gemma-2 norm casts to float32 and
back, so both paths carry float32 rounding; every edge must lie within 1e-5
of the key's largest entry. Negative control: leaving the norm scales
unfrozen misses the key by more than 1e-3 of its largest entry.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import torchlens as tl

transformers = pytest.importorskip("transformers")

N_LAYERS = 2
D_MODEL = 16
N_FEATURES = 8
VOCAB = 64
TOKENS = (2, 17, 5, 41, 9)
EDGE_FRAC = 1e-5
CONTROL_FRAC = 1e-3


class JumpReluTranscoder:
    """A Gemma Scope style transcoder (JumpReLU) in the SAE Lens orientation."""

    def __init__(self, gen: torch.Generator, write: str) -> None:
        self.W_enc = torch.randn(D_MODEL, N_FEATURES, generator=gen) / D_MODEL**0.5
        self.b_enc = 0.1 * torch.randn(N_FEATURES, generator=gen)
        self.threshold = torch.full((N_FEATURES,), 0.05)
        self.W_dec = {write: torch.randn(N_FEATURES, D_MODEL, generator=gen) / N_FEATURES**0.5}
        self.b_dec = {write: 0.05 * torch.randn(D_MODEL, generator=gen)}

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.pre(x)
        return pre * (pre > self.threshold).to(pre.dtype)


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    cfg = transformers.Gemma2Config(
        vocab_size=VOCAB,
        hidden_size=D_MODEL,
        intermediate_size=32,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        query_pre_attn_scalar=8,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        sliding_window=4096,
        attn_implementation="eager",
    )
    cfg._attn_implementation = "eager"
    model = transformers.Gemma2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "layernorm" in name or name.endswith("norm.weight"):
                param.copy_(0.3 * torch.randn_like(param))
    return model


def _transcoders() -> dict[str, JumpReluTranscoder]:
    gen = torch.Generator().manual_seed(1)
    return {
        f"model.layers.{layer}.mlp": JumpReluTranscoder(
            gen, f"model.layers.{layer}.post_feedforward_layernorm"
        )
        for layer in range(N_LAYERS)
    }


def _frozen_ops(trace: tl.Trace, *, scales: bool = True) -> list[str]:
    patterns, norms = [], []
    for op in trace:
        module = str(op.module or "").split(":")[0]
        if op.func_name == "softmax" and module.endswith(".self_attn"):
            patterns.append(str(op.label))
        elif op.func_name == "rsqrt" and (module.endswith("layernorm") or module == "model.norm"):
            norms.append(str(op.label))
    assert len(patterns) == N_LAYERS
    assert len(norms) == 4 * N_LAYERS + 1
    return patterns + (norms if scales else [])


#: The one TorchLens warning a Hugging Face Gemma-2 capture may raise: the
#: attention-mask builder runs under vmap (no effect on the graph's sites).
VMAP_BOUNDARY = "captured a vmap transform as a boundary op"


def _trace(model) -> tl.Trace:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, torch.tensor([TOKENS]))
    for warning in caught:
        if issubclass(warning.category, UserWarning) and "TorchLens" in str(warning.message):
            assert VMAP_BOUNDARY in str(warning.message), str(warning.message)
    return trace


def _build(model, transcoders, *, scales: bool = True):
    trace = _trace(model)
    return tl.attribution.build_graph(
        trace,
        transcoders,
        embedding="model.embed_tokens",
        logits_at=tl.attribution.SiteTarget("lm_head"),
        max_n_logits=3,
        desired_logit_prob=0.99,
        frozen=_frozen_ops(trace, scales=scales),
    )


def _key(model, transcoders, graph, monkeypatch) -> torch.Tensor:
    """Adjacency by plain autograd under detach-style linearisation (never TorchLens)."""

    rsqrt, softmax = torch.rsqrt, torch.nn.functional.softmax
    monkeypatch.setattr(torch, "rsqrt", lambda *a, **k: rsqrt(*a, **k).detach())
    monkeypatch.setattr(torch.nn.functional, "softmax", lambda *a, **k: softmax(*a, **k).detach())
    leaves: dict[str, torch.Tensor] = {}
    inputs: dict[str, torch.Tensor] = {}
    handles = []

    def leaf(name):
        def hook(_module, _args, output):
            fresh = output.detach().requires_grad_(True)
            leaves[name] = fresh
            return fresh

        return hook

    def grab(name):
        def hook(_module, args):
            inputs[name] = args[0]

        return hook

    handles.append(model.model.embed_tokens.register_forward_hook(leaf("embed")))
    for layer in range(N_LAYERS):
        block = model.model.layers[layer]
        handles.append(block.post_feedforward_layernorm.register_forward_hook(leaf(layer)))
        handles.append(block.mlp.register_forward_pre_hook(grab(layer)))
    pre_softcap: dict[str, torch.Tensor] = {}
    handles.append(
        model.lm_head.register_forward_hook(lambda _m, _a, out: pre_softcap.update(z=out))
    )
    try:
        model(torch.tensor([TOKENS]))
    finally:
        for handle in handles:
            handle.remove()
    sources = [leaves[layer] for layer in range(N_LAYERS)] + [leaves["embed"]]
    nodes = graph.nodes
    n_nodes = len(nodes)
    key = torch.zeros(n_nodes, n_nodes, dtype=torch.float64)
    with torch.no_grad():
        acts = {
            layer: transcoders[f"model.layers.{layer}.mlp"].encode(inputs[layer][0])
            for layer in range(N_LAYERS)
        }
    for row, node in enumerate(nodes):
        if node.kind == "feature":
            layer = int(node.site.split(".")[2])
            reader = transcoders[node.site]
            value = reader.pre(inputs[layer][0, node.position])[node.feature]
        elif node.kind == "logit":
            z = pre_softcap["z"][0, -1]
            value = z[node.token] - z.mean()
        else:
            continue
        grads = torch.autograd.grad(value, sources, retain_graph=True, allow_unused=True)
        grads = [torch.zeros_like(s) if g is None else g for g, s in zip(grads, sources)]
        for col, source in enumerate(nodes):
            if source.kind == "token":
                vec = leaves["embed"][0, source.position]
                g = grads[-1][0, source.position]
            elif source.kind in {"feature", "error"}:
                layer = int(source.site.split(".")[2])
                g = grads[layer][0, source.position]
                write = f"model.layers.{layer}.post_feedforward_layernorm"
                reader = transcoders[f"model.layers.{layer}.mlp"]
                if source.kind == "feature":
                    vec = (
                        acts[layer][source.position, source.feature]
                        * reader.W_dec[write][source.feature]
                    )
                else:
                    recon = acts[layer][source.position] @ reader.W_dec[write] + reader.b_dec[write]
                    vec = leaves[layer][0, source.position] - recon
            else:
                continue
            key[row, col] = float((g.double() * vec.detach().double()).sum())
    return key


class TestGemma2Sites:
    """Gemma-2's pattern and norm-scale ops, pre-softcap logits and scaled embedding."""

    def test_edges_match_detach_linearisation_key(self, monkeypatch) -> None:
        model = _model()
        transcoders = _transcoders()
        graph = _build(model, transcoders)
        counts = graph.node_counts
        assert counts["feature"] > 0 and counts["logit"] >= 1
        assert counts["error"] == N_LAYERS * len(TOKENS)
        key = _key(model, transcoders, graph, monkeypatch)
        scale = key.abs().max().item()
        assert scale > 0
        gap = (graph.adjacency_matrix.double() - key).abs().max().item()
        assert gap <= EDGE_FRAC * scale, (gap, scale)

    def test_unfrozen_norm_scales_miss_the_key(self, monkeypatch) -> None:
        model = _model()
        transcoders = _transcoders()
        loose = _build(model, transcoders, scales=False)
        key = _key(model, transcoders, loose, monkeypatch)
        gap = (loose.adjacency_matrix.double() - key).abs().max().item()
        assert gap > CONTROL_FRAC * key.abs().max().item()
