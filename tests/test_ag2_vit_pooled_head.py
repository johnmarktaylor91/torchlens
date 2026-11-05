"""Attribution graphs on a small randomly initialised torchvision vision transformer.

A vision transformer has no token ids (its embedding output is the patch
embedding after the position embedding, ``encoder.dropout``) and no
per-position logits: ``heads`` reads the class token and returns a pooled
``(1, classes)`` head. ``build_graph`` admits both: token nodes carry no id,
logit nodes carry the last position by convention, and both facts appear in
``disclosures``. Its attention is the fused scaled-dot-product kernel, so there
is no pattern op to freeze: ``frozen=()`` differentiates through the patterns
and the layer norms, and the answer key does the same.

The answer key never calls TorchLens: it runs the same torchvision model under
plain autograd with every transcoder-written MLP output replaced by a fresh
leaf and the embedding output a leaf, then contracts each target's gradient
with each source vector.

Tolerance, stated before the first run: both paths run in float32 on the same
module graph, so every edge must lie within 1e-5 of the key's largest entry.
Negative control: the Neuronpedia export refuses a graph whose token nodes
carry no id with the typed code ``attribution_graph_export_unsupported``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import torchlens as tl

torchvision = pytest.importorskip("torchvision")

N_LAYERS = 2
D_MODEL = 32
N_FEATURES = 8
N_CLASSES = 10
IMAGE_SIZE = 16
PATCH = 8
N_POS = 1 + (IMAGE_SIZE // PATCH) ** 2
EDGE_FRAC = 1e-5


def _mlp(layer: int) -> str:
    return f"encoder.layers.encoder_layer_{layer}.mlp"


class JumpReluTranscoder:
    """A JumpReLU transcoder in the SAE Lens orientation, writing the MLP it reads."""

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
    model = torchvision.models.VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=PATCH,
        num_layers=N_LAYERS,
        num_heads=4,
        hidden_dim=D_MODEL,
        mlp_dim=2 * D_MODEL,
        num_classes=N_CLASSES,
    )
    return model.eval()


def _image() -> torch.Tensor:
    gen = torch.Generator().manual_seed(2)
    return torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=gen)


def _transcoders() -> dict[str, JumpReluTranscoder]:
    gen = torch.Generator().manual_seed(1)
    return {_mlp(layer): JumpReluTranscoder(gen, _mlp(layer)) for layer in range(N_LAYERS)}


def _build(model, transcoders):
    trace = tl.trace(model, _image())
    return tl.attribution.build_graph(
        trace,
        transcoders,
        embedding_module="encoder.dropout",
        max_num_logits=3,
        logit_probability_coverage=0.99,
        frozen=(),
    )


def _layer_of(site: str) -> int:
    return int(site.split(".")[2].rsplit("_", 1)[1])


def _key(model, transcoders, graph) -> torch.Tensor:
    """Adjacency by plain autograd with leaf MLP outputs and embedding (never TorchLens)."""

    leaves: dict[object, torch.Tensor] = {}
    inputs: dict[int, torch.Tensor] = {}
    handles = []

    def leaf(name):
        def hook(_module, _args, output):
            fresh = output.detach().requires_grad_(True)
            leaves[name] = fresh
            return fresh

        return hook

    def grab(layer):
        def hook(_module, args):
            inputs[layer] = args[0]

        return hook

    handles.append(model.encoder.dropout.register_forward_hook(leaf("embed")))
    for layer in range(N_LAYERS):
        block = getattr(model.encoder.layers, f"encoder_layer_{layer}")
        handles.append(block.mlp.register_forward_hook(leaf(layer)))
        handles.append(block.mlp.register_forward_pre_hook(grab(layer)))
    try:
        logits = model(_image())
    finally:
        for handle in handles:
            handle.remove()
    sources = [leaves[layer] for layer in range(N_LAYERS)] + [leaves["embed"]]
    nodes = graph.nodes
    key = torch.zeros(len(nodes), len(nodes), dtype=torch.float64)
    with torch.no_grad():
        acts = {
            layer: transcoders[_mlp(layer)].encode(inputs[layer][0]) for layer in range(N_LAYERS)
        }
    for row, node in enumerate(nodes):
        if node.kind == "feature":
            layer = _layer_of(node.site)
            reader = transcoders[node.site]
            value = reader.pre(inputs[layer][0, node.position])[node.feature_index]
        elif node.kind == "logit":
            z = logits[0]
            value = z[node.token_id] - z.mean()
        else:
            continue
        grads = torch.autograd.grad(value, sources, retain_graph=True, allow_unused=True)
        grads = [
            torch.zeros_like(s) if g is None else g for g, s in zip(grads, sources, strict=True)
        ]
        for col, source in enumerate(nodes):
            if source.kind == "token":
                vec = leaves["embed"][0, source.position]
                g = grads[-1][0, source.position]
            elif source.kind in {"feature", "error"}:
                layer = _layer_of(source.site)
                g = grads[layer][0, source.position]
                reader = transcoders[_mlp(layer)]
                write = _mlp(layer)
                if source.kind == "feature":
                    vec = (
                        acts[layer][source.position, source.feature_index]
                        * reader.W_dec[write][source.feature_index]
                    )
                else:
                    recon = acts[layer][source.position] @ reader.W_dec[write] + reader.b_dec[write]
                    vec = leaves[layer][0, source.position] - recon
            else:
                continue
            key[row, col] = float((g.double() * vec.detach().double()).sum())
    return key


class TestVisionTransformerPooledHead:
    """Image input (no token ids) and a pooled class-token head."""

    def test_edges_match_leaf_linearisation_key(self) -> None:
        model = _model()
        transcoders = _transcoders()
        graph = _build(model, transcoders)
        counts = graph.node_counts
        assert counts["feature"] > 0 and counts["logit"] >= 1
        assert counts["error"] == N_LAYERS * N_POS
        assert counts["token"] == N_POS
        key = _key(model, transcoders, graph)
        scale = key.abs().max().item()
        assert scale > 0
        gap = (graph.adjacency_matrix.double() - key).abs().max().item()
        assert gap <= EDGE_FRAC * scale, (gap, scale)

    def test_pooled_head_and_imageless_tokens_are_disclosed(self) -> None:
        graph = _build(_model(), _transcoders())
        assert graph.input_token_ids.numel() == 0
        assert all(graph.nodes[i].token_id is None for i in graph.node_indices("token"))
        assert all(graph.nodes[i].position == N_POS - 1 for i in graph.node_indices("logit"))
        assert graph.logit_token_ids.tolist() == [
            graph.nodes[i].token_id for i in graph.node_indices("logit")
        ]
        notes = " ".join(graph.disclosures)
        assert "pooled (1, classes) head" in notes
        assert "token nodes carry no id" in notes

    def test_neuronpedia_export_refuses_idless_tokens(self) -> None:
        graph = _build(_model(), _transcoders())
        tokenizer = SimpleNamespace(decode=lambda ids: " ".join(str(i) for i in ids))
        with pytest.raises(tl.errors.AttributionError) as info:
            graph.to_neuronpedia(slug="vit", model_id="vit-test", tokenizer=tokenizer)
        assert info.value.fields["code"] == "attribution_graph_export_unsupported"
