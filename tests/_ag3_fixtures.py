"""Fixture F4: a cross-layer transcoder on a tiny Hugging Face Gemma-2, and its answer key.

The transcoder follows circuit-tracer's cross-layer layout on Gemma-2: the layer-l reader
reads ``hook_resid_mid`` (the residual add whose output enters
``model.layers.l.pre_feedforward_layernorm``, an op label) and writes ``hook_mlp_out`` (the
output of ``model.layers.L.post_feedforward_layernorm``) for every layer ``L >= l``. Its
decoders are random, so it does not reconstruct the MLPs and the error nodes are nonzero.

The answer key never calls TorchLens. It runs the same Hugging Face model under plain
autograd with ``torch.rsqrt`` and ``softmax`` patched to detach (the frozen norm scales and
attention patterns), and replaces each MLP output by the transcoder writes plus a fixed
error, each source scaled by a scalar leaf equal to one: one leaf per active feature, per
error node and per token position, and one per decoder-bias write. An edge is the gradient
of a target with respect to its source's leaf.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

N_LAYERS = 3
D_MODEL = 16
N_FEATURES = 8
VOCAB = 64
TOKENS = (2, 17, 5, 41, 9)
TARGETS = [3, 17, 40]
WRITE = "model.layers.{}.post_feedforward_layernorm"
READ = "model.layers.{}.pre_feedforward_layernorm"


def build_model(dtype: torch.dtype = torch.float64) -> torch.nn.Module:
    """Return the tiny randomly initialised Gemma-2 (three layers, eager attention)."""

    import transformers

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
    return model.to(dtype)


class CrossLayerTranscoder:
    """One reader layer of a ReLU cross-layer transcoder (SAE Lens orientation)."""

    def __init__(
        self,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: dict[str, torch.Tensor],
        b_dec: dict[str, torch.Tensor],
    ) -> None:
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.W_dec = W_dec
        self.b_dec = b_dec

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.pre(x))


def layer_transcoders(dtype: torch.dtype = torch.float64) -> list[CrossLayerTranscoder]:
    """Return the F4 readers in layer order; reader ``l`` writes every MLP output from ``l`` on."""

    gen = torch.Generator().manual_seed(7)
    readers = []
    for layer in range(N_LAYERS):
        writes = [WRITE.format(w) for w in range(layer, N_LAYERS)]

        def rand(*shape: int, scale: float) -> torch.Tensor:
            return (scale * torch.randn(*shape, generator=gen, dtype=torch.float64)).to(dtype)

        readers.append(
            CrossLayerTranscoder(
                rand(D_MODEL, N_FEATURES, scale=D_MODEL**-0.5),
                rand(N_FEATURES, scale=0.3),
                {w: rand(N_FEATURES, D_MODEL, scale=0.4) for w in writes},
                {w: rand(D_MODEL, scale=0.05) for w in writes},
            )
        )
    return readers


def read_label(trace, layer: int) -> str:
    """Return the pass-qualified op label of ``hook_resid_mid`` at ``layer``.

    A module's ``input_ops`` are layer labels without a pass, and on this tiny model the
    residual add is a multi-pass layer, so the read is the last op with that layer label before
    the norm's first op.
    """

    name = READ.format(layer)
    (base,) = list(trace.modules[name].input_ops)
    previous = None
    for op in trace:
        if str(op.module or "").split(":")[0] == name:
            break
        if str(op.label).split(":")[0] == str(base):
            previous = str(op.label)
    assert previous is not None, (layer, base)
    return previous


def keyed(trace, readers: list[CrossLayerTranscoder]) -> dict[str, CrossLayerTranscoder]:
    """Key each reader by its ``hook_resid_mid`` op label on ``trace``."""

    return {read_label(trace, layer): reader for layer, reader in enumerate(readers)}


def subset(reader: CrossLayerTranscoder, idx: torch.Tensor) -> CrossLayerTranscoder:
    """Return ``reader`` restricted to the features ``idx`` (in increasing order)."""

    return CrossLayerTranscoder(
        reader.W_enc[:, idx].clone(),
        reader.b_enc[idx].clone(),
        {w: d[idx].clone() for w, d in reader.W_dec.items()},
        {w: b.clone() for w, b in reader.b_dec.items()},
    )


@contextmanager
def detached_norms_and_patterns() -> Iterator[None]:
    """Patch ``torch.rsqrt`` and ``softmax`` to detach their outputs (the frozen sites)."""

    rsqrt, softmax = torch.rsqrt, torch.nn.functional.softmax
    torch.rsqrt = lambda *a, **k: rsqrt(*a, **k).detach()
    torch.nn.functional.softmax = lambda *a, **k: softmax(*a, **k).detach()
    try:
        yield
    finally:
        torch.rsqrt, torch.nn.functional.softmax = rsqrt, softmax


@dataclass
class Clean:
    """The plain forward's values the key holds fixed."""

    reads: list[torch.Tensor]
    mlp_out: list[torch.Tensor]
    embed: torch.Tensor
    logits: torch.Tensor


def hooked_forward(model, write_hook=None, embed_hook=None) -> tuple[list, torch.Tensor]:
    """Run the model once; return the read inputs (live) and the pre-softcap logits."""

    reads: dict[int, torch.Tensor] = {}
    out: dict[str, torch.Tensor] = {}
    handles = []
    for layer, block in enumerate(model.model.layers):
        handles.append(
            block.pre_feedforward_layernorm.register_forward_pre_hook(
                lambda _m, args, layer=layer: reads.__setitem__(layer, args[0][0])
            )
        )
        if write_hook is not None:
            handles.append(
                block.post_feedforward_layernorm.register_forward_hook(
                    lambda _m, _a, o, layer=layer: write_hook(layer, o)
                )
            )
    if embed_hook is not None:
        handles.append(
            model.model.embed_tokens.register_forward_hook(lambda _m, _a, o: embed_hook(o))
        )
    handles.append(
        model.lm_head.register_forward_hook(lambda _m, _a, o: out.__setitem__("z", o[0]))
    )
    try:
        model(torch.tensor([TOKENS]))
    finally:
        for handle in handles:
            handle.remove()
    return [reads[layer] for layer in range(N_LAYERS)], out["z"]


def clean_forward(model) -> Clean:
    """Return the plain forward's read inputs, MLP outputs, embedding output and logits."""

    mlp_out: dict[int, torch.Tensor] = {}
    embed: dict[str, torch.Tensor] = {}

    def grab(layer, o):
        mlp_out[layer] = o.detach()[0].clone()

    def grab_embed(o):
        embed["e"] = o.detach()[0].clone()

    with torch.no_grad():
        reads, z = hooked_forward(model, grab, grab_embed)
    return Clean(
        [r.detach().clone() for r in reads],
        [mlp_out[layer] for layer in range(N_LAYERS)],
        embed["e"],
        z.detach().clone(),
    )


def writes_at(readers: list[CrossLayerTranscoder], clean: Clean, layer: int) -> torch.Tensor:
    """Return every reader's write into MLP output ``layer``, decoder biases included."""

    write = WRITE.format(layer)
    total = torch.zeros_like(clean.mlp_out[layer])
    for reader_layer, reader in enumerate(readers[: layer + 1]):
        acts = reader.encode(clean.reads[reader_layer])
        total = total + acts @ reader.W_dec[write] + reader.b_dec[write]
    return total
