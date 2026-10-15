"""Independent multi-receiver edge patching (plain PyTorch forward hooks, no TorchLens).

The convention (MIB circuit track): in ONE forward, every patched receiver
``t`` reads ``input_t - sum_{s in P_t} (s - s_corrupt)``, where ``input_t``
and each ``s`` are the values of that same patched forward and
``s_corrupt`` is the sender's value on the corrupted input. Senders that are
not patched into ``t`` keep their live values.

``HookedPatch`` registers forward hooks on a model's sender and receiver
modules from a plain description (which module writes which node, which norm
each receiver reads through), runs the forward, and records the outputs of
named modules. Attention receivers ``a{L}.h{H}<q|k|v>`` of a Hugging Face
GPT-2 block are handled by recomputing head ``H``'s role columns of
``c_attn`` from the patched pre-norm input.

This module never imports torchlens.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PatchSpec:
    """Where the nodes and receivers of one model live.

    ``node_writers``: node name -> (module, function of (inputs, output) -> node value),
    called from a forward hook (``inputs`` is the hook's positional args).
    ``norm_receivers``: receiver name -> norm module whose sole input is the receiver's input.
    ``gpt2_blocks``: Hugging Face GPT-2 blocks, for ``a{L}.h{H}<q|k|v>`` receivers.
    ``record``: name -> module whose output is recorded after the forward.
    """

    node_writers: Mapping[str, tuple[nn.Module, Callable[..., torch.Tensor]]]
    norm_receivers: Mapping[str, nn.Module]
    gpt2_blocks: list[nn.Module] = field(default_factory=list)
    record: Mapping[str, nn.Module] = field(default_factory=dict)


def gpt2_spec(model: nn.Module) -> PatchSpec:
    """The node and receiver map of a Hugging Face ``GPT2LMHeadModel``."""

    blocks = list(model.transformer.h)
    n_heads = int(model.config.n_head)
    d_head = int(model.config.n_embd) // n_heads
    writers: dict[str, tuple[nn.Module, Callable[..., torch.Tensor]]] = {
        "input": (model.transformer.drop, lambda args, out: out)
    }
    receivers: dict[str, nn.Module] = {"logits": model.transformer.ln_f}
    record: dict[str, nn.Module] = {"ln_f": model.transformer.ln_f, "lm_head": model.lm_head}
    for layer, block in enumerate(blocks):
        proj = block.attn.c_proj
        for head in range(n_heads):
            cols = slice(head * d_head, (head + 1) * d_head)

            def head_out(args, out, cols=cols, proj=proj):
                return args[0][..., cols] @ proj.weight[cols, :]

            writers[f"a{layer}.h{head}"] = (proj, head_out)
        writers[f"b{layer}"] = (proj, lambda args, out, proj=proj: proj.bias.expand_as(out))
        writers[f"m{layer}"] = (block.mlp, lambda args, out: out)
        receivers[f"m{layer}"] = block.ln_2
        record[f"c_attn{layer}"] = block.attn.c_attn
        record[f"ln_2.{layer}"] = block.ln_2
    return PatchSpec(writers, receivers, blocks, record)


class HookedPatch:
    """One hooked forward under the multi-receiver patching convention."""

    def __init__(self, model: nn.Module, spec: PatchSpec) -> None:
        self.model = model
        self.spec = spec

    def run(
        self,
        ids: torch.Tensor,
        patched: Mapping[str, list[str]] | None = None,
        corrupt: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return ``(node values, recorded module outputs)`` of one forward."""

        patched = dict(patched or {})
        nodes: dict[str, torch.Tensor] = {}
        recorded: dict[str, torch.Tensor] = {}
        handles = []

        def delta(receiver: str) -> torch.Tensor | int:
            total: torch.Tensor | int = 0
            for sender in patched.get(receiver, []):
                total = total + (nodes[sender] - corrupt[sender])
            return total

        for name, (module, fn) in self.spec.node_writers.items():

            def write(mod, args, out, name=name, fn=fn):
                nodes[name] = fn(args, out)

            handles.append(module.register_forward_hook(write))
        for name, norm in self.spec.norm_receivers.items():
            if name in patched:

                def pre(mod, args, name=name):
                    return (args[0] - delta(name),) + tuple(args[1:])

                handles.append(norm.register_forward_pre_hook(pre))
        for layer, block in enumerate(self.spec.gpt2_blocks):
            handles.extend(self._attention_hooks(layer, block, patched, delta))
        for name, module in self.spec.record.items():

            def keep(mod, args, out, name=name):
                recorded[name] = out.detach().clone()

            handles.append(module.register_forward_hook(keep))
        try:
            with torch.no_grad():
                self.model(ids)
        finally:
            for handle in handles:
                handle.remove()
        return nodes, recorded

    def _attention_hooks(self, layer, block, patched, delta):
        """Hooks that rewrite head ``H``'s role columns of ``c_attn`` from a patched input."""

        wanted = [name for name in patched if name.startswith(f"a{layer}.h") and "<" in name]
        if not wanted:
            return []
        n_heads = int(self.model.config.n_head)
        d_model = int(self.model.config.n_embd)
        d_head = d_model // n_heads
        state: dict[str, torch.Tensor] = {}

        def pre_ln(mod, args):
            state["resid"] = args[0]

        def rewrite(mod, args, out):
            out = out.clone()
            ln = block.ln_1
            for name in wanted:
                head = int(name.split(".h")[1].split("<")[0])
                role = "qkv".index(name[-2])
                cols = slice(role * d_model + head * d_head, role * d_model + (head + 1) * d_head)
                x = F.layer_norm(
                    state["resid"] - delta(name), (d_model,), ln.weight, ln.bias, ln.eps
                )
                out[..., cols] = x @ mod.weight[:, cols] + mod.bias[cols]
            return out

        return [
            block.ln_1.register_forward_pre_hook(pre_ln),
            block.attn.c_attn.register_forward_hook(rewrite),
        ]
