"""Independent edge-attribution reference for GPT-2 (plain PyTorch, no TorchLens).

Used by the EAP/EAP-IG definition tests and the brute-force edge-patching
measurement. It calls the Hugging Face GPT-2 submodules (``ln_1``,
``c_attn``, ``c_proj``, ``ln_2``, ``mlp``, ``ln_f``, ``lm_head``) directly and
recomputes eager causal attention per head, so every receiver can read its
own copy of the residual stream (the treeified view of the MIB / EAP-IG
graph): ``a{L}.h{H}<q|k|v>`` reads a private copy through ``ln_1`` and only
head ``H``'s role columns of ``c_attn``; ``m{L}`` and ``logits`` read private
copies through ``ln_2`` and ``ln_f``.

* :meth:`TreeGPT2.forward` returns the node values (``input``,
  ``a{L}.h{H}`` without the output bias, ``b{L}``, ``m{L}``), the logits and,
  when ``grads=True``, a zero leaf added to each receiver copy, so one
  backward gives every receiver's own gradient.
* :meth:`TreeGPT2.patched_metric` runs the forward from one receiver's layer
  with that receiver's input replaced (brute-force edge patching), batched
  over the patched variants.

This module never imports torchlens.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


def edge_names(n_layers: int, n_heads: int) -> list[tuple[str, str]]:
    """MIB vocabulary edges ``(sender, receiver)`` of sequential GPT-2 (no bias nodes)."""

    edges: list[tuple[str, str]] = []
    upstream = ["input"]
    for layer in range(n_layers):
        for head in range(n_heads):
            for role in "qkv":
                edges.extend((s, f"a{layer}.h{head}<{role}>") for s in upstream)
        upstream = upstream + [f"a{layer}.h{h}" for h in range(n_heads)]
        edges.extend((s, f"m{layer}") for s in upstream)
        upstream = upstream + [f"m{layer}"]
    edges.extend((s, "logits") for s in upstream)
    return edges


class TreeGPT2:
    """Treeified GPT-2 forward over a Hugging Face ``GPT2LMHeadModel``."""

    def __init__(self, hf_model: torch.nn.Module) -> None:
        self.model = hf_model
        self.blocks = list(hf_model.transformer.h)
        self.n_layers = len(self.blocks)
        self.n_heads = int(hf_model.config.n_head)
        self.d_model = int(hf_model.config.n_embd)
        self.d_head = self.d_model // self.n_heads

    # -- pieces ---------------------------------------------------------------

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Token plus position embedding (the ``input`` node)."""

        tr = self.model.transformer
        pos = torch.arange(ids.shape[1], device=ids.device)
        return tr.wte(ids) + tr.wpe(pos)[None]

    def _qkv_role(self, block, x: torch.Tensor, role: int, head: int) -> torch.Tensor:
        """Head ``head``'s q (0), k (1) or v (2) from a residual copy ``x``."""

        lo = role * self.d_model + head * self.d_head
        weight = block.attn.c_attn.weight[:, lo : lo + self.d_head]
        bias = block.attn.c_attn.bias[lo : lo + self.d_head]
        return block.ln_1(x) @ weight + bias

    def _head_outputs(self, block, q, k, v) -> torch.Tensor:
        """Per-head outputs ``[H, B, P, D]`` (without the output bias) from ``[H, B, P, d]``."""

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        n = scores.shape[-1]
        causal = torch.ones(n, n, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        z = torch.softmax(scores, dim=-1) @ v
        w_o = block.attn.c_proj.weight.view(self.n_heads, self.d_head, self.d_model)
        return torch.einsum("hbpd,hdm->hbpm", z, w_o)

    def _attention(self, block, layer, x, reads, values, leaves, patch, grads):
        """One attention layer: per-head per-role reads, head nodes and the bias node."""

        if grads:
            qkv = []
            for role, name in enumerate("qkv"):
                per_head = []
                for head in range(self.n_heads):
                    copy = reads(f"a{layer}.h{head}<{name}>", x, leaves)
                    per_head.append(self._qkv_role(block, copy, role, head))
                qkv.append(torch.stack(per_head))
        else:
            fused = block.attn.c_attn(block.ln_1(x))
            parts = fused.split(self.d_model, dim=-1)
            shape = (*x.shape[:-1], self.n_heads, self.d_head)
            qkv = [part.view(shape).movedim(-2, 0).clone() for part in parts]
            if patch is not None and patch[0].startswith(f"a{layer}.h"):
                head = int(patch[0].split(".h")[1].split("<")[0])
                role = "qkv".index(patch[0][-2])
                qkv[role][head] = self._qkv_role(block, patch[1](x), role, head)
        heads = self._head_outputs(block, *qkv)
        for head in range(self.n_heads):
            values[f"a{layer}.h{head}"] = heads[head]
        bias = block.attn.c_proj.bias.expand_as(x)
        values[f"b{layer}"] = bias
        return heads.sum(0) + bias

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        ids: torch.Tensor,
        *,
        embedding: torch.Tensor | None = None,
        grads: bool = False,
        patch: tuple[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        start: tuple[int, torch.Tensor] | None = None,
        last: torch.Tensor | None = None,
    ):
        """Return ``(values, logits, leaves)`` of one treeified forward.

        ``embedding`` replaces the ``input`` node; ``grads`` adds a zero leaf
        to every receiver copy (returned in ``leaves``; its gradient is the
        receiver's own gradient, and the copy stays connected upstream); ``patch`` maps one
        receiver's copy through a function; ``start=(layer, x)`` begins at
        that layer with residual state ``x``; ``last`` (one position per row)
        computes logits ``[rows, vocab]`` at those positions only.
        """

        values: dict[str, torch.Tensor] = {}
        leaves: dict[str, torch.Tensor] = {}

        def reads(receiver, x, store):
            if not grads:
                return x
            probe = torch.zeros_like(x, requires_grad=True)
            store[receiver] = probe
            return x + probe

        first = 0
        if start is not None:
            first, x = start
        else:
            x = self.embed(ids) if embedding is None else embedding
            values["input"] = x
        states = {}
        for layer in range(first, self.n_layers):
            block = self.blocks[layer]
            states[layer] = x
            x = x + self._attention(block, layer, x, reads, values, leaves, patch, grads)
            copy = reads(f"m{layer}", x, leaves)
            if patch is not None and patch[0] == f"m{layer}":
                copy = patch[1](copy)
            out = block.mlp(block.ln_2(copy))
            values[f"m{layer}"] = out
            x = x + out
        values["_final"] = x
        copy = reads("logits", x, leaves)
        if patch is not None and patch[0] == "logits":
            copy = patch[1](copy)
        values["_states"] = states
        if last is not None:
            copy = copy[torch.arange(copy.shape[0], device=copy.device), last]
        logits = self.model.lm_head(self.model.transformer.ln_f(copy))
        return values, logits, leaves

    # -- definitions -------------------------------------------------------------

    def edge_scores(
        self,
        clean: torch.Tensor,
        corrupt: torch.Tensor,
        metric: Callable[[torch.Tensor], torch.Tensor],
        *,
        method: str,
        steps: int = 5,
    ) -> dict[str, float]:
        """EAP (gradient at clean) or EAP-IG (left Riemann, ``alpha = k/m``), mean over examples."""

        with torch.no_grad():
            clean_values, _, _ = self.forward(clean)
            corrupt_values, _, _ = self.forward(corrupt)
        e_clean, e_corrupt = clean_values["input"], corrupt_values["input"]
        if method == "eap":
            points = [e_clean]
        else:
            points = [e_corrupt + (k / steps) * (e_clean - e_corrupt) for k in range(steps)]
        total: dict[str, torch.Tensor] = {}
        for point in points:
            with torch.enable_grad():
                _values, logits, leaves = self.forward(clean, embedding=point, grads=True)
                metric(logits).sum().backward()
            missing = sorted(name for name, leaf in leaves.items() if leaf.grad is None)
            assert not missing, f"receivers with no gradient: {missing[:6]}"
            for name, leaf in leaves.items():
                total[name] = total.get(name, 0) + leaf.grad
        scores = {}
        for sender, receiver in edge_names(self.n_layers, self.n_heads):
            delta = corrupt_values[sender] - clean_values[sender]
            grad = total[receiver] / len(points)
            scores[f"{sender}->{receiver}"] = float((delta * grad).flatten(1).sum(1).mean())
        return scores

    @torch.no_grad()
    def brute_force_effects(
        self,
        clean: torch.Tensor,
        corrupt: torch.Tensor,
        metric: Callable[[torch.Tensor], torch.Tensor],
        *,
        last: torch.Tensor,
        receivers: list[str] | None = None,
        on_receiver: Callable[[str, int, float], None] | None = None,
        chunk: int = 32,
    ) -> dict[str, float]:
        """Effect of patching each edge alone: ``mean(M_patched - M_clean)`` over examples.

        Edge ``s->t``: receiver ``t`` reads ``clean input - s_clean + s_corrupt``;
        every other receiver reads what it reads in that patched forward.
        Variants of one receiver run as one batch from the receiver's layer;
        ``metric`` takes last-position logits ``[rows, vocab]`` (``last`` gives
        each example's position).
        """

        import time

        clean_values, clean_logits, _ = self.forward(clean, last=last)
        corrupt_values, _, _ = self.forward(corrupt, last=last)
        base = metric(clean_logits)
        states = clean_values["_states"]
        batch = clean.shape[0]
        by_receiver: dict[str, list[str]] = {}
        for sender, receiver in edge_names(self.n_layers, self.n_heads):
            by_receiver.setdefault(receiver, []).append(sender)
        effects: dict[str, float] = {}
        for receiver, senders in by_receiver.items():
            if receivers is not None and receiver not in receivers:
                continue
            started = time.perf_counter()
            layer = self.n_layers if receiver == "logits" else int(receiver[1:].split(".")[0])
            for lo in range(0, len(senders), chunk):
                group = senders[lo : lo + chunk]
                n = len(group)
                deltas = torch.cat([corrupt_values[s] - clean_values[s] for s in group])
                state = clean_values["_final"] if receiver == "logits" else states[layer]
                _v, logits, _l = self.forward(
                    clean.repeat(n, 1),
                    patch=(receiver, lambda copy, d=deltas: copy + d),
                    start=(layer, state.repeat(n, 1, 1)),
                    last=last.repeat(n),
                )
                m = metric(logits).view(n, batch) - base.view(1, batch)
                for index, sender in enumerate(group):
                    effects[f"{sender}->{receiver}"] = float(m[index].mean())
            n = len(senders)
            if on_receiver is not None:
                on_receiver(receiver, n, time.perf_counter() - started)
        return effects
