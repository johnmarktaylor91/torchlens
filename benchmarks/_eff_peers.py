"""Private adapters for the installed peer APIs, using the same eager HF model."""

from __future__ import annotations

import contextlib
from typing import Any

import torch


class _Peers:
    def __init__(self, runner: Any) -> None:
        self.runner = runner
        if runner.mode == "nnsight":
            from nnsight import NNsight

            self.model = NNsight(runner.model)
            self.targets = []
            for site in runner.sites:
                target = self.model
                for part in site.split("."):
                    target = target[int(part)] if part.isdigit() else getattr(target, part)
                self.targets.append(target)
        else:
            from transformer_lens import TransformerBridge
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
            self.model = TransformerBridge.boot_transformers(
                "gpt2" if hasattr(runner.model.net, "transformer") else "Qwen/Qwen3-0.6B",
                hf_model=runner.model.net,
                tokenizer=tokenizer,
                device="cpu",
            )
            indices = [int(s.rsplit(".", 1)[1]) for s in runner.sites]
            self.names = [f"blocks.{i}.hook_resid_post" for i in indices]

    def call(self, ids: Any, action: str, strength: float, patch: Any, grad: bool) -> Any:
        runner = self.runner
        cache = {}
        context = contextlib.nullcontext() if grad else torch.no_grad()
        with context:
            if runner.mode == "nnsight":
                with self.model.trace(ids):
                    for index, target in enumerate(self.targets):
                        value = target.output
                        if hasattr(runner.model.net, "transformer"):
                            value = value[0]
                        if index == 1 and action != "none":
                            changed = (
                                value + runner.direction * strength if action == "steer" else patch
                            )
                            if hasattr(runner.model.net, "transformer"):
                                target.output[0] = changed
                            else:
                                target.output = changed
                            value = changed
                        cache[runner.sites[index]] = value.grad.save() if grad else value.save()
                    output = self.model.output.save()
                    if grad:
                        self.model.output.sum().backward()
            else:
                hooks = []
                for index, name in enumerate(self.names):

                    def hook(value: Any, hook: Any, index: int = index) -> Any:
                        if index == 1 and action != "none":
                            value = (
                                value + runner.direction * strength if action == "steer" else patch
                            )
                        if grad:
                            value.retain_grad()
                            cache[runner.sites[index]] = value
                        else:
                            cache[runner.sites[index]] = value.detach().clone()
                        return value

                    hooks.append((name, hook))
                output = self.model.run_with_hooks(ids, fwd_hooks=hooks)[:, -1]
                if grad:
                    output.sum().backward()
                    cache = {s: v.grad for s, v in cache.items()}
            result = output.detach().clone(), {s: v.detach().clone() for s, v in cache.items()}
        runner.model.zero_grad(set_to_none=True)
        runner.bytes = sum(v.numel() * v.element_size() for v in result[1].values())
        return result
