"""Feature probes for the transformers surfaces the test suite depends on.

TorchLens supports ``transformers>=4.45,<6``, and the suite runs on two
bands: CI's pinned 5.x environment and the torch floor row (transformers
4.55.4, the newest 4.x that imports on torch 2.1.2). A test whose expectation
differs between the bands asks the installed package what it can do through
one of these probes; nothing here parses a version string. Each probe names
the one upstream surface its callers depend on, so a test routed by a probe
documents why its expectation differs, and the graceful branch asserts the
behaviour the older surface really has.
"""

from __future__ import annotations

import functools


@functools.cache
def t5_supports_sdpa() -> bool:
    """Return whether transformers' T5 model class declares sdpa support.

    T5 gained the unified attention interface (and ``_supports_sdpa = True``)
    in the 5.x band; earlier releases keep T5 eager-only and resolve a
    requested ``"sdpa"`` implementation to eager or refuse it.
    """

    from transformers import T5PreTrainedModel

    return bool(getattr(T5PreTrainedModel, "_supports_sdpa", False))


@functools.cache
def moe_router_aux_loss_runs_on_cpu(family: str) -> bool:
    """Return whether a MoE family's router auxiliary loss can be computed on CPU.

    ``family`` names the transformers model module (``"olmoe"``,
    ``"qwen2_moe"``, ``"mixtral"``). The CausalLM computes this loss whenever
    ``output_router_logits=True``. The 4.x ``olmoe`` and ``qwen2_moe`` loss
    slices its expert columns at ``int(routing_weights.device.index)``, which
    is ``None`` on CPU, so the CausalLM call raises ``TypeError`` there; the
    5.x band dropped that slice. The probe runs the family's own loss function
    on a one-layer CPU gate tensor.
    """

    import importlib

    import torch

    module = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    try:
        module.load_balancing_loss_func((torch.zeros(2, 4),), num_experts=4, top_k=2)
    except TypeError:
        return False
    return True


@functools.cache
def vit_blocks_path() -> str:
    """Return where transformers' ``ViTModel`` keeps its encoder blocks.

    The 5.x band hangs the block list directly on the model
    (``vit.layers.N``); the 4.x band keeps it inside a ``ViTEncoder``
    (``vit.encoder.layer.N``). The probe builds a one-block ViT and reads its
    module tree.
    """

    from transformers import ViTConfig, ViTModel

    config = ViTConfig(
        image_size=8,
        patch_size=8,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=8,
    )
    names = {name for name, _ in ViTModel(config, add_pooling_layer=False).named_modules()}
    return "layers" if "layers.0" in names else "encoder.layer"
