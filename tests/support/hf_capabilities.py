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
def cache_layers_initialize_lazily() -> bool:
    """Return whether a ``DynamicCache`` layer starts from an empty tensor it concatenates onto.

    The 5.x cache layer allocates empty key/value tensors on its first update
    and ``torch.cat``-s the new states onto them, so every attention layer of a
    traced causal LM records two tensor allocations and two concatenations.
    The 4.x layer stores the first states as they are. Op-count and FLOP
    goldens of models that run with the cache on depend on this surface.
    """

    from transformers.cache_utils import DynamicLayer

    return callable(getattr(DynamicLayer, "lazy_initialization", None))


@functools.cache
def causal_lm_returns_cache_object() -> bool:
    """Return whether a GPT-2 forward with the cache on and no past returns a ``Cache`` object.

    The 5.x forward returns the ``DynamicCache`` it filled (an opaque,
    tensor-holding leaf). The 4.x forward converts it back to the legacy tuple
    of per-layer ``(key, value)`` tensors before returning, so the output's
    cache leaves are plain tensors at ``past_key_values.<layer>.<0|1>``.
    Probed on a one-layer model under a forked RNG so callers' seeding holds.
    """

    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    with torch.random.fork_rng(devices=[]):
        config = GPT2Config(n_layer=1, n_head=1, n_embd=4, vocab_size=8, n_positions=4)
        model = GPT2LMHeadModel(config).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 2, dtype=torch.long), use_cache=True)
    return not isinstance(output.past_key_values, tuple)


@functools.cache
def sdpa_attention_passes_enable_gqa() -> bool:
    """Return whether transformers' sdpa attention hands grouped KV heads to the kernel.

    When transformers passes ``enable_gqa=True`` to
    ``scaled_dot_product_attention`` the KV expansion happens inside the
    kernel and no per-query-head copy exists in the graph. Releases that do
    not (4.x below torch 2.5) expand the KV heads with an explicit
    ``repeat_kv`` before the kernel. Probed on a two-query-head, one-KV-head
    Llama under a forked RNG.
    """

    import torch
    from torch.overrides import TorchFunctionMode
    from transformers import LlamaConfig, LlamaForCausalLM

    class _SdpaKwargs(TorchFunctionMode):
        """Record the keyword names of every sdpa call."""

        def __init__(self) -> None:
            """Start with no recorded calls."""

            super().__init__()
            self.enable_gqa: list[bool] = []

        def __torch_function__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
            """Note ``enable_gqa`` on sdpa calls and run every function unchanged."""

            kwargs = kwargs or {}
            if getattr(func, "__name__", "") == "scaled_dot_product_attention":
                self.enable_gqa.append(bool(kwargs.get("enable_gqa", False)))
            return func(*args, **kwargs)

    with torch.random.fork_rng(devices=[]):
        config = LlamaConfig(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            attn_implementation="sdpa",
        )
        model = LlamaForCausalLM(config).eval()
        recorder = _SdpaKwargs()
        with torch.no_grad(), recorder:
            model(torch.zeros(1, 3, dtype=torch.long))
    return any(recorder.enable_gqa)


@functools.cache
def bert_switches_attention_dynamically() -> bool:
    """Return whether a built BERT model can change its attention implementation.

    5.x BERT routes attention through the ``AttentionInterface``, so
    ``set_attn_implementation`` (and TorchLens's ``register_attention``)
    switch a built model. 4.x BERT keeps its own sdpa/eager classes;
    transformers logs a warning and the model keeps its implementation.
    """

    import torch
    from transformers import BertConfig, BertModel

    with torch.random.fork_rng(devices=[]):
        config = BertConfig(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=8,
            attn_implementation="sdpa",
        )
        model = BertModel(config).eval()
        model.set_attn_implementation("eager")
    return model.config._attn_implementation == "eager"


@functools.cache
def gpt2_attention_keeps_causal_mask_buffer() -> bool:
    """Return whether GPT-2 attention registers its causal mask as a non-persistent buffer.

    4.x ``GPT2Attention`` registers ``bias`` (a boolean lower-triangular mask)
    and ``masked_bias`` as non-persistent buffers; 5.x builds the mask in
    ``masking_utils`` and registers none. A runnable save of a model with used
    non-persistent buffers stores their values and discloses that with a
    ``TorchLensWarning``.
    """

    import torch
    from transformers import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    with torch.random.fork_rng(devices=[]):
        config = GPT2Config(n_layer=1, n_head=1, n_embd=4, vocab_size=8, n_positions=4)
        attention = GPT2Attention(config)
    return "bias" in attention._non_persistent_buffers_set


@functools.cache
def attention_masks_avoid_vmap() -> bool:
    """Return whether transformers builds sdpa/eager attention masks without ``torch.vmap``.

    The 5.x ``sdpa_mask`` takes ``use_vmap`` and builds masks from index
    tensors by default. The 4.x builders always expand the mask function
    through ``torch.vmap``, so a traced causal LM records an op whose callable
    is a vmapped closure with no import path. Without ``masking_utils`` there
    is no vmap construction at all.
    """

    import inspect

    try:
        from transformers.masking_utils import sdpa_mask
    except ImportError:
        return True
    return "use_vmap" in inspect.signature(sdpa_mask).parameters
