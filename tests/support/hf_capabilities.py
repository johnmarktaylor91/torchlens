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
def gpt2_fast_tokenizer_takes_in_memory_vocab() -> bool:
    """Return whether ``GPT2TokenizerFast`` builds from in-memory ``vocab``/``merges``.

    The 5.x band constructs fast tokenizers from ``vocab`` and ``merges``
    objects; the 4.x band only takes ``vocab_file``/``merges_file`` paths
    (it converts through the slow tokenizer, which opens those files).
    """

    import inspect

    from transformers import GPT2TokenizerFast

    params = inspect.signature(GPT2TokenizerFast.__init__).parameters
    return "vocab" in params or "vocab_file" not in params


def from_pretrained_dtype_kwargs(dtype: object) -> dict[str, object]:
    """Return the ``from_pretrained`` keyword that sets the load dtype on this band.

    Configs that carry a ``dtype`` attribute belong to the band whose
    ``from_pretrained`` reads ``dtype=`` (and deprecates ``torch_dtype=``);
    older configs carry only ``torch_dtype`` and ``from_pretrained`` forwards
    an unknown ``dtype=`` to the model constructor, which rejects it.
    """

    from transformers import PretrainedConfig

    key = "dtype" if hasattr(PretrainedConfig(), "dtype") else "torch_dtype"
    return {key: dtype}


@functools.cache
def transformers_loads_pickled_weights() -> bool:
    """Return whether transformers will load a pickled (``.bin``) checkpoint on this torch.

    transformers refuses ``torch.load`` of pickled weights below torch 2.6
    (CVE-2025-32434, its ``check_torch_load_is_safe`` gate), so a checkpoint
    that ships only ``pytorch_model.bin`` cannot load on the torch floor.
    The probe asks that gate itself and fails toward running: if the gate
    is missing or raises anything but its named ``torch.load`` refusal, it
    reports support, so routed tests run and fail loudly instead of skipping.
    """

    try:
        from transformers.utils.import_utils import check_torch_load_is_safe
    except ImportError:
        return True
    try:
        check_torch_load_is_safe()
    except ValueError as refusal:
        return "torch.load" not in str(refusal)
    return True


@functools.cache
def sdpa_mask_builder() -> str:
    """Return the name of the function transformers builds sdpa attention masks with.

    On the 4.x band transformers picks ``sdpa_mask_recent_torch`` (a vmap
    construction) on torch 2.6 or newer and ``sdpa_mask_older_torch`` below
    it, and the two record different op counts for the same model; the 5.x
    band has a single ``sdpa_mask``. Empty when the package has no
    ``masking_utils``.
    """

    try:
        from transformers.masking_utils import sdpa_mask
    except ImportError:
        return ""
    return str(getattr(sdpa_mask, "__name__", ""))
