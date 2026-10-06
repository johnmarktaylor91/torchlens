"""Capability check for real-package tests that load a cached Hugging Face checkpoint.

The real-bridge tests run offline (``HF_HUB_OFFLINE=1``) against tiny
checkpoints that a CI leg warms into the local cache. Whether a checkpoint is
there is checked BEFORE loading, so a missing checkpoint is a visible,
conditional skip, and a cached checkpoint that then fails to load fails the
test loudly instead of being swallowed by a skip inside an ``except`` handler
(``tests/test_proofnet_gate_witness.py`` bans those).
"""

from __future__ import annotations

import pytest


def skip_unless_hf_checkpoint_cached(name: str) -> None:
    """Skip the calling test when checkpoint ``name`` is not in the local HF cache.

    Parameters
    ----------
    name:
        Hub repository id, for example
        ``"hf-internal-testing/tiny-random-gpt2"``. The checkpoint counts as
        cached when its ``config.json`` resolves to a file in the local cache.
    """

    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(name, "config.json"), str):
        pytest.skip(f"checkpoint {name} not cached")
