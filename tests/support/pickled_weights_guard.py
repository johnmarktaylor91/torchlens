"""The pickled-checkpoint route, shared by tests that load ``.bin``-only hub checkpoints.

Some real checkpoints the suite loads (``openai/clip-vit-base-patch32``,
``google/electra-small-discriminator``, ``hf-internal-testing/tiny-random-roberta``,
``sshleifer/tiny-gpt2``) ship only ``pytorch_model.bin`` on their pinned
revision, and transformers refuses ``torch.load`` below torch 2.6
(CVE-2025-32434). On such a torch the tests marked here route by the
``transformers_loads_pickled_weights`` capability instead of a version check,
and ``tests/test_hf_pickled_weights_route.py`` asserts the refusal itself on
every torch.
"""

from __future__ import annotations

import pytest

from tests.support.hf_capabilities import transformers_loads_pickled_weights

requires_pickled_weight_loading = pytest.mark.skipif(
    not transformers_loads_pickled_weights(),
    reason=(
        "the checkpoint ships only pickled weights and transformers refuses torch.load"
        " below torch 2.6 (CVE-2025-32434); tests/test_hf_pickled_weights_route.py"
        " asserts that refusal"
    ),
)
