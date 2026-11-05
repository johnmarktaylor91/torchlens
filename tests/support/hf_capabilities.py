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
