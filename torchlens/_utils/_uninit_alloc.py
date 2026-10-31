"""Uninitialized-memory value-source family (r53 hon_2): ONE closed table.

Split out of ``utils/rng.py`` under the R43 file-size ratchet. The historical
import surface (``torchlens._utils.rng``) re-exports every name here, so all
existing consumers and the aten-namespace drift meta-test keep their spelling.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

_SEEDED_RNG_NAMESPACES = frozenset({"torch", "torch.Tensor", "torch.nn.functional"})

# --- r53 hon_2: uninitialized-memory value-source family (ONE closed table) --------
#
# The ``empty`` factory family and a GROWING ``resize_``/``resize_as_`` produce bytes
# that are not a function of the recorded computation (allocator garbage). PyTorch
# gives these ops NO distinguishing ``torch.Tag`` (verified: ``empty`` tags are only
# ``core``/``generated``/``pt2_compliant_tag``), so the family is a maintained name
# table defended by an aten-namespace drift meta-test
# (``tests/test_tlspec_runnable_r53_uninit_alloc.py``): a new ``empty*``/``resize*``
# aten name that is neither in the family table nor in the test's justified
# non-family allowlist is a FAILING test, never a silent gap.
#
# This block is the SINGLE definition consumed by all three recognition layers:
# the load-side value-source classifier (``_runnable_execution.py``), the producer
# origin ledger (``backends/torch/completeness_witness.py``), and the pruned-orphan
# control walk (``postprocess/graph_traversal.py``). No other call site may
# re-derive uninitialized-memory nondeterminism from qualnames (source-scan
# meta-test).

_UNINIT_ALLOC_FACTORY_TAILS = frozenset(
    {
        "empty",
        "empty_like",
        "empty_permuted",
        "empty_strided",
        "empty_quantized",
        "new_empty",
        "new_empty_strided",
        # Private quantized spellings: never captured through the public wrap
        # surface, tabled so the aten drift meta-test stays exhaustive.
        "_empty_affine_quantized",
        "_empty_per_channel_affine_quantized",
    }
)
"""Factory ops whose freshly allocated bytes are uninitialized."""

_UNINIT_ALLOC_SIZE_GATED_TAILS = frozenset({"new"})
"""Python-level ``torch.Tensor`` allocators whose UNINIT semantics depend on ARG FORM.

r55 hon_1: the legacy ``Tensor.new(*sizes)`` allocator returns byte-identical
uninitialized memory to ``new_empty`` (probed: distinct ``.sum()`` across
allocations; redispatches ``aten.empty.memory_format``), but the SAME name
called with DATA (``Tensor.new([values])`` / ``Tensor.new(tensor)``) is a
deterministic copy constructor (redispatches ``aten.lift_fresh``/``aten.alias``).
``new`` has NO aten spelling (``hasattr(torch.ops.aten, "new")`` is ``False``), so
the aten drift meta-test cannot see it; the Python-``torch.Tensor``-method drift
meta-test (``tests/test_tlspec_runnable_r53_uninit_alloc.py``) defends this table
instead. Membership here is NECESSARY but not SUFFICIENT for uninit taint: every
consumer must additionally gate on :func:`uninit_new_call_is_size_form` (the
size-vs-data argument-form predicate) so the data spelling is never over-ceilinged.
"""

_UNINIT_ALLOC_RESIZE_TAILS = frozenset({"resize_", "resize_as_", "resize", "resize_as"})
"""Resize spellings that expose stale allocator bytes ONLY when they GROW.

``resize_``/``resize_as_`` preserve the element prefix (probed: shrink/same-size
is byte-deterministic) but a grow beyond the pre-call element count exposes
uninitialized tail bytes (probed: 4088/4092 stale bytes recovered through a
4096-element grow). The non-underscore spellings are the deprecated
``Tensor.resize``/``resize_as`` aliases and the functionalized aten variants,
which share the grow semantics. The GROW gate is decided by the caller (shapes
are layer-local facts); an undecidable grow fails closed to tainted.
"""

_UNINIT_TOTAL_WRITER_TAILS = frozenset({"copy_", "zero_", "fill_"})
"""In-place ops whose result bytes are a TOTAL write independent of prior content."""

_UNINIT_RNG_FILL_TAILS = frozenset(
    {
        "uniform_",
        "normal_",
        "bernoulli_",
        "random_",
        "exponential_",
        "geometric_",
        "cauchy_",
        "log_normal_",
    }
)
"""In-place RNG fills: total writes that REPLACE uninit taint with the RNG source
classification (they are ``nondeterministic_seeded``-tagged, so the seeded nets
own their products)."""


def qualname_is_uninitialized_alloc(namespace: str | None, qualname: str | None) -> bool:
    """Return whether a captured callable is an uninitialized-memory FACTORY op.

    Parameters
    ----------
    namespace:
        Captured callable namespace (e.g. ``"torch"``, ``"torch.Tensor"``).
    qualname:
        Captured callable qualified name (e.g. ``"empty_like"``,
        ``"Tensor.new_empty"``).

    Returns
    -------
    bool
        Whether the callable belongs to the ``empty`` factory family. Growing
        resizes are matched separately by :func:`qualname_is_uninit_growth_resize`
        because their taint additionally requires the grow refinement.
    """

    if namespace not in _SEEDED_RNG_NAMESPACES or not qualname:
        return False
    return qualname.rsplit(".", 1)[-1] in _UNINIT_ALLOC_FACTORY_TAILS


def qualname_is_uninit_size_gated_alloc(namespace: str | None, qualname: str | None) -> bool:
    """Return whether a captured callable is an ARG-FORM-GATED uninit allocator.

    r55 hon_1: matches the ``Tensor.new`` legacy allocator family
    (:data:`_UNINIT_ALLOC_SIZE_GATED_TAILS`). A ``True`` here means the call is
    uninitialized-memory-producing ONLY in its size-argument form; the caller
    OWNS the form refinement via :func:`uninit_new_call_is_size_form` and must
    fail closed to tainted on an undecidable form (the grow-gate precedent).
    """

    if namespace not in _SEEDED_RNG_NAMESPACES or not qualname:
        return False
    return qualname.rsplit(".", 1)[-1] in _UNINIT_ALLOC_SIZE_GATED_TAILS


def uninit_new_call_is_size_form(args: Sequence[Any]) -> bool | None:
    """Classify a ``Tensor.new(...)`` positional-argument tuple as SIZE vs DATA form.

    Probed legacy-constructor semantics (the size form allocates UNINITIALIZED
    memory; the data form is a deterministic copy):

    - ``new()`` / ``new(int...)`` / ``new(np.integer...)`` / ``new(torch.Size)``
      -> SIZE form (``aten.empty.memory_format``): returns ``True``.
    - ``new(list)`` / ``new(tensor)`` / ``new(ndarray)`` / ``new(range)``
      -> DATA form (``aten.lift_fresh`` / ``aten.alias``): returns ``False``.
    - a single plain ``tuple`` of ints -> ``None`` (UNDECIDABLE): a LIVE plain
      tuple is the data form (probed), but the portable literal grammar erases
      ``torch.Size`` to a plain tuple, so a DECODED int-tuple may have been a
      capture-time SIZE call. A caller holding live runtime types sees
      ``torch.Size`` classified ``True`` before this case can fire; a caller
      holding decoded literals must fail closed to tainted on ``None``.
    - any other spelling -> ``None`` (fail closed to tainted; an invalid
      spelling raises at execution time and never produces a value to classify).

    ``bool`` is NOT an integral size argument (``Tensor.new(True)`` raises,
    probed), so ``True``/``False`` literals fall through to ``None``.
    """

    positional = tuple(args)
    if not positional:
        return True  # zero-size uninit alloc; the zero-numel refinement is downstream
    if all(
        (isinstance(item, int) and not isinstance(item, bool)) or isinstance(item, np.integer)
        for item in positional
    ):
        return True
    if len(positional) == 1:
        head = positional[0]
        if isinstance(head, torch.Size):
            return True
        if isinstance(head, (torch.Tensor, np.ndarray, list, range)):
            return False
        if isinstance(head, tuple):
            return None  # torch.Size erased by the portable grammar: undecidable
    return None


def qualname_is_uninit_growth_resize(namespace: str | None, qualname: str | None) -> bool:
    """Return whether a captured callable is a resize op that CAN expose stale bytes.

    The caller owns the grow refinement (``new numel > pre-call numel``); a
    matching name with an undecidable grow fact must fail closed to tainted.
    """

    if namespace not in _SEEDED_RNG_NAMESPACES or not qualname:
        return False
    return qualname.rsplit(".", 1)[-1] in _UNINIT_ALLOC_RESIZE_TAILS


def qualname_is_uninit_total_writer(namespace: str | None, qualname: str | None) -> bool:
    """Return whether a captured callable totally overwrites its destination's bytes.

    Total writers (``copy_``/``zero_``/``fill_`` and the in-place RNG fills)
    REMOVE prior uninitialized-memory taint from their destination: the
    post-call value is independent of the destination's prior content. Partial
    or unprovable in-place writers (``index_put_``, ``masked_fill_``,
    ``scatter_``, a sliced ``copy_`` through a view's base) are deliberately
    NOT in this table -- unprovable coverage propagates taint (fail closed).
    The namespace gate keeps the sanitizer NARROW: a custom callable that
    merely shares a total-writer name never removes taint.
    """

    if namespace not in _SEEDED_RNG_NAMESPACES or not qualname:
        return False
    tail = qualname.rsplit(".", 1)[-1]
    return tail in _UNINIT_TOTAL_WRITER_TAILS or tail in _UNINIT_RNG_FILL_TAILS


def deterministic_fill_governs(
    deterministic_algorithms: bool | None,
    fill_uninitialized_memory: bool | None,
) -> bool:
    """Return whether a governing context proves deterministic uninit-memory fill.

    Under ``torch.use_deterministic_algorithms(True)`` with
    ``torch.utils.deterministic.fill_uninitialized_memory`` not ``False``
    (default ``True``; ``None`` means the runtime predates the disable knob and
    always fills under deterministic mode), torch deterministically fills the
    ``empty`` family (NaN for floating point, the max value for int dtypes;
    probed), so the family's bytes ARE a function of the recorded computation
    and carry no taint.
    """

    return deterministic_algorithms is True and fill_uninitialized_memory is not False
