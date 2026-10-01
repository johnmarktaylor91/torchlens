"""Memory-timeline category contract + parity oracle (torchnative W0.7/6.3).

Torch deprecated its only categorized memory view; the replacement answers a
different question. TorchLens's rebuilt timeline (observe implements, W3.2)
uses torch's category words ONLY where they map 1:1 to an observed record
fact and NEVER emits torch's heuristic-only categories -- this module is the
contract of record for that mapping, plus the machinery of the DEADLINE
parity oracle: torch's still-alive private categorizer is snapshotted on a
pinned deterministic scenario and frozen as a torch-2.13 golden fixture,
because after upstream deletes the categorizer the oracle can never be
built again.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from ..utils import _torch_compat

__tl_layer__ = "L5"

#: Torch category words the rebuilt timeline emits, each mapping 1:1 to an
#: observed TorchLens record fact (section 6.3 contract).
EMITTED_CATEGORIES: dict[str, str] = {
    "PARAMETER": "parameter (declared parameter inventory)",
    "GRADIENT": "gradient (captured grad_fn/parameter gradient records)",
    "INPUT": "input (captured input records)",
    "ACTIVATION": "activation (captured op output records)",
    "OPTIMIZER_STATE": "optimizer_state (declared optimizer state inventory)",
    "None": "unknown (observed allocation with no record fact; never guessed)",
}

#: Torch categories the rebuilt timeline NEVER emits, with the reason: they
#: are heuristic INFERENCES (intra-operator temporaries, autograd
#: bookkeeping) with no observed record fact behind them. Our totals are
#: correspondingly lower than torch's deprecated view and the migration
#: table says so; the allocator snapshot is the process-level account.
NEVER_EMITTED_CATEGORIES: dict[str, str] = {
    "TEMPORARY": "heuristic-only intra-operator temporary inference",
    "AUTOGRAD_DETAIL": "heuristic-only autograd bookkeeping inference",
}


def pinned_parity_scenario() -> Any:
    """Run the pinned deterministic scenario under torch's memory profiler.

    Seed-0 two-step SGD-with-momentum training loop on a small MLP: small
    enough for smoke CI, rich enough to exercise every category torch's
    categorizer can assign in eager (parameter, gradient, input, activation,
    optimizer state, and both heuristic-only classes). The scenario is
    FROZEN -- changing it invalidates the torch-2.13 golden fixture.

    Returns
    -------
    Any
        The CLOSED profiler (memory profiling enabled).
    """

    from ._session import session

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    inputs = torch.randn(4, 16)
    with session(
        mode="owned",
        profiler_kwargs={"profile_memory": True, "record_shapes": True, "with_stack": True},
    ) as active:
        for _ in range(2):
            optimizer.zero_grad()
            loss = model(inputs).sum()
            loss.backward()
            optimizer.step()
    return active.closed_profiler


def categorized_key_counts(profiler: Any) -> dict[str, int] | None:
    """Return torch's per-category keyed-tensor counts for one profiler.

    ``None`` when the private categorizer is unavailable on this torch
    build (``HAS_MEMORY_PROFILE`` flipped; callers skip with that named
    reason -- the frozen golden then carries the preserved behavior).

    Parameters
    ----------
    profiler:
        A closed profiler from :func:`pinned_parity_scenario`.

    Returns
    -------
    dict[str, int] | None
        Category name -> count of distinct ``(TensorKey, version)`` keyed
        tensors torch assigned that category.
    """

    memory_profile = _torch_compat.memory_profile_from_profiler(profiler)
    if memory_profile is None:
        return None
    try:
        keys = {
            (entry[2][0], entry[2][1])
            for entry in memory_profile.timeline
            if entry[2][0] is not None
        }
        counts: Counter[str] = Counter()
        for key, version in keys:
            category = memory_profile._categories.get(key, version)
            counts[getattr(category, "name", str(category))] += 1
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        # Private-API drift reads as unavailable (callers skip with the
        # named capability reason); the frozen golden stays the authority.
        return None
    return dict(sorted(counts.items()))


def category_vocabulary_report(counts: dict[str, int]) -> dict[str, Any]:
    """Classify observed torch categories against the 6.3 contract.

    Returns
    -------
    dict[str, Any]
        ``emitted`` / ``never_emitted`` / ``unmapped`` category name lists.
        A non-empty ``unmapped`` list means torch grew a category the
        migration table does not know -- the review trigger.
    """

    emitted, never, unmapped = [], [], []
    for name in counts:
        if name in EMITTED_CATEGORIES:
            emitted.append(name)
        elif name in NEVER_EMITTED_CATEGORIES:
            never.append(name)
        else:
            unmapped.append(name)
    return {"emitted": emitted, "never_emitted": never, "unmapped": unmapped}


__all__ = [
    "EMITTED_CATEGORIES",
    "NEVER_EMITTED_CATEGORIES",
    "categorized_key_counts",
    "category_vocabulary_report",
    "pinned_parity_scenario",
]
