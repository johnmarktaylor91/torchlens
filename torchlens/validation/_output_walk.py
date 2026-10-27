"""Validation-owned independent output-tensor traversal (b9 R74/75-1).

Ground-truth output enumeration used to share its root with capture: both
``backends/torch/backend.py`` (output extraction) and the validation oracle
resolved through ``backends.torch.ops._walk_output_tensors_with_paths``, so a
walker defect made capture AND the oracle drop the same output leaf and
``validate_forward_pass`` returned True over a missing output (sol's live
plant). This module is the independent second root: a deliberately small,
self-contained traversal that must NEVER import from ``torchlens.backends``
-- ``tests/test_output_walker_independence.py`` enforces that structurally.

It is a CROSS-CHECK, not a replacement: capture's adapter defines the
agreed container semantics (registered containers, structseqs, dedup rules),
so validation still feeds the adapter's enumeration to the replay pipeline
and FAILS the run when this walker disagrees about the tensor-leaf identity
set, instead of silently inheriting the adapter's blind spot.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import torch

# Must be >= the capture-side output ceiling (``ir.container_registry.
# OUTPUT_TREE_MAX_DEPTH``, 200) or the cross-check is ONE-SIDED: the consumer
# fails only on leaves the independent walk finds that the adapter missed, so
# a walker that stops shallower than capture cannot see -- and can never flag
# -- a dropped deep leaf (a planted drop-last-leaf capture mutation validated
# True for outputs nested past the old ceiling of 8). Kept as a literal, not
# an import, so the traversal stays a genuinely independent second root;
# ``tests/test_output_walker_independence.py`` locksteps the two ceilings.
_MAX_DEPTH = 200


def independent_output_tensor_ids(output: Any) -> list[int]:
    """Return ``id()``s of every distinct tensor leaf reachable in an output.

    The traversal covers the generic Python container shapes an eager forward
    can return -- tuples/lists (namedtuples and torch structseqs included),
    dicts, sets, and dataclasses -- to a bounded depth, counting each tensor
    OBJECT once no matter how many positions alias it (capture's agreed
    dedup rule).

    Parameters
    ----------
    output:
        Arbitrary model output tree.

    Returns
    -------
    list[int]
        Sorted distinct tensor object ids.
    """

    seen: set[int] = set()

    def _walk(value: Any, depth: int) -> None:
        """Accumulate tensor ids from one subtree.

        Parameters
        ----------
        value:
            Subtree to inspect.
        depth:
            Remaining recursion budget guard.
        """

        if isinstance(value, torch.nn.Parameter):
            # Capture's agreed semantics: a Parameter returned as-is is model
            # state, not a computed output leaf, and the adapter excludes it
            # by design. Counting it here reported that design as a dropped
            # leaf. Only Parameter instances are skipped; any other tensor the
            # adapter misses still fails the cross-check.
            return
        if isinstance(value, torch.Tensor):
            seen.add(id(value))
            return
        if depth >= _MAX_DEPTH:
            return
        if isinstance(value, dict):
            for item in value.values():
                _walk(item, depth + 1)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                _walk(item, depth + 1)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field_info in fields(value):
                _walk(getattr(value, field_info.name, None), depth + 1)
            return

    _walk(output, 0)
    return sorted(seen)
