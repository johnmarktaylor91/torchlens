"""glance: the stats line as a function on arbitrary tensors (F10; FORK-C).

The panel's answer to "why not monkey-patch": TorchLens ships the FUNCTION,
never the mutation (``lt.monkey_patch``-style global ``Tensor.__repr__``
replacement is a permanent skip -- out of identity, hostile in a library,
fights ``torch.compile``). ``glance(t)`` renders the same byte-checked core
grammar the record surfaces use, over any tensor, captured or not.

DOCUMENTED STRICT-SUBSET caveat (memo section 13, FORK-C): a glanced line
has NO envelope (no trace/address/pass provenance), NO deviation marks,
NO graph-proven semantic role, and NO trace-declared sampling budget; the
sample seed cannot derive from record identity, so sampled families seed
from the tensor's geometry -- deterministic per shape/dtype, NOT per
record identity across processes.

Submodule spelling (``torchlens.stats.glance``): the root-name budget is
frozen (C01 surface law); hoisting to ``tl.glance`` is a naming-session
decision through the sanctioned surface generator, not a lane's.
Spellings DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import torch


def glance(
    tensor: torch.Tensor,
    *,
    style: str = "ascii",
    max_width: int | None = None,
) -> str:
    """Render the one-line stats core for any dense torch tensor.

    Parameters
    ----------
    tensor:
        Tensor to summarize. Never mutated; autograd/RNG state untouched;
        the kernel is allocation-free and single-sync per the C02 contract.
    style:
        ``"ascii"`` (canonical) or ``"unicode"`` -- the two byte-checked
        renderings of identical content (``ascii == degrade(unicode)``).
    max_width:
        Optional width bound; degradation follows the D14 retention order
        (bytes, then ``n=``, then the sparkline -- never
        shape/dtype/extrema/health).

    Returns
    -------
    str
        The core stats line, e.g.
        ``f32[2,8]@cpu n=16 [-1.442 .. 2.118] mean=-0.02398 sd=0.9027``.

    Raises
    ------
    TypeError
        When ``tensor`` is not a ``torch.Tensor`` (teaching message naming
        the captured-record spelling for record payloads).
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "glance() renders torch tensors; for captured records use the record "
            "surfaces (repr(log[label]) / log.stats_table()) which add provenance "
            f"and honesty envelopes. Got {type(tensor).__name__}."
        )
    from ._stats_render import render_core_line
    from ._tensor_stats import tensor_stats

    identity = f"glance:{tuple(tensor.shape)}:{tensor.dtype}"
    stats = tensor_stats(tensor, identity=identity)
    return render_core_line(stats, style=style, max_width=max_width)
