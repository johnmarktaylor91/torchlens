"""Select every site, take a complement, and grow a region from it.

What this demonstrates
----------------------
``tl.all_sites()`` selects every op of a trace. Subtracting a region from it
gives the sites outside that region. ``tl.neighborhood`` and ``tl.between``
grow only from sites that still carry selected elements, so a neighborhood of
the complement grows from the sites outside the region, not from the sites
the subtraction emptied.

How to run
----------
``python examples/intervention/18_complement_under_reach.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class TwoPath(nn.Module):
    """Two paths from ``a`` to the final add: a -> h -> d -> b and a -> b."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        a = torch.relu(x)
        h = a * 2
        d = h + 1
        return d + a


def main() -> None:
    """Grow a one-hop region from the sites not downstream of the ReLU."""

    torch.manual_seed(0)
    log = tl.trace(TwoPath(), torch.randn(1, 3))
    try:
        everything = log[tl.all_sites()]
        print("all sites:", everything.labels)

        downstream = tl.neighborhood(tl.func("relu"), hops=len(everything), direction="downstream")
        outside = log[tl.all_sites() - downstream]
        # Subtraction keeps every site as a row; the emptied rows select nothing.
        kept = [label for label, site in zip(outside.labels, outside) if site.num_selected]
        print("not downstream of relu:", kept)

        region = log[tl.neighborhood(outside, hops=1, direction="downstream")]
        print("one hop downstream of that:", region.labels)
        assert len(region) == 2  # the input and the ReLU it feeds
    finally:
        log.cleanup()


if __name__ == "__main__":
    main()
