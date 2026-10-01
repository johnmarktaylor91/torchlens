"""Relocated cross-layer vocabulary closures (L0 BASIS; private home).

Rule V2 (architecture memo 3.3): a vocabulary component needed by two or
more strata below its home physically moves down as its whole transitive
vocabulary closure. This package is the physical L0 home for the ratified
relocations; each source module stays behind as a re-export shim so no
public spelling or pickle-visible identity changes, and lower-layer
consumers import from here so the killed inversions stay dead.

Module names here are PRIVATE plumbing; the naming sprint owns any public
spellings.
"""

from __future__ import annotations

__tl_layer__ = "L0"
__tl_vocabulary__ = True
