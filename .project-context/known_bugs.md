# Known Bugs

## ELK-IF-THEN Edge Label Dedup

Status: internal bug, not fixed in Phase 1d.

The ELK layout backend is now internal-only under
`torchlens.visualization._elk_internal`. Because `layout="elk"` remains accepted
as an internal backend escape hatch, the old conditional/argument edge-label
dedup behavior can still matter for explicit ELK renders. Phase 1d retires ELK
from the public API surface and documents this as an internal renderer bug
instead of changing rendering output.
