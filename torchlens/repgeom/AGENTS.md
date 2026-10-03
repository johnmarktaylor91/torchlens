# repgeom/ - Implementation Guide

Representation-geometry analysis over saved activations (`tl.repgeom`, lazy;
one module, 12-name `__all__`). Sprint-B/C provisional surface — names are
review-day provisional per `docs/agent-reference/package/public-surface.md`.

## Surface

- Matrix/analysis primitives: `activation_distance_matrix`, `rdm`,
  `classical_mds`, `procrustes_align`, `scree`, `effective_dimensionality`.
- Per-layer evolution tables: `mds_evolution`, `rdm_evolution`,
  `scree_evolution` — these read activations ALREADY retained by the
  capture-time `save=` decision (use a curated predicate, not exhaustive
  saves, for image batches).
- Node-visual factories for `Trace.draw(node_spec_fn=...)`:
  `mds_scatter_node_spec`, `rdm_node_spec`, `scree_node_spec` (PIL-only
  render-time images composed from `tl.viz.render_*` primitives).

## Gotchas

- Evolution/table functions require the target layers' activations to have
  been saved; unsaved payload reads refuse — they never trigger recapture.
- `_SYMMETRY_TOLERANCE` and the numeric tolerances here are
  validation-adjacent: treat loosening as a tripwire change, not a tweak
  (round-7 R13 flagged the scale-blindness of the symmetry tolerance — see
  the fixplan's replay-tolerance lane before touching it).

## Tests

`tests/` files matching `repgeom`/`mds`/`rdm` (e.g. `pytest -k repgeom`).
