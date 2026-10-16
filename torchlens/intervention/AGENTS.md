# torchlens/intervention/ - Intervention Tools and Bundle Facade

## What This Is
Intervention owns selector resolution, hook/site helpers, replay/rerun/save workflows, and
the compatibility facade for Bundle-level comparison tools. Bundle members are ordinary
`Trace` objects; `tl.Bundle` projects sub-Trace objects across members through
internal Super* views.

The capture-free lane is `binding.py` (`spec.bind(model)`) with `steering.py`
(`steer_generate`); which intervention path to use for speed is in
`docs/guides/fast_paths.md`.

## Internal Layout
The package's PRIMARY surface lives in flat modules this layout previously
omitted entirely: `selectors.py` (site selectors incl. `without_op`),
`sites.py`, `resolver.py` (trusted-callable resolution), `hooks.py`,
`predicates.py`, `replay.py` (`push`/`push_from`), `rerun.py` (`run`),
`save.py` (intervention specs), `runtime.py`, `sweep.py`, `types.py`,
`handles.py`, `helpers.py`, and `errors.py`.
- Facet selectors (`tl.head`, `tl.facet`, `.latent`) on a finished trace: a
  bare `FacetSelector` in `resolver.resolve_sites` (and so `trace[selector]`)
  resolves through `hooks._resolve_facet_specs`, the same resolution the write
  doors use, and returns `Facet` entries (`SiteTable.labels()` names their home
  ops); composites and `find_sites` still refuse. `tl.trace(save=<facet
  selector>)` goes through `selectors._facet_save_predicate`, a save-only
  predicate wired in `_trace_selector_helpers._split_save_options_and_predicate`;
  `intervene=`, `bind` and capture `hooks=` evaluate the selector itself and keep
  refusing, so no facet edit lands on a whole home tensor. `FacetSelector`
  hashes by its spelling.
- `_super/` - generic `Super[T]`, tensor-bearing Super mixins, SuperOp/SuperLayer, the
  remaining Super* log wrappers, and Bundle accessors.
- `_topology/` - Supergraph, SupergraphNode, TopologyDiff, `build_supergraph()`, and
  `compare_topology()`.
- `_metrics.py` - tensor distance metrics shared by Super* diff helpers.
- `bundle.py` - compatibility shim for the top-level `tl.Bundle` type and
  `AmbiguousLabelError`.

## Bundle Super Rules
Every sub-Trace accessor has a Bundle counterpart. `bundle.ops`, `bundle.layers`,
`bundle.modules`, `bundle.params`, `bundle.buffers`, `bundle.grad_fns`,
`bundle.module_calls`, and `bundle.grad_fn_calls` each return aligned Super* views.

`bundle.at(label)` dispatches to the matching Super accessor from a label string and raises a
clear ambiguity error when a label matches more than one accessor. Structural-agreement
predicates such as `is_structurally_consistent`, `shared_*_labels`, and
`divergent_*_labels` live on Bundle so callers can check alignment readiness before projecting
across members.
