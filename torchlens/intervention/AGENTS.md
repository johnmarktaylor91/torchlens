# torchlens/intervention/ - Intervention Tools and Bundle Facade

## What This Is
Intervention owns selector resolution, hook/site helpers, replay/rerun/save workflows, and
the compatibility facade for Bundle-level comparison tools. Bundle members are ordinary
`Trace` objects; `tl.Bundle` projects sub-Trace objects across members through
internal Super* views.

## Internal Layout
The package's PRIMARY surface lives in flat modules this layout previously
omitted entirely: `selectors.py` (site selectors incl. `without_op`),
`sites.py`, `resolver.py` (trusted-callable resolution), `hooks.py`,
`predicates.py`, `replay.py` (`push`/`push_from`), `rerun.py` (`run`),
`save.py` (intervention specs), `runtime.py`, `sweep.py`, `types.py`,
`handles.py`, `helpers.py`, and `errors.py`.
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
