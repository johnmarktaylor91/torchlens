# receptive_field/ - Implementation Guide

Lazy public submodule (`tl.receptive_field` resolves through the top-level lazy
`__getattr__` table). Influence geometry is solved on FIRST PROPERTY ACCESS:
the `op.receptive_field` / `op.projective_field` property bodies call
`_engine.solve(trace)` / `solve_projective(trace)` EAGERLY (with the solution
cached on the trace) and then return `ReceptiveFieldView`s — the laziness
boundary is the property access itself, not the first view query.

## __init__.py
- `verify(trace, ...)` is the public tripwire entry point; `self_check()` aliases it.
- `ReceptiveFieldVerification.verdict` is tri-state PASS / FAIL / INDETERMINATE;
  INDETERMINATE (unarmed) never reads as a pass via `.passed`.
- Arming the gradient tripwire needs `requires_grad` inputs, `backward_ready=True`,
  and `save_mode="reference"`; unarmed containment checks stay INDETERMINATE.
- `_empirical_adjoint_checks()` compares signed receptive vs projective derivatives.
- Importing the package imports `rules/`, registering the built-in rule pack before
  any engine path can run.

## Engines
- `_engine.py` - `solve(trace)` / `solve_from(trace, source)` / `lookup(trace, op)`
  drive the backward (receptive) DAG solve.
- `_engine_forward.py` - `solve_projective(trace, target_ops)` is the forward
  (projective) transpose solve.
- `_engine_geometry.py` - `_AxisState` / `_InputState` and the affine window algebra
  (`_compose`, `_transpose_mapped`).
- `_engine_descriptor.py` - `_descriptor()` builds public `ReceptiveField` records.

## Queries and gradients
- `_query.py` - `box_for_unit()` (receptive box for one output unit).
- `_forward_query.py` - `box_for_source_unit()` (projective box for one input unit).
- Window and clip maps (`_standard_window_index_set`, `map_transposed_convolution_index_set`,
  `_IndexSet.clipped`) compute a one-progression index set in closed form and enumerate
  otherwise; both paths must return the same canonical set (`tests/test_perf5_rf_invariant_cost.py`).
- `_query._shared_query_scope()` lets many per-unit queries over one solution share graph
  facts and rule results (the always-on metadata invariants open it); outside it every
  query derives its own.
- `_path.py` - `resolve_graph_point()`, `ancestor_labels()`, `descendant_labels()`,
  `require_path()` own graph-point resolution and path checks.
- `_gradient.py` - `gradient_for_unit()`, the empirical VJP probe; snapshots and
  restores probe state (`_ProbeSnapshot`) so probing never mutates the trace.
- `_gradient_forward.py` - `projective_gradient_for_unit()` (double-VJP column).

## Views, tables, viz
- `_view.py` - `ReceptiveFieldView` with `at()`, `center_unit()`, `check()`,
  `gradient()`, and `show()`; `show(grad=True)` recomputes WITHOUT retain_graph
  and frees the autograd graph, so call it last or re-capture.
- `_table.py` - `build_rf_profile()` backs `Trace.receptive_fields()` /
  `Trace.projective_fields()` (pandas required at call time).
- `_viz.py` - `show()` and `node_spec()` render overlays.
- `_types.py` - frozen public dataclasses/enums (`ReceptiveField`,
  `ReceptiveFieldAxis`, `ReceptiveFieldBox`, `ReceptiveFieldStatus`,
  `ReceptiveFieldDirection`, `GridLayout`, ...).
- `_validation.py` - `cross_validate()` containment checks consumed by `verify()`.
- `_errors.py` - typed error family rooted at `ReceptiveFieldError`
  (`AmbiguousInputError`, `NoInfluencePathError`, `ReceptiveFieldUnavailableError`, ...).

## rules/ subpackage
- `_rules.py` (at the PACKAGE level, `receptive_field/_rules.py` — not inside
  `rules/`) holds the registry: `register_rf_rule()`, `rules()`,
  `ReceptiveFieldRule` / `ReceptiveFieldRuleContext`.
- `rules/__init__.py` imports the eight built-in modules (`attention`, `conv_pool`,
  `elementwise`, `interpolation`, `linear`, `norms`, `sequence`, `transforms`) and
  calls `_install_builtin_rule_pack()` idempotently.
- `rules/_utils.py` - shared parsers (`spatial_rank()`, `int_tuple()`, `int_config()`).
- Rules return `_RuleResult` geometry per function family; unknown functions fall
  back to conservative engine defaults rather than guessing.

## Gotchas
- `verify()` probes backprop through the ONE capture-time autograd graph: it catches
  derivation/indexing/rule bugs but does not re-attest capture fidelity (that is
  `torchlens.validation`'s replay tripwire).
- Poisoned (diverged-run) traces are refused by RF verification
  (`_refuse_poisoned_rf_verification`).
- Empirical probes must leave trace state byte-identical; keep the snapshot/assert
  pair in `_gradient.py` intact when touching probe paths.
