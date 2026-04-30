# Intervention API Reference

This page documents the public v2 intervention API that shipped in the current
branch. It avoids proposed naming changes from separate design workstreams.

## Selectors

Selectors resolve against completed `ModelLog.layer_list` records.

| Selector | Signature | Use |
| --- | --- | --- |
| `tl.label` | `label(name: str)` | Exact final, raw, short, or pass-qualified label. |
| `tl.func` | `func(name: str)` | Match captured function name such as `"relu"` or `"linear"`. |
| `tl.module` | `module(address: str)` | Match a module output boundary. |
| `tl.contains` | `contains(substring: str)` | Case-insensitive label substring search. |
| `tl.where` | `where(predicate, *, name_hint=None)` | Predicate over layer pass records; non-portable. |
| `tl.in_module` | `in_module(address: str)` | Match sites contained in a module address. |

Selectors compose with `&` and `|` for in-memory discovery:

```python
candidate = tl.in_module("block") & tl.func("relu")
sites = log.find_sites(candidate, max_fanout=8)
exact = tl.label(sites.labels()[0])
```

For saved or replayed specs, prefer exact `tl.label(...)` or another simple
portable selector after discovery.

## Forward Helpers

Forward helpers return `HelperSpec` objects that can be passed to `set`,
`attach_hooks`, `do`, live capture `hooks=...`, replay, or rerun.

| Helper | Signature | Portability |
| --- | --- | --- |
| `tl.zero_ablate` | `zero_ablate(*, force_shape_change=False)` | Portable built-in; append-compatible unless shape changes are forced. |
| `tl.mean_ablate` | `mean_ablate(source=None, *, over="self", force_shape_change=False)` | Portable for supported tensor/self sources; batch-dependent policies can block append. |
| `tl.resample_ablate` | `resample_ablate(source=None, *, from_=None, seed=None, force_shape_change=False)` | Built-in, stochastic; seeded runs are reproducible, append-incompatible. |
| `tl.steer` | `steer(direction, magnitude=1.0, *, coef=None, feature_axis=None, force_shape_change=False)` | Portable when `direction` is serializable tensor data. |
| `tl.scale` | `scale(factor, *, force_shape_change=False)` | Portable built-in and append-compatible. |
| `tl.clamp` | `clamp(*, min=None, max=None, force_shape_change=False)` | Portable built-in and append-compatible. |
| `tl.noise` | `noise(std, *, seed=None, force_shape_change=False)` | Built-in stochastic helper; seeded runs avoid global RNG consumption. |
| `tl.project_onto` | `project_onto(direction, *, feature_axis=None, force_shape_change=False)` | Portable when `direction` is tensor data. |
| `tl.project_off` | `project_off(direction, *, feature_axis=None, force_shape_change=False)` | Portable when `direction` is tensor data. |
| `tl.swap_with` | `swap_with(other_label, *, force_shape_change=False)` | Tensor and LayerPassLog-like sources work in memory; label resolution is runtime-dependent. |
| `tl.splice_module` | `splice_module(module, *, input="activation", output="activation", force_shape_change=False)` | Executable in the same environment; not portable and not append-compatible. |

## Backward Helpers

Backward helpers are Tier-1 live/rerun-only helpers.

| Helper | Signature | Portability |
| --- | --- | --- |
| `tl.bwd_hook` | `bwd_hook(fn)` | Live/rerun-only; not portable. |
| `tl.gradient_zero` | `gradient_zero(*, force_shape_change=False)` | Live/rerun-only; not portable. |
| `tl.gradient_scale` | `gradient_scale(factor, *, force_shape_change=False)` | Live/rerun-only; not portable. |

## ModelLog Mutators

| Method | Use |
| --- | --- |
| `log.set(site, value)` | Record a one-shot tensor or callable replacement and mark the recipe stale. |
| `log.attach_hooks(site, hook)` | Add sticky helper/callable hooks to the recipe. |
| `log.do(...)` / `tl.do(log, ...)` | Apply an intervention and dispatch to `replay`, `rerun`, or `set_only`. |
| `log.fork(name=None)` | Create an isolated branch for experiments. |
| `log.replay(hooks=None)` | Propagate over the saved DAG without calling `model.forward`. |
| `log.rerun(model, x, append=False)` | Re-execute the model under the active spec. |
| `log.save_intervention(path, level=...)` | Write a `.tlspec/` intervention recipe. |

## Bundle

Construct with `tl.bundle(...)` or `tl.Bundle(...)`:

```python
bundle = tl.bundle({"clean": clean_log, "patched": patched_log}, baseline="clean")
```

Common operations:

| Operation | Use |
| --- | --- |
| `bundle.names` | Member names in order. |
| `bundle["clean"]` | Access one `ModelLog`. |
| `bundle.node(site)` | Return a `NodeView` across members after relationship checks. |
| `bundle.compare_at(site)` | Pairwise comparison matrix at a shared site. |
| `bundle.metric(fn)` | Apply a per-member metric. |
| `bundle.joint_metric(fn)` | Apply a metric to the whole bundle. |
| `bundle.do(...)`, `bundle.attach_hooks(...)`, `bundle.replay()`, `bundle.rerun(model, x)` | Apply mutator/propagation calls to each member. |
| `bundle.fork(name=None)` | Fork all members into a new bundle. |

Relationship gates are intentional. Operations that require shared topology or
same-input evidence fail when TorchLens cannot prove enough compatibility.
