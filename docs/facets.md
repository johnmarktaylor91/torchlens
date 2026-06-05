# Facets

Facets are named views of a captured activation record. They provide one access
surface for structural outputs, such as an LSTM's `output`, `h_n`, and `c_n`,
and semantic recipe outputs, such as attention `q`, `k`, `v`, MLP activations,
normalization parameters, and embedding weights.

## Access

Use item access as the canonical form:

```python
log = tl.trace(model, x)
q = log.modules["blocks.0.attn"].facets["q"]
h_n = log.modules["lstm"].facets["h_n"]
```

Attribute access is convenience sugar for valid Python identifiers that do not
collide with `FacetView` methods:

```python
q = log.modules["blocks.0.attn"].facets.q
```

Item access always works for names that collide with methods, such as `keys`, and
for dotted structural names such as `out1.0`. Typed container paths are also
accepted when string names would collide.

Structural output names come from the captured `ContainerSpec`:

```text
single tensor output -> out
named output         -> values, indices, output, h_n, c_n
positional output    -> out0, out1
nested positional    -> out1.0
```

`module.outs` remains a thin alias for raw structural outputs. `module.facets` is
the unified named access layer. If a module was called more than once,
`module.facets` raises; choose an explicit pass with `module.calls[n].facets`.

## Specs

Recipes now return `FacetSpec` objects rather than computed tensors. A spec
records:

```text
home kind: op, module_output, module_input, parameter, computed
home label/address and pass/call index
output path
transform primitive chain
capability class and flags
value version
conflict or alias group
recipe id and version
```

The supported structural primitive vocabulary is:

```python
spec[key]
spec.heads(n_heads, d_head)
spec.split(n_sections, dim=-1)
spec.reshape(...)
spec.transpose(dim0, dim1)
spec.select(dim, index)
```

Read is lazy: `facet.value` applies the transform chain to the home value.
Facet objects are tensor-like, so most tensor reads can use the facet directly:

```python
torch.equal(module.facets["q"], manual_q)
module.facets["q"].shape
```

## Gradients

`facet.grad` applies the same transform chain to the home op's saved output
gradient. It is available only for grad-capable op-anchored structural facets and
only when the home op has a saved gradient.

Default traces do not save gradients. When a gradient is unavailable,
`facet.grad` returns a `MissingGradient` sentinel with an exact recapture
instruction. The sentinel does not raise merely because it was returned; tensor
use raises:

```python
log = tl.trace(model, x)
missing = log.modules["attn"].facets["q"].grad
print(missing.reason)

log = tl.trace(model, x, backward_ready=True, gradients_to_save="all")
log.log_backward(log[log.output_layers[0]].out.sum())
q_grad = log.modules["attn"].facets["q"].grad
```

Parameter facets, such as LayerNorm `gamma` and Embedding `weight`, read the
parameter value. Their parameter gradients are not exposed through `facet.grad`
in P1; this path is intentionally read-only.

## Registration

The process registry remains additive:

```python
@tl.facets.register(class_name="MyAttention", target_scope="module")
def my_attention(module):
    ...
```

Registration affects future traces. Each trace captures an immutable registry
snapshot with a version and provenance id, so lazy facet access is stable even if
the global registry changes later.

Per-trace and context additions are also capture-time only:

```python
log = tl.trace(model, x, recipes=[my_attention])

with tl.facets.using(my_attention):
    log = tl.trace(model, x)
```

`tl.facets.reset()` restores the process registry to built-ins. `tl.facets.list()`
and `tl.facets.info(class_name)` expose registry metadata.

Conflict resolution is deterministic: `class_qualname` is more specific than
`class_name`, which is more specific than a predicate; user recipes beat built-ins
only within the same specificity tier. True same-tier facet collisions warn.

## Capability Classes

Facet transform chains fail closed. The chain capability is the intersection of
its primitive flags.

```text
bijective_view      reshape, transpose, permute-style exact views; read, grad, write
selection           slice, select, split; read, grad, write
aliasing_selection  grouped-query or broadcast-style aliases; read and grad only
computed            callable or multi-home values; read-only
```

Only `bijective_view` and `selection` chains are writable. `aliasing_selection`
facets, such as grouped-query K/V heads selected through a query-head index,
require an explicit alias policy and are refused by default. `computed` facets
are never writable.

## Intervention

Facet intervention reuses the existing TorchLens hook machinery. A facet target
resolves to its home op, and TorchLens installs a normal whole-output hook on
that home. The wrapper reads the facet slice from the home output, calls the
user hook on that slice, then scatters the edited slice back into the full home
output returned by the hook.

Use `tl.facet(name)` for a named facet and `tl.head(index, name)` or
`tl.facet(name).head(index)` for attention-head slices:

```python
log = tl.trace(model, x, layers_to_save="all", save_arg_values=True)
edited = log.fork("ablated")
edited.attach_hooks(tl.head(3, "q"), tl.zero_ablate())
edited.rerun(model, x)
```

Static patching uses the same scatter-back path:

```python
patch = torch.zeros_like(log.modules["blocks.0.attn"].facets.head(3).q)
edited = log.fork("patched")
edited.set(tl.facet("q").head(3), patch)
edited.rerun(model, x)
```

`tl.head(index)` without a facet name targets the default attention projection
facets `q`, `k`, and `v` wherever they are present:

```python
edited.attach_hooks(tl.head(3), tl.zero_ablate())
```

Scatter-back preserves the home tensor dtype, device, shape, and memory format.
`bijective_view` primitives write through exact view operations such as reshape
and transpose. `selection` primitives scatter into the selected region, such as
a split third of a fused QKV projection or one selected head.

When multiple facet edits share one home op, TorchLens applies them in attach
order and checks boolean write masks before rerun. Non-overlapping writes, such
as GPT-2 `c_attn` edits to `q` head 0 and `k` head 1, compose on the shared
home. Overlapping same-home writes fail at attach time with a conflict error.

Facet intervention fails closed in these cases:

```text
computed facet        refused; read-only
aliasing_selection    refused unless a future explicit alias policy is supplied
non-op home           refused
non-raw value version refused as not intervention-safe
overlapping writes    refused before rerun
```

Attention reconstruction, paired `grad_fn` input-side gradients, and
per-module-eager reconstruction remain later phases.

## Built-In Inventory

TorchLens records built-in facet capabilities as data in
`torchlens.semantic.recipes.BUILTIN_FACET_CAPABILITY_INVENTORY`. Current classes
are:

```text
op_structural
parameter
module_input
module_output
computed_read_only
missing
```

Only `op_structural` built-in facets may claim facet-gradient capability in P1.

## Glossary

`FacetSpec`: ABI record describing a facet home, transform chain, capability
flags, value version, alias/conflict group, and recipe provenance.

`tl.facet(name)`: selector for facet-level intervention on a named facet.

`tl.head(index, name=None)`: selector for facet-level intervention on one
attention head. With `name=None`, targets default `q`, `k`, and `v` facets.

`scatter_update`: ABI method that writes an edited facet slice back into the
full home-op output and returns the edited home tensor.

`MissingGradient`: typed sentinel returned by `facet.grad` when the home op has
no saved gradient or the facet is not grad-capable.

`bijective_view`: transform class for exact view operations such as reshape and
transpose.

`selection`: transform class for slice/select/split operations.

`aliasing_selection`: transform class for repeated or shared storage selections,
such as grouped-query K/V head access.

`computed`: transform class for callable, nonlinear, or multi-home values.

Write capability: a facet may be edited only when every primitive in its chain
is scatter-back capable and the home is an op raw output.

Alias policy: explicit rule required before an `aliasing_selection` facet can be
written; P2 provides no public policy and refuses these writes by default.

Conflict policy: same-home facet writes compose in attach order when their write
masks do not overlap; overlapping writes raise before rerun.

`recipes=`: `tl.trace` argument for additive per-trace facet recipes captured in
that trace's registry snapshot.

`using()`: `tl.facets.using(...)` context manager for additive context-local
recipes used only by traces captured inside the context.

`reset()`: `tl.facets.reset()` restores the process registry to built-in recipes.

Item-access naming: canonical `record.facets[...]` lookup for structural and
recipe facets, including dotted names and method-name collisions.
