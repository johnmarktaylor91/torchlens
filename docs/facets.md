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

P1 defines write capability flags for the ABI, but does not implement
intervention or scatter-back. Attention reconstruction, paired `grad_fn`
input-side gradients, and intervention are later phases.

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

`MissingGradient`: typed sentinel returned by `facet.grad` when the home op has
no saved gradient or the facet is not grad-capable.

`bijective_view`: transform class for exact view operations such as reshape and
transpose.

`selection`: transform class for slice/select/split operations.

`aliasing_selection`: transform class for repeated or shared storage selections,
such as grouped-query K/V head access.

`computed`: transform class for callable, nonlinear, or multi-home values.

`recipes=`: `tl.trace` argument for additive per-trace facet recipes captured in
that trace's registry snapshot.

`using()`: `tl.facets.using(...)` context manager for additive context-local
recipes used only by traces captured inside the context.

`reset()`: `tl.facets.reset()` restores the process registry to built-in recipes.

Item-access naming: canonical `record.facets[...]` lookup for structural and
recipe facets, including dotted names and method-name collisions.
