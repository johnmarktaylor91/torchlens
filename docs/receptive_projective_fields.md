# Receptive & projective fields

TorchLens answers an influence question on the graph that actually executed: which source
positions can affect this target position, and which target positions can a source position
affect? The first is a receptive field; the second is a projective field. Both can be queried
between model endpoints or between any two reachable graph points.

This matters for modern networks. A sequential receptive-field formula can silently misstate a
ResNet, an encoder-decoder, or a model with branches. TorchLens follows the captured DAG, so
skip connections and merges are part of the answer rather than an afterthought.

## Start with an operation

```python
import torch
import torchlens as tl
from torchvision.models import resnet18

model = resnet18(weights=None).eval()
x = torch.randn(1, 3, 224, 224, requires_grad=True)
trace = tl.trace(model, x, backward_ready=True)

op = trace["layer4.1.conv2"]
rf = op.receptive_field
print(rf.size, rf.jump, rf.center0)

# .at() uses coordinates over only the target's derived windowed axes.
box = rf.at((3, 3))

# Gradient-bearing actions use a complete target-element index, including batch.
unit = rf.center_unit(batch_index=0)
empirical = rf.gradient(unit)
checked = rf.check(unit)
overlay = rf.show(unit, gradient=True)
overlay.save("resnet18_rf.png")
```

`ReceptiveFieldView.at()` is the cheap geometric query. `.gradient()` probes exactly one output
element and returns its empirical support mask. `.check()` is the zero-tolerance containment
tripwire, and `.show()` renders an input overlay; with `gradient=True`, it also overlays the
empirical heatmap. Gradient operations require a backward-ready trace and are deliberately
explicit because they consume autograd work.

## Two complementary methods

| Method | Use it for | Cost | What it returns |
|---|---|---:|---|
| Geometric | Convolutions, pooling, common transforms, trace-wide tables | One cached DAG solve | Exact geometry where the captured rules prove it; otherwise an honest status or bound |
| Gradient | Arbitrary differentiable behavior, holes, masks, and auditing | One or more autograd probes | The observed nonzero support mask for one seeded output element |

The geometric method retains exact rational geometry when it can. The gradient method is more
general, but a support mask is local to the captured inputs, parameters, and control path. Use
the two together: the empirical support must stay inside the geometric box.

## Status is part of the answer

Never read a numeric box without its `status`.

| `ReceptiveFieldStatus` | Meaning |
|---|---|
| `EXACT` | Exact integer hull of potential support for the captured execution. |
| `WHOLE_INPUT` | The whole far endpoint is the exact structural envelope. In a projective query, the display means the whole output/target even though the stable enum name remains `WHOLE_INPUT`. |
| `UPPER_BOUND` | Sound envelope, not an exact claim. |
| `DATA_DEPENDENT` | Routing depends on data; use `.gradient()`. |
| `UNKNOWN` | The operation is recognized, but captured facts or axis roles are insufficient. |
| `UNSUPPORTED` | A contributing operation has no registered geometric rule. |

This is deliberately conservative. Skip connections are composed over the real DAG; norms that
couple examples, unmasked attention, and global mixing can widen an answer to the whole relevant
extent. Transposed convolutions, masked attention, interpolation, gathers, and data-dependent
routing are never promoted to `EXACT` without a proof. A gradient mask can preserve sparse or
masked structure that an interval box cannot express.

## Receptive, projective, and layer-to-layer queries

The receptive view is target-anchored by default and looks backward to model input(s). The
projective sibling is source-anchored and looks forward to model output(s).

```python
# Backward-looking: input positions that can affect this target unit.
incoming = op.receptive_field.at((3, 3))

# Forward-looking: output positions reached by this source unit.
outgoing = op.projective_field.at((3, 3))

# State the same direction through the hybrid view when composing a workflow.
outgoing_again = op.receptive_field.at((3, 3), direction="projective")

# Restrict the far endpoint. source= selects an ancestor for a receptive query;
# target= selects a descendant for a projective query.
early = trace["layer1.0.conv1"]
late = trace["layer4.1.conv2"]
between = late.receptive_field.at((3, 3), source=early)
forward_between = early.projective_field.at((3, 3), target=late)
```

`source=`, `direction=`, and `target=` make the endpoints explicit. For receptive queries,
`source=` names the ancestor result grid; for projective queries, `target=` names the descendant
result grid. The `unit` always belongs to the view owner: the target for receptive queries and
the source for projective queries.

Trace tables are geometric and do not invoke backward:

```python
incoming_table = trace.receptive_fields(level="layer")
outgoing_table = trace.projective_fields(level="layer")
print(incoming_table.to_pandas()[["name", "status", "size", "jump"]])
```

`Op`, `Layer`, `ModuleCall`, and `Module` all expose the two sibling properties. A multi-pass
layer, multi-call module, or multi-output module call is intentionally ambiguous rather than
silently selecting a representative; use the trace table to enumerate boundary outputs.

## Namespace reference

`tl.receptive_field` is lazy and deliberately outside `tl.__all__`. Its result vocabulary is
`GridLayout`, `ReceptiveField`, `ReceptiveFieldAxis`, `ReceptiveFieldBox`,
`ReceptiveFieldBoxAxis`, `GradientReceptiveField`, `ReceptiveFieldProfile`,
`ReceptiveFieldValidation`, `ReceptiveFieldValidationStatus`, `ReceptiveFieldViolation`,
`ReceptiveFieldStatus`, `ReceptiveFieldAlignment`, and `ReceptiveFieldDirection`. The query
view type is `ReceptiveFieldView`.

For extension, use `register_rf_rule()` and `rules()`; `ReceptiveFieldRule` and
`ReceptiveFieldRuleContext` are the corresponding rule types, and `rules()` is a read-only
snapshot. `node_spec()` creates a `Trace.draw(node_spec_fn=...)` callback. `verify()` and
`self_check()` run the model-facing diagnostics, while `cross_validate()` gives the batch sweep.
The typed error surface is `ReceptiveFieldError`, `ReceptiveFieldUnavailableError`,
`ReceptiveFieldValidationError`, `AmbiguousInputError`, `AmbiguousPassError`,
`AmbiguousCallError`, `AmbiguousTargetError`, `NoInfluencePathError`, and
`BackendUnsupportedError`.

## Verify the tripwire

```python
# Exhaustive/sampled trace sweep over center units (and optionally corners).
results = tl.receptive_field.cross_validate(trace, units="center")

# The named model-facing convenience entry point.
results = tl.receptive_field.verify(trace, units="center")

# Capture a backward-ready trace internally, then sample both directions.
results = tl.validate(model, x, scope="receptive_field")
```

Each check is `PASS`, `FAIL`, or `INDETERMINATE`. There is no coordinate tolerance: a gradient
support position outside a claimed geometric box is a bug to investigate, not a value to relax.
`INDETERMINATE` is honest when geometry cannot support a containment claim.

The `tl.validate(..., scope="receptive_field")` scope is the sampled validator entry point. It
is useful in model integration checks; use `cross_validate()` or `verify()` when choosing the
operations, endpoints, or units yourself.

## Visual audit: cone, box, and heatmap

```python
unit = op.receptive_field.center_unit(batch_index=0)

# Highlight the ancestor cone in the graph drawing.
trace.draw(
    node_spec_fn=tl.receptive_field.node_spec(op, unit=unit),
    vis_save_only=True,
    vis_fileformat="svg",
)

# Save the source image overlay and the empirical gradient heatmap together.
op.receptive_field.show(unit, gradient=True).save("rf_overlay.png")
```

Use a complete index for `.gradient()`, `.check()`, and `.show()`; `.at()` alone takes the
shorter coordinate tuple over windowed axes. This separation prevents a guessed batch coordinate
from becoming an apparently precise answer.

## References

- Araujo, Norris, and Sim, [*Computing Receptive Fields of Convolutional Neural Networks*](https://distill.pub/2019/computing-receptive-fields/), Distill, 2019.
- Luo, Li, Urtasun, and Zemel, [*Understanding the Effective Receptive Field in Deep Convolutional Neural Networks*](https://papers.nips.cc/paper_files/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Paper.pdf), NeurIPS 2016.
- Lehky and Sejnowski, [*Network model of shape-from-shading: neural function arises from both receptive and projective fields*](https://www.nature.com/articles/333452a0), Nature, 1988.
