# Conditional Branch Attribution

TorchLens can attribute captured tensor ops to Python conditional branches in eager
`forward()` code. This metadata is populated even when `save_source_context=False`:
rich source text is optional, but the call-site identity fields needed for attribution
are always captured.

## What TorchLens Attributes

- `if` / `elif` / `else` chains in eager Python `forward()`
- Ternary expressions (`x if cond else y`)
- The code after an `if` / `elif` chain that has no `else` when every arm provably ends
  (see "Fall-through arms" below)
- Graphviz forward-edge labels for `THEN`, `ELIF`, `ELSE` and `FALLTHROUGH`

Ternary attribution uses `(line, col_offset)` matching. On Python 3.11+ this gives full
same-line arm attribution; on Python 3.10 ambiguous same-line ternaries fail closed
instead of guessing.

## Fall-through arms

Arms come from the source: TorchLens never invents an `else`. When a chain has no `else`
and every arm body ends in `return`, `raise`, `break` or `continue` (or in a nested `if`
whose two sides both end that way), the statements after the chain in the same block run
only when every test was false. TorchLens records them as one more arm of kind
`fallthrough`, appended after the source arms:

```python
import torch
from torch import nn

import torchlens as tl


class EarlyReturn(nn.Module):
    def forward(self, x):
        if x.mean() > 0:
            return x * 2
        return -x  # the fallthrough arm


trace = tl.trace(EarlyReturn(), torch.full((2, 2), -1.0))
conditional = trace.graph.conditionals[0]
assert conditional.fired_arm_kind == "fallthrough"
assert conditional.outcome == "fallthrough"
assert not conditional.has_else
```

On an input whose test is false, `conditional.fired_arm_kind == "fallthrough"`, the `neg`
op's `conditional_branch_stack` holds `(conditional, "fallthrough")` and the drawing labels
its entry edge `FALLTHROUGH`, never `ELSE`. `has_else` stays `False` and `num_arms` counts
the fall-through arm. The rule is conservative: `try`, `with`, `match` and loop bodies are
never proven to end, so a chain whose arm ends in one of them records no fall-through arm,
and neither does a plain `if` whose body does not end (its following code runs either way).

## Which way did it go: `Conditional.outcome`

`Conditional.outcome` is a derived, unpersisted string read from the recorded arms:

| Value | Meaning |
| --- | --- |
| `then`, `elif`, `else`, `fallthrough` | that arm fired (or, with no tensor op in it, its test was witnessed selecting it) |
| `skipped` | every test was witnessed false and the chain has no `else` or fall-through arm |
| `multiple` | more than one arm fired (a conditional evaluated on several passes) |
| `unknown` | no evidence decides it (a pure-Python or compound test with no arm fired) |

`fired_arm_kind` is `None` when no single source arm's body was witnessed firing;
`outcome` says why.

## Classified Only, Not Branch-Attributed

These bool consumers are recognized and classified, but TorchLens does not treat them as
branch arms yet:

- `assert tensor_cond`
- standalone `bool(x)`
- comprehension filters
- `while cond:`
- `match case ... if guard:`

## Documented False Negatives

TorchLens cannot attribute branches when the predicate bypasses captured tensor-bool flow:

- `if python_bool:`
- `if self.training:`
- `if tensor.item() > 0:`
- shape or metadata predicates such as `if x.shape[0] > 0:`
- functional conditionals such as `torch.where`, `torch.cond`, or masked blending

## Unsupported / Source-Unavailable Cases

Conditional attribution is unsupported or intentionally fail-soft in these cases:

- Jupyter or REPL cells
- `exec` / `eval`
- `torch.compile`, `torch.jit.script`, `torch.jit.trace`
- `nn.DataParallel` / `DistributedDataParallel`
- monkey-patched `forward` methods

When source resolution is unavailable, TorchLens classifies the site as unknown and does
not emit confident branch attribution.

## Deferred Features

Not yet supported:

- dagua conditional-edge rendering
- while-loop body attribution
