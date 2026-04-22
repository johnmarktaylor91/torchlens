# Conditional Branch Attribution

TorchLens can attribute captured tensor ops to Python conditional branches in eager
`forward()` code. This metadata is populated even when `save_source_context=False`:
rich source text is optional, but the call-site identity fields needed for attribution
are always captured.

## What TorchLens Attributes

- `if` / `elif` / `else` chains in eager Python `forward()`
- Ternary expressions (`x if cond else y`)
- Graphviz forward-edge labels for `THEN`, `ELIF`, and `ELSE`

Ternary attribution uses `(line, col_offset)` matching. On Python 3.11+ this gives full
same-line arm attribution; on Python 3.9/3.10 ambiguous same-line ternaries fail closed
instead of guessing.

## Classified Only, Not Branch-Attributed

These bool consumers are recognized and classified, but TorchLens does not treat them as
branch arms in this sprint:

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

The sprint intentionally deferred a few follow-on items:

- dagua conditional-edge rendering
- ELK conditional-edge rendering
- while-loop body attribution

Reference: `.project-context/plans/if-else-attribution/plan.md` (v7).
