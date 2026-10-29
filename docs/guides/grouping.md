# How TorchLens groups operations into layers

Every operation TorchLens records is an op. Ops that are "the same layer run again" are grouped
into one layer with several passes: `linear_1` with passes `linear_1:1`, `linear_1:2`, and so on.
This page says exactly when two ops are passes of one layer, so you can predict every label in a
trace.

## The rule in plain words

A layer is one operation written at one place in a module's code. Running that same place again
is another pass of the same layer. That covers a loop, a recursion, a comprehension, a later
generation step, and the module itself being called again. Two different places are two layers,
even when they compute the same operation with the same weights.

A module object is a scope. Everything inside one module instance is grouped by where it sits in
that module's code, and calling the module again, from anywhere, gives every op inside it its next
pass. So `self.block(x)` called three times from a loop gives every op inside `block` three
passes, and the pass number equals the module call number (`block:1`, `block:2`, `block:3`).

Argument values, tensor shapes, shared weights and the shape of the graph are never evidence
either way.

**A loop written out by hand is N separate layers, not one layer with N passes, and that is
intended.** If you write

```python
x = torch.tanh(x) * 0.5
x = torch.tanh(x) * 0.5
```

you get four single-pass layers (two `tanh`, two `mul`), because those are two places in your code.
Write the same body inside `for _ in range(2):` and you get one `tanh` layer and one `mul` layer,
two passes each.

## The rule, precisely

For every recorded op TorchLens forms a key from four parts. Two ops are passes of one layer if
and only if their keys are equal.

1. **Scope**: the innermost module instance running when the op ran, by object identity (the root
   model when no submodule is running). One instance reached through two parents is one scope.
2. **Call site**: the chain of Python frames inside that module's current `forward` call, from the
   `forward` down to the line that called the torch function. Each frame is reduced to its code
   (file, function, first line and bytecode) and the instruction that was running in it. Helper
   functions, decorators, lambdas and torch's own Python (`torch.nn.functional`) are all part of
   the chain; TorchLens's own frames are not.
   - On Python 3.11 and later the instruction is its source span (line and columns), so
     `torch.sin(a) + torch.sin(a)` on one line is two places.
   - On Python 3.10 the instruction is its bytecode offset. The compiler writes a `while`
     condition twice; the second copy is folded onto the first, so a tensor op in a `while`
     condition is one layer.
   - Recursion collapses: every depth of a recursive function maps to the same chain, so a
     recursive body gives one layer with a pass per depth. Entering the recursion from two
     different places still gives two layers.
3. **Operation**: the torch function name (`linear`, `__add__`, `tanh`).
4. **Output slot**: which output of a multi-output op (`max` gives values and indices).

Passes are numbered 1 to N in capture order, flat across nested loops. Because weights are not
part of the key, the passes of one layer can use different weights; the layer then lists every
weight its passes used (`layer.params`), and each pass lists its own (`layer.ops[k].params`).

When no Python frame sits between the module boundary and the op (a module whose `forward` is a C
function, for example), TorchLens falls back to the op's position: the same operation type at the
same position within the module call. Buffers are grouped by the buffer itself (one layer whose
passes are its versions). Inputs and outputs are always single-pass layers.

## Cases

| Code | Layers |
| --- | --- |
| `for _ in range(3): x = torch.tanh(x) * 0.5` | `tanh` 3 passes, `mul` 3 passes |
| `for _ in range(3): x = torch.tanh(x)` | `tanh` 3 passes |
| The two-op statement written out twice by hand | 4 single-pass layers |
| `torch.tanh(torch.tanh(torch.tanh(x)))` on one line | 3 single-pass layers |
| `nn.RNNCell` or `nn.LSTMCell` applied per time step, with `x[:, t]` | every op, one pass per step |
| Two residual adds in a block called once | 2 single-pass `add` layers |
| That block called 3 times from a loop | each `add` 3 passes, kept apart |
| A `ModuleList` of 3 such blocks (a transformer decoder) | 6 single-pass `add` layers |
| `relu(relu(x))` written on two lines | 2 single-pass layers |
| A helper function called from two lines | its ops: one layer per calling line |
| A helper called from one loop line | its ops: one layer, one pass per iteration |
| A helper called from two different loops (2 and 3 iterations) | 2 passes and 3 passes, two layers |
| Two same-type ops on one line, inside a loop | two layers, one pass per iteration each |
| One raw weight used on two lines, `x @ W` twice | 2 single-pass `matmul` layers |
| Embedding and output head with tied weights | separate layers |
| One module called 3 times from a loop | 3 passes |
| One module instance called from 3 different lines | 3 passes |
| One `nn.ReLU` instance reused on 3 lines, with other ops between | `relu` 3 passes |
| One module instance shared by two parent modules | 1 layer, 2 passes |
| A `ModuleList` of distinct but identical modules | separate layers |
| Siamese encoder (one encoder on two inputs) | every encoder op 2 passes |
| `[torch.exp(c) for c in cs]` | `exp` one pass per element |
| `sum(h(x) for h in self.heads)` over a `ModuleList` | each head its own layer; builtin `sum` gives `radd` once and `add` N-1 times |
| `map(torch.sin, xs)`, `map(lambda t: torch.sin(t), xs)` | `sin` one pass per element |
| `torch.func.vmap` or `torch.func.grad` in a loop | the transform boundary op, one pass per iteration |
| Greedy generation calling a model from one line | every op, one pass per step |
| Nested loops, 2 outer by 3 inner, no module | inner ops 6 passes, outer ops 2 passes |
| `break` partway through a loop | executed passes only |
| A branch taken on some iterations | each line counts its own executions |
| A tensor op in a `while` condition, body runs N times | condition ops N+1 passes |
| Recursion of depth 3 | body ops 3 passes |
| Recursion entered from two places | two layers |
| A buffer updated in place in a loop | body ops one pass per iteration; the buffer one layer of versions |
| A buffer updated on two different lines | 2 single-pass `add_` layers |
| `x * i` with a varying scalar, or a growing `cat`, in a loop | 1 layer, one pass per iteration |
| A loop over a `ParameterList`, `x = torch.tanh(x @ w)` | `matmul` and `tanh` one pass per weight |
| Root `ModuleList` loop with glue `x = x + blk(x)` | blocks separate; the glue `add` one pass per block |
| `torch.utils.checkpoint` of one block from two lines | the block's ops 2 passes each |
| A forward hook on a module called in a loop | hook ops one pass per call |
| One hook on two distinct modules called from one loop line | hook ops: one layer per module |
| A loop over a `ParameterList`: which weights does the layer list? | all of them; each pass lists its own |
| A special first step before a loop that uses the same weight | first-step ops single; loop ops one pass per iteration |
| A function built with `exec` and called in a loop | one pass per iteration |
| Two separately written identical functions | separate layers |
| A decorator whose wrapper calls `f(x)` on two lines | two layers per op |
| A TorchScript submodule called in a loop | its boundary op, one pass per call |
| `nn.MultiheadAttention` (many `linear` calls inside torch's code) | each internal call site its own layer |
| Two separate traces of one model | each trace numbers its own passes |

## Limits

- One Python instruction that makes several same-function torch calls through C code (builtin
  `sum`, `map`, `functools.reduce`, a C extension calling the torch Python API) gives passes of
  one layer, because there is no Python frame to tell the calls apart.
- A forward hook (or pre-hook) runs outside the module's own `forward`, so its ops belong to the
  caller's scope, with the hooked module as part of the place: the same hook on two distinct
  modules gives two layers (two weight-normed layers in one `nn.Sequential` have separate
  weight-norm layers), a hooked module called from one loop line gives hook ops one pass per call,
  and called from two lines, two hook-op layers.
- TorchScript modules and `torch.func` transforms are recorded as single boundary ops, keyed by
  their Python call site; TorchLens does not see inside them.
- Capturing across concurrent Python threads, including `nn.DataParallel`, is not supported.
- Captures without Python call-site evidence (the JAX and preview backends, and the predicate mode
  of `tl.record`) use the older structural grouper. `trace.grouping_policy["detector"]` says which
  ran: `"callsite_v1"` for the call-site rule, `"grouper_v1"` for the structural grouper.
- Saved traces keep the labels they were saved with; loading never regroups.

## How this relates to `site_key`

`op.site_key` names an op's position within one module call (module address, op type, output
slot and an ordinal), and it is the same whichever grouping ran. Grouping uses the call site, not
the position. Most layers have one site key; a loop inside one module call gives a layer whose
passes sit at different positions, and `layer.site_key` refuses for it with
`layer_site_ambiguous`. Read per-pass keys with `layer.ops[k].site_key`.
