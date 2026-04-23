# Prior Art Research: Python-level Control-flow Attribution for TorchLens

Date: 2026-04-22

## Scope

Question: how should TorchLens attribute each captured PyTorch op to the enclosing
Python `if` / `elif` / `else` branch in eager execution?

Goal of this report:

1. Survey prior art in PyTorch internals and adjacent Python tooling.
2. Compare AST, bytecode, `sys.settrace`, and frame-inspection approaches.
3. Recommend a low-risk stdlib-first design for TorchLens.

Legend:

- `Documented`: directly supported by cited docs or source.
- `Inference`: reasoned conclusion drawn from the cited material.

## Executive Summary

- No surveyed activation/introspection tool already gives eager-mode per-op branch attribution for arbitrary Python `if` / `elif` / `else`.
- `torch.fx.symbolic_trace` and `make_fx` do not preserve dynamic Python branches; they either reject data-dependent control flow, specialize it away, or capture only the executed path. [Documented: `torch/fx/proxy.py:382-390`, `torch/fx/_symbolic_trace.py:1280-1326`, https://docs.pytorch.org/docs/stable/fx.html, https://docs.pytorch.org/docs/2.9/compile/programming_model.non_strict_tracing_model.html]
- TorchDynamo is the strongest prior art for Python-level control-flow handling in PyTorch. It works at CPython bytecode level, recognizes conditional jumps, specializes many branches with guards, and graph-breaks on data-dependent tensor branching. [Documented: `torch/_dynamo/symbolic_convert.py:575-631`, `torch/_dynamo/symbolic_convert.py:723-726`, `torch/_dynamo/symbolic_convert.py:2921-2924`, https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamo_overview.html, https://pytorch.org/assets/pytorch2-2.pdf]
- TorchScript is the strongest prior art for source-based handling: it requires source access, parses Python source with `ast.parse`, and lowers `if` statements into its own IR. [Documented: `torch/_sources.py:12-35`, `torch/_sources.py:120-138`, `torch/jit/frontend.py:326-343`, `torch/jit/frontend.py:804-811`, https://docs.pytorch.org/docs/2.9/generated/torch.jit.script.html]
- `coverage.py` is the best general-Python prior art for combining static branch structure with runtime execution. It enumerates possible branch arcs from AST and records runtime line-to-line transitions via `sys.settrace`. [Documented: `coverage/parser.py:1001-1011`, `coverage/pytracer.py:313-325`, https://coverage.readthedocs.io/en/latest/branch.html]
- Recommendation: use AST as the primary control-flow model, with per-op frame inspection at TorchLens capture points. Do not make `sys.settrace` the default. Keep bytecode/disassembly as a fallback/debug aid, not the primary implementation. [Inference]

## Comparison Matrix

| Tool / family | Detection mechanism | Branch granularity | Runtime cost | Activation mode | Applicability to TorchLens |
| --- | --- | --- | --- | --- | --- |
| `torch.fx.symbolic_trace` | Proxy-based symbolic execution | None for dynamic branches; static branches specialized | Moderate tracing-time cost | On-demand | Negative prior art: does not preserve eager branch identity |
| `make_fx` / non-strict tracing | Python execution + operator overloading | Executed path only | Moderate tracing-time cost | On-demand | Negative prior art: single-path capture only |
| TorchDynamo | Bytecode interpreter + guards + graph breaks | Conditional jump / continuation boundary | High implementation complexity; lower runtime cost than `settrace` | On-demand compiler path | Strong conceptual prior art; too heavyweight/version-sensitive to reuse directly |
| TorchScript `script()` | Source extraction + AST parse + compiler IR | Statement / expression in TorchScript subset | Compile-time cost only | On-demand | Strong AST prior art; private helpers, subset semantics |
| TorchScript `trace()` | Execute once and record ops | Executed path only | Tracing-time cost | On-demand | Negative prior art for branch attribution |
| Captum | Hooks / forward-backward instrumentation | Module/layer execution only | Moderate to high depending on method | On-demand | No branch model |
| DeepSpeed FLOPs profiler | Forward hooks + monkey-patched functionals | Module/function call only | Moderate | On-demand | No branch model |
| `coverage.py` | AST for possible arcs + `sys.settrace` runtime events | Line-to-line arcs | High | Opt-in, global | Best Python prior art, but too heavy/conflicting for default TorchLens path |
| `sys.settrace` directly | Line/opcode callbacks | Line/opcode; no branch event | High | Global or per-thread | Useful optional debug mode, not default |
| `sys.setprofile` | Call/return profiler callbacks | No line/branch detail | Lower than `settrace`, but insufficient | Global or per-thread | Not enough information |
| `trace` / `bdb` | Thin wrappers over tracing callbacks | Statement stepping, breakpoints | High | Opt-in | No extra branch semantics |
| `dis` / `co_lines()` | Static bytecode and line-table inspection | Jump op / bytecode offset | Low offline cost | On-demand | Good fallback/debug aid only |
| mypy / pytype / radon | Static AST / CFG analysis | Static only | Offline only | On-demand | Useful for design intuition, not runtime attribution |

## PyTorch Prior Art

### 1. `torch.fx.symbolic_trace`

What it does:

- FX rejects converting a `Proxy` to `bool` by default. `Tracer.to_bool()` raises `TraceError("symbolically traced variables cannot be used as inputs to control flow")`. [Documented: `torch/fx/proxy.py:382-390`]
- The FX docs say dynamic control flow is the main limitation of symbolic tracing. Static control flow can be specialized, and `concrete_args` can partially specialize branch conditions. [Documented: https://docs.pytorch.org/docs/stable/fx.html, https://docs.pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing, `torch/fx/_symbolic_trace.py:1280-1326`, `torch.fx` docs lines around `concrete_args`: https://docs.pytorch.org/docs/stable/fx.html]

Evaluation:

- Detection mechanism: symbolic execution via proxy objects, not AST or bytecode. [Documented]
- Branch granularity: none for dynamic `if` / `elif` / `else`; static branches are burned in. [Documented]
- Performance cost: moderate tracing-time overhead, but not always-on. [Inference]
- Runtime activation: on-demand. [Documented]
- TorchLens relevance: mostly a "what not to copy" for branch attribution. FX either errors, specializes away the branch, or leaves a leaf call. It does not preserve "this op belonged to else-arm N" in eager Python. [Inference]

TorchLens takeaway:

- Good negative prior art.
- Reusable idea: treat source-level control flow and op-level tracing as separate concerns. [Inference]

### 2. `make_fx` / functorch-style non-strict tracing

What it does:

- PyTorch's non-strict tracing docs explicitly say it runs a Python function and records tensor ops through operator overloading. The docs show an `if x.shape[0] > 2:` example where only the top branch is captured for the given input. [Documented: https://docs.pytorch.org/docs/2.9/compile/programming_model.non_strict_tracing_model.html]
- `make_fx` is the main entry point for this style of tracing. [Documented: `torch/fx/experimental/proxy_tensor.py:2281+`, https://docs.pytorch.org/docs/2.9/compile/programming_model.non_strict_tracing_model.html]

Evaluation:

- Detection mechanism: Python execution plus operator overloading / proxy tensor machinery. [Documented]
- Branch granularity: executed path only; no branch alternatives preserved. [Documented]
- Performance cost: moderate tracing-time cost. [Inference]
- Runtime activation: on-demand. [Documented]
- TorchLens relevance: not suitable as the primary control-flow attribution mechanism. It observes "what happened", but not the unexecuted branch structure. [Inference]

TorchLens takeaway:

- Another strong negative prior art: "single executed path" tracing is insufficient if the product requirement is explicit branch attribution rather than just op capture. [Inference]

### 3. TorchDynamo (`torch.compile`, `torch._dynamo`)

What it does:

- TorchDynamo works by interpreting Python bytecode and rewriting execution into compiled segments plus continuation frames. The overview docs show generated `__resume_at_<offset>` functions that resume after graph breaks. [Documented: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamo_overview.html]
- In source, conditional bytecodes are routed through `generic_jump(...)`, and Python 3.11 jump opcodes are bound with `POP_JUMP_FORWARD_IF_TRUE/FALSE = generic_jump(...)`. [Documented: `torch/_dynamo/symbolic_convert.py:575-631`, `torch/_dynamo/symbolic_convert.py:2921-2924`]
- `generic_jump` explicitly labels "Data-dependent branching", says Dynamo does not support tracing dynamic control flow, and suggests `torch.cond`. If the branch condition is tensor-dependent and partial-graph compilation is allowed, Dynamo graph-breaks, emits resume code, and continues after the jump. [Documented: `torch/_dynamo/symbolic_convert.py:575-585`, `torch/_dynamo/symbolic_convert.py:587-631`, `torch/_dynamo/symbolic_convert.py:723-726`]
- The PyTorch 2 paper states that most control flow is optimized away through specialization/guards, while less common truly dynamic control flow triggers a graph break and then resumes analysis after the jump. [Documented: https://pytorch.org/assets/pytorch2-2.pdf]

Evaluation:

- Detection mechanism: bytecode interpretation plus guard system. [Documented]
- Branch granularity: conditional-jump level, with explicit resume points. [Documented]
- Performance cost: much better than `sys.settrace` for compiler use, but implementation complexity is very high and tied to CPython bytecode details. [Inference]
- Runtime activation: on-demand via compile path. [Documented]
- TorchLens relevance: excellent conceptual prior art, poor direct implementation fit. [Inference]

Reusable ideas for TorchLens:

- Distinguish guardable/static branches from truly data-dependent branches. [Documented + Inference]
- Treat the current branch decision as a runtime fact derived from the executing frame, not from static source alone. [Inference]
- Avoid reproducing Dynamo's bytecode interpreter unless source is unavailable and the fallback benefit is worth the maintenance cost. Dynamo is tightly coupled to Python opcode variants (`POP_JUMP_*`, `TO_BOOL`, resume bytecode generation). [Inference]

### 4. TorchScript (`torch.jit.script`, `torch.jit.trace`)

What `script()` does:

- The public docs say `torch.jit.script` inspects source code and compiles it as TorchScript, supporting control-dependent operations. The docs include an `if x.max() > y.max(): ... else: ...` example. [Documented: https://docs.pytorch.org/docs/2.9/generated/torch.jit.script.html]
- Internally, TorchScript source extraction uses `inspect.getsourcefile` and `inspect.getsourcelines`, errors if source is unavailable, normalizes indentation, and parses with `ast.parse`. [Documented: `torch/_sources.py:12-35`, `torch/_sources.py:120-138`]
- In the frontend, `get_jit_def()` calls `parse_def()`, and `StmtBuilder.build_If()` lowers Python `ast.If` to TorchScript `If(...)`. [Documented: `torch/jit/frontend.py:326-343`, `torch/jit/frontend.py:804-811`]

What `trace()` does:

- The tracing docs say tracing only records operations seen for the provided example inputs and explicitly warn that it will not record control flow like `if` statements or loops. [Documented: https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html]

Evaluation:

- Detection mechanism:
  - `script()`: source + AST + compiler frontend. [Documented]
  - `trace()`: execute once and record ops. [Documented]
- Branch granularity:
  - `script()`: statement/expression in TorchScript subset; preserves `if` in TorchScript IR. [Documented]
  - `trace()`: executed path only. [Documented]
- Performance cost:
  - `script()`: compile-time source parse and frontend cost. [Documented + Inference]
  - `trace()`: tracing-time cost only. [Documented]
- Runtime activation: on-demand. [Documented]
- TorchLens relevance: `script()` is strong proof that source-plus-AST is viable for Python control flow, but TorchScript helpers are private and tied to compiler semantics/subset constraints. [Inference]

TorchLens takeaway:

- Copy the pattern, not the implementation:
  - use `inspect.getsourcefile` / `inspect.getsourcelines`
  - use `ast.parse`
  - cache parsed source by code object / file span
- Do not depend on `torch._sources.parse_def` or other private TorchScript internals as stable APIs. [Inference]

### 5. Captum / DeepSpeed / activation tooling

#### Captum

- Captum FAQ says some methods rely on hooks during back-propagation; reused modules can cause layer or neuron attribution methods to compute attributions for the last execution of the module; JIT models do not support hooks for hook-based methods. [Documented: https://captum.ai/docs/faq]

Evaluation:

- Detection mechanism: hooks around modules/layers, not source, bytecode, or branch-aware tracing. [Documented]
- Branch granularity: module execution only; repeated-module reuse is already ambiguous. [Documented]
- TorchLens relevance: useful negative prior art. Hook-based activation attribution is orthogonal to Python branch attribution. [Inference]

#### DeepSpeed FLOPs profiler

- DeepSpeed's FLOPs profiler docs/source say it profiles forward passes by recursively adding attributes, monkey-patching functionals/tensor methods, and registering `register_forward_pre_hook` / `register_forward_hook` handlers. [Documented: https://deepspeed.readthedocs.io/en/stable/_modules/deepspeed/profiling/flops_profiler/profiler.html]

Evaluation:

- Detection mechanism: forward hooks plus monkey-patched functionals. [Documented]
- Branch granularity: module/function-call level only. [Documented]
- TorchLens relevance: no branch awareness; good evidence that adjacent profiling tools generally stop at module/op call accounting, not enclosing Python branch attribution. [Inference]

## Python Stdlib and Coverage Prior Art

### 6. `coverage.py`

What it does:

- Coverage docs say branch mode collects pairs of line numbers (source, destination), and compares measured transitions with statically possible transitions. [Documented: https://coverage.readthedocs.io/en/latest/branch.html]
- In source, the AST parser's `_handle__If` enumerates possible exits for true and false arms. [Documented: `coverage/parser.py:1001-1011`]
- Runtime collection uses `sys.settrace` in `PyTracer.start()`. [Documented: `coverage/pytracer.py:313-325`]
- Coverage warns that `sys.settrace` conflicts with other tracers, and that `sys.setprofile` code execution is invisible to coverage's tracing. [Documented: https://coverage.readthedocs.io/en/latest/trouble.html]

Evaluation:

- Detection mechanism: hybrid AST + runtime tracing. [Documented]
- Branch granularity: line-to-line arcs, not expression-level branch-arm identity. [Documented]
- Performance cost: high relative to normal execution; coverage even exposes a slower `timid` trace mode. [Documented: `coverage/control.py:183-188`]
- Runtime activation: opt-in, process-wide / thread-wide. [Documented]
- TorchLens relevance: this is the best prior art outside PyTorch for "static branch possibilities + runtime execution facts." [Inference]

TorchLens takeaway:

- Strongly reuse the idea of separating:
  - static branch structure from AST
  - runtime location from execution events
- But do not copy the "always-on global `sys.settrace`" part for default TorchLens usage. TorchLens already intercepts ops, so it can usually sample the active frame only when an op is captured. That should be much cheaper and less conflicting. [Inference]

### 7. `sys.settrace`

- Python docs say `sys.settrace` emits `call`, `line`, `return`, `exception`, and `opcode` events. Per-opcode events require setting `frame.f_trace_opcodes = True`. [Documented: https://docs.python.org/3.11/library/sys.html]
- There is no branch event. You must infer branch decisions from line movement or opcode/jump execution. [Documented + Inference]
- The docs state it is intended for debuggers, profilers, coverage tools, and similar implementation-platform tooling. [Documented: https://docs.python.org/3.11/library/sys.html]

Evaluation:

- Detection mechanism: interpreter callbacks on every traced event. [Documented]
- Branch granularity: line-level by default; opcode-level with extra work. [Documented]
- Performance cost: high. [Inference]
- Runtime activation: global/per-thread while enabled. [Documented]
- TorchLens relevance: good optional fallback or debug mode, poor default path. [Inference]

Why not default to `settrace`:

- No direct branch semantic; branch identity must still be reconstructed.
- Conflicts with coverage/debuggers and other tracer users. [Documented: https://coverage.readthedocs.io/en/latest/trouble.html]
- Overhead is paid for all executed Python, not just TorchLens-captured ops. [Inference]

### 8. `sys.setprofile`

- Python docs say `setprofile` is called on call/return and C-call events, not for each executed line. [Documented: https://docs.python.org/3.11/library/sys.html]
- Coverage docs explicitly say code run under `sys.setprofile` does not fire trace events, so coverage cannot see it. [Documented: https://coverage.readthedocs.io/en/latest/trouble.html]

Evaluation:

- Detection mechanism: profiler callbacks. [Documented]
- Branch granularity: none. [Documented]
- TorchLens relevance: insufficient for branch attribution. [Inference]

### 9. `trace` module and `bdb`

- `trace` documents statement execution tracing and counting, and explicitly points users to coverage.py for advanced branch coverage. [Documented: https://docs.python.org/3.11/library/trace.html]
- `bdb` dispatches `line`, `call`, `return`, and `exception` events and builds debugger stepping on top of `sys.settrace`, not on branch-specific events. [Documented: https://docs.python.org/3.11/library/bdb.html]

Evaluation:

- Detection mechanism: wrappers over trace callbacks. [Documented]
- Branch granularity: none beyond line stepping / breakpoints. [Documented]
- TorchLens relevance: not a direct solution. [Inference]

### 10. `dis`, `code.co_lines()`, and bytecode inspection

- Python's `dis` docs expose conditional jump opcodes such as `POP_JUMP_FORWARD_IF_FALSE` / `POP_JUMP_BACKWARD_IF_FALSE`. [Documented: https://docs.python.org/3.11/library/dis.html]
- PEP 626 introduces precise line number semantics and `co_lines()` for mapping bytecode ranges to source lines. [Documented: https://peps.python.org/pep-0626/]

Evaluation:

- Detection mechanism: static bytecode inspection. [Documented]
- Branch granularity: jump-op level; still requires runtime state to know which branch was taken. [Documented + Inference]
- Performance cost: low offline cost. [Documented + Inference]
- TorchLens relevance: useful as a source-unavailable fallback or debugging/validation aid, but not ideal as the primary implementation because CPython bytecode is version-sensitive and lower-level than the user-facing requirement. [Inference]

## Static Analysis Prior Art

### 11. mypy

- `mypy.reachability.infer_reachability_of_if_statement()` statically marks `if` / `elif` / `else` blocks unreachable when conditions are provably constant. [Documented: `mypy/reachability.py:53-76`]
- mypy's binder tracks frames for branching and type narrowing, explicitly noting a new frame for control-flow branching. [Documented: `mypy/binder.py:50-77`]

Evaluation:

- Detection mechanism: static AST/control-flow analysis. [Documented]
- Branch granularity: static blocks and type states, not runtime branch choice. [Documented]
- TorchLens relevance: useful for data structures and terminology, not runtime attribution. [Inference]

### 12. pytype

- Pytype developer docs say it builds a control-flow graph where each node roughly correlates with a statement, and models `if` / `else` as distinct CFG paths that merge later. [Documented: https://google.github.io/pytype/developers/typegraph.html]

Evaluation:

- Detection mechanism: static CFG/typegraph. [Documented]
- Branch granularity: statement-level static paths. [Documented]
- TorchLens relevance: confirms AST/CFG representations are conventional, but pytype is not a runtime attribution model. [Inference]

### 13. Radon

- Radon analyzes cyclomatic complexity from AST using `ComplexityVisitor`, and exposes `cc_visit_ast`. [Documented: https://radon.readthedocs.io/en/stable/api.html]

Evaluation:

- Detection mechanism: static AST metrics. [Documented]
- Branch granularity: complexity counts, not runtime branch identity. [Documented]
- TorchLens relevance: minimal. It can count branches, not attribute executed ops to one. [Inference]

### 14. Pyright

- I did not find a public, branch-oriented runtime API in pyright during this research pass. Public material around pyright is mostly about type checking and narrowing, not exposing executed-branch metadata to other tools. [Inference]

Evaluation:

- Detection mechanism: static type/narrowing analysis. [Inference]
- TorchLens relevance: likely similar to mypy/pytype in spirit, but not directly reusable for runtime attribution. [Inference]

## Research and Compiler-adjacent Material

### 15. PyTorch 2 paper

- The PyTorch 2 paper is the most directly relevant research-style writeup for Python control flow in PyTorch. It states that TorchDynamo specializes most control flow away and graph-breaks on truly data-dependent control flow, resuming after the jump. [Documented: https://pytorch.org/assets/pytorch2-2.pdf]

TorchLens takeaway:

- Good conceptual split between:
  - static/specializable control flow
  - truly runtime/data-dependent control flow
- For TorchLens attribution, we do not need to compile or transform the program, only to name the enclosing branch of each already-captured op. That strongly argues for a simpler source-based approach than Dynamo. [Inference]

### 16. GraphMend

- GraphMend presents source-level transformations to remove graph breaks from dynamic control flow and Python side effects in PyTorch 2. [Documented: https://arxiv.org/abs/2509.16248]

TorchLens takeaway:

- Recent research treats unresolved Python control flow primarily as a source-transformation / CFG problem, not as an activation-hook problem. That supports AST-first thinking. [Inference]

### 17. PyTorch/XLA / LazyTensor docs

- XLA docs explicitly say Python `if/else/while/for` based on runtime values forces materialization and that two obvious solutions are either explicit control-flow ops or parsing Python source like TorchScript. [Documented: https://docs.pytorch.org/xla/release/r2.8/perf/recompilation.html]

TorchLens takeaway:

- This is an unusually direct external endorsement of the AST/source-parsing direction for Python control flow in PyTorch-adjacent tooling. [Inference]

## Recommendation

### Final recommendation

Choose **AST as the primary mechanism**, with **op-time frame inspection** at TorchLens capture points.

Do **not** make `sys.settrace` the default.

Do **not** make bytecode interpretation the primary implementation.

This is closest to "pure AST" in the choice set, but with an important clarification:

- pure AST alone is not enough
- the runtime fact should come from the frame already available when TorchLens captures an op
- the branch model should come from cached AST/source analysis

That is lower risk than an `AST + sys.settrace` hybrid and far lower maintenance than a bytecode-primary design. [Inference]

### Why this is the best fit

1. TorchLens already has op capture points.
   At each wrapped torch op, TorchLens can inspect the active Python frame (`frame.f_code`, `frame.f_lineno`) without paying global tracing overhead. [Inference]

2. Branch identity is a source-structure question.
   The user wants "which `if`/`elif`/`else` arm enclosed this op?", which maps naturally to AST nodes and source spans, not bytecode offsets. TorchScript and coverage.py both reinforce that source structure is the right static model. [Documented + Inference]

3. Bytecode is the wrong abstraction for the product requirement.
   Dynamo proves bytecode works, but only with substantial interpreter emulation, guard machinery, and Python-version opcode maintenance. That is overkill for TorchLens attribution. [Inference]

4. `sys.settrace` is too expensive/conflict-prone for default use.
   Coverage.py uses it because it must observe all Python execution. TorchLens does not have that requirement; it only needs branch context when a captured op occurs. [Documented + Inference]

### Concrete stdlib-only APIs TorchLens should use

Primary path:

- `inspect.getsourcefile(obj)` and `inspect.getsourcelines(obj)` for source recovery.
- `ast.parse(source)` for parsing.
- AST node location fields: `lineno`, `end_lineno`, `col_offset`, `end_col_offset`.
- Runtime frame fields: `frame.f_code`, `frame.f_lineno`.
- `linecache` as a resilience layer when source lookup is slightly messy. [Inference]

Useful supporting APIs:

- `code.co_firstlineno` for anchoring code objects to source.
- `code.co_lines()` for validation/debugging on Python 3.10+ (PEP 626). [Documented: https://peps.python.org/pep-0626/]
- `dis.get_instructions()` only for fallback/debugging, not as the primary path. [Documented: https://docs.python.org/3.11/library/dis.html]

### Suggested attribution semantics

Recommended labels:

- `if.test` for ops executed while evaluating the predicate itself.
- `if.body`
- `if.orelse`
- `elif[i].test`
- `elif[i].body`
- `else.body`

Reason:

- An op inside `if torch.sum(x) > 0:` occurs before branch choice is known; it should not be mislabeled as belonging to the taken arm. [Inference]

### Fallback modes when source is unavailable

Recommended degradation order:

1. **Best effort source recovery**
   Use `inspect.getsourcefile`, `inspect.getsourcelines`, `linecache`.

2. **No source, but code object available**
   Record coarse location only: filename, qualname, line number, and mark branch attribution as unavailable.

3. **Optional debug fallback**
   Use `dis` plus `co_lines()` to identify nearby conditional bytecode/jump structure for diagnostics, but do not synthesize a confident branch label unless the mapping is unambiguous.

4. **Do not silently guess**
   Prefer `branch_id=None` / `branch_unknown` over a wrong branch label. [Inference]

### When to consider optional `sys.settrace`

Only as an opt-in debug or research mode, for cases like:

- validating the AST/frame implementation against line-to-line transitions
- investigating tricky multi-line predicates, comprehensions, or exception-heavy code
- collecting more exact path arcs than TorchLens needs in normal operation

This should not be the default runtime path. [Inference]

## What TorchLens Can Learn or Reuse

Directly reusable ideas:

- From TorchScript: source acquisition -> normalization -> AST parse -> branch-node lowering. [Documented: `torch/_sources.py:12-35`, `torch/_sources.py:120-138`, `torch/jit/frontend.py:804-811`]
- From coverage.py: keep static branch structure separate from runtime execution evidence. [Documented: `coverage/parser.py:1001-1011`, `coverage/pytracer.py:313-325`, https://coverage.readthedocs.io/en/latest/branch.html]
- From Dynamo: distinguish specialization-friendly branches from truly dynamic branches; keep fallbacks explicit rather than magical. [Documented: `torch/_dynamo/symbolic_convert.py:575-631`, https://pytorch.org/assets/pytorch2-2.pdf]

Things to avoid:

- Re-implementing a bytecode interpreter like Dynamo. [Inference]
- Global `sys.settrace` for the common path. [Inference]
- Depending on private TorchScript helpers as public API. [Inference]
- Hook-only thinking from Captum/DeepSpeed; hooks tell you which module ran, not which Python branch enclosed the op. [Inference]

## Bottom Line

The prior art converges on a simple conclusion:

- branch structure is best modeled from source/AST
- runtime branch choice should come from execution state
- for TorchLens, execution state can be sampled at existing op-capture sites

So the lowest-risk design is:

**AST-first branch modeling + per-op frame inspection, with explicit graceful degradation when source is unavailable.**
