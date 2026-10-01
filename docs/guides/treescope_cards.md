# Notebook cards and the treescope bridge

*(F16, treescope memo B2-B4/B7. Every spelling on this page is
DOCUMENTED-UNSTABLE pending the naming session.)*

TorchLens objects carry dependency-free rich HTML cards as their primary
notebook experience; the treescope bridge is an **opt-in upgrade** that adds
interactive array visualization. This ordering is not a preference but a
mechanical fact: a live IPython kernel calls `_repr_html_` before treescope's
machinery ever runs, so the card IS the top-level experience whether or not
treescope is installed.

## The four cards

Print any of these in a notebook cell and the card renders — stdlib-only
generation, no IPython or notebook extra required, readable with JavaScript
disabled:

| Object | Card |
|---|---|
| `Trace` | model/backend/counts, save coverage, outcome badge, NaN/Inf disclosure, params/FLOPs, budgeted foldable index of copyable lookup keys |
| `Layer` | header + one row PER PASS (per-pass facts read via `ops[k]`; a single-pass layer degenerates to the Op card) |
| `Op` | identity (function, pass k/N, module, site key), stats line, budgeted array grid with truncation-mask motifs, grad stats when captured, graph neighbors as copyable keys, cost wrappers untouched |
| `PartialTrace` | failure-first: FAILED CAPTURE banner, phase/reason, escaped exception, committed prefix and frontier, what remains safe to inspect |

Every card carries the **PROOF row** — capture outcome, verification badge,
nonfinite coverage basis, save policy. A TorchLens card asserts "this is a
verified execution", not "this is an object".

Hard rules: a card never raises (internal failures degrade to a one-line
`card unavailable: <reason>`); honesty states are never folded; display paths
never materialize disk payloads, never do unbounded device transfers, and
never recompute stats.

### Copy-expression roots

Card keys are executable access expressions (`log['conv2d_1_1']`). The root
name resolves through a ladder: an explicit `root=` wins; nested treescope
renders use treescope's own path; in a live notebook an identity scan of the
user namespace picks the shortest non-underscore binding (IPython history
names `_`, `_N`, `_i*`, `In`, `Out` are excluded — without the exclusions the
tie-break picks `_`, rebound by the next cell); with no root, keys copy as
literals and the card says why. Offline artifacts never scan a namespace.

## The treescope bridge

```python
import torchlens as tl
from torchlens.bridge import treescope as tl_treescope

tl_treescope.register()        # idempotent; typed conflict on occupied slots
tl_treescope.status()          # version, handlers, tensor capability, suppression state
tl_treescope.display(op)       # the explicit door: interactive arrayviz when capable
with tl_treescope.disabled():  # scope: treescope's own default rendering
    ...
tl_treescope.unregister()      # removes exactly our handlers, by identity
```

What the bridge buys: without it, an explicit `treescope.display(trace)` or a
Trace nested in a rendered container reflects the dataclass field-by-field
into a **megabyte internals dump** (measured 1.3–2.75 MB, up to 82.9 s on
gpt2/128 tokens). With it: bounded foldable summaries with lookup-key indexes
(~10 KB), and — when the installed treescope can render tensors — the REAL
saved payload through treescope's interactive array visualizer with TorchLens
truncation budgets.

Rules the bridge keeps:

- **Never `register_as_default()`**, never a `torch.Tensor` handler — the
  user keeps control of global display state, and treescope owns the tensor
  slot.
- **Container rule:** bridged containers emit scalars, strings, and a
  budgeted key index with `shown K of N`; they never recurse into child
  TorchLens objects (the recursive version measured WORSE than no bridge).
- **Capability probe, never a version pin:** the only treescope release on
  PyPI (0.1.10) reads the removed `Tensor.names` attribute and raises on
  every torch>=2.13 tensor. The bridge probes behaviorally at first use;
  a broken adapter means strict degradation — the object card stays, the
  reason is named, and a `treescope_bridge_degraded` warning fires once.
- **Top-level display prefers our card** (`_repr_html_` short-circuits
  treescope). `tl_treescope.display(obj)` is the explicit door to the
  interactive render; `show(method="treescope")` is the candidate method
  spelling for the UI sprint.
- **The duplicate box:** treescope appends any object's `_repr_html_` as a
  collapsed "Rich HTML representation" box after our handler renders. The
  cards suppress the duplicate payload with a narrow, fail-safe check (a
  treescope postprocessor frame on the stack AND the bridge having just
  rendered that exact object); if the check ever misfires, the worst case is
  a redundant collapsed box, never a missing card.

## Version compatibility

| treescope | torch | behavior |
|---|---|---|
| absent | any | cards only; `register()`/`display()` refuse typed (`treescope_bridge_unavailable`) |
| 0.1.10 (released) | < 2.13 | full bridge incl. tensor leaves (upstream adapter works) |
| 0.1.10 (released) | >= 2.13 | bridge with STRICT DEGRADATION on tensor leaves (`Tensor.names` crash; probe-detected), object summaries still bounded |
| git main / first release with the `names` fix | any supported | full bridge incl. tensor leaves |

## Remote capture recipe (single-file report)

```bash
# on the cluster
python -c "
import torch, torchlens as tl
from mymodel import build; model, x = build()
log = tl.trace(model, x)
tl.export.html(log, 'run_report.html')   # one file, zero network
"
scp cluster:run_report.html .            # open from file://, no Python needed
```

See `docs/guides/offline_report.md` for the report's anatomy and options.

## Upstream filings (recorded, not yet filed)

Three goodwill filings for the treescope repository, to be filed by a human
or with explicit approval (this repo never auto-posts):

1. **Release nudge:** the `Tensor.names` getattr guard is committed upstream
   but unreleased; every torch>=2.13 user of released 0.1.10 crashes on any
   tensor render.
2. **`for_tokenizer` HuggingFace fix:** upstream's helper calls HF tokenizers
   with integer ids and raises; a duck-typed `int -> str` lookup works.
3. **Per-type repr-HTML suppression hook (PRIORITY):** a public hook to skip
   the appended "Rich HTML representation" box per type would retire our
   stack-based duplicate suppression and its residual collapsed label.

## Credits

The card and report conventions adopt ideas from **treescope / penzai
(Daniel D. Johnson, Google DeepMind; Apache-2.0)**: external type
registries, collapsed/expanded balanced layouts, copy-buttons that emit
code, faceted array visualization conventions, truncation budgets with edge
items, diverging-around-zero defaults, and pattern-coded exceptional values.
Named code borrow: `infer_balanced_truncation` (including `doubling_bonus`)
and the two-stage slice-then-convert truncation protocol, ported with license
notice in `torchlens/notebook/_truncation.py`.
