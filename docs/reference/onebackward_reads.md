# One-backward attribution reads

One backward pass, a table of "how much does each site matter" -- batched
over targets, with arbitrary targets, an inspectable linearization policy, a
common result carrier, and a `by=` door so "select the k most important
sites" is one line. Every SPELLING on this page is DOCUMENTED-UNSTABLE
pending the naming sprint; the semantics and the stable error codes
(`docs/reference/error_refusal_contract.md`) are the contract. Design of
record: the M(reads) tri-lab memo (F04).

```python
import torchlens as tl
from torchlens import attribution as attr

log = tl.trace(model, x)                       # any save mode; zero preparation
table = attr.read(
    log,
    target=attr.seed("output_1", index=(0, -1, 464)),   # edge-seeded: no payload needed
    method="activation_x_grad",                # {activation_x_grad, grad, activation}
    reduce="sum",                              # named scalar reductions; None = element grain
    # frozen= omitted -> the default MLP-output linearization (see below)
)
circuit = tl.top_k(k=50, by=table)             # the one-liner: rank sites by the table
```

## The substrate is already there

Every op on a finished live torch trace records its real autograd producer
(`op.grad_fn_handle`), a durable object id, and its output slot
(`op.multi_output_index`). The read resolves sites to
`GradientEdge(node, slot)` addresses from those EXISTING fields -- no arming
flag, no capture option, any save mode. Saved payloads are clone-insulated
from the graph on every default capture, so the registry route is the only
correct one: a naive `autograd.grad` over payloads returns `None` at every
site.

The id rejoin is a fallback: a user `log_backward` before the read nulls the
per-op handles, and the read transparently rejoins through the trace's
pinned refs, bitwise-equal.

Liveness is a closed typed gate (`read_addressing_unavailable` with
`fields["reason"]`): loaded, cleaned, inference-only, chunked, structure-only,
detached, and graph-freed traces refuse with the exact recapture or ordering
remedy. The v1 read is live-PyTorch-only.

## Suppression: the read never contaminates its own capture

Every read backward runs inside a suppression context generalizing the
shipped receptive-field probe context: TorchLens backward capture and
gradient hooks are gated off, capture counters are snapshotted and asserted
unchanged, pinned-ref growth is a tripwire, and state restores in `finally`.
Without it the read's own first backward would destroy its own addressing
and leak autograd nodes per target. User-installed PyTorch hooks still run
inside the context (documented boundary).

## Targets: two families, exactly-stated requirements

- **Edge-seeded (primary)**: `attr.seed(site, index=...)` (one-hot from
  recorded shape metadata) or `attr.seed(site, cotangent=...)` (explicit
  projection). Needs the site's registry node only -- no payload, no
  retention: a `layers_to_save=[]` capture serves full gradient tables.
  Wrong-shape cotangents refuse naming the site's RECORDED shape.
- **Tensor-expression (power-user)**: a graph-connected finite real scalar
  Tensor, or a pure `Trace -> Tensor` callable (a 1-D result is a target
  batch, never silently summed). Requires graph-connected payloads
  (`backward_ready=True` captures); detached targets refuse teachably,
  naming the edge-seeded spelling.

## `frozen=`: the inspectable linearization policy

Three states:

- **Omitted / `attr.DEFAULT`**: freeze every trustworthy executed
  transformer MLP-output facet home -- the published attribution-graph
  direct-edge semantics. Resolved by SEMANTIC FACET EVIDENCE only (the
  recipe-matched "output" facet homes), never name substrings. On
  attention-bearing graphs with unresolvable MLP homes: typed refusal
  `default_frozen_sites_underivable`. On MLP-free architectures (ResNet):
  the disclosed empty set.
- **`frozen=None`**: the explicit total derivative -- the Captum-familiar
  number.
- **Explicit ACT selection / site specs**: freeze exactly that, REPLACING
  the default.

The default deliberately changes transformer saliency numbers relative to
Captum-family tools; the disclosure warning
(`read_frozen_default_linearization`) fires once per process when `frozen=`
is omitted on an attention-bearing model under a gradient-bearing method.

Mechanism: temporary slot-targeted `Node.register_prehook` hooks
(batch-compatible arithmetic; removal restores gradients bit-exact).
Documented semantics: "prehook at this node, post-freeze for downstream
frozen sites" -- a frozen site's own row is NOT its gradient from an
otherwise-unfrozen graph. `frozen_applied` is cone-dependent: `<=` requested
and non-empty, never an exact count. Freezing one `(node, slot)` alias
freezes the whole group and says so.

## The population law (explicit contract, implicit filter)

An explicit `within=` naming an unretained site refuses at preflight with
the first missing addresses and a concrete recapture recipe
(`read_payload_unretained`); an explicit site that is not upstream of the
target keeps an honest `unreachable` row. An implicit population
(`within=None`) serves the eligible sites and records excluded counts by
reason -- under an intermediate target the population defaults to the
target's ancestor cone with `not_upstream_of_target` counts. A numeric zero
stays `ok`; autograd's `None` never becomes a zero.

## `ReadTable`: honesty as columns

Immutable, trace-bound Mapping keyed `(target_id, kind, address)`. Rows
carry the closed status vocabulary (`ok` / `unreachable` /
`not_differentiable` / `unsupported_grain` / `unavailable`), retention and
capture status, resolution grain, frozen requested/resolved/reached, the
policy digest, a nullable `sample_id` (dataset-mean seam), and the portable
structural site key as a SEPARATE field. Table provenance carries the
population counts, excluded-status counts, alias disclosure, the batching
plan with its reason, exact `autograd_calls`, timing, and bytes.

Operations: `to_pandas(values='omit'|'object'|'explode')`, `for_target`,
`column`, `aggregate_targets(reducer)` (no implicit mean/max/sum anywhere),
`collapse_alias_groups()` (explicit opt-in; per-address rows are the
default). Scalar tables persist as standalone `read_table_v1` JSON
(fail-closed load; loaded tables are `rescorable=False`); tensor-bearing
tables refuse `read_table_not_portable` until sidecars exist.

## The `by=` door

`tl.top_k`, `tl.top_fraction`, and `tl.threshold` accept a single-target
`ReadTable`: site-grain tables rank SITES (`k` counts sites; a win selects
the complete site mask), element-grain tables rank elements against exactly-
matching index spaces. Multi-target tables refuse (narrow with `for_target`
or fold with `aggregate_targets`); foreign and stale tables refuse; NaN is
unrankable; ties break canonical site order then flat index. Selection
provenance carries the schema, method, reduction, target, frozen digest, and
excluded counts.

## Batching: plan-vs-refusal, honest claims

`target_batch_size='auto'` plans per device and stamps the plan with its
reason on the table. CPU batching is a measured single-digit win (1.7-4.3x
across labs and substrates); the CUDA envelope is UNMEASURED and the CUDA
auto default stays sequential until the C-READ cluster row lands. Chunks
group by target CONE (same seed site / same expression tensor), because a
mixed-cone batched call fabricates zero rows for not-upstream pairs -- the
exact silent-wrongness class this design forbids. A batched chunk hitting an
operator without a batching rule refuses typed
(`batched_attribution_unsupported`), teaching `target_batch_size=1` --
never a silent fallback. `autograd_calls == ceil(T/B)` per cone group,
exactly.

Honest performance framing (per the panel's measurements): one capture plus
one backward for every retained site replaces N model re-runs and per-site
calls (352x vs re-running `layer_attribution` at 36 sites x 256 targets as a
forward-cost floor; 438x vs per-site `autograd.grad` calls, extrapolated
leg disclosed). Do NOT claim orders of magnitude from target batching; the
orders of magnitude live on the per-site and per-capture axes.
Gradient-only reads retain NOTHING; `activation_x_grad` retains only the
sites you multiply. Capture itself costs ~70x a bare forward with
`layers_to_save='all'` -- selective retention is the natural spelling.

## Deferred, with the plumbing bought now

- **EDGE-grain reads (EAP)**: `ReadTable` admits the EDGE kind;
  `mint_grad_input_use_map` builds the fail-closed backward-slot ->
  forward-consumption contract (`exact`/`ambiguous`/`unsupported`; the
  unmated census kills positional zipping). EAP remains honestly blocked
  until the correspondence map is trustworthy; the map is session-time only
  (its persistence is the EDGE-MAP-SCHEMA coordination with the next
  trace-schema bump).
- **Dataset-mean attribution**: nullable `sample_id`, `sample_count`
  provenance, and the streaming-reduction seam ship now.
- **Frozen-LN / full published linearization (R10)**: the frozen-policy
  registry ships; the v1 vocabulary is `{'mlp_out'}` and there is NO
  `'attribution_graph'` preset name until R10 lands.
- **R4 position targets**: the closed `METHODS` tuple is the vocabulary
  seam; no semantics claimed in v1.
- Recorded read passes will stamp `(origin, target_ids, chunk_id)` through
  `stamp_backward_pass_origin` (v1 reads run suppressed and are never
  recorded).

## Credit

The MLP stop-gradient linearization discipline, the batching-is-mandatory
lesson, and the nightly parity reference follow **circuit-tracer**
(Anthropic). The `frozen=None` parity target and vocabulary follow
**Captum**'s `LayerGradientXActivation`. Many-sites-one-run ergonomics
follow **TransformerLens** and **nnsight**. The edge-grain requirement this
build plumbs for follows **EAP / EAP-IG / ACDC**. Dataset/channel Taylor
importance follows **VISCNN**.
