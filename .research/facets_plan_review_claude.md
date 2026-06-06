# Facets P1 plan — adversarial design review (Claude)

Reviewer stance: skeptical PyTorch-internals architect. Every claim below is checked against
current code with file:line evidence. Focus is P1 (the foundation everything rests on).

**Headline:** Two of the eight P1 work-items rest on premises that are FACTUALLY WRONG about the
current code (the multi-output "bug" is already fixed; `container_spec` already preserves names).
Two flagship features (per-facet gradients, attention reconstruction) silently depend on capture
flags that are **OFF by default**, so as specified they degrade to nothing for a normal
`tl.trace(model, x)`. The "facet = (home op, structural transform)" model has real anchoring holes
for every fused/RoPE/GQA attention recipe that ships TODAY.

---

## BLOCKING (must resolve before building P1)

### B1. `op.grad` does not exist by default — the flagship grad feature is a no-op on a plain trace
**Evidence:**
- `torchlens/options.py:626` — `save_gradients: bool = False` (default).
- `torchlens/backends/torch/model_prep.py:1174` — the backward hook (`_add_tensor_backward_hook`)
  is registered **only** `if trace.save_gradients:`. No hook -> `Op.log_tensor_grad` never fires
  (`op.py:2097`) -> `op.grad` stays `None`, `has_grad` stays `False`.
- `op.py:687-696` — `__getattribute__` returns `state.get("grad")` (None) when no `grad_ref`.

**Why it matters:** The proposal sells `facet.grad = transform(op.grad)` as "the flagship
differentiator... beats both rivals" (proposal §3 Pillar C). But on the canonical call
`tl.trace(model, x)`, `op.grad is None` for every op, so EVERY `facet.grad` is `MissingFacet`/None.
nnsight's `tensor.grad` works after a normal `loss.backward()` with no special flag — so as
specified TorchLens is strictly *worse* here, not better, for the default user. The plan's gate
"facet.grad validated where checkable" (sprint §P1 GATES) is trivially vacuous if nothing is
checkable by default.

**Suggested resolution:** Decide and DOCUMENT the contract explicitly: either (a) `facet.grad`
requires `save_gradients=True` (or `backward_ready=True`) and raises a `MissingFacet` with an
actionable "re-run with save_gradients=True" message (mirroring the fused-attention message in
`_helpers.py:117`), OR (b) make grad facets work off a user-driven `loss.backward()` by having the
facet pull from the live autograd graph. The spec currently implies grads are "already-saved"
(Fork D(a): "lazy slicing of the already-saved op gradient") — but they are NOT saved unless opted
in. Pin this down before building, and make the P1 grad gate run with `save_gradients=True` so it
is non-vacuous.

### B2. Attention reconstruction needs inputs that are NOT saved by default
**Evidence:**
- `torchlens/options.py:625` — `save_arg_values: bool = False` (default).
- The recipes reconstruct from CHILD-OP OUTPUTS, not saved args: `_helpers.child_out` reads
  `child.calls[0].out` (`_helpers.py:34-43`); `gpt2_attention` splits `c_attn.out`
  (`attention.py:140-144`). Those are saved activations (always on), good. BUT the proposal §3c
  describes reconstruction from "the fused op's recorded INPUTS (Q,K,V,mask,scale as upstream saved
  activations)" and replay-validation `pattern@V (+proj) == captured output`. The mask and scale
  are non-tensor / kwarg arguments to `F.scaled_dot_product_attention`; recovering them requires
  `save_arg_values=True` (off by default) — `op.saved_args`/`saved_kwargs` are populated only when
  `has_saved_args` (gated by that flag; see `op.py:830-832`, `options.py:560-561`).

**Why it matters:** The "validated reconstruction" READ tier (proposal §3c, the headline
"look at patterns -> free") cannot recompute `scores = Q@Kᵀ*scale + mask` without scale and the
causal/padding mask, and cannot replay-validate `pattern@V == output` faithfully. Today the fused
recipes don't even attempt it — they return a `MissingFacet` telling the user to switch to eager
(`_helpers.py:117-125`). So the "free validated reconstruction" is currently VAPOR; P1 has to build
it from scratch AND it depends on a non-default flag.

**Suggested resolution:** Either scope the P1 reconstruction to the inputs that ARE always saved
(Q/K/V child outputs) and EXPLICITLY defer mask/scale-dependent `scores`/`pattern` to "requires
`save_arg_values=True`" with a `MissingFacet`, or make attention recipes opt into arg-saving for
their home op. Do not let the plan imply pattern reconstruction is free on a default trace.

### B3. The "single home op + structural transform" model is FALSE for the attention recipes that ship today
**Evidence (the q/k/v facets are NOT a pure slice of one op's `out`):**
- GPT-2 (`attention.py:140-144`): `q,k,v = c_attn_out.split(dim//3); reshape_heads(...)`. This is
  split (selection, OK) **then** `.view()` (`_helpers.py:98`). A `.view` to `(B,S,n,d)` is
  structural/invertible — fine. So GPT-2 q IS expressible as `home=c_attn`, `transform=[:, :, slice]
  .reshape`. Acceptable.
- Llama/Mistral (`attention.py:166-175`): q = `reshape_heads(child_out(q_proj))`. Anchored to
  `q_proj.out` + view. BUT these are **pre-RoPE**. RoPE is applied *after* `q_proj`/`k_proj` inside
  the attention forward. So the q/k facets do NOT equal the tensors that actually enter `QKᵀ`.
- BERT/ViT/T5/DistilBERT: q = `reshape_heads(child_out(query/q/q_lin))` — anchored + view. OK
  structurally, but see B2/B4 for the pattern path.

**Why it matters two ways:**
1. **RoPE breaks reconstruction (proposal §3c "no pre/post-RoPE ambiguity" is the OPPOSITE of
   reality for these recipes).** The proposal claims reconstruction uses "the fused op's OWN recorded
   inputs (exact tensors the kernel used — no pre/post-RoPE ambiguity)". But the recipes source q/k
   from `q_proj`/`k_proj` CHILD outputs = pre-RoPE. If reconstruction uses the recipe's q/k facets,
   `softmax(q@kᵀ)` will NOT match the captured attention output for any RoPE model (Llama, Mistral,
   most modern LLMs) and the replay-validation will (correctly) FAIL — turning a CORRECT capture into
   a `MissingFacet`. If instead reconstruction uses the fused op's saved args (post-RoPE), then the
   `pattern`/`scores` facets are anchored to a DIFFERENT tensor than the `q`/`k` facets the user sees
   — an internal inconsistency the plan never reconciles.
2. **Write-back through `.view`/`.split` for GQA shares one home op for k and v differently than q.**
   See B4.

**Suggested resolution:** State explicitly in the spec that the home op for `scores`/`pattern` is the
fused SDPA op (post-RoPE args), DISTINCT from the `q_proj`/`k_proj`-anchored `q`/`k` facets, and that
the two are NOT interchangeable. For RoPE models, either (a) anchor q/k facets to a post-RoPE op if
one is captured, or (b) document that `q`/`k` facets are pre-RoPE projections (still useful) and the
reconstructed `pattern` uses the fused op's post-RoPE inputs. Pick one; do not claim "no ambiguity."

### B4. GQA/MQA write-back and per-head grad index alignment is underspecified and the existing head-slice is read-only-shaped
**Evidence:**
- `facets.py:118-131` (`AttentionHeadView._slice`): for k/v it maps a query-head index to a kv-head
  via `head_index // group_size` with `group_size = n_q_heads // n_kv_heads`. This is a READ slice.
  There is no inverse: writing head 3 of q is a clean `[:, :, 3, :]` write, but "writing head 3 of k"
  maps MANY q-heads onto ONE kv-head — write-back is many-to-one and ambiguous. The proposal's
  "intervene = write_back via inverse" (§2) has no defined semantics for the GQA k/v case.
- q/k/v sharing one home op (GPT-2 `c_attn`): editing q-head 3 and k-head 3 are TWO writes into the
  SAME `c_attn.out` tensor at different column ranges. Proposal §2 lists "q/k/v sharing one home op
  (cumulative write-back)" as a known case but the plan gives no mechanism for composing two
  independent slice-writes into one parent reconstruction within a single rerun.

**Why it matters:** P2 intervention is built on P1's `FacetSpec` inverse. If the inverse is undefined
for GQA k/v and for cumulative same-op writes, P2 is blocked, and worse, a naive implementation will
silently produce WRONG edited tensors (exactly the silent-wrong-data the validation tripwire exists
to catch — project CLAUDE.md "Validation Integrity").

**Suggested resolution:** In P1, define `FacetSpec` write-back as operating on the HOME OP'S raw
output columns, not the reshaped head view, so GQA is a plain column-slice on `k_proj.out`
(`[:, :, kv_head_cols]`) with NO query-head expansion — editing "kv head j" is unambiguous; editing
"query head i's k" is explicitly disallowed (raise). For shared-home-op (c_attn), require all facet
edits on one op to be collected and applied as a single composite slice-assignment before the rerun.
Write the regression test for both in P1, even though intervention lands in P2.

### B5. P1 task #5 ("fix the LSTM-mislabel-as-recurrent bug") describes a bug that is ALREADY FIXED
**Evidence:**
- `tests/test_multi_output_modules.py:215-225` — `test_lstm_three_outputs_distinct_layers` (a
  `@pytest.mark.smoke` test) ASSERTS `lstm.num_calls == 1`, 3 distinct layer labels, and
  `multi_output_name == ["output", "h_n", "c_n"]`. `test_gru_two_outputs_distinct_equivalence_classes`
  (:249-256) asserts distinct equivalence classes per output. `test_bilstm_outputs_preserve_single_
  call_structure` (:259-267) asserts multi-output not multi-pass.
- I ran the file: **10 passed** (exit 0). So `num_calls` is correct and outputs are not mislabeled as
  recurrent passes TODAY.

**Why it matters:** The plan budgets a "careful, regression-tested, loop-detection-adjacent" fix
(sprint §P1.5, §Risks) for something already shipped. Building against a non-existent bug wastes the
phase's highest-risk budget and risks a "fix" that REGRESSES the working behavior these smoke tests
lock in.

**Suggested resolution:** Replace task #5 with "VERIFY current multi-output behavior is correct
(it is — see the passing smoke tests) and add any missing per-facet coverage." If there is a SPECIFIC
residual mislabel case (e.g. a model where multi-output still collides with genuine recurrence),
name it with a failing repro before allocating the loop-detection risk budget. Do not touch loop
detection without a red test first.

### B6. `container_spec` ALREADY preserves namedtuple/dataclass/HF field names — P1 task #4 premise is wrong
**Evidence:**
- `torchlens/backends/torch/ops.py:460-545` (`_build_container_spec`) already emits
  `kind="namedtuple"` with `fields=tuple(value._fields)` (:490-503), `kind="dataclass"` with
  `dataclasses.fields` names (:505-519), `kind="hf_model_output"` with `keys=tuple(value.keys())`
  (:475-489), and dict keys (:520-531).
- `torchlens/intervention/types.py:206-216` — `ContainerSpec` carries `keys`, `fields`,
  `type_module`, `type_qualname`. `tests/test_multi_output_modules.py:285-294` asserts dict
  `output_structure.keys == ("logits","hidden")`; the MHA test (:271-280) asserts
  `multi_output_name == ["attn_output", "attn_output_weights"]`.

**Why it matters:** Plan §P1.4 says "extend `container_spec` to keep dict keys / `NamedTuple._fields`
/ dataclass / structseq (today tracks tuple/dict shape, not the names)". That parenthetical is FALSE
for namedtuple/dataclass/dict/HF — names are already there. The only genuinely open case is
**structseq** (`torch.return_types.max` etc.): `_is_namedtuple_instance` checks
`isinstance(value, tuple) and hasattr(value, "_fields")` (`ops.py:397-411`). structseq returns
(`torch.return_types.*`) are tuple subclasses but expose field names via `_fields` in modern torch,
so they MAY already be covered — this needs an explicit test, not a from-scratch build.

**Suggested resolution:** Rescope task #4 to (a) ADD a structseq test (`torch.max(x, dim=0)` ->
`values`/`indices` field names surfaced as facets) and fix only if it fails; (b) wire the
ALREADY-CAPTURED names into the `.facets` door. Do not rebuild name capture that exists.

---

## MAJOR risks

### M1. paired grad_fn `grad_inputs`/`grad_outputs` index alignment onto a facet slice is genuinely undefined
**Evidence:** `GradFnCall.grad_inputs`/`grad_outputs` (`grad_fn_call.py:33-34`) are nested tuples
cloned verbatim from the autograd hook (`grad_fn.py:443-463`, `_clone_grad_value`). Their element
ORDER is autograd's internal next-functions order, NOT the op's logical output order, and there is no
recorded mapping from "facet slice of op.out" to "which element of grad_inputs/grad_outputs". The
proposal §3 Pillar C claims "Same slice spec applied to the right tuple element (trivial for single-
output ops; index bookkeeping for multi-in/out)" — the "index bookkeeping" is the hard part and is
entirely unspecified. For multi-output ops (LSTM!) the grad_outputs tuple alignment to
`output`/`h_n`/`c_n` is exactly the case that will be wrong if guessed.
**Resolution:** For P1, restrict paired-grad_fn sourcing to single-output ops where alignment is
unambiguous; for multi-in/out, fall back to `op.grad` (output-side) only, and `MissingFacet` the
input-side grad until a verified index map exists. Add a small oracle test against a hand-computed
2-input op.

### M2. In-place ops and views poison the "home op" anchor
**Evidence:** `Op` tracks `is_inplace` (`op.py:888`) and `out_versions_by_child`
(`op.py:852-853`, used by `input_activations` :1411) precisely because in-place mutation means an op's
saved `out` may not equal what a given child consumed. A facet anchored to such an op's `.out` reads
the FINAL post-mutation value; write-back through it can corrupt a tensor other ops alias.
**Resolution:** `FacetSpec` should refuse to anchor to `is_inplace` ops (or to ops with
`has_out_variations=True`) for the INTERVENE tier, degrading to read-only. Document it. Add a test
with an in-place ReLU attention variant.

### M3. Recipe facets are NOT structural transforms today — they RUN torch ops at access time
**Evidence:** `reshape_heads` calls `value.view(...)` (`_helpers.py:98`); `gpt2_attention` calls
`.split` (`attention.py:141`); `_with_attention_common` even calls `module.facets.head`
(`attention.py:49`) recursively. These execute under no `pause_logging` guard inside the recipe
(unlike `apply_transform` in `op.py:276-278`). The migration to `FacetSpec` (P1.2 "replace, no compat")
must convert every one of these imperative bodies into a recorded structural spec — that is the bulk
of P1's real work and the plan undersells it as "migrate built-in recipes."
**Resolution:** Budget P1.2 as the dominant task. Verify each recipe's transform is expressible in the
`__getitem__/.heads/.split/.reshape/.transpose/.select` vocab; anything that isn't (e.g. a recipe that
applies GELU, see `_helpers.activation_gelu` :101) is non-structural -> read-only by the proposal's own
rule, and the MLP-neuron facets that need an activation fn will LOSE grad/intervene. Confirm that is
intended.

### M4. `keys()`/declared-facets uses AST literal extraction — fragile foundation for the unified door
**Evidence:** `facets.py:368-390` (`_literal_return_keys`) parses the recipe SOURCE with `ast` to
guess facet names without running it. The new attention recipes pass explicit `facets=` tuples
(`attention.py:103`), but the proposal's "all outputs become auto-facets" (§2) means the `.facets`
door must also enumerate STRUCTURAL outputs (out0/out1/named) that come from `container_spec`, not from
a recipe's literal dict. Merging "recipe-declared names" with "container-derived names" in `keys()` is
unspecified.
**Resolution:** Specify how `FacetView.keys()` unions (a) container_spec-derived output facets and
(b) recipe-declared facets, including precedence when a recipe aliases `out0 -> output` (proposal §2).

### M5. Snapshot-at-capture vs lazy facet computation is contradictory as written
**Evidence:** `FacetView` is fully lazy and reads the GLOBAL `_REGISTRY` at construction
(`facets.py:147` `_matching_recipes(record)` -> `facets.py:393-396` iterates module-global `_REGISTRY`).
There is no per-trace snapshot today. Proposal Fork A requires "each Trace SNAPSHOTS the active recipe
set at capture time (immune to later global mutation; lazy facets stay stable)". But facets are
computed LAZILY on access, potentially long after capture, against the LIVE registry. Snapshot +
lazy-compute requires storing the snapshot ON the trace and routing `_matching_recipes` through it —
a real change to `FacetView.__init__`'s registry source, not a minor add.
**Resolution:** P1 must give `FacetView` a registry source parameter (the trace's snapshot), and the
trace must capture `list(_REGISTRY)` (plus contextvar overrides) at trace-finalization time. Specify
this; it is load-bearing for the reproducibility guarantee and currently absent.

---

## Recommendations / underspecified points

### R1. Replay-validation tolerance for reconstruction must be defined up front
The proposal wants `pattern@V (+proj) == captured output` to gate reconstruction, but lists no
tolerance. Real divergences for a CORRECT reconstruction: fp32 softmax upcast (SDPA computes softmax
in fp32 then downcasts), attention dropout (stochastic — `pattern@V` will NOT match a dropped-out
captured output), sliding-window/ALiBi/soft-cap biases, and scale conventions (`1/sqrt(d)` vs a model-
specific `scaling`). Per the tripwire principle (CLAUDE.md), you cannot just widen a tolerance to pass
— so you must DECIDE which of these are "correct-by-design outside the check" (dropout in train mode ->
reconstruction simply unavailable, `MissingFacet`) vs genuine failures. Enumerate them in the spec
with the exact rtol/atol and the eval-mode requirement (reconstruction only valid with
`model.eval()` / dropout off). Without this the gate is either vacuous or wrongly red.

### R2. `head()` returns an `AttentionHeadView` that is attribute-magic, not a `FacetSpec`
`facets.py:91-131` — the current head view does ad-hoc tensor slicing. The proposal's `.facets.head(i).q`
must become a composed `FacetSpec` (home=q_proj, transform=[..., i, :]) to get grad/intervene. The plan
should state the head accessor returns a spec, and that `AttentionHeadView` is replaced (it currently
returns raw sliced tensors with no provenance).

### R3. Entry-point plugin security is asserted but the loader is unbuilt
Proposal Fork A and §4 describe a `torchlens.recipes` entry-point group "installed = always loaded".
There is no entry-point loader in `facets.py` today (registry is populated only by `@register`
decorators at import). Recipes are arbitrary CODE executed at import of the plugin; "no auto-exec'd
`~/.torchlens/recipes/` dir" is good, but an installed entry-point plugin is the SAME trust level as
any pip dependency — fine, but say so, and confirm the loader fails CLOSED (a broken plugin must not
abort `import torchlens`). Specify load order vs specificity resolution (Fork A says specificity, not
registration order — but entry-point load order still affects the `_REGISTRY` list that `keys()`
iterates).

### R4. Specificity-ordering conflict resolution is specified but the code is last-wins
`facets.py:238-254` (`_compute`) merges recipes in `_REGISTRY` order and WARNS on collision
("recipe overrides facet") — i.e. last-registered wins, exactly the "current last-wins fragility"
Fork A says to defang. P1 must implement the qualname > class_name > predicate, user > built-in
ordering in `_matching_recipes`/`_compute`. This is net-new sorting logic, not present today.

### R5. `MissingFacet` raising semantics will surprise `.get()`/membership users
`facets.py:75-88` — `MissingFacet.__getattr__`/`__getitem__` RAISE. `FacetView.get` (:185-201) only
swallows `KeyError`, not the `RuntimeError` from a `MissingFacet`. So `view.get("pattern")` on a fused
model RAISES instead of returning the default. With "all outputs become facets" widening the surface,
nail down whether `MissingFacet` is returned (truthy sentinel) or raised, consistently across
`__getitem__`, `get`, `head().x`, and iteration.

### R6. Multi-pass / recurrent home op: which pass does a facet read?
A recipe sources `child.calls[0].out` (`_helpers.py:41`) — hardcoded FIRST call. For a genuinely
recurrent attention (same module called N times), `.facets.q` silently returns pass-0 only. The
proposal's "multi-pass/recurrent ops where the home op has multiple passes" (review prompt) is real:
`FacetSpec` needs to carry a pass index, and `module.facets` on a multi-call module must either pick a
pass or raise like `_single_pass_or_error` (`module.py:1542-1557`). Specify.

---

## Summary of top BLOCKING findings
1. **B1** — `save_gradients=False` by default + grad hook gated on it (`model_prep.py:1174`,
   `options.py:626`): `op.grad` is `None` on a plain trace, so `facet.grad` (the flagship) is a no-op
   unless the user opts in. Define the contract; make the grad gate non-vacuous.
2. **B2** — `save_arg_values=False` by default (`options.py:625`): the fused-attention scale/mask
   needed for `pattern`/`scores` reconstruction + replay-validation aren't saved by default. "Free
   validated reconstruction" is not free.
3. **B3** — The shipped Llama/Mistral recipes anchor q/k to PRE-RoPE `q_proj`/`k_proj` outputs
   (`attention.py:166-175`), so reconstruction's "no pre/post-RoPE ambiguity" claim (§3c) is inverted;
   reconstruction will fail-validate correct captures or anchor pattern to a different tensor than the
   user-visible q/k.
4. **B4** — GQA k/v write-back is many-to-one (`facets.py:118-131`) and q/k/v-share-one-op
   (`attention.py:140-144`) cumulative write-back have no defined inverse — undefined `FacetSpec`
   inverse = silent-wrong edits, which the validation tripwire forbids.
5. **B5/B6** — Two P1 tasks target non-problems: the LSTM-as-recurrent bug is ALREADY FIXED (10/10
   smoke tests pass, `test_multi_output_modules.py:215-225`), and `container_spec` ALREADY preserves
   namedtuple/dataclass/dict/HF names (`ops.py:460-545`). Rescope to structseq coverage + wiring,
   and do NOT touch loop detection without a red repro.
