# The mech-interp kit (`torchlens.mechinterp`)

TransformerLens's daily-driver analysis tools rebuilt on torchlens: residual
decomposition, direct logit contributions (DLA), per-head results, patching
grids, head-taxonomy scores, prompt utilities, an alias layer, and retention
plans -- derived from the captured graph (no architecture registry), with
verification IN the API, not only in CI. Every spelling is
DOCUMENTED-UNSTABLE pending the naming sprint; the facade name is a
placeholder.

Axis disambiguation: `torchlens.attribution` attributes model OUTPUTS to
INPUTS; this kit attributes logits to internal WRITERS. No kit name contains
the word "attribution".

## The engine and its gate

`residual_decomposition(trace)` walks BACKWARD from the captured tensor the
final norm consumes: each qualifying addition is a spine step, the other
operand is the writer; hops across value-preserving ops are grammar-proposed
and payload-verified; every step carries the op's pass-qualified label and
live site key. The built-in oracle is an EQUALITY assertion, not a
tolerance: accumulating the literal captured writers in execution order
replays the forward's own fp32 add sequence, so all partial sums equal the
captured spine states under `torch.equal` (fp32/CPU; the per-dtype degrade
clause is disclosed in the identity receipt off that envelope). A failing
strict walk refuses OPEN; `strict=False` appends an `unresolved_remainder`
row, stamps the stack diagnostic-only, and is refused by DLA, gallery, and
launch claims.

Stack grading (D7): `complete` (every row a named writer) /
`closed_constant` (non-writer rows are VERIFIED constants) /
`closed_unresolved` (diagnostic). OPEN never constructs.

## DLA

`direct_logit_contributions(trace, answer=..., vs=...)`: directions-first
(peak memory scales with requested tokens, never vocabulary), frozen
validated norm scale (bit-exact against TLens's own `ln_final.hook_scale` on
real gpt2), ONE minimal constant row (final-norm beta + unembedding bias --
writer-side biases are already inside the literal writers; double-counting
costs a measured 6.35 logits), and the identity `sum(rows) + constant ==
the captured native logit` runs on EVERY call. Convention, in one sentence:
DLA linearly decomposes the ACTUAL logits at the actual operating point; it
is not a set of counterfactuals.

## Coverage and honesty surfaces

- `attention_head_contributions`: per-QUERY-head rows from the validated
  `result` facet + a remainder labelled `output_bias` only when PROVEN.
- `head_weight_views`: `[H, E, D]`/`[H, D, E]` views with orientation decided
  by module class (Conv1D `[in, out]` vs Linear `[out, in]`), fused-QKV
  slice evidence from the captured graph, GQA counts, shared-storage
  disclosure; payload-verified before serving.
- `resolve_alias` / `alias_report`: TLens spellings (both 2.x and 3.x-bridge
  generations, `translation.json`) resolve to native coordinates with an
  honest status: `real` / `reconstructed_read_only` / `needs_capture` /
  `structurally_absent` / `ambiguous`, each with a remedy.
- `retention_plan`: printable, priced, `save=`-composable plans; every kit
  function refuses with the COMPLETE missing-site set when payloads are
  absent -- never a partial table, never a silent widening.
- `grid` + presets: receipted patching cells over the live-hook rerun route
  (a hook that did not fire is an exception, never a number), pre-run cost
  disclosure, hard `budget=`, `engine="replay"` reserved typed.
- `lowered_counterfactual`: by-head and by-pattern patching on DEFAULT-loaded
  fused-SDPA models with no eager reload -- the delta lowers EXACTLY to an
  add at the real post-projection op, with a receipt naming the virtual
  site, the real site, the formula version, and the invalidated facets;
  reading an invalidated facet on the patched rerun refuses typed.
- `head_scores` / `inspect_prompt` / `test_prompt`: taxonomy masks with
  TLens-compatible conventions (credited), structured prompt records with
  explicit BOS disclosure and teacher-forced multi-token scoring.

Fused attention (torch SDPA) needs
`capture=CaptureOptions(save_arg_values=True)` for reconstructed
pattern/z/result reads; the facets stay READ-only -- edits are the lowered
counterfactual's job, or an eager recapture.

## Refusal vocabulary

Every kit refusal carries a stable `fields["code"]` (`mi_*` rows in
`docs/reference/error_refusal_contract.md`) and a remedy; callers branch on
codes, never message text. The any-architecture claim stays bounded: any
executed architecture whose target has a certifiable additive residual
topology works; everything else refuses typed with the frontier named.
