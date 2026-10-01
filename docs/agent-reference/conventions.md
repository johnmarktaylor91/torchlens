## Conventions

- **This repo is PUBLIC; internal notes stay private.** Never commit `.research/` or
  `.project-context` (except `architecture.md` / `state_of_torchlens.md`) — they are
  gitignored and a `no-internal-notes` pre-commit hook hard-fails on them. Never `git add -f`
  to bypass. See AGENTS.md "Internal notes stay PRIVATE (LOCKED)".
- Conventional commits: prefer `docs(scope):`, `chore(scope):`, `test(scope):` for
  non-release changes; never use major-bump markers casually.
- TorchLens host-object metadata lives under `obj._tl`; sub-fields are snake_case and
  new metadata should extend a `TorchLensMeta` subclass rather than adding `tl_*` attrs.
- `_raw_` prefix for pre-postprocessing state; `_final_` for post-processed state
- FIELD_ORDER constants in `constants.py` define canonical field sets; update both class
  fields and constants when adding fields
- NumPy-format docstrings on all functions
- Type hints on all functions
- Import order: stdlib -> third-party -> local (enforced by ruff)
- Line length: 100
- `tl.receptive_field` is lazy; entity-level `receptive_field` / `projective_field` siblings
  pair with `Trace.receptive_fields()` / `Trace.projective_fields()` tables.
- BOUND-METHOD ROOTS (F41, spellings DOCUMENTED-UNSTABLE): `tl.trace`/`tl.validate` accept an
  `nn.Module` OR a bound method of one (`tl.trace(model.generate, ids, ...)`): owner via
  `method.__self__`, registered as the `owner` submodule of a TL-authored wrapper root, method
  called exactly once; identity reads `type(owner).__name__`;
  `root_entry_point="bound_method:..."` + `tl_authored_root=True` on the root op record persist.
  `EpisodeSpec.stepped_module` defaults to the owner on bound-method episode captures. The
  rerun/append identity gate is FAIL-CLOSED on the fact: absent fact refuses
  `root_entry_point_unavailable`, non-`module_call` roots refuse `rerun_entry_point_unsupported`
  (interim posture; replay engines unaffected). Other callables refuse `model_type_unsupported`
  teaching the ruled spelling.
- EPISODE CAPTURE (torch-only, spellings DOCUMENTED-UNSTABLE): `tl.trace(episode_root, x,
  episode=tl.options.EpisodeSpec(stepped_module=model, n_steps=N))` captures one wrapped
  multi-step generation run as ONE product with a per-step status ledger at
  `trace.annotations["episode"]` (disclosure, never a settlement authority; persists
  plainly as of the tlspec v8 coordinated bump, load-validated fail-closed). Cross-step
  joins are MEASURED (lane F40c): per-row `step_join` grades (continuous/forced/
  transformed/declared/exogenous/unchecked) in the `episode_step_join_v1` header envelope; a
  measured break refuses episode-dependent claims (series reads, whole-episode replay,
  the blessing fold, escalation) while ops/values/graph/per-segment reads stay usable;
  `EpisodeSpec(feed="closed")` is the strict halt-at-next-entry arm,
  `on_feed_break="refuse"` the built FORK-4 typed-refusal arm, `crossings=(k,...)` the
  declared tool-call crossings; envelope-less artifacts read unmeasured, never
  measured. ATTESTED COUPLING (lane F42): `episode=` x `intervene=` runs COUPLED —
  per-row measured `fire_count` (zero and multiple fires first-class), the
  deterministic header `intervention_digest`, `fidelity_basis="perturbed"` on
  replacing fires, the step-qualified selector `torchlens.intervention.at_step(...)`
  (live + post hoc via `Op.episode_step` stamps), `trace.episode_coupling`
  (recompute-and-compare capture-digest binding + per-segment facts that never span a
  measured break), and the replay law: `run()` on coupled products refuses
  `episode_coupled_replay_underivable` (both engines) and `do()` edits quarantine
  inherited episode evidence. DIAGNOSTIC-TIER: cost is superlinear
  in step count — tens of steps, never hundreds. Bundles carry the optional S6
  member-relation table (`member_relations=`, `Bundle.relate`,
  `Bundle.derive_episode_status`). Doc of record: `docs/reference/episode_capture.md`;
  refusal codes in `docs/reference/error_refusal_contract.md`.
- Experiments remember (F03; spellings documented-unstable; doc of record
  `docs/reference/experiment_ledger.md`): Bundles carry persisted lineage
  (random `bundle_id`, `member_construction` origin anchors, hash-chained
  `operations` chronology; fork carries relations + anchors both ways);
  `bundle.why(member)` / `bundle.provenance()` are the derived provenance
  join (exact-or-refused ancestry, four honesty axes, additive wording only
  on an empty reference suffix); `bundle.vary(mapping)` names one spec or
  None identity per member with complete-coverage preflight and no-rollback
  partial disclosure; `tl.sweep(include_baseline=True)` mints the pristine
  baseline; `torchlens.experiment.site_sweep` is the serial candidate
  engine (edit=/metric=/retain= REQUIRED, undeclared multi-site candidates
  refused, complete effect table persisted incl. released/refused/failed
  rows, `bundle.effects()`/`measure_members()` reads);
  `head_ablation_candidates` is the module-scoped v-facet sugar (GQA
  refuses; hand-written-hook oracle in its test file); the experiment
  ledger (`torchlens.experiment.ledger`) is the opt-in event-sourced
  hash-chained semantic record (per-event fsync, kill-9-safe, quarantine
  default, closed verdict vocabulary, `ledger_overview`/`ledger_entry`/
  `ledger_evidence` read-only serving, MCP `torchlens_ledger_*`).
- Checkpoint live-ref guard (spellings documented-unstable): cross-member parameter
  value/difference/trajectory reads on a Bundle (`SuperParam` views with >= 2 members:
  `weight_norm_diff`, `diff_pair`, `aggregate`, `out`/`grad`) refuse BEFORE tensor lookup
  with stable code `checkpoint_series_live_params` — parameter reads resolve through live
  model handles, never capture-time bytes, so a checkpoint series would otherwise report
  identical weights (or NaN/empty after load). Claim-keyed, never object identity; guard
  site is the one `_TensorBearing._tensor_dict` funnel. `Param.value_basis` is the derived
  read-time disclosure (`live_ref` / `absent(not_persisted)`; `snapshot` only with future
  R8(b) snapshots, the only basis that passes). `Param.value` keeps its live-handle
  contract; version-axis relation rows still order members.
