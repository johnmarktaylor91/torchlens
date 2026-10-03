# Structure-Only Capture: Capability Contract

THE capability table file for structure-only capture (`structure_only=True`),
owned by L7a; the in-code authority is
`torchlens.capture.structure_only.STRUCTURE_ONLY_CAPABILITIES` with the ONE
chokepoint `require_structure_only_capability`, and this file is its human
mirror (lockstep-tested in `tests/test_structure_only_capabilities.py`).

RULE: Rows may change STATUS by amendment from their amend_owner at their
named flip_event (with evidence). No amendment may widen a CLAIM beyond its
conditional wording; a wave that wants a bigger claim writes a NEW row
through S2. This file never promises for a wave it hasn't shipped.

NAMING: every spelling below is DOCUMENTED-UNSTABLE pending naming-session
ratification, and every refusal code is additionally S2-gated (no
deprecation shim owed on rename). The D8 decision point (meta-tensor
admission scoped to structure-only captures) is QUEUED; every cell below is
the D8-DEFAULT state — the entry-gate refusal stands until an explicit
decision.

Grammar of `status_v1` (closed, five members): `supported_structural` |
`supported_hypothesis` | `refuse:<code>` | `verify:<Vn>` |
`out_of_scope:<contract-ref>`.

| key | claim (conditional wording) | status_v1 | flip_event | evidence | amend_owner | refusal_code |
|---|---|---|---|---|---|---|
| `graph_structure` | The op graph, edges, order, and module nesting of THIS meta execution are recorded exactly. | `supported_structural` | never | tests/test_structure_only_entry.py (E-1 structural pins) | L7a | — |
| `param_geometry` | Parameter/buffer names, shapes, dtypes are recorded as declared; the persistence partition is recorded IF V8 verifies meta-compatibility. | `supported_structural` | V8 verdict | pending: V8 verification test | L7a | — |
| `shapes_dtypes` | Per-op shapes/dtypes are HYPOTHESES: valid under meta propagation, unproven until discharged by a real capture of the same graph. | `supported_hypothesis` | discharge | tests/test_structure_only_discharge.py | L7a | — |
| `flops_estimates` | FLOPs/MACs are derived from hypothesis shapes; they inherit hypothesis status and are labelled estimated, never measured. | `supported_hypothesis` | discharge | tests/test_structure_only_honesty.py | L7a | — |
| `memory_estimates` | Memory figures are geometry estimates; measured-memory columns render unknown, never zero. | `supported_hypothesis` | discharge | tests/test_structure_only_honesty.py | L7a | — |
| `taken_path_conditionals` | Conditional structure of the taken path is recorded; any VALUE-dependent branch through the enumerated escape surface refuses typed at the user's source line REGARDLESS of the tensor's device; unenumerated meta deaths refuse typed via the backstop; unenumerated REAL-value escapes in form (b) are undetectable and are priced by hypothesis status (coverage claim exactly per memo sec 2.1 C-ENUM/C-BACKSTOP/C-RESIDUAL). | `supported_structural` | never | tests/test_structure_only_teaching.py | L7a | — |
| `meta_admission` | Meta-materialized models (form (a)) are ADMITTED under the D8 grant (decided 2026-08-26), if and only if structure-only is in force (scoped admission, W2): the graph, module nesting, parameter geometry, and shape/dtype HYPOTHESES are recorded with no tensor values. Without structure_only the entry gate refuses meta unchanged. | `supported_structural` | D8 granted 2026-08-26 (this row IS the flip; last merge) | tests/test_weightsfree_admission.py; parity gate: tests/test_weightsfree_parity.py (real digest == meta digest AND discharge CORROBORATED on every fixture) | S2-amendment | — |
| `value_payloads` | Activations, argument values, output values are never recorded; requests refuse typed. | `refuse:structure_only_values_unsupported` | never | tests/test_structure_only_entry.py | L7a | `structure_only_values_unsupported` |
| `previews` | Value previews/thumbnails require values; refused. | `refuse:structure_only_values_unsupported` | never | tests/test_structure_only_entry.py | L7a | `structure_only_values_unsupported` |
| `nonfinite_predicates` | raise_on_nan and nonfinite halt predicates have no values to test; the combination refuses typed at entry. | `refuse:structure_only_option_conflict` | never | tests/test_structure_only_entry.py | L7a | `structure_only_option_conflict` |
| `runnable_ready_composition` | structure_only + runnable_ready is refused at entry: runnable eligibility and the structure substrate are incompatible in v1. | `refuse:structure_only_option_conflict` | L7b amendment lands | tests/test_structure_only_entry.py; conflict lift rides the S2 StateSource amendment (request R-L7B-1) with the belt-coverage pin re-authored in the same change | L7b | `structure_only_option_conflict` |
| `substrate_uniformity` | Admission requires a UNIFORM substrate: every input tensor leaf meta AND every registered parameter/buffer meta (tied objects deduplicated by identity; parameterless models judged by inputs). Mixed cells refuse typed in BOTH directions at entry with structure_only_substrate_mismatch, naming which side is which; a REAL tensor discovered mid-forward (stale pre-wrap factory reference, device='cpu' literal) refuses through the same family at the user's source line (W1-CLS). | `supported_structural` | D8 granted 2026-08-26 | tests/test_weightsfree_admission.py (mixed cells, tamper rows) | S2-amendment | — |
| `plan_shape_check` | The audit-only plan checker (Trace.check_plan, D14) resolves selectors, checks multiplicity, and compares declared replacement geometry against hypothesis shapes/dtypes; the report is executable=false ALWAYS and never arms, replays, or lifts the late-bind refusal. REFUTED sources refuse through this row (G5); callables and value-derived selection refuse typed (plan_check_unsupported). | `supported_hypothesis` | discharge | tests/test_weightsfree_plan_check.py | L7a | — |
| `measurement_exports` | Measurement-shaped exports (chrome_trace, speedscope, flamegraph, memory_timeline) refuse typed: timings and allocator peaks are MEASUREMENTS, not payload values, and a value-free capture has none — meta-dispatch overhead rendered as 'measured' inverts real cost rankings (weightsfree memo D15; the existing values code must not silently widen). | `refuse:structure_only_measurements_unsupported` | never | tests/test_weightsfree_disclosure.py | S2-amendment | `structure_only_measurements_unsupported` |
| `viz_graph_render` | Graph rendering (incl. size_by consuming hypothesis shapes) works, carrying the structure-only banner. | `supported_hypothesis` | never | tests/test_structure_only_honesty.py | L7a | — |
| `viz_payload_visualizers` | Payload-consuming visualizers (activation heatmaps, custom value visualizers) require values; refused typed. | `refuse:structure_only_values_unsupported` | never | tests/test_structure_only_entry.py | L7a | `structure_only_values_unsupported` |
| `structure_digests` | Graph-shape and meta-domain content digests are always computed; they can never collide with value-bearing digests. | `supported_structural` | never | tests/test_structure_only_honesty.py (G2 domain pin) | L7a | — |
| `discharge` | A real capture of the same graph upgrades hypothesis rows to corroborated or refutes them; upgrades happen ONLY via the discharge authority. | `supported_structural` | never | tests/test_structure_only_discharge.py | L7a | — |
| `refuted_rows` | Consumers that tolerate hypothesis rows refuse REFUTED rows typed (G5). | `supported_structural` | never | tests/test_structure_only_discharge.py | L7a | — |
| `teaching_refusals` | Enumerated value escapes (device-neutral, both forms) and missing meta kernels refuse typed with the user source line; other meta-mechanism deaths are typed via the backstop without branch classification; unenumerated real-value escapes are outside the detectable surface (C-RESIDUAL); user exceptions propagate unchanged. | `supported_structural` | never | tests/test_structure_only_teaching.py | L7a | — |
| `save_analysis_artifact` | Analysis-level artifacts persist the structure_only marker plainly (tlspec v8); loads validate marker coherence (M-C2/M-C3 in torchlens/_io/forgery_validation.py) and every value-claim on the loaded trace stays a HYPOTHESIS. | `supported_structural` | never | tests/test_structure_only_capabilities.py | P1 | — |
| `save_runnable` | Runnable save is refused in v1; IF the L7b late-bind posture lands (wave 1, post-L4, S1-serialized), declared late-bind slots replace this refusal for eligible models. | `refuse:structure_only_runnable_unsupported` | L7b amendment lands | tests/test_structure_only_capabilities.py; entry-dark bridge + mandatory bind-digest authority shipped: tests/test_structure_only_bridge.py (flip blocked on the S2 StateSource amendment, request R-L7B-1) | L7b | `structure_only_runnable_unsupported` |
| `live_replay` | Replay requires values; refused unless and until late-bind (see save_runnable) provides them at bind time. | `refuse:structure_only_replay_unsupported` | L7b amendment lands | tests/test_structure_only_capabilities.py; entry-dark bridge + S1 validator-reuse binding path shipped: tests/test_structure_only_bridge.py (flip blocked on the S2 StateSource amendment, request R-L7B-1) | L7b | `structure_only_replay_unsupported` |
| `validation_entry` | There is nothing to validate against; refused permanently by design (discharge is the verification story). | `refuse:structure_only_validation_unsupported` | never | tests/test_structure_only_capabilities.py | L7a | `structure_only_validation_unsupported` |
| `backward_grads` | Backward/gradient capture is refused in v1; any future support is an L9-adjacent S2 amendment, not implied here. | `refuse:structure_only_backward_unsupported` | S2 amendment | tests/test_structure_only_capabilities.py | S2-amendment | `structure_only_backward_unsupported` |
| `episode_composition` | Episode capture composes with structure-only ONLY if a later S2 amendment rules it; refused in v1. | `refuse:structure_only_episode_unsupported` | S2 amendment | reserved: no episode surface exists on this branch yet | S2-amendment | `structure_only_episode_unsupported` |
| `distributed` | Distributed structure-only capture is out of scope this sprint; the existing distributed refusal contract governs. | `out_of_scope:distributed-contract` | S2 amendment | torchlens/_distributed.py refusal surface | S2-amendment | — |
| `fake_tensor_substrate` | Capturing a REAL model structure-only via FakeTensorMode is a VERIFY item, not a capability. | `verify:V1` | V1 verdict | pending: V1 spike | L7a | — |
| `symbolic_shapes` | Symbolic/dynamic shapes remain refused at the variant gate; a ShapeEnv-backed range hypothesis is a VERIFY item. | `verify:V2` | V2 verdict | pending: V2 verification | L7a | — |
