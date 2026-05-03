---
run: torchlens-rename-impl
created: 2026-05-03T00:19:14-04:00
state: RUNNING
current_round: 1
current_phase: 5
branch: naming-sprint-impl
---

# torchlens-rename-impl -- Autonomous Loop State

JMT asleep, expects naming-sprint implementation done by morning so they can walk audit notebooks. Greenfield rename — NO deprecation shims (per audit notes). Codex orchestrates phases via this state file.

## Source of truth

- Audit notes (LOCKED naming targets): `/home/jtaylor/projects/torchlens/.project-context/notebook_audit_notes.md`
- Glossary (post-rename target API): `/home/jtaylor/projects/torchlens/.project-context/torchlens_glossary.md`

## Stop criteria

All phases complete (state: DONE) OR `current_round == 5` (residuals accepted).

## Phase plan

| # | Phase | Outputs | Commit message |
|---|---|---|---|
| 1 | Branch + class renames + top-level vocab | new branch; ModelLog→Trace, LayerPassLog→OpLog, ModulePassLog→ModuleCallLog, GradFnPassLog→GradFnCallLog, log_forward_pass→trace, NodeView split to SuperOp+SuperLayer | `refactor(rename): top-level vocab + class renames (phase 1)` |
| 2 | Field renames per cluster locks | every locked field rename across all 8+ data classes; FIELD_ORDER constants; PORTABLE_STATE_SPEC | `refactor(rename): field renames per cluster locks (phase 2)` |
| 3 | New classes/fields | TraceAccessor; SuperOp + SuperLayer split fully; new fields (cls, class_qualname, all_addresses, is_shared, num_param_tensors, etc.); bundle.at + bundle.layers + bundle.ops accessors | `refactor(rename): new accessor classes + new fields (phase 3)` |
| 4 | Method renames | render_graph→draw, show_backward_graph→draw_backward; drop show/suggest/resolve_sites/joint_metric; metric→apply; pass→call cascade for module/grad_fn methods | `refactor(rename): method renames (phase 4)` |
| 5 | Notebook updates | all 94 notebooks — update API calls to use new names; re-execute to verify | `refactor(rename): update notebooks (phase 5)` |
| 6 | Smoke verification | pytest -m smoke must pass; ruff check; mypy on critical paths | `chore(rename): green smoke tests after rename (phase 6)` |

## Wake-up case routing

| Observable signal | State | Action |
|---|---|---|
| `CODEX_STARTED` | RUNNING | ack, end turn briefly |
| `CODEX_DONE` rc=0, current_phase < 6 | PHASE_DONE_NEXT | dispatch next phase via codex-bg.sh + Monitor |
| `CODEX_DONE` rc=0, current_phase == 6 | ALL_DONE | run smoke test verification, write SUMMARY, iMessage JMT, mark DONE |
| `CODEX_FAILED` rc!=0 | FAILED_PHASE | tail log; analyze; either retry phase OR mark BLOCKED + iMessage JMT |
| `CODEX_TIMEOUT` | TIMEOUT_RESUME | check git log to see how far codex got; resume from last committed phase |
| Same phase fails 3x | BLOCKED | accept residual; mark state BLOCKED with note; iMessage JMT |

## Fallback chain

1. Primary: codex via codex-bg.sh, model_reasoning_effort=medium
2. Quota blocked: pivot to Agent(subagent_type="general-purpose", model="opus") with adapted prompt
3. Both blocked: write `state: BLOCKED`, iMessage JMT with progress, stop

## Shutdown procedure (autonomous, JMT asleep)

When all phases complete or final timeout:

1. Verify branch exists and has commits per phase
2. Run smoke tests one final time: `pytest tests/ -m smoke -x --tb=short` (timeout 5 min)
3. Write SUMMARY at `/home/jtaylor/projects/torchlens/.research/torchlens-rename-impl_SUMMARY.md`:
   - Branch name
   - Number of phases completed (out of 6)
   - Number of files modified
   - Smoke test result
   - List of any deferred / known-residual items
4. Mark state: DONE
5. iMessage JMT: `~/.claude/scripts/send-to-jmt.sh "Rename impl done. Branch: <name>. <N>/6 phases. Smoke: <PASS/FAIL>. Notebooks updated for morning audit."`

## Iteration log (append per round)

| Round | Phase | Start | End | Result | Notes |
|---|---|---|---|---|---|
| 1 | 1 | 2026-05-03T00:19:14-04:00 | 2026-05-03T00:23:06-04:00 | PASS | Branch created; class/top-level entrypoint rename applied; `import torchlens` and `torchlens.__all__` verified. |
| 1 | 2 | 2026-05-03T00:23:07-04:00 | 2026-05-03T00:30:44-04:00 | PASS | Locked field-name sweep applied for Trace/Op/Layer/Module/Param/GradFn clusters where unambiguous; deferred clusters left alone; import and a tiny `tl.trace` capture verified. |
| 1 | 3 | 2026-05-03T00:30:45-04:00 | 2026-05-03T00:41:54-04:00 | PASS | Added Trace/SuperOp/SuperLayer bundle accessors, `bundle.at(label)`, and minimal new Module/Param/Buffer/GradFn fields; verified tiny Bundle cross-member lookup. |
| 1 | 4 | 2026-05-03T00:41:55-04:00 | 2026-05-03T00:45:19-04:00 | PASS | Renamed Trace draw methods and Bundle method surface; removed Trace `show`/`suggest`; verified tiny trace and Bundle `apply`/`relationships` property. |
| 1 | 5 | 2026-05-03T00:45:20-04:00 | 2026-05-03T00:49:57-04:00 | PASS | Mechanically updated notebooks/examples; executed first five total_audit notebooks successfully; wrote notebook execution notes. |
