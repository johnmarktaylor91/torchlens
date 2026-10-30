# Agent benchmark harness

Measures whether a coding agent reimplements recent interpretability methods more often
correctly, faster and with fewer tokens with an interpretability library than with plain PyTorch
or a peer library. The primary track builds each method from primitives: any ready-made
implementation of the task's target method is disabled in every arm.

Task specs, reference solutions, graders and held-out inputs are written by a separate seat that
never sees the library under test; they live in `oracles/<task-id>/`. This directory holds the
harness that runs subject agents against them. Live agent runs call a model and are never run in
CI; the tests in `tests/test_agent_bench.py` use a scripted fake agent instead.

## Parts

| Module | Job |
| --- | --- |
| `registry.py` | Loads each task directory: spec, public inputs, held-out inputs, grader command, arms |
| `arms.py` | The four arms; builds each arm's environment from one base and audits its imports |
| `guards.py` | Import guards that disable natives; the post-run audit of the agent's code |
| `natives.json` | Per task and arm, the native implementations the guard disables (draft) |
| `runner.py` | One run: work folder, agent command, caps, checkpoint snapshots, grading, record |
| `telemetry.py` | Event-stream parsing, tokens, tool calls, executions, code metrics, record schema |
| `scheduler.py` | Seeded, blocked, interleaved run order; the pilot's cells |
| `fake_agent.py` | Scripted stand-in agent for tests and model-free smoke runs |
| `stub/T0/` | A harness stub task in the oracle layout (not a benchmark task) |
| `pipeline_smoke.sh` | Model-free end-to-end smoke: environments, guards, stub oracle, one fake run |

## Arms

| Arm | Environment | Agent material |
| --- | --- | --- |
| A | Base: Python, torch, torchvision, transformers, numpy, scipy | Plain-PyTorch index (hooks, `torch.func`, autograd) |
| B | A plus the library under test | Its agent index, as a user gets it from its setup command |
| C | A plus nnsight, local execution only | nnsight's maintainers' own material, filled to the template |
| D | A plus TransformerLens, on the tasks it covers | TransformerLens's best public material, filled to the template |

All material follows one template and size budget, so the study measures the library and not
the context file. Each arm is a separate virtual environment; `build-env` checks that the arm can
import its own library and none of the others, records the frozen package list and its hash, and
reports any base package a peer library moved.

## Task contract

A task directory holds `spec.md` (the prompt), the inputs the agent may see (default `public/`),
and private material the agent never sees: `reference.py`, `grader.py`, `heldout/` and seeded
wrong solutions in `wrong/`. An optional `task.json` names each part; `registry.py` documents
every field and the grade-command placeholders. The grader writes a JSON verdict with `passed`
and a list of bars (`name`, `passed`, `value`, `threshold`). A verdict without a boolean `passed`,
a grader that exits nonzero and any failing bar all count as a failed run; the harness never
loosens a bar.

## A run

1. A fresh work folder gets `TASK.md` (the spec), the public inputs and the arm's material.
   Oracle files stay outside it.
2. The agent command for the tier (from a runtime configuration kept outside the repository)
   starts in the work folder with the arm's interpreter first on `PATH` and the run's guard file
   named by `AGENT_BENCH_GUARD`. Its event stream (JSON lines on stdout) is saved and read live.
3. The run stops at 45 minutes, past 60 substantive tool calls or past the token cap. The work
   folder is copied at 10, 20 and 45 minutes; a checkpoint the run did not reach takes the final
   state.
4. Each checkpoint is graded. The agent's Python files are audited for guarded names; a hit fails
   the run.
5. One `record.json` is written for every run, failures included: elapsed time, tokens (uncached,
   cache read, cache write, output), tool calls (total, substantive, by name), executions,
   load-bearing lines, checkpoint verdicts, guard entries and violations, the first call into
   the arm's library, the environment hash, and paths to the transcript and code.

Load-bearing lines count lines in the agent's Python files that are not blank, comment-only or
docstring, excluding provided inputs and self-test files named `test_*.py`.

## Commands

From `tools/`:

```bash
python -m agent_evals.bench list
python -m agent_evals.bench build-env --arm A --dest ENVS/A
python -m agent_evals.bench check-task --task T0 --python ENVS/A/bin/python --oracles agent_evals/bench/stub
python -m agent_evals.bench plan --pilot --repeats 3 --seed 1 --out plan.jsonl
python -m agent_evals.bench run-plan --plan plan.jsonl --config TIERS.json --envs ENVS --out RESULTS
python -m agent_evals.bench summarize --results RESULTS
```

`TIERS.json` maps `small`, `mid` and `frontier` to an agent command (argv list; placeholders
`{prompt_file}`, `{prompt}`, `{workdir}`, `{arm_python}`, `{arm_bin}`, `{material}`, `{guard}`)
and optional caps. The command must keep the agent's tools inside the work folder, with no
network, and write its event stream on stdout.

## Isolation limits

The import guard deters; the code audit and the environment audit are the checks. An agent that
reaches a native through a path the audit cannot see (code generated at run time, a subprocess
with a cleaned environment) is caught only by transcript review, which is published with the
results. The runner does not sandbox the agent itself: the agent command is responsible for
filesystem and network confinement.
