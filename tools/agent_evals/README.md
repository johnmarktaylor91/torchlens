# Agent-from-docs harness

Tests whether a coding agent can drive TorchLens from its shipped documentation alone. An agent
gets the docs bundle and a short task, writes one script (`solution.py`), and the harness executes
the script and grades it with deterministic checks.

## Tasks

| Task | Surface | Checked against |
| --- | --- | --- |
| `capture_inspect` | capture and inspect | relu count and shapes from forward hooks; output sum from a plain forward |
| `save_load` | save and load | op count from leaf-module calls; first relu sum from a hook; the artifact reloads |
| `intervention` | capture-time edit | a forward hook that zeroes `act1`; the edit must change the output |
| `report_export` | report and export | `Capture status` kept, at most 300 tokens (chars / 4); HTML file; one CSV row per op |
| `refusal` | a documented refusal | the stable code `input_rung_conflict` the quickstart names, plus a remedy |
| `agent_tools` | agent tool surface | manifest schema id; paging with `max_rows=2` reaches every row once |

Every task uses the fixture model in `tasks.py` (`MODEL_SOURCE`, written into each task directory
as `model.py`). Oracles come from plain torch or from codes and schema ids the docs state; a
grader never takes its expected value from the script under test.

## Offline mode (runs in the test suite)

Each task has a reference solution in `references/`, written from the docs alone; its docstring
names the pages it used. `tests/test_agent_evals.py` executes every reference against the current
code and runs the task's checks, so a docs or API change that breaks a documented spelling fails
with no model call. Negative controls check that every grader fails a script that does nothing.

```bash
python tools/agent_evals/harness.py offline
pytest tests/test_agent_evals.py
```

## Live mode (opt-in, never in CI)

```bash
python tools/agent_evals/harness.py live --out RUN_DIR \
    --agent-cmd 'my-agent --prompt-file {prompt} --docs {docs} --write {solution}'
```

`live` builds the docs bundle (`README.md`, every Markdown page under `docs/`, and the text
`torchlens.agent.guide()` serves, as `agent_guide.md`), writes one directory per task, runs the
agent command once per task from that directory, then grades. The template may use `{prompt}`,
`{docs}`, `{workdir}` and `{solution}`. To run the agent somewhere else, split the steps:
`prepare --out RUN_DIR` writes the task directories (the prompt names the docs directory by a
relative path, so the tree can move), the agent writes each `solution.py`, and
`grade --dir RUN_DIR` executes and grades them. Results go to `RUN_DIR/results.json`.

The prompt tells the agent to use only the docs and not to run Python. The harness does not
sandbox the agent, so a run that must prove docs-only use should restrict the agent's file access
or check its tool log for reads outside the bundle.

A task an agent fails because the docs are wrong or missing is a docs finding: fix the page, and
if the reference solution relied on the same gap, fix the reference too.

## Adding a task

Add a `Task` to `TASKS` in `tasks.py` with a grader that computes its oracle independently, write
`references/<task_id>.py` from the docs alone (cite the pages in its docstring), and run the
offline tests.

## Budget measurement

`measure_budgets.py` measures the agent surface's responses over the classics corpus against the
budget configuration in `torchlens/agent/_budgets.py`; the method and the latest numbers are in
`BUDGETS.md`.
