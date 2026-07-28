# Author subagent pool -- operator runbook

The author lane is the crawler's throughput bottleneck: ~4,463 author-hours across 28,482
models, needing 10-16 sessions in flight to finish inside 30 days. This runbook is how a
human starts, watches, pauses, and resumes that pool for one campaign.

**One managing Claude session per campaign.** Four campaigns run concurrently in four
clones, each with its own driver, its own author queue, and its own managing session. The
campaigns never share a queue.

| campaign | rows | workload | author tier | Agent-tool model |
|---|---:|---|---|---|
| `c1-mech` | ~7,150 | library-zoo mechanical | `claude-sonnet` | `sonnet` |
| `c2-disco` | ~14,798 | discovered PyTorch tail | `claude-sonnet` | `sonnet` |
| `c3-classics` | ~5,487 | unregistered classics | `claude-opus-5` | `opus` |
| `c4-native` | ~1,047 | native TF / Keras / JAX | `claude-sonnet` | `sonnet` |

The tier is a **campaign** property, not a per-model decision. A campaign's
`author_model_identity` is frozen for its entire run, so a model a sonnet campaign finds
genuinely hard is emitted as a typed `BLOCKED` result and requeued into `c3-classics`.
Escalating in place would fail authority validation on the first opus result.

---

## 1. Start

### 1.1 Environment

```bash
export CAMPAIGN=c1-mech
export CLONE=$HOME/campaigns/$CAMPAIGN            # this campaign's clone
export QUEUE=$CLONE/.crawl-local/author-queue     # campaign-local, never shared
export PY=$CLONE/.venv-crawler/bin/python

export MENAGERIE_AUTHOR_QUEUE="$QUEUE"
export MENAGERIE_CAMPAIGN_ID="$CAMPAIGN"
export MENAGERIE_AUTHOR_COMMAND="$PY -m menagerie.crawler.operator_author --queue $QUEUE --campaign $CAMPAIGN"
export MENAGERIE_CHECKER_COMMAND="$PY -m menagerie.crawler.operator_checker"
export MENAGERIE_ENVIRONMENT_COMMAND="$PY -m menagerie.crawler.operator_environment"
export MENAGERIE_PUBLIC_MIRROR=/path/to/public-mirror
export MENAGERIE_PRIVATE_MIRROR=/path/to/private-mirror
```

`menagerie` is not yet a declared wheel package (W0.1), so every wrapper must be invoked
with the clone as the working directory or on `PYTHONPATH`. The driver already runs
wrappers with the repo root as cwd; keep it that way.

### 1.2 Start the pool **before** the doctor

The strict doctor's `author-web-tools` check runs a live capability probe down the real
author path. That probe is answered by the pool, so the pool must already be servicing the
queue when `doctor` runs. Starting the doctor first will fail the check -- correctly, since
at that moment nothing is servicing the queue.

In the managing Claude session:

1. `$PY -m menagerie.crawler.author_pool --queue "$QUEUE" list` -- confirm the queue is
   reachable and see what is waiting.
2. Keep that loop running (see section 2).

### 1.3 Preflight

```bash
cd "$CLONE" && $PY -m menagerie.crawler doctor --target osx-arm64 --strict
```

Two of its checks now depend on this pool:

- **`author-web-tools`** -- the doctor mints a nonce, invokes `MENAGERIE_AUTHOR_COMMAND`
  with a `author-capability-probe.v1` request, and requires a receipt naming `WebSearch`,
  `web_search_exa`, and `web_fetch_exa`, each bound to that nonce. The pool derives an
  unpredictable challenge from the nonce and only writes a receipt if all three tools
  independently agree on a live fact. **A failure here means the author path cannot
  research. Do not start the campaign; fix the tools.**
- **`notifier-delivery`** -- the doctor sends a nonce through the notifier and requires a
  receipt at `MENAGERIE_NOTIFICATION_RECEIPT_PATH`. The receipt is written only after the
  underlying delivery script exits zero, so this fails exactly when notifications would
  silently vanish -- which is the state in which a month-long unattended campaign goes dark.

### 1.4 Freeze the campaign policy

Only `c1-mech` is allowed to carry the anti-slop review gate. Its launch must include the
three durable advance notifications:

```bash
cd "$CLONE" && $PY -m menagerie.crawler run \
  --intake .crawl-local/intake --target osx-arm64 \
  --review-checkpoint-at 1000 --progress-milestones 900,950,1000,2000,3000,5000,10000 \
  --author-command "$MENAGERIE_AUTHOR_COMMAND" \
  --checker-command "$MENAGERIE_CHECKER_COMMAND" \
  --environment-command "$MENAGERIE_ENVIRONMENT_COMMAND"
```

This first invocation writes the mode-0600 campaign config. Stop it once that config
exists, then install the supervisor in section 1.5. For `c2-disco`, `c3-classics`, and
`c4-native`, do not launch until the C1 review is signed off, and replace the checkpoint
flag with `--review-checkpoint-at 0`. The production campaign manifest and CLI both reject
the inverse policy: C1 cannot silently lose its gate, and C2-C4 cannot accidentally inherit
it.

### 1.5 Install the per-campaign launchd supervisor

```bash
CONFIG=$(find "$CLONE/.crawl-local/campaign-configs" -name '*.json' -print -quit)
PLIST="$HOME/Library/LaunchAgents/org.torchlens.menagerie-crawler.$CAMPAIGN.plist"
"$PY" -m menagerie.crawler.supervisor render-launchd \
  --campaign-id "$CAMPAIGN" --repo-root "$CLONE" \
  --campaign-config "$CONFIG" --author-queue "$QUEUE" \
  --python "$PY" --output "$PLIST"
launchctl bootstrap "gui/$(id -u)" "$PLIST"
```

The installed agent runs `tools/crawler_supervisor.sh`. Unexpected driver exits restart
with exponential backoff; five crashes in 30 minutes open a circuit and produce a
receipt-backed alert. Exit 3 backs off because another driver owns the campaign. Exit 4
stops the outer agent and delegates exactly once to the existing reset-time wake episode;
that callback re-enters through the same supervisor. Exit 5 stops for human review. Exit 6
is retryable operator infrastructure and alerts before stopping. Exit 0 notifies and
stops. launchd restarts only an unexpectedly dead *supervisor*, not any of those handled
states.

---

## 2. Run the pool loop

The pool is not a daemon. It is a loop the managing session runs, and each iteration is
four steps.

**Step 1 -- see what is waiting.**

```bash
$PY -m menagerie.crawler.author_pool --queue "$QUEUE" list
```

Each row carries `job_id`, `kind` (`source-request`, `author`, or `capability-probe`),
`stable_id`, `campaign_id`, `subagent_model`, and any live lease.

**Step 2 -- claim and get the brief.**

```bash
$PY -m menagerie.crawler.author_pool --queue "$QUEUE" claim --job "$JOB"
```

This writes a lease and prints JSON containing `claimed_at`, `deadline_at`,
`subagent_model`, and `brief` -- the complete, ready-to-send dispatch prompt, already
carrying the job's absolute paths, its effort grant, and the campaign's standards.

**Step 3 -- dispatch a subagent** with the Agent tool: `model` = the printed
`subagent_model`, `prompt` = the printed `brief`, `run_in_background: true`. Run 10-16
concurrently; tune from day-2 telemetry.

**One subagent serves both stages of a model.** After the `source-request` job completes
and the coordinator's controlled fetch freezes the manifest, the matching `author` job
appears. Continue the **same subagent** with `SendMessage` and its `author` brief rather
than dispatching a cold one -- the research context is the expensive part, and reloading it
is the whole cost this architecture exists to avoid.

**Step 4 -- commit the answer.**

```bash
$PY -m menagerie.crawler.author_pool --queue "$QUEUE" complete \
  --job "$JOB" --claimed-at "$CLAIMED_AT" --tool-calls "$OBSERVED_TOOL_CALLS"
```

`--tool-calls` is **mandatory and must be the number you actually observed.** The pool is
the only boundary that sees Agent-tool events, so it declares them to the engine, and the
engine refuses any receipt declaring more than the grant. Wall time and fetch-target count
are measured by the pool itself and are not yours to state. Do not round a count down to
make a job land: an over-budget session is a real finding, and the honest failure it
produces is requeued, whereas a false receipt corrupts the effort ledger for the run.

For long sessions, extend the lease so another servicer does not take the job:

```bash
$PY -m menagerie.crawler.author_pool --queue "$QUEUE" renew --job "$JOB"
```

### Handling a bad outcome

| what happened | what to run |
|---|---|
| Claude usage limit | `backoff --job "$JOB" --excerpt "<verbatim provider text>"` -- pauses the scheduler with a reset time; never a model failure |
| subagent crashed, tools glitched, transient | `fail --job "$JOB" --reason subagent-transient --retryable --detail "..."` |
| the model genuinely cannot be authored | prefer a valid `BLOCKED` **result** from the subagent + `complete`; use `fail ... --permanent` only when no result exists |
| ran out of budget with nothing to show | `fail --job "$JOB" --reason effort-cap-exhausted --permanent` |
| you claimed it and cannot service it | `release --job "$JOB"` |

`--retryable` / `--permanent` is required, not defaulted: the engine refuses a failure
sidecar without an explicit classification rather than guessing, because guessing turns an
infrastructure blip into a permanently burned model.

### Isolation

Subagents get the clone **read-only** plus their per-job write root
(`allowed_output_root` / `allowed_model_dir` in the envelope). They must never write to
ledgers, checkpoints, queue state, environment specs, or another model's tree. Before each
review checkpoint, diff the clone's tracked tree and the authority roots; any modification
outside a per-job root is a hard stop, not a warning.

---

## 3. Monitor

- `list` and `expired` every few minutes. `expired` returns jobs whose lease lapsed with no
  answer -- a dead subagent. Re-claim with `--force` and re-dispatch; the job's answer files
  are nonce-bound, so a late file from the dead attempt can never be mistaken for the new
  one.
- Watch the **queue depth**. A queue that never drains means the pool is under-provisioned;
  a queue that is always empty means the driver, not the author lane, is the bottleneck.
- Watch for **stalls**. If the pool stops answering, the engine's lane hits its 45-minute
  deadline and raises retryable infrastructure -- a stalled queue, never a failed model.
  The supervisor independently watches the durable pending queue and alerts as soon as the
  same deadline is crossed, before the bounded engine retry finishes. That is the safe
  behavior, but it costs a full retry cycle, so treat repeated stalls as an incident.
- The driver notifies at 900/950/1000 before and at the hard 1,000-model review checkpoint, and on
  quota pauses. Those notifications now carry receipts; if a notification is expected and
  no receipt exists, the notifier is broken -- re-run the doctor.

---

## 4. Pause

**Provider pause (automatic).** Publish a backoff sidecar for the in-flight job. The lane
raises a typed pause, the driver routes it to the usage-limit path with
`provider="anthropic"`, and the campaign schedules a wake at the reset time. Stop
dispatching; leave the driver alone.

**Operator pause (manual).** Stop claiming new jobs and let the in-flight ones finish.
Unclaimed jobs simply stay pending -- the queue is durable. If you must stop immediately,
`release` every job you hold so a later session can re-claim them; releasing is always
safe, because a released job has produced no receipt.

Do **not** delete pending jobs to "clear" a queue. The lane is blocked on them, and
removing one makes the lane wait out its full stall deadline for an answer that will never
come.

---

## 5. Resume

1. Start a replacement managing Claude session in the same campaign clone and export the
   queue variables from section 1.1. Do not delete or recreate the queue. The pending job
   descriptor is the exact work the dead session left behind, and its nonce prevents a late
   answer from the old session being accepted.
2. `list` -- pending jobs are exactly where they were.
3. `expired` -- clear any lease left behind by the stopped session; re-claim with `--force`.
4. Re-run the doctor's two receipts if the machine or the tool configuration changed. They
   are cheap and they are the only evidence that the author path still works.
5. Resume the loop in section 2. If the driver has already exited 6, kickstart the installed
   agent with
   `launchctl kickstart "gui/$(id -u)/org.torchlens.menagerie-crawler.$CAMPAIGN"`.

The 45-minute incident is operational-only: it appends campaign health and supervisor
state, but no model attempt or terminal failure. Never requeue the affected model merely
because its managing session died.

---

## 6. What the receipts actually prove

Both new doctor checks were built to replace probes that could not fail usefully. Their
value is entirely in being able to fail, so neither may be weakened to get a campaign
started.

**Author capability.** The nonce selects an unpredictable package from a fixed corpus; the
session must report that package's current version and release serial via all three tools;
the fetched document must hash to its own declared digest and literally contain the
reported facts; the two search tools must not have returned identical result lists; and
every timestamp must fall inside the probe window. An author path with no live web reach
cannot produce any of this. It does **not** defend against a subagent deliberately
fabricating a self-consistent document -- that agent is inside our trust boundary and is
the thing being configured. It does catch the real failure mode: tools missing,
unconfigured, MCP-disconnected, or silently erroring.

**Notifier delivery.** The receipt records the resolved transport, its exit status, the
delivery instant, and the digest of the exact bytes sent. It is written only after the
transport exits zero, so "no transport installed", "transport failed", and "transport timed
out" all leave no receipt and fail the check.

If either check fails, the answer is to fix the capability. Editing the check is how a
month-long campaign ships ungrounded proposals and nobody finds out.

---

## 2b. Batch the loop (`tools/pool_batch.py`)

Section 2's four steps are correct but not operable one job at a time -- at 28,482 models
that is roughly three shell commands plus one dispatch per model, and every hand-run step is
a chance to mistype a lease owner or lose a `claimed_at`. Use the batch tool instead. It
collapses the mechanical steps into two calls and leaves exactly one thing to the managing
session: dispatching the subagents, which is the only part it alone can do.

```bash
PB="$PY -m menagerie.crawler.tools.pool_batch --queue $QUEUE --repo-root $CLONE --python $PY"

$PB status                                     # compact queue summary
$PB claim --count 12 --out .crawl-local/rounds/r1
```

`claim` leases up to N unleased jobs, writes each brief to its own file, and emits
`manifest.json` with `job_id`, `kind`, `stable_id`, `subagent_model`, `claimed_at`, and
`brief_path`. Dispatch one Agent subagent per row **in a single message**, using the row's
`subagent_model` and the contents of its `brief_path`.

Then commit the round:

```bash
$PB complete --manifest .crawl-local/rounds/r1/manifest.json \
             --counts   .crawl-local/rounds/r1/counts.json
```

`counts.json` maps each `job_id` to either a completion or a typed failure:

```json
{
  "author-abc123": {"tool_calls": 14},
  "author-def456": {"tool_calls": 9, "evidence": "/abs/receipt.json"},
  "author-ghi789": {"failed": true, "reason": "subagent-transient", "retryable": true}
}
```

Three properties this deliberately enforces, matching section 2:

- **The lease owner is pinned** (`--owner`, default `pool-manager`). Without a stable owner
  each shell invocation gets a fresh `host:pid` and a job claimed in one call cannot be
  completed by the next -- the lease looks foreign.
- **`tool_calls` is required and never defaulted.** A missing count is an error, not a zero.
  The engine refuses a receipt declaring more than the grant, and rounding a count down to
  make a job land corrupts the effort ledger for the whole run.
- **Failures must be classified.** `retryable` has no default, because guessing turns an
  infrastructure blip into a permanently burned model.

A job that cannot be claimed this round is reported and skipped rather than aborting the
batch, so one bad row cannot stall a whole wave.
