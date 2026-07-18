# Campaign run

The scheduler is single-writer and environments are created, used, checkpointed, and removed one
at a time. Set `INTAKE` to the immutable snapshot directory printed by intake.

## Phase 1: PyTorch

```bash
python -m menagerie.crawler plan --intake "$INTAKE" --target osx-arm64 --phase pytorch
python -m menagerie.crawler run --intake "$INTAKE" --target osx-arm64 --phase pytorch --sequential-envs
```

Routing assigns PyTorch work from declared framework and dependency requirements to the thick
`core`, `graph`, `audio`, `mmlab`, `tabular-ts`, `detect-misc`, `legacy-torch`, and `oddballs`
intents. R2/R3/R4 normally route to `core` unless exact source requirements prove otherwise. The
planner may split an incompatible thick intent, but it does not create per-model convenience
environments.

## Phase 2: native framework tail

Start this only after Phase 1 has no pending PyTorch workflow rows.

```bash
python -m menagerie.crawler plan --intake "$INTAKE" --target osx-arm64 --phase native-tail
python -m menagerie.crawler run --intake "$INTAKE" --target osx-arm64 --phase native-tail --sequential-envs
```

Native TensorFlow/Keras, JAX/Flax, and Paddle work routes to `tf-keras-arm64`,
`jax-flax-arm64`, and `paddle-arm64`. A different original and run framework is recorded only when
a transparent adapter delegates to the native implementation.

Static hints only prioritize probes. `deferred:needs-cuda` or `deferred:needs-x86` requires stored
source or focused probe evidence. The designated arm64 heavy-build pool gets at most two recorded
CPU source-build attempts before an x86 deferral; source that proves unavoidable CUDA can defer
earlier. Failed commands, logs, hashes, host/environment identity, and defer evidence stay in the
append-only attempt history.

Inspect progress without taking the writer role:

```bash
python -m menagerie.crawler status --intake "$INTAKE" --full --verify-partition
```

## Tiny-model acceptance dry-run

The opt-in acceptance harness exercises the real driver and isolated worker on four tiny local
PyTorch architectures across the ten rows required by the frozen metadata-batch minimum. It uses the
selected lock-built fixture prefix, shipped environment binder/compiler/supervisor, canonical attempt
sink, and reducer. Only the author, checker, and notifier remain deterministic. The explicit prefix
must have the exact fixture lock, resolved export, export digest, and probe receipt artifacts beside
it; there is no current-interpreter fallback. The commands write campaign state only beneath the
selected disposable root and print the canonical current funnel plus structured acceptance status as
JSON.

```bash
DRY_RUN_ENV_PREFIX=/path/to/round19-real-environment/prefix
python -m menagerie.crawler run --dry-run --dry-run-root /tmp/menagerie-crawler-dry-run \
  --dry-run-environment-prefix "$DRY_RUN_ENV_PREFIX" \
  --review-checkpoint-at 2 --progress-milestones 3
python -m menagerie.crawler resume --dry-run --dry-run-root /tmp/menagerie-crawler-dry-run \
  --dry-run-environment-prefix "$DRY_RUN_ENV_PREFIX" \
  --review-checkpoint-at 2 --progress-milestones 3 --after-review
```

The first command must exit with the documented paused code after two real `runs` revisions. The
resume command succeeds only after all ten expected rows have authenticated forward attempts and
current `runs` revisions. A terminal partition of source failures is an acceptance failure and exits
nonzero even though the deterministic driver itself reached terminality.

Environment freshness checks cover the complete sealed tree and exact external targets with full mode,
size, `mtime_ns`, `ctime_ns`, device, and inode triggers. A hardlink clone may change shared-inode ctime;
the first subsequent check rehashes once and, when content is byte-identical, refreshes only the local
authority's cheap baseline without changing its authority or generation. A changed content digest
invalidates the authority before spawn. Only a privileged/out-of-band actor preserving every cheap
field, a release-rejected coarse filesystem, or the final verify/read TOCTOU can defer detection.
