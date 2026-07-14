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

The opt-in test harness exercises the real driver and isolated worker on four tiny local PyTorch
architectures across the ten rows required by the frozen metadata-batch minimum. It replaces only
the author, checker, notifier, and environment solve with deterministic fakes, writes only beneath
the selected disposable root, and prints the current funnel as JSON.

```bash
python -m menagerie.crawler run --dry-run --dry-run-root /tmp/menagerie-crawler-dry-run \
  --review-checkpoint-at 2 --progress-milestones 3
python -m menagerie.crawler resume --dry-run --dry-run-root /tmp/menagerie-crawler-dry-run \
  --review-checkpoint-at 2 --progress-milestones 3 --after-review
```
