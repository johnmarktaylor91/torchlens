# Resume and crash recovery

Canonical JSONL and the kernel lock determine recovery. SQLite, cursor files, views, caches, and
installed environments are disposable accelerators.

Before resuming, inspect status and verify the three canonical ledgers:

```bash
python -m menagerie.crawler status --intake "$INTAKE" --full --verify-partition
python -m menagerie.crawler checkpoint --intake "$INTAKE" --verify-ledgers --verify-views
python -m menagerie.crawler resume --intake "$INTAKE" --target osx-arm64
```

Startup scans every complete JSONL line. A non-newline-terminated tail may be truncated only after
its byte offset, byte count, digest, time, and recovery location are durably recorded. A malformed
complete line is corruption and is not repaired. The driver then rebuilds disposable state and
continues the first unsatisfied work identity; accepted history is not rewritten.

The OS advisory lock at `.crawl-local/locks/driver.lock` is authoritative. PID metadata alone does
not permit breaking a live lock. A wakeup that meets a live owner records or prints an idempotent
already-running result and exits.

Pending metadata gates remain visible; eligible mechanical work may continue, but completion does
not become true. R3/R4 forward observations remain blocked until fidelity is current. Claude or
checker usage limits record the provider, exact reset time, queued counts, and current environment.
The configured recurring wake episode reacquires the same lock until a durable resume, completion,
cancellation, or supersession fact resolves it, then removes its scheduler projection.
Production wake callbacks enter through `tools/crawler_supervisor.sh`, so a post-reset run keeps the
same outer liveness and queue-stall watchdog. The reset-time episode remains the only component that
decides when to resume; the persistent launchd agent does not race it.

At the blocking 1000-model review, ordinary resume remains paused. After JMT signs off, use:

```bash
python -m menagerie.crawler resume --intake "$INTAKE" --target osx-arm64 --after-review
```

This checkpoint belongs only to `c1-mech`. The other three campaign configs are accepted only with
`--review-checkpoint-at 0`, and are launched after this sign-off.
