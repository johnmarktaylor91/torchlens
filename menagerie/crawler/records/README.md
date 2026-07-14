# Canonical crawler records

The committed record store is the campaign source of truth. Intake snapshots and manifests are
immutable. Model revisions, attempts, gates, and operational events are append-only JSONL, sharded
by the repository's declared model, machine, or campaign naming policy. A single active driver owns
all appends; shard merging is deterministic and conflicting immutable identities fail loudly.

Every accepted line ends with a newline and carries its schema identity and canonical payload hash
or revision hash. Startup validates complete lines in byte order. Recovery may remove only a torn
final partial line, and only after durable recovery evidence records the removed bytes. Complete
malformed lines, hash mismatches, forks, stale parents, and conflicting replays are never repaired
in place. Corrections and human grants append new records or superseding revisions; history is not
rewritten.

SQLite, cursors, current-model files, release files, deferred files, and status summaries are
derived views. Rebuild them from the immutable intake plus canonical model, attempt, and gate JSONL;
never treat a derived view as evidence.

Records are public-repository material. They may contain bounded logs, tracebacks, literal grounding
excerpts, URLs, revisions, and hashes, but never credentials, tokens, private keys, SSH-agent data,
private mirror bytes, or unrestricted third-party source archives. Restricted bytes belong only in
the private content-addressed mirror; committed records retain their hash-bound public metadata.
