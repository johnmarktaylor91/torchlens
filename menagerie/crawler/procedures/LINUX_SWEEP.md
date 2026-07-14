# Linux and NVIDIA deferred sweep

Transfer only the crawler branch and committed facts, code, prompts, target lock products, and
manifests. Configure the physically separate mirrors on the Linux x86_64 CUDA box. Solve exact
Linux locks there during setup; never reuse installed Mac environments or hand-author lock hashes.

After `INTAKE` points to the transferred immutable snapshot, the deferred-bucket sweep is one
crawler command:

```bash
python -m menagerie.crawler handoff-linux --intake "$INTAKE" --resume --only-status 'deferred:*'
```

The handoff target is fixed to `linux-x86_64-cuda`. The selected bucket is only
`deferred:needs-cuda` and `deferred:needs-x86`. The Linux lifecycle fetches required artifacts by
hash, validates target-solved locks and resolved exports, creates each environment sequentially,
and uses the same author, checker, offline worker, gate, checkpoint, and teardown rules.

Successful or newly terminal Linux observations append superseding model revisions. They preserve
the Mac model revisions, attempt evidence, deferral decisions, and mirror history. Mutable SQLite,
installed environments, caches, and scratch state do not transfer between machines. The crawler
does not push; perform any policy-required fast-forward pull before the one-command handoff.
