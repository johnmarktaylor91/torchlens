# TransformerLens arm material: provenance

TransformerLens publishes no agent-usage file (no `llms.txt`; the repository's
`AGENTS.md`, kept verbatim in `raw/AGENTS.md`, is guidance for agents *contributing*
to the library, not for using it). The benchmark authors therefore assembled
`transformerlens.md` to the template from the project's public material, snapshot
2026-10-30:

- `README.md` at `main` (kept in `raw/README.md`): `TransformerBridge.boot_transformers`,
  `run_with_cache`, install.
- `AGENTS.md` at commit `82e1068069aa9c29064e4865099d1a4bc161ce12` (2026-09-25):
  the two hook-name systems (canonical `blocks.{i}.hook_resid_post` aliases versus
  Bridge-native `blocks.{i}.hook_out`), `enable_compatibility_mode()`, the fp32 and
  eager-attention load configuration, `hook_points.py`, `tools/analysis`.
- Hook function signature, `run_with_hooks`, `add_hook`/`reset_hooks`, backward hooks,
  `hook_z` shape, `to_tokens` BOS behaviour and the cache helpers
  (`decompose_resid`, `stack_head_results`, `apply_ln_to_stack`) come from the
  library's long-standing public documentation and the ARENA tutorials it links.

Licence: MIT (Copyright 2022 TransformerLensOrg). Everything in `transformerlens.md`
was written by the benchmark authors (marked "added" in full); nothing was trimmed
from a maintainers' usage file because none exists. The file is thinner in
maintainer-authored pitfalls than the nnsight one; the pitfalls listed are the
ones the public documentation itself warns about. A reviewer who knows the 4.x
API should check the hook names before the first run.

Pitfall parity (30 Oct 2026, see `../PITFALL-PARITY.md`): added the final-norm and
unwrapped-model sentences in "Reading intermediate values", the graph-root line in the
gradient example (it raised on a frozen model, as the plain-PyTorch example did before
its fix), the batched vector-Jacobian sentence, the batch-size sentence and pitfall 8
(in-place overwrites), so this file covers the same task traps as the plain-PyTorch file.

API check against transformer-lens 4.3.0 (30 Oct 2026, a T1 run with processing off on the
benchmark base): canonical hook names exist without compatibility mode; BOS follows
`cfg.default_prepend_bos` (off for Pythia); `enable_compatibility_mode(no_processing=True)`
keeps raw weights; the bridge forces eager attention; `run_with_cache` returns detached
tensors; `hook.name` is the native name. The file was corrected to match.
