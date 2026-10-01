# Security Policy

## Reporting a vulnerability

Report suspected vulnerabilities privately through GitHub's security advisory
form ("Report a vulnerability" under the repository's Security tab) at
https://github.com/johnmarktaylor91/torchlens/security. Please do not open a
public issue for an unpatched vulnerability. Reports are acknowledged on a
best-effort basis; include a minimal reproduction where possible.

## Supported versions

| Version | Supported |
|---------|-----------|
| Latest 2.x release | Yes |
| Older 2.x releases | Upgrade to the latest release before reporting |
| 1.x | No |

## The trust model, in one paragraph

TorchLens traces models by RUNNING them: `tl.trace(model, x)` executes the
model's `forward()`, so tracing an untrusted model executes untrusted code by
construction, before any TorchLens defense is relevant. Every hardening layer
below applies to TorchLens's own artifact formats and dependencies; none of it
can make running someone else's model safe. Only trace models you would run
anyway.

## Artifact loading (`.tlspec` bundles)

Portable bundles contain a `metadata.pkl` payload. It is treated as UNTRUSTED
input and is read through a restricted unpickler with a default-deny class
allowlist (`torchlens/_io/_safe_unpickle.py`), mirroring the intent of
`torch.load(..., weights_only=True)`:

- Code-execution gadgets (`os` / `sys` / `subprocess` / `builtins` /
  `importlib` and the `torch.serialization` loaders) are denied and execute
  nothing; trust is decided by the resolved object's real module, never the
  pickled path.
- Foreign (non-torchlens) callables embedded in an artifact are NEVER imported
  at load time. By default they deserialize as inert placeholders: the trace
  loads structurally, and any attempt to CALL such a recipe fails closed.
- Tensor payloads travel as safetensors blobs, not pickles.

Executing what an artifact DESCRIBES is a separate, explicit step with its own
opt-ins:

- Intervention-spec loads tolerate foreign `custom` callable keys for
  structural analysis without importing their modules. Resolution FOR
  EXECUTION denies foreign imports by default, because importing a module
  executes its top-level code. Trusted execution opts in with
  `trust_custom_callables=True`; prefer the narrower
  `allowed_custom_callable_modules={"my_trusted_module"}`, which stays
  enforced even alongside broad trust. TorchLens-owned `torchlens.*` callables
  and the fixed `torch`, `torch.Tensor`, `torch.nn.functional`, and `operator`
  namespaces always resolve.
- Runnable artifacts (`trace.run()` on a loaded sparse trace) execute the
  recorded computation DAG. Treat a runnable artifact from an untrusted source
  exactly like an untrusted model checkpoint: do not run it.

The practical rule is unchanged from the README: only load bundles from
sources you trust, and never enable the trust opt-ins for artifacts you did
not produce.

## Dependency advisories (the optional-extra RCE surface)

`pip install torchlens` pulls only the base runtime dependencies (numpy,
packaging, torch, safetensors, tqdm, graphviz, typing_extensions, pillow). CI
runs `pip-audit` over that runtime tree with NO suppressions: an advisory in a
runtime dependency blocks release.

The OPTIONAL extras tree is audited separately, with each accepted advisory
documented here and in the workflow (`.github/workflows/quality.yml`):

- **transformers 4.x band (four advisories, fixed only in 5.x).** The `hf`
  and `test` extras admit `transformers>=4.45,<6`. Installs resolving to the
  4.x floor carry four advisories of one class -- remote code execution from
  LOADING AN UNTRUSTED MODEL ARTIFACT:
  - PYSEC-2025-217: X-CLIP checkpoint-conversion deserialization RCE
  - PYSEC-2026-2288: `Trainer._load_rng_state` torch.load RCE (fixed 5.0.0)
  - PYSEC-2026-2289: malicious `config.json` RCE (fixed 5.3.0)
  - PYSEC-2026-2290: LightGlue model-loading RCE (fixed 5.5.0)

  Remedy: install `transformers>=5.5` (inside the admitted band) to clear all
  four. Exposure through TorchLens itself is limited: transformers is never a
  runtime dependency, and TorchLens has no `Trainer`, checkpoint-conversion,
  or LightGlue call sites. The one path TorchLens exercises is
  `from_pretrained` config parsing, and its own suite only loads pinned,
  well-known public models. The general rule stands regardless of version:
  loading an attacker-supplied model repository is code execution.

- **lightning (PYSEC-2026-3624 / CVE-2026-58659, dated waiver).** RCE in
  PyTorch Lightning's checkpoint loading via `load_from_checkpoint`; no
  released lightning version contains the upstream fix at the time of the
  waiver (2026-08-19). `lightning` is an optional extra, and TorchLens has
  zero `load_from_checkpoint` call sites. The waiver is scoped to the
  extras-tree audit job and drops the moment a fixed release ships.

## The MCP server

`torchlens.bridge.mcp` (extra `torchlens[mcp]`) is a read-only stdio server
over saved `.tlspec` artifacts and environment diagnostics. It executes no
user code and mutates nothing; it wraps the same public read surface as the
Python API, including the restricted-unpickler load path above.

## Hardening expectations for contributors

- Never widen the safe-unpickler allowlist without the fail-closed analysis
  documented in `torchlens/_io/_safe_unpickle.py`; a pickle `REDUCE` invokes
  what you admit, with attacker-controlled arguments.
- New optional integrations that LOAD anything (checkpoints, traces, configs)
  route through the bounded readers and declare their trust boundary in their
  module docstring.
- Dependency advisories are never suppressed silently: a waiver carries its
  exposure analysis and a review trigger, in the workflow file, or it does not
  land.
