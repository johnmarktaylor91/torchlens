# Target setup

## Prerequisites

Use the `menagerie/crawler-pipeline` branch on an Apple Silicon macOS target or the ruled
Linux x86_64 CUDA target. Claude Code must expose WebSearch plus the Exa `web_search_exa` and
`web_fetch_exa` tools to the author agent. Configure readable, physically separate public and
private mirror roots through `MENAGERIE_PUBLIC_MIRROR` and `MENAGERIE_PRIVATE_MIRROR`. Configure
the author, checker, and exact-lock lifecycle wrappers through `MENAGERIE_AUTHOR_COMMAND`,
`MENAGERIE_CHECKER_COMMAND`, and `MENAGERIE_ENVIRONMENT_COMMAND` or the corresponding CLI flags.

```bash
git switch menagerie/crawler-pipeline
export MENAGERIE_PUBLIC_MIRROR=/absolute/path/to/public-mirror
export MENAGERIE_PRIVATE_MIRROR=/absolute/path/to/private-mirror
export MENAGERIE_AUTHOR_COMMAND='/absolute/path/to/claude-author-wrapper'
export MENAGERIE_CHECKER_COMMAND='/absolute/path/to/codex-checker-wrapper'
export MENAGERIE_ENVIRONMENT_COMMAND='/absolute/path/to/exact-lock-wrapper'
python -m menagerie.crawler doctor --target osx-arm64 --strict
```

The crawler branch is the only branch for campaign records. The crawler never pushes. An operator
may pull with `git pull --ff-only`, but publishing remains a separate human action.

## Exact locks

Each intent starts from its readable `environment.yml`. During setup, the configured lifecycle
tool solves the exact lock on the actual target, captures artifact URLs and hashes, creates the
environment from that lock, captures the resolved export and its hash, and runs declared probes.
Locks are target-solved at SETUP. They are never hand-authored, guessed, copied from another
platform, or shipped with invented hashes. A changed lock, export, toolchain, or probe contract
creates a new environment generation.

## Network and execution boundaries

There are exactly three network postures:

1. Agent research is web-enabled. Only the Claude author lane uses WebSearch and Exa to locate and
   ground sources.
2. Controlled fetch is pinned-URL only. `fetcher.py` retrieves exact URLs and revisions into the
   local content-addressed store and verifies hashes.
3. Model execution is offline. Worker subprocesses receive no research tools or network access.

Offline execution uses a credential-scrubbed environment, offline framework flags, a socket
tripwire, write auditing outside the per-attempt scratch root, read-only source/environment inputs,
and a static and dynamic TorchLens-import ban. A policy observation blocks a run award. The doctor
self-tests the socket and write tripwires; it does not claim VM or fail-closed host-firewall
isolation.
