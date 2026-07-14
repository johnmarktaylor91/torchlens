# Mini quickstart

Run this from the TorchLens worktree on the Apple Silicon mini. The campaign branch is
`menagerie/crawler-pipeline`; crawler automation never pushes it.

```bash
git switch menagerie/crawler-pipeline
git pull --ff-only origin menagerie/crawler-pipeline
python3 -m venv .venv-crawler
source .venv-crawler/bin/activate
python -m pip install -e '.[dev,test]'
export MENAGERIE_PUBLIC_MIRROR=/absolute/path/to/public-mirror
export MENAGERIE_PRIVATE_MIRROR=/absolute/path/to/private-mirror
export MENAGERIE_AUTHOR_COMMAND='/absolute/path/to/claude-author-wrapper'
export MENAGERIE_CHECKER_COMMAND='/absolute/path/to/codex-checker-wrapper'
export MENAGERIE_ENVIRONMENT_COMMAND='/absolute/path/to/exact-lock-wrapper'
python -m menagerie.crawler doctor --target osx-arm64 --strict
INTAKE=$(python -m menagerie.crawler intake --all-existing | python -c 'import json,sys; print(json.load(sys.stdin)["path"])')
python -m menagerie.crawler plan --intake "$INTAKE" --target osx-arm64 --phase pytorch
python -m menagerie.crawler run --intake "$INTAKE" --target osx-arm64 --phase pytorch --sequential-envs
```

Strict doctor is the go/no-go check. Among its checks are the crawler branch, target, disk,
single-writer lock, both mirror roots, Claude Code, author WebSearch, Exa `web_search_exa` and
`web_fetch_exa`, secrets, offline/socket/write tripwires, wakeup support, and the TorchLens-import
ban. Fix every reported failure before pressing go.

The driver pauses after 1000 terminal models for JMT review and sign-off. After JMT approves the
generated review report, continue with:

```bash
python -m menagerie.crawler resume --intake "$INTAKE" --target osx-arm64 --phase pytorch --after-review
```

The later 2000, 3000, 5000, 10000, 15000, and 20000 milestones notify JMT and continue without
stopping. The 1000-model review is a one-shot blocking checkpoint.
