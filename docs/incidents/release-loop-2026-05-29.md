---
name: incident-release-loop-2026-05-29
description: Runaway semantic-release loop (1599 commits) root cause + fix; LFS tag-push + missing skip-ci
metadata: 
  node_type: memory
  type: project
  originSessionId: 41945be0-4dae-41fe-bb6b-db977cb13d8f
---

2026-05-29: a `fix:` push to main lit a **1599-commit runaway** of `chore(release): 2.18.0`
bot commits (one every ~40s for ~15.5h). PyPI was unharmed (stayed 2.17.0) because every
release FAILED at the tag step. Two compounding latent bugs:

**Bug A (loop engine):** `.github/workflows/release.yml` runs on EVERY push to main, and the
bot's own `chore(release):` commit re-triggered it. No `[skip ci]` / actor exclusion.

**Bug B (loop fuse):** the version TAG push failed under git-lfs. Repo tracks a few
`.research/docs-plan-megasprint_PLAN*.md` via LFS (added after v2.17.0 -- that's why v2.17.0
released fine but the next release looped). The Release job authenticates as GitHub App
`torchlens-release`, whose push URL embeds username `torchlens-release[bot]`. The `[bot]`
brackets break git-lfs endpoint parsing: `batch request: missing protocol: "<unknown>"`.
Plain commit pushes tolerate it; the TAG push invokes LFS and dies -> no `v2.18.0` tag ->
semantic-release sees the same unreleased feats since the last TAG (v2.17.0) every run ->
re-releases forever.

**Fix (commits 0b24b6e + 056193d on main; 2.18.0 shipped clean):**
- `pyproject.toml` `commit_message = "chore(release): {version} [skip ci]"` (loop-proof; GitHub
  skips push-triggered workflows on `[skip ci]` head commits).
- `release.yml` job guard `if: ${{ !startsWith(github.event.head_commit.message, 'chore(release):') }}`
  (belt-and-suspenders).
- `release.yml` step before semantic-release: `git config lfs.allowincompletepush true` +
  `git config lfs.locksverify false`. (First tried `git lfs uninstall` -- it cleared the
  bracketed-URL "missing protocol" error but the tag push then hit "missing or corrupt local
  objects" because CI checks out lfs:false / pointers-only; the objects are already on the
  remote, so allowincompletepush lets the tag ref get created.) VALIDATED: next push shipped
  2.18.0 -> v2.18.0 tag + PyPI + GitHub release, single `[skip ci]` release commit, NO loop.

**Remediation done:** `gh workflow disable Release` to stop the bleeding; force-pushed clean
main (`--force-with-lease`, bypassed `non_fast_forward` ruleset as admin) to wipe the 1599
junk commits + de-bloat pyproject; tags v2.16.0/v2.17.0 intact; re-enabled Release.

**Carry-forward:**
- Force-push to main works for JMT's account (admin bypass actor id 5 + App integration on
  ruleset `main-updates`); the bot also bypasses branch protection (that's why bad commits
  reached main).
- PyPI was at 2.17.0 after cleanup; main has unreleased feats -> next push to main cuts 2.18.0
  and VALIDATES the LFS fix (untested in CI). If the LFS fix is imperfect it fails ONCE, no loop.
- Optional cleanup: those `.research/*.md` are markdown and probably should NOT be in LFS at all.
- Watcher lesson: a single `fix:` push to a repo with broken release tagging can runaway
  overnight -- check `git log HEAD..origin/main` count before integrating after any gap.
