# Contributing to TorchLens

TorchLens records a PyTorch forward pass into `Trace`, `Op` and `Layer` objects. This page
is the shortest path from a clone to a change that passes review. The maintainer pages
under `docs/maintainers/` hold the longer form: the
[architecture map](docs/maintainers/architecture.md), the generated
[where-does-X-live index](docs/maintainers/where.md), the
[testing philosophy and tiers](docs/maintainers/testing.md) and the
[review checklist](docs/maintainers/review_checklist.md).

## Setup

```bash
git clone https://github.com/johnmarktaylor91/torchlens
cd torchlens
pip install -e ".[dev,test,tabular,viz]"   # the extras CI installs
```

Graphviz rendering needs the Graphviz binaries (`apt install graphviz` on Debian and
Ubuntu). Optional extras gate the appliance and bridge namespaces; `pyproject.toml` lists
them. Supported torch versions are 2.1 and later for eager capture.

To see what your environment supports, run the environment report:

```python
import torchlens as tl

print(tl.utils.doctor())     # installed extras, torch capability flags, degraded probes
print(tl.compat.report())    # the compatibility view of the same facts
```

Include its output in any bug report about capture or torch compatibility.

## Making a change

1. Read `AGENTS.md` and the reference pages it names for the area you touch; the
   [architecture map](docs/maintainers/architecture.md) tells you which package that is.
2. Work on a `wip/` branch in its own worktree. Commit named paths only
   (`git commit --only -m "type: description" -- <files>`), with a conventional type
   (`feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`), no `!` marker and no
   breaking-change footer. Never touch the version in `pyproject.toml` or
   `torchlens/__init__.py`.
3. Add tests by the admission rule in [testing.md](docs/maintainers/testing.md): a behaviour
   row, a regression row that fails before the fix, a corpus entry or a property test.
4. Run the gates below and go through the [review checklist](docs/maintainers/review_checklist.md).

## Gates

The tiers are declared in `test-tiers.json`; run them in this order.

```bash
ruff format .
ruff check . --fix
mypy torchlens/
pytest tests/<files for the code you touched> -x --tb=short   # per step
python scripts/smoke_ci_parity.py -n 4                        # commit gate, CI's pinned environment
pytest tests/ -m "not rare and not slow" -x --tb=short        # before merging a boundary or API change
```

Byte goldens enforce only in the pinned environment (python 3.10, torch 2.13.0), so the
parity script is the commit gate, not a plain smoke run in your own environment. Smoke is
about 1,400 coverage-chosen tests under a fixed size ceiling; a new smoke test needs a
matching demotion. Run `-n N` with `N` equal to the cores you have, never `-n auto`.

## The lockstep rule

The glossary (`docs/reference/glossary.md`) is the canonical API specification and code
conforms to it. Renaming, adding or removing any public name (a dataclass field, a property,
a method, a top-level `tl.*` name, a keyword argument) updates, in the same change, the
glossary entry, the examples under `docs/agent-reference/`, and the audit notebooks under
`notebooks/audit/` and `examples/` that use it. A change that leaves them stale is
incomplete. `python tools/name_lockstep check --base ref:<base-branch>` lists every new,
renamed and removed public name against a base and exits 1 for each missing glossary entry,
missing reference mention or stale old spelling.

## Validation integrity

The `validation/` pipeline (forward replay, backward checks, metadata invariants) is a
tripwire whose purpose is to catch capture bugs. Never weaken, loosen, exempt, broaden a
tolerance or skip a validation check to make a test pass: a validation failure is the system
working, so find and fix the capture bug. The only legitimate exemption is behaviour that is
correct by design and provably outside the check's contract, and even then the carve-out is
narrow and still fails on the unintended case.

## Public repository

This repository is public. Internal notes, plans, sprint records and run logs are never
committed; a pre-commit hook refuses the private paths and must not be bypassed. Tracked
files name no private machine, folder or credential.

## Where to file

- Bugs and feature requests: https://github.com/johnmarktaylor91/torchlens/issues, with the
  environment report above, the torch version and a minimal model.
- Security findings about saved-file loading: open an issue marked security, or email the
  maintainer listed in `pyproject.toml`, before publishing details.
