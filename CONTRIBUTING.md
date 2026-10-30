# Contributing to TorchLens

Thanks for helping. Bug reports, docs fixes and pull requests are all welcome.

## Questions and bug reports

- Questions about how to do something: [GitHub Discussions](https://github.com/johnmarktaylor91/torchlens/discussions/categories/q-a).
- Bugs: open an issue with the bug report form. Include a minimal script and the output of
  `python -c "import torchlens as tl; print(tl.utils.doctor())"`.
- Security problems: report them privately, as described in [SECURITY.md](SECURITY.md).

## Set up a development install

```bash
git clone https://github.com/johnmarktaylor91/torchlens
cd torchlens
pip install -e ".[dev,test,tabular,viz]"
```

Graph rendering needs the Graphviz system package (`apt install graphviz` on Debian and
Ubuntu, `brew install graphviz` on macOS). On Debian forky and Ubuntu 26.04 or later, also
install `libgvplugin-neato-layout8`.

## Make a change

1. Branch from `main`.
2. Write or update a test that fails without your change.
3. Run the checks below.
4. Open a pull request that says what changed and how you tested it.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
`fix: ...`, `feat: ...`, `docs: ...`, `test: ...`, `refactor: ...`, `chore: ...`.

## Checks

```bash
ruff format .
ruff check . --fix
mypy torchlens/
pytest tests/<files for the code you touched> -x --tb=short
pytest tests/ -m smoke -n 4
```

The test tiers, their markers and their time budgets are described in
[tests/AGENTS.md](tests/AGENTS.md). In short: run the test files for the code you changed
while you work, the `smoke` tier before you push, and
`pytest tests/ -m "not rare and not slow"` when you change a public API or a module
boundary. A few byte-exact golden tests only pass in CI's pinned environment
(Python 3.10, torch 2.13.0); `python scripts/smoke_ci_parity.py -n 4` builds that
environment and runs the smoke tier in it.

## Rules reviewers check

- **Validation checks are not loosened to make a test pass.** The validation pipeline
  (forward replay, backward checks, metadata invariants) exists to catch capture bugs. If
  it fails, find and fix the cause. Do not widen a tolerance, add an exemption or skip the
  check.
- **Docs change with the code.** Renaming, adding or removing a public name (a `tl.*`
  function, a method, a property, a field or a keyword argument) updates
  [docs/reference/glossary.md](docs/reference/glossary.md), the examples in
  `docs/agent-reference/`, and any notebook or example that uses it, in the same pull
  request.
- **Check torch capabilities in one module.** Code that depends on a torch
  version checks for the capability in `torchlens/utils/_torch_compat.py`, not on
  `torch.__version__`.
- **No new top-level names without discussion.** New features go in a submodule first.
- **Public claims need evidence.** A sentence in the README or docs that states a number,
  a comparison or an "every" or "any" claim needs a row in
  `docs/migration/_claims_ledger.json` pointing at the test or artifact behind it.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
