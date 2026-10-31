# Packaging matrix

`matrix.toml` is the one reviewed input: the named extras sets (none, each extra alone, `all`,
and the intentionally incompatible pairs), the interpreter per set, shared and per-extra pins,
and the typed dependency refusals each set must show.
`python tools/packaging/generate_constraints.py` writes `constraints/<set>.txt` (sorted, no
timestamps); `--check` exits 1 when they are stale; `--list-sets --group pr` prints the CI matrix.
`tests/packaging/` (markers `slow` and `packaging`) builds the wheel and sdist, installs each into
a fresh venv per set, and probes import, `tl.report.doctor()`, a trace, a save and load round
trip, and the refusals. Select sets with `TORCHLENS_PACKAGING_SETS` (names, `pr`, `extended`, `all`).
CI: the `packaging-sets` and `packaging-install` jobs in `.github/workflows/pr.yml`.
