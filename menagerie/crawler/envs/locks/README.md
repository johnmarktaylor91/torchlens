# Target-solved environment locks

Exact `*.lock`, `*.resolved.json`, and `*.resolved.sha256` files are target-solve outputs. The release
lock family also includes solver provenance and an observed probe receipt. Locks capture exact artifact
URLs and SHA-256 hashes, and must clean-create an installed inventory byte-identical to the committed
resolved export before they can be committed. These files are never hand-authored or guessed.

`round19-linux-64.*` is the genuine Linux release family solved and clean-validated on Linux from
`../specs/round21-release.yml`. `round19-osx-arm64.*` is the genuine macOS release family
cross-solved on Linux with `conda-lock` and `../specs/round21-release.virtual-packages.yml`.
CI materializes both with `menagerie.crawler.tools.release_lock`, which checks every downloaded archive
SHA-256 and the complete installed inventory. The release fixture consumes these committed artifacts
and reruns the committed probe contract; it never derives release provenance from the live prefix.
Both targets include committed probe receipts from native validation. The macOS receipt was produced
on Apple Silicon with the crawler environment operator against the committed lock-built prefix.

From the repository root, regenerate the macOS receipt with the production command adapter and
canonical serializer:

```bash
CRAWLER_PY=/Users/jmt/projects/torchlens/.venv-crawler/bin/python
MACOS_PREFIX=/Users/jmt/.crawler-envs/round19-osx-arm64
"$CRAWLER_PY" -c 'import json,sys; from pathlib import Path; from menagerie.crawler.driver_admission import CommandEnvironmentBackend; from menagerie.crawler.env_lifecycle import canonical_probe_receipt_bytes; from menagerie.crawler.envs import ExportCheck,IntentProbes; contract_path,prefix_path,receipt_path=map(Path,sys.argv[1:]); raw=json.loads(contract_path.read_bytes()); probes=IntentProbes(tuple(raw["imports"]),tuple(ExportCheck(str(row["module"]),str(row["attribute"])) for row in raw["export_checks"]),()); backend=CommandEnvironmentBackend((sys.executable,"-m","menagerie.crawler.operator_environment")); results=tuple(backend.probe(prefix_path,probes)); receipt_path.write_bytes(canonical_probe_receipt_bytes(results))' "$PWD/menagerie/crawler/envs/specs/round21-release.probes.json" "$MACOS_PREFIX" "$PWD/menagerie/crawler/envs/locks/round19-osx-arm64.probes.json"
```

Other intent families remain unlocked until their target setup occurs. Any changed lock, export,
provenance, or probe contract creates a new environment generation.
