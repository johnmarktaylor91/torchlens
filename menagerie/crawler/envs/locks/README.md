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
Linux includes a committed probe receipt from local validation. macOS probe observations are produced
only on hosted macOS release CI and retained in that job's pass attestation.

Other intent families remain unlocked until their target setup occurs. Any changed lock, export,
provenance, or probe contract creates a new environment generation.
