# Target-solved environment locks

Exact `*.lock`, `*.resolved.json`, and `*.resolved.sha256` files are setup-time outputs. The lifecycle
solver must run on the actual `osx-arm64` or `linux-x86_64-cuda` target, capture exact artifact URLs and
hashes, create the environment from that lock, and pass all declared probes. These files are never
hand-authored, guessed, or generated on a different platform.

Until that on-target setup occurs, each intent intentionally reports an `unlocked` status and no
environment generation. A changed lock, resolved export, or probe contract creates a new generation.
