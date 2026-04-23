# TorchLens I/O Architecture

Reference plan: `.project-context/plans/io-sprint/plan.md`

## Bundle Layout

Portable save writes a directory bundle, not a tar archive:

```text
<bundle_dir>/
  manifest.json
  metadata.pkl
  blobs/
    0000000001.safetensors
    0000000002.safetensors
    ...
```

- `manifest.json`: authoritative index, version policy input, and blob metadata.
- `metadata.pkl`: scrubbed `ModelLog` state with tensor payloads replaced by refs.
- `blobs/`: one safetensors file per persisted tensor, keyed by zero-padded blob id.

## Manifest Schema v1

Top-level fields:

```text
io_format_version: int
torchlens_version: str
torch_version: str
python_version: str
platform: str
created_at: str
bundle_format: "directory"
n_layers: int
n_activation_blobs: int
n_gradient_blobs: int
n_auxiliary_blobs: int
tensors: list[TensorEntry]
unsupported_tensors: list[dict[str, str]]
```

`TensorEntry` fields:

```text
blob_id: str
kind: str
label: str
relative_path: str
backend: "safetensors"
shape: list[int]
dtype: str
device_at_save: str
layout: str
bytes: int
sha256: str
```

Supported `kind` values this sprint: `activation`, `gradient`, `captured_arg`,
`child_version`, `rng_state`, `module_arg`, `func_config`.

## Tensor Policy Matrix

| Tensor kind | `strict=True` | `strict=False` |
| --- | --- | --- |
| Dense CPU/CUDA tensors with supported dtypes | Save | Save |
| Non-contiguous tensors | Save after `.contiguous()` | Save after `.contiguous()` |
| Sparse COO/CSR/CSC/BSR/BSC tensors | Raise `TorchLensIOError` | Skip and record in `unsupported_tensors` |
| Quantized tensors | Raise `TorchLensIOError` | Skip and record in `unsupported_tensors` |
| Nested tensors | Raise `TorchLensIOError` | Skip and record in `unsupported_tensors` |
| Tensor subclasses | Raise `TorchLensIOError` | Skip and record in `unsupported_tensors` |
| DTensor/FSDP shard tensors | Raise `TorchLensIOError` | Skip and record in `unsupported_tensors` |
| `complex32` tensors | Raise `TorchLensIOError` | Skip and record in `unsupported_tensors` |
| Meta tensors | Raise `TorchLensIOError` | Skip and record in `unsupported_tensors` |
| Non-CPU/CUDA device tensors | Raise `TorchLensIOError` | Skip and record in `unsupported_tensors` |

## Version Policy

| Scenario | Load behavior |
| --- | --- |
| `io_format_version > current` | Raise `TorchLensIOError` before opening `metadata.pkl` |
| `io_format_version < current` | Load with `DeprecationWarning` listing missing fields |
| `io_format_version == current` | Load normally |
| `torch_version` major mismatch | Raise `TorchLensIOError` before opening `metadata.pkl` |
| `torch_version` minor mismatch | Load with `UserWarning` |
| `torchlens_version` newer than runtime | Load with `UserWarning` |
| `torchlens_version` older than runtime | Load with info log only |
| `python_version` major mismatch | Attempt load; wrap pickle errors in `TorchLensIOError` with a version hint |
| Safetensors backend missing | Raise `TorchLensIOError` with install hint |
| Manifest unparseable | Raise `TorchLensIOError` |
| Manifest blob id missing on disk | Raise `TorchLensIOError` listing missing blobs |
| Blob checksum mismatch | Raise `TorchLensIOError` naming the blob id |
| Unknown extra files in `blobs/` | Warn and continue |

## Fork Callouts

- Fork L: portable bundles are for archival, analysis, and sharing. Portable-loaded
  logs do not support `validate_forward_pass()` or replay-oriented validation.
- Fork M: `torchlens.load(..., lazy=True, materialize_nested=False)` leaves nested
  `BlobRef` values in place. Call `torchlens.rehydrate_nested(model_log)` before
  resaving or expecting nested tensors to behave like normal tensors.
- Fork H: lazy resave uses a two-level drift guard. TorchLens records the source
  manifest sha256 at load time, then rechecks the manifest and each copied blob
  sha256 before fast-copying lazy refs into a new bundle.
- Fork K: lazy refs do not keep shared readers open. Each materialization opens,
  verifies, reads, and closes its blob file independently.

## Operational Shape

The current format intentionally creates many small files. For models with more
than 10,000 saved layers, or for bundles stored on network filesystems, this can
be slower than a chunked archive format. That tradeoff is explicit for this
sprint: random-access lazy loads were prioritized over file-count optimization.

See the sprint plan for deeper rationale, deferred work, and the full fork
history: `.project-context/plans/io-sprint/plan.md`.
