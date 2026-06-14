# Op __slots__ retained-memory baseline

| Model | n_ops | Before B/op | After B/op | Reduction % | Before total | After total |
|---|---:|---:|---:|---:|---:|---:|
| small_mlp | 7 | 52,565.7 | 46,899.4 | 10.8 | 367,960 | 328,296 |
| small_cnn | 19 | 37,970.1 | 32,158.3 | 15.3 | 721,432 | 611,008 |
| small_transformer_block | 10 | 47,899.2 | 42,517.6 | 11.2 | 478,992 | 425,176 |

## Method

Each row traces an inline deterministic PyTorch fixture with `tl.trace(model, input)` and measures retained memory for the resulting full TorchLens trace object. The before state is the parent of the Op `__slots__` commit; the after state is the current checkout. Byte counts use the same script and environment for both states.

## Environment

- Before checkout: `/tmp/preslots`
- After checkout: `/home/jtaylor/projects/torchlens`
- Python: `3.11.6`
- Torch: `2.8.0+cu128`
- TorchLens before: `2.18.0`
- TorchLens after: `2.18.0`
- Sizing method: `pympler.asizeof`
- Platform: `Linux-5.15.0-139-generic-x86_64-with-glibc2.31`
