# TorchLens agent index (2.x)

A package for extracting activations from PyTorch models and saving portable bundles

## Core calls

```python
# (no core cards yet)
```

## Tasks

| Task | Call | Avoid | Tier | Page |
|---|---|---|---|---|
| Draw a reproducible subsample before extracting features | `tl.extract_dataset(model, stimuli[chosen], sites, output_dir="subsample", stimulus_ids=ids)` | Looking for sample_size= or seed= on tl.extract_dataset: it has neither, and the manifest records ids, not seeds. | kit | [Field guide: seeded_subsample](https://johnmarktaylor91.github.io/torchlens/2.x/field-guide/seeded_subsample/) |
