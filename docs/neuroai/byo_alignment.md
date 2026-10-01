# Bring your own alignment (learned transforms over exported features)

Representational-alignment methods -- linear probes, Procrustes maps, and
learned transforms such as gLocal -- are analysis methods, not capture
machinery, and TorchLens deliberately does not reimplement them. What it
guarantees instead is the JOIN: a 2-D "stimuli x features" matrix whose row
order is tied to your stimulus ids by the artifact manifest, produced by one
recorded shaping operation, loadable in a process that imports neither
TorchLens nor the extraction environment.

## The file round trip

Extract once, on the TorchLens side:

```python
import torchlens as tl

tl.extract_dataset(
    model, images, ["avgpool"],
    output_dir="features_run",
    input_transform=pre,            # resolved authority: verified provenance
    stimulus_ids=image_ids,
)
```

Apply YOUR transform anywhere -- the shards are plain `torch.save` payloads
and the manifest is plain JSON, so the analysis side needs only torch (or
numpy after your own conversion):

```python
# analysis environment: no torchlens import required
import json, pathlib, torch

run = pathlib.Path("features_run")
manifest = json.loads((run / "manifest.json").read_text())
ids = json.loads((run / "stimulus_ids.json").read_text())["ids"]
rows = [torch.load(run / e["file"], weights_only=True)["avgpool"]
        for e in map(json.loads, (run / "ledger.jsonl").read_text().splitlines())]
features = torch.cat(rows).reshape(len(ids), -1)

aligned = features @ my_learned_transform    # gLocal weights, a probe, anything
```

## gLocal, credited

gLocal (Muttenthaler et al.; shipped with
[thingsvision](https://github.com/ViCCo-Group/thingsvision)) is the worked
instance this page is shaped around: a learned transform that aligns model
representations with human similarity judgments. Use THEIR implementation and
weights over the exported matrix above -- the projects pin very different
dependency sets, so the file hand-off is not a workaround, it is the honest
workflow: extraction runs in your model environment, alignment runs in
theirs, and the manifest's stimulus ids are the join key that keeps the rows
meaning what you think they mean.
