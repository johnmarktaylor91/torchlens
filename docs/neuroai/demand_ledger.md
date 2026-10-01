# NeuroAI demand ledger

A dated, bounded record of what the NeuroAI feature-extraction audience has
actually asked for, mined from public issue trackers (thingsvision and the
adjacent THINGS-ecosystem tools), and where each demand lands in TorchLens.
Refreshed before roadmap revisions, not continuously. Last refresh:
2026-08-26 (panel review of the trackers below); this snapshot: 2026-08-28.

Reading the table: "status" is the demand's state upstream at refresh time;
"TorchLens answer" names the shipped surface or the deliberate non-goal with
its composition story.

| Source | Demand | Status upstream | TorchLens answer | Doc |
| --- | --- | --- | --- | --- |
| thingsvision #176, #180, #183 | bundled preprocessing recipes drifted from upstream (recurring bug class) | fixed individually, class recurs | no hosted recipes, by design: resolve the loader's OWN authority and verify against it (`torchlens.preprocessing`) | [loaders](loaders.md), [journey s2](journey.md) |
| thingsvision #181, #177, #18 | hosted model/asset sources rotted | intrinsic to hosting | no zoo, no hosted assets; two-line loading per ecosystem | [loaders](loaders.md) |
| thingsvision #24, #46 | recurrent-model support: per-timestep features from reused blocks | long-standing ask | pass-qualified sites are first-class (`"block:2"`); recurrent time-courses are one extraction | [journey showpiece](journey.md) |
| thingsvision #162 | CLI extraction (their two-verb CLI was also their buggiest surface) | shipped upstream, buggy | deferred: the site inventory returns structured rows so a CLI is a formatter, not a redesign | -- |
| thingsvision #67 | multi-GPU / sharded extraction at THINGS scale | open ask | out of this surface's scope; the artifact schema is versioned so a sharded writer extends rather than breaks readers | -- |
| rsatoolbox (usage pattern) | layer-wise RSA needs per-LAYER matrices, not final outputs | n/a | per-site `bridge.rsatoolbox.dataset(source, site=...)`; file and in-memory routes identical | [journey s5](journey.md) |
| Brain-Score / Net2Brain (usage pattern) | features as presentation-x-neuroid assemblies | n/a | `bridge.xarray.data_array` / `neuroid_assembly`; `bridge.brain_score` serves the extractor seam | [journey s5](journey.md) |
| gLocal users | apply learned alignment transforms to extracted features | shipped in thingsvision | bring-your-own-alignment file round trip; gLocal stays theirs, credited | [byo_alignment](byo_alignment.md) |

Non-goals confirmed against this ledger (each with its composition story):
model zoos and checkpoint downloaders (the loaders page is the two-line
answer), bundled matched preprocessing and ANY hosted constants table (the
verifier replaces it), stimulus/dataset management (`extract_dataset` takes
any iterable), new RSA/CKA math (rsatoolbox, Net2Brain, and Brain-Score own
that analysis layer -- see their documentation), and a
thingsvision-compatible wrapper facade (familiar workflow lives in the
[migration page](../migration/from_thingsvision.md); duplicate APIs drift).
