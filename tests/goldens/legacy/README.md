# Artifacts saved by torchlens 2.x (refusal inputs)

Torchlens 3 reads no file saved by torchlens 2.x. Each entry below is one small real artifact
from one format family; `tests/test_load_refuses_saved_by_2x.py` loads every one and checks the
single typed refusal. They are never regenerated: they ARE the old formats.

| Entry | Family | Stamp |
|---|---|---|
| `tlspec_v1.tlspec` | unified directory written by torchlens 2.17.0 (the io v3 sample) | 1 |
| `tlspec_v6_trace.tlspec` | analysis trace, pre-columnar baseline writer | 6 |
| `tlspec_v6_runnable.tlspec` | runnable trace (weights and activations payload) | 6 |
| `tlspec_v7.tlspec` | unified directory, capture-outcome era | 7 |
| `tlspec_v9.tlspec` | unified directory, torchlens 2.34.1 | 9 |
| `modellog_v2_16_genuine.tlspec` | the genuine 2.16 portable ModelLog format (`io_format_version` 2); manifest only, which is all the refusal reads | none |
| `intervention_v2_16.tlspec` | intervention spec of torchlens 2.16 (`format_version` 1 in `spec.json`; torchlens 2.29 began writing 2) | none |
| `extraction_v1/` | dataset-extraction artifact of torchlens 2.34 (`schema` `tl_extract_manifest_v1`); manifest only, which is all the refusal reads | none |

The 2.16 ModelLog bundle with modern spellings stays at
`tests/fixtures/tlspec_v2_16/F2_modellog_tiny_cnn.tlspec`, which `tests/test_tlspec_unified.py`
also uses.
