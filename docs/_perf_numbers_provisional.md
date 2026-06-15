<!-- generated from TorchLens P6 perf gate JSON; do not hand-edit numbers -->

Measured at SHA `a1df040` on `2026-06-14`.

| Model | Device | Row | Median ms | vs raw forward | Status |
|---|---|---|---:|---:|---|
| tinynet | cpu | raw_forward | 0.9 | 1.00x | ok |
| tinynet | cpu | raw_tl_import | 1.0 | 1.04x | ok |
| tinynet | cpu | raw_global_wrapped | 0.9 | 1.01x | ok |
| tinynet | cpu | raw_target_prepared | 0.9 | 1.02x | ok |
| tinynet | cpu | raw_inference_mode | 1.1 | 1.25x | ok |
| tinynet | cpu | global_wrap_dummy | 1660.7 | 1812.43x | ok |
| tinynet | cpu | first_capture_target | 344.0 | 375.45x | ok |
| tinynet | cpu | tl_trace | 38.8 | 42.32x | ok |
| tinynet | cpu | tl_trace_profile | 41.0 | 44.71x | ok |
| tinynet | cpu | tl_rerun | 49.4 | 53.96x | ok |
| tinynet | cpu | fastlog_module | 15.7 | 17.11x | ok |
| tinynet | cpu | aux_save | 29.0 | 31.67x | ok |
| tinynet | cpu | aux_load | 30.2 | 32.95x | ok |
