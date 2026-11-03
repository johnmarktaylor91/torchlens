<!-- generated from TorchLens P6 perf gate JSON; do not hand-edit numbers -->

Measured at SHA `f46f78e` on `2026-11-03`.

| Model | Device | Row | Median ms | vs raw forward | Status |
|---|---|---|---:|---:|---|
| tinynet | cpu | raw_forward | 2.1 | 1.00x | ok |
| tinynet | cpu | raw_tl_import | 1.6 | 0.76x | ok |
| tinynet | cpu | raw_global_wrapped | 1.7 | 0.80x | ok |
| tinynet | cpu | raw_target_prepared | 1.6 | 0.76x | ok |
| tinynet | cpu | raw_inference_mode | 1.7 | 0.79x | ok |
| tinynet | cpu | global_wrap_dummy | 1301.3 | 618.06x | ok |
| tinynet | cpu | first_capture_target | 1348.8 | 640.61x | ok |
| tinynet | cpu | tl_trace | 65.3 | 31.03x | ok |
| tinynet | cpu | tl_trace_profile | 68.9 | 32.74x | ok |
| tinynet | cpu | tl_rerun | 330.8 | 157.10x | ok |
| tinynet | cpu | fastlog_module | 29.7 | 14.10x | ok |
| tinynet | cpu | aux_save | 185.1 | 87.89x | ok |
| tinynet | cpu | aux_load | 106.3 | 50.51x | ok |
