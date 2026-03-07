# Test Results

Public summary of TorchLens test suite outcomes. Updated after each release.

**Last updated**: v0.15.14 · 2026-03-06 · PyTorch 2.1 · CPU

---

## Suite Overview

| Metric | Value |
|--------|-------|
| Total tests | 736 |
| Smoke tests (`-m smoke`) | 18 |
| Test files | 14 |
| Example models (toy) | 139 |
| Real-world models | 92 |

**Run the suite:**
```bash
pytest tests/ -v                    # full suite
pytest tests/ -m smoke              # ~6s critical-path check
pytest tests/ -m "not slow"         # skip long-running tests
pytest tests/test_profiling.py -vs  # profiling report
```

---

## Test Files

| File | Tests | What it covers |
|------|------:|----------------|
| test_toy_models.py | 183 | API coverage on 139 example models (log, validate, visualize, metadata) |
| test_metadata.py | 107 | Field invariants, FLOPs, timing, RNG, func_call_location, corruption detection |
| test_real_world_models.py | 92 | Real-world architectures: validation + visualization |
| test_param_log.py | 70 | ParamLog, ParamAccessor, shared params, grad metadata |
| test_decoration.py | 61 | Toggle state, detached imports, pause_logging, JIT compat, signal safety |
| test_validation.py | 59 | Perturbation checks, metadata invariants, edge cases |
| test_module_log.py | 45 | ModuleLog, ModulePassLog, ModuleAccessor, module hierarchy |
| test_internals.py | 36 | Field order sync, safe_copy, internal algorithms |
| test_layer_log.py | 34 | LayerLog aggregates, multi-pass delegation, loop detection |
| test_save_new_activations.py | 21 | Fast re-logging, state reset, buffer handling |
| test_output_aesthetics.py | 12 | Visual report generation (PDF/TeX/text) |
| test_gc.py | 10 | GC correctness, memory leak detection, param ref release |
| test_profiling.py | 1 | Overhead benchmarks (generates profiling_report.txt) |
| test_arg_positions.py | 1 | ArgSpec lookup table coverage (runs last) |

---

## Model Compatibility

### Toy Models (139 architectures)

All 139 example models in `tests/example_models.py` pass `validate_forward_pass`.
Covers: simple feedforward, branching, conditionals, 48 loop/recurrence variants,
in-place ops, attention, view mutations, autoencoders (vanilla/conv/sparse/denoising/
VQ-VAE/beta-VAE/CVAE), state space models (simple/selective/stacked SSM), GCN, GAT,
MLP-Mixer, Siamese, diffusion, normalizing flow, capsule network, and edge cases.

### Real-World Models

| Category | Models | Status |
|----------|--------|--------|
| **TorchVision Classification** | AlexNet, VGG16, ViT, GoogLeNet, ResNet50, ConvNeXt, DenseNet121, EfficientNet-B6, SqueezeNet, MobileNetV2, Wide ResNet, MNASNet, ShuffleNet, ResNeXt, RegNet, Swin-v2-B, MaxViT, Inception-v3 | 18/18 pass |
| **CORnet** | Z, S, R, RT | 4/4 pass |
| **timm** | BEiT, GluonResNeXt, ECAResNet, MobileViT, ADV-Inception, CaiT, CoAT, ConViT, DarkNet, GhostNet, MixNet, PoolFormer, ResNeSt, EdgeNeXt, HardCoreNAS, SEMNASNet, XCiT, SEResNet | 18/18 pass |
| **Audio** | Conv-TasNet, Wav2Letter, HuBERT, Wav2Vec2, DeepSpeech, Conformer, Whisper-tiny | 7/7 pass |
| **Language** | LSTM, RNN, GPT-2, BERT, DistilBERT, T5-small, RoBERTa, Sentence-BERT | 8/8 pass |
| **State Space Models** | Mamba, Mamba-2, RWKV, Falcon-Mamba | 4/4 pass |
| **Autoencoders** | ViT-MAE (ForPreTraining) | 1/1 pass |
| **Multimodal / Special** | Stable Diffusion (UNet), StyleTTS, QML, Lightning, CLIP, BLIP, ViT-MAE | 7/7 pass |
| **Segmentation** | DeepLab-v3 (ResNet50), DeepLab-v3 (MobileNet), LRASPP, FCN-ResNet50 | 4/4 pass |
| **Detection** | Faster R-CNN (train+eval), FCOS (train+eval), RetinaNet (train+eval), SSD300 (train+eval) | 8/8 pass |
| **Quantized** | ResNet50 (quantized) | 1/1 pass |
| **Video** | R(2+1)D-18, MC3-18, MViT-v2-S, R3D-18, S3D | 5/5 pass |
| **Optical Flow** | RAFT-Small, RAFT-Large | 2/2 pass |
| **Other** | Taskonomy, DimeNet (GNN), MoE | 3/3 pass |
| | **Total** | **90/90 pass** |

*2 tests (Taskonomy, DimeNet) require optional packages and show as SKIPPED when unavailable.*

---

## Profiling Baselines

Overhead of `log_forward_pass` vs raw `model.forward()`. See `test_outputs/reports/profiling_report.txt` for full detail.

| Model | Params | Layers | Raw | log_forward_pass | Overhead |
|-------|-------:|-------:|----:|-------:|-------:|
| SimpleFF | <1K | 3 | <1ms | ~3ms | ~10x |
| ResidualBlock | 5K | 15 | <1ms | ~5ms | ~15x |
| MultiheadAttention | 4K | 40 | <1ms | ~15ms | ~30x |
| ResNet18 | 11.7M | ~70 | ~15ms | ~60ms | ~4x |
| MobileNetV2 | 3.4M | ~200 | ~25ms | ~100ms | ~4x |
| EfficientNet-B0 | 5.3M | ~350 | ~40ms | ~150ms | ~4x |
| VGG16 | 138M | ~40 | ~80ms | ~120ms | ~1.5x |

*Overhead is dominated by per-operation bookkeeping. Large models with fewer, heavier ops (VGG16) show lower relative overhead. Small models with many lightweight ops show higher relative overhead. All measurements on CPU.*

---

## Coverage

Run with `pytest tests/ --cov=torchlens --cov-branch` to generate detailed reports in `test_outputs/reports/`.

---

## Generating Reports Locally

```bash
# Profiling report
pytest tests/test_profiling.py -vs

# Visual aesthetics report
pytest tests/test_output_aesthetics.py -v

# Coverage (text + HTML)
pytest tests/ --cov=torchlens --cov-branch

# ArgSpec function usage coverage
pytest tests/ -v  # test_arg_positions runs last automatically
```

Reports are written to `tests/test_outputs/reports/` (gitignored).
