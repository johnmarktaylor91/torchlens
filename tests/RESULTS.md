# Test Results

Public summary of TorchLens test suite outcomes. Updated after each release.

**Last updated**: v0.15.14 · 2026-03-06 · PyTorch 2.8 · CPU + CUDA

---

## Suite Overview

| Metric | Value |
|--------|-------|
| Total tests | 888 |
| Smoke tests (`-m smoke`) | 18 |
| Test files | 14 |
| Example models (toy) | 247 |
| Real-world models | 183 |

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
| test_toy_models.py | 248 | API coverage on 247 example models (log, validate, visualize, metadata) |
| test_real_world_models.py | 183 | Real-world architectures: validation + visualization |
| test_metadata.py | 107 | Field invariants, FLOPs, timing, RNG, func_call_location, corruption detection |
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

### Toy Models (247 architectures)

All 247 example models in `tests/example_models.py` pass `validate_forward_pass`.

**Core patterns:** simple feedforward (incl. LeNet-5), branching, conditionals,
48 loop/recurrence variants, in-place ops, view mutations, edge cases.

**Attention variants:** multi-head, multi-query (MQA), grouped-query (GQA), RoPE,
ALiBi, slot attention, cross-attention (Perceiver-style), axial attention,
CBAM (channel+spatial), scaled dot-product, transformer encoder/decoder,
embedding+positional, differential attention (noise cancellation),
relative position bias (T5-style), coordinate attention (factorized H/W),
efficient channel attention (ECA, 1D conv), linear attention (kernel-based,
no softmax).

**Gating & skip patterns:** highway network, squeeze-and-excitation, depthwise
separable conv, inverted residual (MobileNetV2), feature pyramid network (FPN),
residual blocks, shared-param branching, channel shuffle (ShuffleNet-style).

**Generative & self-supervised:** VAE, hierarchical VAE, VQ-VAE, beta-VAE, CVAE,
Gumbel-Softmax VQ, GAN (generator + discriminator), diffusion, normalizing flow,
WaveNet-style gated convolutions, PixelCNN masked convolutions, SimCLR contrastive,
BYOL-style stop-gradient, Barlow Twins (cross-correlation), adaptive instance
normalization (AdaIN).

**Sequence models:** GRU, BiLSTM (bidirectional), seq2seq with Bahdanau attention.

**Exotic architectures:** hypernetwork (weight generation), deep equilibrium model
(DEQ, fixed-point iteration), neural ODE (Euler integration), NTM-style memory
augmented network, end-to-end memory network (multi-hop), SwiGLU FFN, Fourier
mixing (FNet-style), Fourier Neural Operator (spectral convolution), spatial
transformer network, SIREN (sinusoidal activations), radial basis function
network (RBF), Perceiver (cross-attention bottleneck).

**Graph neural networks:** GCN, GAT, GraphSAGE, GIN, EdgeConv (DGCNN), graph
transformer, Chebyshev spectral GCN, E(n) equivariant GNN (EGNN).

**Dense prediction:** ASPP (multi-rate dilated convolution), ControlNet-style
parallel encoder with zero-conv injection.

**Architecture patterns:** MLP-Mixer, Siamese, triplet network (metric learning),
prototypical network (few-shot), capsule network, U-Net, TCN (temporal conv net),
super-resolution (PixelShuffle), PointNet, actor-critic, two-tower recommender,
deep & cross network (recommender), wide & deep (recommender), depth estimator,
dueling DQN, mixture of experts (MoE), RMS normalization, sparse/pruned networks,
early exit (multi-head), multi-scale parallel streams (HRNet-style), multi-task
(shared trunk + task heads), FiLM conditioning, partial convolution (inpainting),
Network in Network (1x1 conv + GAP), pixel shuffle upsampling.

**Autoencoders:** vanilla, convolutional, sparse, denoising, VQ-VAE, beta-VAE, CVAE.

**State space models:** simple SSM, selective SSM (Mamba-style), gated SSM, stacked SSM.

### Real-World Models

| Category | Models | Status |
|----------|--------|--------|
| **TorchVision Classification** | AlexNet, VGG16, ViT, GoogLeNet, ResNet50, ConvNeXt, DenseNet121, EfficientNet-B6, SqueezeNet, MobileNetV2, MobileNetV3, Wide ResNet, MNASNet, ShuffleNet, ResNeXt, RegNet, Swin-v2-B, MaxViT, Inception-v3 | 19/19 pass |
| **CORnet** | Z, S, R, RT | 4/4 pass |
| **timm (original)** | BEiT, GluonResNeXt, ECAResNet, MobileViT, ADV-Inception, CaiT, CoAT, ConViT, DarkNet, GhostNet, MixNet, PoolFormer, ResNeSt, EdgeNeXt, HardCoreNAS, SEMNASNet, XCiT, SEResNet | 18/18 pass |
| **timm (additional)** | HRNet, EfficientNetV2, LeViT, CrossViT, PVT-v2, Twins-SVT, FocalNet, Res2Net, gMLP, ResMLP, EVA-02 | 11/11 pass |
| **timm (set 3)** | ConvNeXt-v2, NFNet, DaViT, CoAtNet, RepVGG, ReXNet, PiT, Visformer, GC-ViT, EfficientFormer, FastViT, NesT, Sequencer2D, TResNet | 14/14 pass |
| **Audio (original)** | Conv-TasNet, Wav2Letter, HuBERT, Wav2Vec2, DeepSpeech, Conformer, Whisper-tiny | 7/7 pass |
| **Audio (additional)** | AST, CLAP, EnCodec, SEW, SpeechT5, VITS | 6/6 pass |
| **Audio (set 3)** | WavLM, Data2VecAudio, UniSpeech | 3/3 pass |
| **Language (original)** | LSTM, RNN, GPT-2, BERT, DistilBERT, ELECTRA, T5-small, BART, RoBERTa, Sentence-BERT | 10/10 pass |
| **Decoder-Only LLMs** | LLaMA, Mistral, Phi, Gemma, Qwen2, Falcon, BLOOM, OPT, OLMo, GPT-J, GPTBigCode, GPT-NeoX | 12/12 pass |
| **Encoder-Only (additional)** | ALBERT, DeBERTa-v2, XLM-RoBERTa, Funnel Transformer, CANINE, MobileBERT | 6/6 pass |
| **Encoder-Decoder (additional)** | Pegasus, LED, mBART, ProphetNet | 4/4 pass |
| **Efficient Transformers** | FNet, Nystromformer, BigBird, Longformer, Reformer | 5/5 pass |
| **State Space Models** | Mamba, Mamba-2, RWKV, Falcon-Mamba | 4/4 pass |
| **Mixture of Experts** | Mixtral, Switch Transformer, MoE (toy) | 3/3 pass |
| **Autoencoders** | ViT-MAE (ForPreTraining) | 1/1 pass |
| **Multimodal / Special** | Stable Diffusion (UNet), StyleTTS, QML, Lightning, CLIP, BLIP, ViT-MAE, SigLIP, BLIP-2 | 9/9 pass |
| **Vision Transformers (HF)** | DeiT, CvT, SegFormer, DINOv2 | 4/4 pass |
| **Perceiver** | Perceiver IO | 1/1 pass |
| **Segmentation** | DeepLab-v3 (ResNet50), DeepLab-v3 (MobileNet), LRASPP, FCN-ResNet50 | 4/4 pass |
| **Detection (original)** | Faster R-CNN (train+eval), FCOS (train+eval), RetinaNet (train+eval), SSD300 (train+eval) | 8/8 pass |
| **Detection (additional)** | DETR, Mask R-CNN (train+eval), Keypoint R-CNN (train+eval), Deformable DETR | 6/6 pass |
| **Quantized** | ResNet50 (quantized) | 1/1 pass |
| **Video** | R(2+1)D-18, MC3-18, MViT-v2-S, R3D-18, S3D | 5/5 pass |
| **Optical Flow** | RAFT-Small, RAFT-Large | 2/2 pass |
| **Time Series** | PatchTST, Informer, Autoformer, TimeSeriesTransformer | 4/4 pass |
| **Reinforcement Learning** | Decision Transformer | 1/1 pass |
| **Graph Neural Networks** | DimeNet, GraphSAGE (PyG), GIN (PyG), Graph Transformer (PyG), GATv2 (PyG), R-GCN (PyG), ChebConv (PyG), SGConv (PyG), TAGConv (PyG) | 9/9 pass |
| **Document Understanding** | LayoutLM | 1/1 pass |
| **Other** | Taskonomy | 1/1 pass |
| | **Total** | **183/183 pass** |

*Tests requiring optional packages (torch_geometric, taskonomy) may show as SKIPPED.*

---

## Computational Pattern Coverage

The test suite explicitly covers these distinct computational motifs:

| Pattern | Toy Model(s) | Real-World Model(s) |
|---------|-------------|---------------------|
| Pure sequential | SimpleFF, VGG-style | VGG16, AlexNet |
| Additive skip (residual) | ResidualBlockModel, InvertedResidualBlock | ResNet50, Wide ResNet |
| Concatenation skip | SmallUNet, FeaturePyramidNet | DenseNet121, U-Net |
| Branching + merging | SimpleBranching, SqueezeExcitationBlock | GoogLeNet, Inception-v3 |
| Shared weights | SimCLRModel, SiameseNetwork | ALBERT (cross-layer sharing) |
| Recurrence / loops | LSTM, RNN, 48 loop variants | LSTM, RNN, RWKV |
| Multi-head attention | MultiheadAttentionModel | BERT, GPT-2, ViT |
| Multi-query attention (MQA) | MultiQueryAttentionModel | Falcon |
| Grouped-query attention (GQA) | GroupedQueryAttentionModel | LLaMA, Mistral, Gemma, Qwen2 |
| Rotary position embedding (RoPE) | RoPEAttentionModel | LLaMA, Mistral, Gemma |
| ALiBi positional encoding | ALiBiAttentionModel | BLOOM |
| Slot attention | SlotAttentionModel | — |
| Cross-attention (Perceiver) | CrossAttentionModel | Perceiver IO, DETR |
| Dynamic graph (MoE routing) | MoEModel | Mixtral, Switch Transformer |
| Reparameterization (VAE) | SimpleVAE, HierarchicalVAE | ViT-MAE |
| External memory (NTM) | MemoryAugmentedNet | — |
| ODE integration | SimpleNeuralODE | — |
| Fixed-point iteration (DEQ) | DEQModel | — |
| Stop-gradient (BYOL-style) | StopGradientModel | — |
| In-place operations | InPlaceFuncs, InPlaceChainModel | — |
| Custom autograd | CustomAutogradModel | — |
| Sparse operations (scatter/gather) | ScatterGatherModel, SparsePrunedModel | GNN models |
| FFT operations | FourierMixingModel | FNet |
| Masked convolutions | MaskedConvModel | — |
| Weight generation (hypernetwork) | HyperNetwork | — |
| SwiGLU gated FFN | SwiGLUFFN | LLaMA, Mistral |
| Spatial transform | SpatialTransformerNet | — |
| Multi-scale FPN | FeaturePyramidNet | Faster R-CNN, Mask R-CNN |
| Depthwise separable conv | DepthwiseSeparableConv | MobileNetV2, EfficientNet |
| Channel attention (SE) | SqueezeExcitationBlock | SEResNet, EfficientNet |
| Highway gating | HighwayNetwork | — |
| Graph message passing | SimpleGCN, GraphSAGEModel, GINModel | DimeNet, GraphSAGE-PyG |
| Graph attention | SimpleGAT, GraphTransformerModel | Graph Transformer (PyG) |
| Sliding window attention | — | Mistral, Longformer |
| Disentangled attention | — | DeBERTa-v2 |
| Nystrom attention approximation | — | Nystromformer |
| LSH attention (reversible) | — | Reformer |
| Axial attention | AxialAttentionModel | — |
| Channel+spatial attention (CBAM) | CBAMBlock | — |
| Bidirectional RNN | BiLSTMModel, BidirectionalGRUModel | — |
| Seq2seq + Bahdanau attention | Seq2SeqWithAttention | — |
| Triplet (metric learning) | TripletNetwork | — |
| Cross-correlation (Barlow Twins) | BarlowTwinsModel | — |
| Feature crossing (DCN) | DeepCrossNetwork | — |
| Hierarchical residual (Res2Net) | — | Res2Net |
| Spatial gating MLP | — | gMLP, ResMLP |
| Contrastive audio-text | — | CLAP |
| Neural audio codec (RVQ) | — | EnCodec |
| TTS (flow + adversarial) | — | VITS |
| Audio spectrogram ViT | — | AST |
| Relational message passing | — | R-GCN (PyG) |
| Dynamic graph attention (GATv2) | — | GATv2 (PyG) |
| Keypoint detection | — | Keypoint R-CNN |
| Time series decomposition | — | Autoformer, Informer |
| Self-supervised ViT | — | DINOv2 |
| GRU (gated recurrent) | GRUModel | — |
| Network in Network (1×1 conv + GAP) | NiNModel | — |
| Channel shuffle (group conv) | ChannelShuffleModel | ShuffleNet |
| Pixel shuffle upsampling | PixelShuffleModel | — |
| Partial convolution (inpainting) | PartialConvModel | — |
| FiLM conditioning (affine modulation) | FiLMModel | — |
| Coordinate attention (factorized H/W) | CoordinateAttentionModel | — |
| Differential attention (noise cancel) | DifferentialAttentionModel | — |
| Relative position bias (T5-style) | RelativePositionAttentionModel | — |
| Early exit (multi-head) | EarlyExitModel | — |
| Multi-scale parallel streams (HRNet) | MultiScaleParallelModel | HRNet |
| Gumbel-Softmax VQ | GumbelVQModel | — |
| Multi-hop memory network | EndToEndMemoryNetwork | — |
| Radial basis function (RBF) | RBFNetwork | — |
| Sinusoidal activations (SIREN) | SIRENModel | — |
| Multi-task (shared trunk + heads) | MultiTaskModel | — |
| Wide & deep (recommender) | WideAndDeepModel | — |
| Chebyshev spectral GCN | ChebGCN | ChebConv (PyG) |
| Prototypical network (few-shot) | PrototypicalNetwork | — |
| Efficient channel attention (ECA) | ECAModel | — |
| Parallel attention + FFN | — | GPT-J |
| Sequence reduction (funnel) | — | Funnel Transformer |
| Character-level tokenization | — | CANINE |
| Bottleneck BERT | — | MobileBERT |
| N-gram prediction | — | ProphetNet |
| Deformable attention | — | Deformable DETR |
| Document layout embeddings | — | LayoutLM |
| Normalizer-free training | — | NFNet |
| Reparameterizable convolution | — | RepVGG |
| Sigmoid contrastive loss | — | SigLIP |
| Q-Former (querying transformer) | — | BLIP-2 |
| Spectral graph convolution (Chebyshev) | — | ChebConv (PyG) |
| Simple graph convolution (SGC) | — | SGConv (PyG) |
| Topology adaptive graph conv | — | TAGConv (PyG) |
| LSTM spatial mixing | — | Sequencer2D |
| Pooling-based ViT | — | PiT |
| Dual attention (spatial+channel) | — | DaViT |
| Linear attention (kernel-based) | LinearAttentionModel | RWKV |
| Spectral convolution (FNO) | SimpleFNO | — |
| Cross-attention bottleneck (Perceiver) | PerceiverModel | Perceiver IO |
| Multi-rate dilated conv (ASPP) | ASPPModel | DeepLab-v3 |
| Parallel encoder + zero-conv (ControlNet) | ControlNetModel | — |
| E(n) equivariant message passing | SimpleEGNN | DimeNet |

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
