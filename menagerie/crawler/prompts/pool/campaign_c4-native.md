## CAMPAIGN c4-native -- native TensorFlow / Keras / JAX / Flax tail (author tier: sonnet)

~1,047 models whose real implementation is **not** PyTorch: 751 TensorFlow, 171 JAX/Flax,
125 Keras. They stay native.

### The rule that governs this campaign

**Do not port a native model to PyTorch.** A native model runs behind a transparent native
adapter that records its native object type and the method it delegates to. Record both
`original_framework` and `run_framework` honestly -- they will differ, and that difference
is data, not a defect.

A rewrite into PyTorch would silently change what the atlas claims to contain, and it is
exactly the "approximation instead of the real source" failure the project forbids.

### What to establish

- The exact native library and version that ships the architecture (`tf.keras.applications`,
  `keras_cv`, `flax.linen` model modules, `transformers`' Flax classes, a maintained
  research repo, and so on), plus the exact symbol.
- The native construction call with every pretrained/weights flag explicitly disabled --
  in Keras that usually means `weights=None`, in Flax it means constructing the module and
  initializing parameters with a seeded PRNG key rather than loading a checkpoint.
- The dummy call's real semantics in the **native** convention: channels-last input for
  TF/Keras, an explicit PRNG key and `init`/`apply` pair for Flax, and the framework's own
  training/inference mode flag. Do not silently translate a PyTorch idiom.
- The intent this model needs (`tf-keras-arm64`, `jax-flax-arm64`, `paddle-arm64`). If the
  assigned intent's environment cannot import it, say precisely which packages are
  missing; that finding drives the requeue.

### Do not

- Do not port, translate, or "equivalently reimplement" into PyTorch.
- Do not convert weights or seek checkpoints; construction is random-initialized.
- Do not escalate in place. This campaign is frozen to sonnet; a genuinely hard row gets a
  typed `BLOCKED` for requeue into `c3-classics`.

### Budget shape

Target roughly 6 minutes. The common real difficulty here is version skew in the native
stack, not architectural ambiguity -- pin versions precisely and the rest usually follows.
