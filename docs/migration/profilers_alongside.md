# Profilers to use alongside TorchLens

TorchLens records what a forward computed: every op, its inputs and outputs, shapes,
bytes and the module that ran it. Three tools measure things TorchLens does not, and
they work well next to it:

- **Holistic Trace Analysis (HTA)** reads PyTorch profiler (Kineto) traces, across ranks,
  and reports where GPU time goes: kernel time by category, idle time, overlap of
  communication and computation, and kernel launch statistics. TorchLens's
  `torchlens.bridge.profiler.join(trace, kineto_trace)` lands that same profiler trace's
  timing on TorchLens's named ops, so the two views share one trace file.
- **Mosaic** analyzes CUDA memory snapshots (from
  `torch.cuda.memory._record_memory_history`): it breaks peak memory down by allocation
  stack and annotation, and compares snapshots. TorchLens reports activation bytes per op
  of a captured forward, not allocator behavior.
- **torchprofile** counts a model's MACs from a `torch.jit.trace` graph with one call,
  `profile_macs(model, args)`. TorchLens counts FLOPs per executed op of an eager run
  (`torchlens.report.flops_report`).
