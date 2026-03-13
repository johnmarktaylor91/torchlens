# TorchLens TODO

- Add a first-class "undecorate / suspend decoration" escape hatch so users can
  restore a clean PyTorch environment when tracing is not needed.
- Explore decorating on the first `log_forward_pass(...)` instead of at import
  time, while still leaving torch decorated afterward once the user opts in.
