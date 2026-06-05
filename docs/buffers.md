# Buffers

PyTorch registered buffers are tensors that belong to a module but are not
parameters. BatchNorm running statistics are the common example, but buffers
also show up as recurrent state, masks, counters, cached constants, and custom
module state registered with `module.register_buffer(...)`.

TorchLens treats registered buffers as model state with versions. A buffer read
or write is not a separate compute operation. Instead, the graph contains one
plain `Op` node for each observed buffer version, marked with `is_buffer=True`.
Edges carry the read/write relationship:

```text
producer op -> buffer version -> reader op
```

The persistent `Buffer` object exposed through `trace.buffers` is a projection
over those graph nodes. It is a sibling of `Module` and `Param` metadata, not a
subclass of `Op`. The graph nodes keep per-version fields such as `out`,
`parents`, `children`, `buffer_write_kind`, and `buffer_value_changed`; the
`Buffer` entity keeps address-level state such as `initial_value`,
`final_value`, `versions`, and `num_overwrites`.

## Version Model

Each buffer address has an initial value captured before the forward pass. Every
write creates a new graph version node. The final value is the last observed
version at the end of capture.

For a recurrent state buffer:

```python
class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("h", torch.zeros(2))

    def forward(self, x):
        self.h = torch.tanh(self.h + x)
        self.h = torch.tanh(self.h + x)
        return self.h
```

`trace.buffers["h"].initial_value` is the pre-forward zero tensor.
`trace.buffers["h"].versions` contains the initial read version plus one version
for each reassignment that is read or returned. `num_overwrites == 2`, and
`final_value` is the second written state.

For BatchNorm in training mode, `running_mean`, `running_var`, and
`num_batches_tracked` are buffer entities. The running-stat versions are emitted
when the fused native BatchNorm call updates them. `num_batches_tracked` is
included even when it is not read later, because it is a real registered-buffer
state transition.

Useful accessors:

```python
trace.buffers                  # address -> Buffer
trace.buffers["bn.running_mean"]
trace["bn.running_mean:2"]      # graph Op for a specific buffer version
trace.buffers["bn.running_mean"].versions
trace.modules["bn"].buffers
```

`Buffer.value_at(n)` returns a 1-based observed version value.
`Buffer.value_after(n)` returns the value after the 1-based write index.

## Write Capture

TorchLens captures three kinds of registered-buffer writes.

**Reassignment** covers code like `self.h = new_tensor`. TorchLens installs a
capture-scoped class `__setattr__` patch for prepared module instances, records
the assignment once, and restores the class in capture cleanup. This avoids the
double-count that would happen if both `__setattr__` and PyTorch's internal
`_buffers[name] = value` path were journaled.

**Explicit in-place writes** cover `buf.mul_()`, `buf.add_()`, `buf.copy_()`,
`buf[...] = value`, and `buf.data.copy_(value)`. TorchLens snapshots registered
buffer storage around wrapped torch calls and attributes writes through views or
slices back to the registered buffer whose storage range was touched.

**Fused/native writes** cover BatchNorm and InstanceNorm running-stat updates.
Those kernels can update buffers without bumping PyTorch's normal tensor
`_version` counter, so TorchLens always compares post-op values for known fused
mutators. Fused versions are emitted per train/update op execution and carry a
`value_changed` flag.

## Validation

Buffer version nodes participate in `validate_forward_pass`. Explicit full-buffer
writes replay as identity transitions from the producer op to the buffer version.
For fused/native writes, TorchLens validates the state transition by rerunning
the kernel under restored state when saved argument values are available.

Some explicit writes, such as writing through a slice, produce an op output that
is only the slice while the buffer version stores the full registered buffer.
Those versions are marked as state-transition-only for replay validation. The
forward state and downstream readers still validate.

## Limitations

Assigning through `.data = new_tensor` is unsupported. It bypasses both normal
module reassignment and in-place operation hooks. TorchLens checks registered
buffers at the end of capture and raises a diagnostic if a buffer changed
without a journaled write. Use `self.buffer = tensor` or `self.buffer.copy_(...)`
instead.

If a fused native kernel writes a buffer and that intermediate value is never
read before a later overwrite, TorchLens may not display that intermediate
state. This is computationally inert: it has no effect on model output or replay.
It is only an introspection gap.

Python attributes that are not registered buffers are out of scope. Register
state with `register_buffer()` when you want TorchLens to track it as model
state.
