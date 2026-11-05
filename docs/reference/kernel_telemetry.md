# CUDA kernel telemetry

Status: documented unstable -- no deprecation shim owed.

Kernel telemetry is an optional, detachable extension of the gated ATen execution profile. It
measures CUDA kernels and memory copies from one concrete `torch.profiler`/Kineto session and
correlates them to observed `AtenOp` rows. It is not a finer portable model graph, an equivalence
key, a grouping key, or replay evidence.

The adapter is CUDA-only. It brackets each redispatched ATen call with a unique profiler marker,
then uses marker containment to identify CUDA runtime calls and Kineto runtime correlation IDs to
follow asynchronous work onto the device. It does not inspect operator-name or kernel-name
substrings. The implementation is designed to represent one ATen call launching many kernels and a
single fused launch related to several ATen rows without pretending either shape is one-to-one.

Real-CUDA/CUPTI correlation is **UNVERIFIED on the environment used for this audit**: its real-device
test is NOT-RUN here. The following cases are therefore also NOT-RUN: one ATen call to many kernels,
many ATen calls to one fused launch, multiple streams, asynchronous launches, memory copies,
profiler warmup, and profiler teardown when capture raises. The design and CPU/fake-profiler tests
must not be read as empirical validation of those real-device cases.

The public record facade is `KernelLaunch`. Its measured fields are `launch_name`, `device`,
`stream`, `duration`, `runtime_correlation`, and `attribution_status`. The launch rows and their
relation to primitive calls persist in the artifact's kernel annotation. This release has no public
reader for them: the former `AtenOp.gpu_kernels` and `Op.gpu_kernels` views were removed and refuse
with the typed `facade_redirect` error; a public kernel-recording switch is planned for a later
release. All names and status tokens are listed in the glossary's documented-unstable index.

## Availability and lower bounds

If CUDA or CUPTI is unavailable, telemetry returns a single fact-free disclosure row with
`attribution_status="unavailable"`. That row is not a kernel launch and is never counted as one.
An empty tuple is reserved for a profiler session that ran and observed no related device event.

`mode_paused_interior` remains an observation gap. Wherever it is non-empty, the observed kernel set
and every count derived from it remain lower bounds. TorchLens does not invent a hidden ATen row,
kernel, duration, stream, or correlation ID for the paused region.

## Persistence and shed boundary

Launch rows and their private primitive-sequence relation are declared `FieldPolicy.KEEP`: as of
the tlspec v8 coordinated bump ordinary artifacts persist the telemetry annotation plainly, with
closed-schema load validation (`torchlens/_io/forgery_validation.py`). The pre-v8
prerelease-registrar/pytest-switch path is retired.

No package module imports the adapter, and no non-telemetry behavioral test depends on it. The
frozen public-surface oracle is the only governance reference: the documented computed properties
must remain visible in the public-surface snapshot. Importing the adapter installs that view for
that process. This one-way dependency is the mechanical shed boundary: removing the telemetry
lane and its governance reference leaves capture, ATen recording, validation, visualization,
FLOPs, predicates, runnable execution, and persistence acceptance unchanged.
