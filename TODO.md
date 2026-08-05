# Python on GPUs JAX and multi-GPU extension TODO

## Goal

Add JAX as the high-level compiled-array and multi-device component of
**Python on GPUs**:

- use NumPy-compatible numerical examples to introduce JAX array computing;
- teach `jit` and `vmap` as transformations of whole numerical functions;
- progress from CPU to one GPU and then explicitly sharded multi-GPU
  execution;
- retain CuPy, Numba, and PyCUDA for the concepts that require a clearer view
  of GPU execution, data movement, kernels, and memory access;
- keep distributed neural-network training with native PyTorch and PyTorch
  Lightning in **Machine Learning with Python**.

The preferred first delivery is an optional two-hour JAX module appended to
the existing training, not an additional topic squeezed into the nominal
four-hour core.  The current documented four-hour schedule already lists 270
minutes and must be reconciled independently.

## Agreed scope and ownership

### JAX should own in this training

- [ ] NumPy-like accelerator-oriented array computation with `jax.numpy`.
- [ ] Function-level compilation with `jax.jit`.
- [ ] Automatic batching and vectorization with `jax.vmap`.
- [ ] JAX's pure-function, immutable-array, and explicit-random-key model.
- [ ] Device discovery, placement, and visualization of array sharding.
- [ ] Single-host multi-GPU execution using a device mesh and explicit
      sharding.
- [ ] A concise introduction to `shard_map` and collective communication.
- [ ] Correct benchmarking of compiled, asynchronously dispatched work.
- [ ] Portability and numerical differences across CPU and accelerator
      backends.

### Existing GPU tools should retain

- [ ] Use CuPy for the simplest NumPy-to-GPU transition and for illustrating
      eager array operations.
- [ ] Use Numba and PyCUDA for threads, blocks, grids, kernel launches, memory
      access patterns, coalescing, synchronization, and explicit transfers.
- [ ] Use RAPIDS for GPU-accelerated data-science workflows when it remains a
      stated learning outcome.
- [ ] Use cuPyNumeric only where it supports a distinct learning objective;
      avoid presenting it as another interchangeable library in a catalogue.

### Related-training boundary

- [ ] Keep neural-network architecture, loss functions, optimizers,
      convergence, global batch size, distributed validation, and checkpoint
      semantics in **Machine Learning with Python**.
- [ ] Let that training use native PyTorch and Lightning for distributed
      deep-learning examples.
- [ ] Let **Python on GPUs** explain the underlying general concepts that
      transfer beyond machine learning: devices, topology, sharding,
      collectives, communication cost, synchronization, and scaling.
- [ ] Re-check **Python for HPC** before revising the Numba, benchmarking, or
      profiling material; cross-link shared concepts and avoid teaching the
      same Numba workflow in two mandatory courses.
- [ ] Use numerical methods already taught in **Scientific Python** as prepared
      workloads rather than reteaching those methods here.
- [ ] Add reciprocal links between both trainings rather than duplicating a
      Lightning or neural-network exercise here.

### Explicit non-goals for the first JAX module

- [ ] Do not turn the module into an introduction to Flax, Keras, Optax, or
      another JAX machine-learning stack.
- [ ] Do not teach model, tensor, pipeline, or fully sharded neural-network
      parallelism.
- [ ] Do not claim that ordinary JAX expressions automatically use every
      visible GPU; array and computation sharding must be made explicit.
- [ ] Do not use Pallas as the introductory custom-kernel framework while its
      API and supported GPU generations remain experimental and restrictive.
- [ ] Do not treat successful CPU execution as validation of the GPU or
      multi-GPU path.

## Priority 0: audit and protect the current course

- [ ] Record which slides, notebooks, scripts, and examples are part of the
      current live delivery and which are supplementary.
- [ ] Reconcile the stated four-hour duration with the current 270-minute
      schedule before adding JAX to any mandatory path.
- [ ] Map every current learning outcome to the material and exercise that
      supports it.
- [ ] Decide whether scikit-cuda, cuRAND, RAPIDS, cuPyNumeric, and both Numba
      and PyCUDA still merit mandatory time in one course.
- [ ] Reconcile documentation that refers to slides in this repository with
      the current location of the presentation material.
- [ ] Preserve the current course while the JAX module is prototyped and
      validated.
- [ ] Perform implementation work on a dedicated branch once development
      begins; do not mix it with the existing untracked cuNumeric,
      cuPyNumeric, or metadata work.

## Priority 1: define JAX learning outcomes

Participants completing the optional module should be able to:

- [ ] Explain how JAX differs from NumPy, CuPy, and explicit GPU-kernel tools.
- [ ] Port a suitable NumPy array computation to `jax.numpy`.
- [ ] Identify which part of a program should be compiled with `jax.jit`.
- [ ] Use `vmap` for an independent batch or ensemble dimension.
- [ ] Explain why pure functions, immutable arrays, and stable shapes matter to
      JAX transformations.
- [ ] Identify compilation, recompilation, transfer, synchronization, and
      communication costs in a measurement.
- [ ] Discover local devices and determine where an array is stored.
- [ ] Explain a device `Mesh`, named axes, and a partition specification.
- [ ] Shard a global array across multiple GPUs and run a computation on its
      shards.
- [ ] Distinguish replication, data sharding, and domain decomposition.
- [ ] Interpret one-, two-, and four-GPU scaling results without assuming
      linear speedup.
- [ ] State which backend, hardware, precision, and JAX versions were used for
      a result.

## Candidate optional two-hour module

Use this as a candidate schedule until a trainer rehearsal has been timed:

| Topic | Time |
|---|---:|
| Position JAX among NumPy, CuPy, Numba, and PyCUDA | 10 min |
| Port a NumPy computation to `jax.numpy` | 15 min |
| `jit`, `vmap`, pure functions, and common sharp edges | 20 min |
| Compilation, asynchronous execution, precision, and benchmarking | 15 min |
| Break | 10 min |
| Devices, meshes, and explicit array sharding | 15 min |
| One-, two-, and four-GPU hands-on experiment | 25 min |
| Scaling interpretation, portability, and wrap-up | 10 min |
| **Total** | **120 min** |

- [ ] Rehearse the complete module including environment startup, compilation,
      and scheduler latency.
- [ ] Reduce content rather than speaking faster if the rehearsal exceeds 120
      minutes.
- [ ] Decide whether to offer the module as a continuous six-hour workshop or
      as a separate follow-up session; prefer the separate session when GPU
      allocation or queueing makes timing uncertain.

## Priority 2: select representative workloads

- [ ] Choose one small baseline that runs quickly with NumPy and JAX on CPU
      and demonstrates correctness before performance.
- [ ] Choose one fixed-shape, compute-intensive scientific workload for the
      accelerator and multi-GPU experiment.
- [ ] Prefer a non-machine-learning workload, for example:
  - [ ] a batched Monte Carlo or parameter-ensemble calculation;
  - [ ] an N-body force calculation;
  - [ ] a stencil or iterative grid computation;
  - [ ] another existing course example that maps cleanly to multiple
        backends.
- [ ] Reuse one workload across NumPy, CuPy, and JAX where that comparison is
      scientifically and computationally fair.
- [ ] Avoid using matrix multiplication alone as evidence of general
      application performance; it mostly benchmarks a highly tuned library
      kernel.
- [ ] Include a workload size for which compilation or multi-GPU overhead
      dominates and explain the negative result.
- [ ] Keep reference inputs small enough for routine CPU correctness tests.
- [ ] Define expected outputs, invariants, and tolerances before optimizing or
      sharding the implementation.

## Priority 3: create the single-device JAX material

- [ ] Create `source-code/jax/README.md` describing scope, prerequisites,
      environments, and expected runtimes.
- [ ] Create a concise notebook that progresses from NumPy to `jax.numpy`,
      `jit`, and `vmap`.
- [ ] Create a script version of the final computation for repeatable timing
      and scheduler execution.
- [ ] Explain the difference between eager-looking Python and compiled JAX
      execution.
- [ ] Show at least one failed or inappropriate transformation involving:
  - [ ] a side effect;
  - [ ] value-dependent Python control flow;
  - [ ] a changing array shape;
  - [ ] repeated recompilation caused by changing static values or shapes.
- [ ] Show the appropriate JAX mechanism or redesign for each selected sharp
      edge without turning the module into an exhaustive JAX language course.
- [ ] Introduce explicit random keys if the selected workload is stochastic.
- [ ] Make host-to-device placement and result transfer observable.
- [ ] Demonstrate CPU and single-GPU execution using the same scientific
      computation.
- [ ] Preserve a CPU fallback that can validate correctness without pretending
      to validate accelerator performance.

## Priority 4: teach performance measurement correctly

- [ ] Measure compilation separately from steady-state execution.
- [ ] Warm up compiled functions before collecting steady-state timings.
- [ ] Call `block_until_ready()` or otherwise synchronize before stopping an
      accelerator timer.
- [ ] State whether host-device transfer is included in each measurement.
- [ ] Compare matching shapes, dtypes, algorithms, and convergence criteria.
- [ ] Address JAX's default precision explicitly; do not compare JAX
      `float32` with NumPy `float64` and call the result a fair speedup.
- [ ] Check numerical results with justified tolerances rather than bitwise
      equality.
- [ ] Record first-call, warm-call, and end-to-end timings where they answer
      different questions.
- [ ] Separate data-transfer, compilation, computation, synchronization, and
      inter-device communication costs where practical.
- [ ] Record memory use or out-of-memory behaviour for the scaling workload.

## Priority 5: add single-host multi-GPU execution

- [ ] Discover and display the available local devices.
- [ ] Arrange devices in a named `Mesh` that reflects the intended logical
      and, where possible, physical topology.
- [ ] Introduce `PartitionSpec` or the current equivalent through one clear
      global-array example.
- [ ] Visualize or print the sharding of inputs and results.
- [ ] First shard an embarrassingly parallel batch or ensemble dimension.
- [ ] Replicate only the data that must be available on every device.
- [ ] Add one explicit collective operation when it supports the selected
      scientific calculation.
- [ ] Introduce `shard_map` as the modern manual SPMD mechanism; do not build
      new teaching material around legacy `pmap` patterns.
- [ ] Compare one-, two-, and four-GPU execution when the infrastructure is
      available.
- [ ] Report throughput, wall time, speedup, and parallel efficiency.
- [ ] Check correctness at every device count.
- [ ] Explain communication-to-computation ratio and why additional devices
      may make small workloads slower.
- [ ] Explain that sharding can increase aggregate capacity but does not make
      all distributed memory behave like one low-latency memory space.

## Optional follow-on: multi-host JAX

- [ ] Keep multi-host execution out of the first live module unless reliable
      reservations on multiple GPU nodes are available.
- [ ] Provide a prepared script that calls `jax.distributed.initialize()`
      before any JAX computation.
- [ ] Provide a Slurm launcher whose task and GPU layout matches JAX's process
      and local-device configuration.
- [ ] Compare one process per GPU with one process per node for the selected
      infrastructure rather than assuming one layout is always superior.
- [ ] Document coordinator selection, process indices, local devices, and
      global devices.
- [ ] Fail clearly when the requested process or device topology is
      inconsistent with the allocation.
- [ ] Record interconnect and network information with multi-host results.
- [ ] Add timeouts and small smoke runs so configuration failures do not waste
      large allocations.

## Optional follow-on: custom kernels and Pallas

- [ ] Re-evaluate Pallas only after the high-level JAX module is stable.
- [ ] Verify the current Pallas API, supported GPU generations, and backend
      status before preparing participant material.
- [ ] Treat Pallas as experimental while upstream documents it as such.
- [ ] Compare one small Pallas kernel with the corresponding Numba or PyCUDA
      kernel only if the comparison teaches a distinct abstraction trade-off.
- [ ] Keep explicit CUDA execution concepts in the Numba or PyCUDA path even
      if Pallas becomes sufficiently stable for optional use.
- [ ] Do not advertise multi-vendor kernel portability without executing the
      same kernel on the claimed backends.

## Environments and backend portability

- [ ] Create a CPU environment that supports routine notebook and correctness
      validation without a GPU driver.
- [ ] Create or document a separate NVIDIA GPU environment whose JAX package,
      CUDA runtime, driver, and hardware requirements are explicit.
- [ ] Avoid assuming that one conda environment can reliably own the host GPU
      driver stack.
- [ ] Decide whether official pip-based JAX accelerator packages should be
      installed inside the course environment rather than forcing an
      unsuitable all-conda solution.
- [ ] Document NVIDIA/CUDA as the primary backend only if it is the backend
      actually tested before delivery.
- [ ] List AMD/ROCm support as documented but untested until exercised on AMD
      hardware with a recorded ROCm JAX plugin version.
- [ ] Treat TPU execution as a portability note rather than a supported course
      path unless TPU access and validation are added deliberately.
- [ ] Keep backend-specific setup isolated from portable lesson code.
- [ ] Ensure missing accelerator support produces a clear message and a CPU
      fallback rather than a confusing import or runtime failure.

## Coordination with Machine Learning with Python

- [ ] Add a link from this JAX module to the optional multi-GPU deep-learning
      module in `../Machine-learning-with-Python/TODO.md` or its eventual
      participant-facing successor.
- [ ] Add a reciprocal explanation to Machine Learning with Python:
  - [ ] JAX demonstrates general array sharding and SPMD computation;
  - [ ] PyTorch DDP and Lightning apply related ideas to neural-network
        training;
  - [ ] the examples have different scientific and pedagogical goals.
- [ ] Use consistent terminology for device, process, rank, replica, shard,
      collective, data parallelism, and scaling efficiency where the concepts
      overlap.
- [ ] Avoid requiring participants to take the machine-learning course before
      the JAX module.
- [ ] Avoid importing a neural-network workload merely to demonstrate that JAX
      can use multiple GPUs.

## Documentation and quality assurance

- [ ] Update `README.md`, `docs/README.md`, `source-code/README.md`, and
      `training.toml` together when the JAX module is accepted.
- [ ] Add JAX and multi-GPU execution to the learning outcomes only for the
      delivery variants that actually include the optional module.
- [ ] Document that ordinary Colab access is not a dependable substitute for
      a reserved multi-GPU node.
- [ ] Add a repeatable clean-kernel execution check for the introductory JAX
      notebook.
- [ ] Add CPU syntax and correctness tests for the script.
- [ ] Run a single-GPU smoke test on actual hardware before advertising GPU
      support.
- [ ] Run multi-GPU checks on actual devices; do not simulate success through
      artificial CPU device counts in release validation.
- [ ] Test device discovery and fail clearly when fewer devices are available
      than requested.
- [ ] Run the reference correctness case on CPU and every claimed accelerator
      backend.
- [ ] Record JAX, jaxlib, accelerator plugin, driver, runtime, device model,
      device count, precision, and workload shape with benchmark artifacts.
- [ ] Remove stored exceptions and machine-specific paths from
      participant-facing notebooks.
- [ ] Check that slides, notebooks, scripts, environments, scheduler files,
      and website text describe the same supported configurations.
- [ ] Repeat the complete multi-GPU rehearsal after JAX or accelerator-runtime
      upgrades; sharding and experimental APIs can change materially.

## Completion criteria

- [ ] The optional module schedule totals exactly 120 minutes including its
      break.
- [ ] Every JAX learning outcome has participant-facing material and an
      exercise or decision point.
- [ ] Participants can explain when JAX is preferable to CuPy and when an
      explicit Numba or PyCUDA kernel is more appropriate.
- [ ] The reference calculation agrees with the baseline within justified
      tolerances on CPU and the tested accelerator configuration.
- [ ] Steady-state timings exclude compilation unless compilation is the
      quantity being discussed and synchronize asynchronous execution.
- [ ] Participants can inspect a sharded array and explain its device layout.
- [ ] The multi-GPU exercise checks correctness and reports results for one,
      two, and four GPUs, or clearly documents unavailable configurations.
- [ ] At least one exercise demonstrates that adding GPUs can reduce
      performance for an unsuitable workload.
- [ ] Backend and hardware claims match configurations that were actually
      tested.
- [ ] The JAX module contains no neural-network framework material duplicated
      from Machine Learning with Python.
- [ ] The existing four-hour schedule has been reconciled before JAX is added
      to any mandatory delivery.
- [ ] Repository documentation, source code, environments, and training
      metadata agree on whether the JAX module is optional or included.
