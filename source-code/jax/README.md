# JAX for numerical computing

This directory contains self-paced introductory tutorials on using JAX for
general numerical computing.  The worked examples cover partial differential
equations and particle dynamics; no machine-learning background is required.

## Contents

1. `jax_features.ipynb`: concise overview of the shared JAX programming model,
   transformations, control flow, precision, timing, and trade-offs.  Start
   here if JAX is new to you.
1. `heat_equation.ipynb`: introductory tutorial using the heat equation,
   compilation, vectorization, and automatic differentiation.
1. `wave_equation.ipynb`: tutorial on second-order state, CFL stability,
   `jax.lax.scan`, conservation, numerical dispersion, and sensitivities.
1. `n_body.ipynb`: tutorial on pairwise interactions, symplectic integration,
   conserved quantities, batched simulations, and sensitivity analysis.
1. `multi_device_wave_equation.ipynb`: multi-device tutorial using a 2-D wave
   equation, spatial slabs, explicit halo exchange, a global reduction, and
   strong-scaling methodology.
1. `multi_device_n_body.ipynb`: focused multi-device N-body tutorial using
   equal particle ownership and explicit `all_gather` communication.
1. `multi_device_lattice_boltzmann.ipynb`: D2Q9 Lattice Boltzmann tutorial
   using a decaying periodic shear wave, local collision, packed neighbour
   exchanges, independent streaming tests, and optional GPU strong scaling.
1. `multi_device_lattice_boltzmann.py`: batch version of the Lattice Boltzmann
   example, including multi-node execution, shard-local initialization,
   slowest-process timing, and sharded final-state output.
1. `multi_device_wave_equation.py`: non-interactive command-line version of
   the multi-device wave solver for local runs and scheduler batch jobs.
1. `environment.yml`: Conda environment for an NVIDIA GPU using the
   community-supported Conda packages for JAX.
1. `python_on_gpus_jax_spec.txt`: Conda environment specification for an NVIDIA
   GPU using the community-supported Conda packages for JAX.

## Environment setup

Create and activate the environment with Mamba or Conda:

```bash
mamba env create --file environment.yml
conda activate python_on_gpus_jax
jupyter lab
```

The environment selects a CUDA-enabled `jaxlib` build.  For a CPU-only
environment, omit the `jaxlib=*=*cuda*` constraint:

```bash
mamba create --name python_on_gpus_jax_cpu --channel conda-forge \
    jax jupyterlab matplotlib numpy
```

The multi-device notebooks need at least two visible devices.
For correctness testing on a CPU-only machine, create four virtual CPU devices
before starting Jupyter:

```bash
JAX_MULTI_DEVICE_CPU_EMULATION=1 jupyter lab
```

This mode validates sharding and collective operations, but its timings say
nothing about multi-GPU performance.  The notebook keeps its real-GPU scaling
experiment disabled unless `RUN_JAX_MULTI_GPU_BENCHMARK=1` is set before
Jupyter starts.

The Lattice Boltzmann notebook uses float32 by default; set `JAX_ENABLE_X64=1`
for a double-precision validation run. Its optional benchmark uses a fixed
1024-by-1024 grid, configurable through `LBM_BENCH_NX` and `LBM_BENCH_NY`.
Keep the grid and precision fixed when comparing device counts. The worked
example targets one process on one node and includes a separate optional
hands-on block with complete solutions.

## Wave-equation batch script

`multi_device_wave_equation.py` runs the notebook's standing-wave problem as
a batch job. It computes the final displacement field, numerical diagnostics,
and timings. It does not generate plots, store a time history, or automatically
run a scaling study. Compare separate runs with different device counts while
keeping the global grid, precision, and physical parameters fixed.

### Physical problem and defaults

The scalar displacement $u(x,y,t)$ satisfies the two-dimensional wave equation

$$
\frac{\partial^2 u}{\partial t^2}
=c^2\left(\frac{\partial^2 u}{\partial x^2}
+\frac{\partial^2 u}{\partial y^2}\right)
$$

on a periodic rectangle of lengths $L_x,L_y$. Here $x,y$ are spatial coordinates,
$t$ is time, and $c$ is the wave speed. There are no sources or damping. The
initial displacement has unit amplitude and zero initial velocity:

$$
u(x,y,0)=\sin(k_x x)\sin(k_y y),\qquad
\frac{\partial u}{\partial t}(x,y,0)=0.
$$

The wave numbers are $k_x=2\pi m_x/L_x$, $k_y=2\pi m_y/L_y$, where the integers
$m_x,m_y$ count periods along each direction. The continuum solution is

$$
u_{\mathrm{exact}}(x,y,t)
=\cos(\omega t)\sin(k_x x)\sin(k_y y),\qquad
\omega=c\sqrt{k_x^2+k_y^2},
$$

with angular frequency $\omega$. This is a standing wave: the spatial pattern
stays fixed while its amplitude oscillates.

| Option | Default | Meaning |
| --- | --- | --- |
| `--nx`, `--ny` | 128, 128 | Grid points in each direction |
| `--nx-per-device` | Unset | Set global `nx` to this count times the selected device count; mutually exclusive with `--nx` |
| `--length-x`, `--length-y` | $2\pi$, $2\pi$ | Periodic domain lengths |
| `--wave-speed` | 1 | Speed $c$ |
| `--mode-x`, `--mode-y` | 1, 2 | Period counts $m_x,m_y$ |
| `--final-time` | 0.5 | Requested final time; mutually exclusive with `--steps` |
| `--steps` | Unset | Fixed number of time steps, including the Taylor starting step |
| `--dt` | Unset | Explicit time step with `--steps`; otherwise use the CFL-based step |
| `--cfl` | 0.5 | Factor used to choose the time step |
| `--precision` | `float64` | Arithmetic precision |
| `--num-gpus`, `--num-devices` | All visible | Global number of devices in the simulation mesh; these options are aliases |
| `--device-ids` | Unset | Ordered indices in `jax.devices()`, for single-process selection; mutually exclusive with the device count |
| `--warmup-runs`, `--repetitions` | 1, 1 | Untimed warmups and timed solves |
| `--validation` | `full` | Checks described below |

No physical unit system is imposed: lengths, speed, and time must use consistent
units. Mode counts must be positive and below half their respective grid sizes;
use counts well below that limit to resolve the wave. Zero counts would make
the sine-product initial field vanish and are rejected.

With the defaults, the script advances to $t=0.5$ in 21 time steps. The exact
final field is $\cos(\sqrt{5}/2)\sin(x)\sin(2y)$; the numerical field approximates
it with spatial and temporal discretization error.

### Numerical method and communication

The periodic grid omits duplicate endpoints and has spacings
$\Delta x=L_x/N_x$, $\Delta y=L_y/N_y$, where $N_x,N_y$ are the grid sizes.
A five-point central-difference Laplacian and a centered, second-order time
update advance the field. A second-order Taylor step starts the recurrence
using the zero initial velocity.

Writing $T$ for the final time and $q$ for `--cfl`, the script chooses
$\Delta t_{\mathrm{limit}}=q\min(\Delta x,\Delta y)/c$, takes
$N_t=\lceil T/\Delta t_{\mathrm{limit}}\rceil$ steps, and uses
$\Delta t=T/N_t$ to reach $T$ exactly when using `--final-time`. Here $N_t$ is
the number of time steps. With `--steps`, $N_t$ is fixed, $\Delta t$ is either
`--dt` or $\Delta t_{\mathrm{limit}}$, and $T=N_t\Delta t$. An explicit `--dt`
requires `--steps`. Both modes check the stability condition
$(c\Delta t/\Delta x)^2+(c\Delta t/\Delta y)^2\leq 1$.

The global field is split into contiguous slabs along $x$ across the selected
devices. `--nx` must be divisible by their count, with at least two rows per
device. Each device owns the complete $y$ direction of its slab. JAX `shard_map`
runs the local stencil; `lax.ppermute` exchanges boundary rows with periodic
neighbors. Collective reductions compute global errors and energy. These JAX
operations also carry communication between processes in a multi-node run.

### Diagnostics, validation, and output

Progress goes to standard error. Process 0 writes a JSON summary to standard
output or `--output`. Its sections are `configuration`, `runtime`, `derived`,
`timing`, `diagnostics`, and `outputs`, together with `status`.

- `continuous_exact_max_error` and `continuous_exact_rms_error`: maximum
  absolute and root-mean-square errors against the continuum solution at the
  final time. These include discretization error and have no pass/fail limit.
- `relative_energy_drift`: absolute change in the discrete energy divided by
  its initial magnitude. The conserved quantity uses two adjacent time levels:
  a squared time difference and products of forward spatial gradients at those
  levels, as defined in the notebook. It is evaluated initially at
  $\Delta t/2$ and finally at $T-\Delta t/2$.
- `laplacian_max_error`: maximum absolute difference between the distributed
  initial Laplacian and a NumPy periodic stencil. This checks halo exchange.
- `reference_max_error`: maximum absolute difference between the final field
  and an unsharded JAX solve using the same discrete method. This checks the
  distributed implementation independently of continuum discretization error.

`--validation standard` checks the Laplacian and energy drift; `full` also runs
and checks the unsharded reference solve. A failed check returns a nonzero exit
status. For float64, the limits are respectively $10^{-11}$, $2\times10^{-11}$,
and $2\times10^{-11}$; for float32 they are $2\times10^{-5}$, $5\times10^{-5}$,
and $2\times10^{-4}$. `none` skips these checks and the extra stencil/reference
comparisons, but still computes continuum errors and energy diagnostics. Use
`standard` for large grids after validating a smaller case with `full`.

Compilation is timed separately. Each warmup and timed repetition solves the
same initial-value problem from scratch. Execution timings include all $N_t$
steps, including the Taylor starting step, wait for both output time levels,
and exclude initialization, compilation, warmups, diagnostics, and file output.
Multi-process runs synchronize before each repetition and report its maximum
duration across processes; timing barriers and reductions are outside the
measured interval. Compilation time is also the maximum across processes.

`execution_seconds` contains the individual repetition times; the summary also
reports their minimum, median, maximum, and population standard deviation.
`grid_point_updates_per_second` is $N_xN_yN_t$ divided by the median duration.
`derived.grid_point_updates` records the update count. Runtime metadata includes
the available and selected device counts, and `mesh_devices` lists device IDs,
process indices, and descriptions in mesh order. Configuration records both
requested controls and the resolved grid size and final time.

`--field-output wave.npy` saves only the final numerical displacement as a
NumPy array of shape `(nx, ny)`, indexed by $x$ then $y$, in the selected
precision. Grid spacings and time are in the JSON summary, not in the array.
This option supports single-process runs only. Files are never replaced unless
`--overwrite` is passed.

Test the distributed code without GPUs:

```bash
python multi_device_wave_equation.py \
    --emulate-cpu-devices 4 \
    --nx 128 --ny 128 \
    --validation full \
    --output wave-summary.json \
    --field-output wave-final.npy
```

### Scaling experiments

Run the following commands from this directory in the same JAX environment.
First validate a small case with `--validation full`; large timing experiments
below use `none`, which still reports energy and continuum errors. Full
validation repeats the solve and allocates a full reference field on every
process, so it is intended for small correctness cases. CPU emulation tests
the implementation but does not measure GPU scaling.

On one process, `--num-gpus N` selects the first `N` JAX-visible devices.
`--num-devices` is the same control for any backend. `--device-ids 0,2` instead
selects those indices in `jax.devices()`, in the supplied order; these are not
necessarily physical GPU ordinals after visibility restrictions. Selection
changes the simulation mesh, not the scheduler allocation or
`CUDA_VISIBLE_DEVICES`. Requesting more devices than are visible is an error.

For **strong scaling**, keep the global grid, step count, precision, and physical
parameters fixed, and vary the selected device count:

```bash
for gpus in 1 2 4; do
    python multi_device_wave_equation.py \
        --backend gpu --num-gpus "$gpus" \
        --nx 2048 --ny 2048 --steps 1000 \
        --precision float32 \
        --warmup-runs 2 --repetitions 5 --validation none \
        --output "wave-strong-${gpus}.json"
done
```

For **weak scaling**, keep each device's slab size and the step count fixed:

```bash
for gpus in 1 2 4; do
    python multi_device_wave_equation.py \
        --backend gpu --num-gpus "$gpus" \
        --nx-per-device 512 --ny 1024 --steps 1000 \
        --precision float32 \
        --warmup-runs 2 --repetitions 5 --validation none \
        --output "wave-weak-${gpus}.json"
done
```

Here the global row count grows with the device count. At fixed domain lengths,
this refines the $x$ grid and can change $\Delta t$ and $T$; the computational
work per device stays fixed. To keep the spatial spacing and physical duration
fixed as well, increase `--length-x` proportionally to the device count, with
`--length-y`, the CFL factor, and the step count fixed. The selected mode counts
then determine the wavelengths in that enlarged domain.

Let $P$ be the selected device count and $t_P$ the
`timing.median_execution_seconds` for that run. Strong-scaling speedup is
$S_P=t_1/t_P$ and efficiency is $S_P/P$, where $t_1$ is the one-device baseline
and $S_P$ denotes speedup. Weak-scaling efficiency is $t_1/t_P$. Retain the
individual timing samples to assess variability, and avoid field output during
timing experiments to reduce memory and I/O costs outside the measured solve.

A single-node Slurm job can expose all allocated GPUs to one Python process:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00

python multi_device_wave_equation.py \
    --backend gpu --num-gpus 4 \
    --nx 4096 --ny 4096 --steps 1000 \
    --precision float32 \
    --warmup-runs 2 --repetitions 5 --validation none \
    --output "wave-${SLURM_JOB_ID}.json"
```

For multiple nodes, launch one process per allocated GPU and let JAX obtain
the coordinator, process count, and process index from Slurm:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00

srun python multi_device_wave_equation.py \
    --backend gpu --num-gpus 8 \
    --distributed auto \
    --nx 4096 --ny 4096 --steps 1000 \
    --precision float32 \
    --warmup-runs 2 --repetitions 5 --validation none \
    --output "wave-${SLURM_JOB_ID}.json"
```

The resource directives shown here are common Slurm spellings, not a portable
cluster standard; adapt the GPU request, modules, environment activation, and
account or partition settings to the target site.

Every process must use the same simulation, device-count, and timing options;
the script checks agreement before the solve. In multi-process runs the global
selected count must divide equally across all launched processes, and each
process uses its first local devices. With one process per GPU, reduce the
launcher's process count and adjust the resource request to use fewer GPUs;
`--num-gpus` cannot exclude an already launched process. Explicit `--device-ids`
selection is limited to single-process runs. Manual initialization's
`--local-device-ids` is a separate control for the devices visible to each
process.

Distributed initialization must occur before any JAX device query; the script
handles that ordering. The
simple `--field-output` path gathers a fully addressable array and is therefore
limited to single-process runs.  A multi-node production code should use a
sharded checkpoint format instead of gathering the global field on rank 0.

Initialization and exact-solution arrays are constructed from process-local
slabs, without replicating the global host field. Only `full` validation builds
the global reference arrays on each process. Multi-node final-state output
still requires a sharded checkpoint implementation.

JAX accelerator support changes more rapidly than its array API.  Consult the
[official installation guide](https://docs.jax.dev/en/latest/installation.html)
if the solver cannot find the expected GPU, or when targeting AMD, Intel, Apple,
or TPU hardware.

## Lattice Boltzmann batch script

Run the small reference-validated case on four emulated CPU devices:

```bash
python multi_device_lattice_boltzmann.py --emulate-cpu-devices 4
```

The script simulates the notebook's periodic shear wave using lattice units,
with float32 by default. `--precision float64` enables double precision.
`--nx` must divide evenly over the global device count; one-row slabs are
supported. `--tau` must exceed 0.5, and `--amplitude` is restricted to a Mach
number of at most 0.1. These input conditions do not establish stability for
arbitrary grids or relaxation times.

`--validation full` runs independent routing tests, conservation checks, and
an unsharded reference on each process. It is restricted to at most 65536 sites
to avoid accidentally replicating a large reference problem. `standard`
retains routing and conservation checks without a global reference. Continuum
decay checks apply for `nx >= 64` and `0.7 <= tau <= 1.0`; the summary records
whether those checks were applied. `none` still rejects non-finite populations
and non-positive density. Long float32 runs may need a precision review if
accumulated conservation error exceeds the fixed validation tolerances.

For two nodes with four GPUs each, launch one process per GPU:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

# Activate the same JAX environment on every node before this command.
srun --distribution=block:block python multi_device_lattice_boltzmann.py \
    --backend gpu --distributed auto \
    --nx 4096 --ny 4096 --steps 1000 \
    --precision float32 --validation standard \
    --output "lbm-${SLURM_JOB_ID}.json" \
    --checkpoint-dir "lbm-${SLURM_JOB_ID}-state"
```

Adapt GPU requests and binding to the site's Slurm configuration. Verify that
each process sees its allocated GPU. The mesh groups devices by process ID;
block placement groups adjacent slab owners on the same node. Under Slurm,
JAX discovers the distributed configuration automatically. With other launch
environments, use `--distributed manual` and supply `--coordinator-address`,
`--num-processes`, `--process-id`, and `--local-device-ids`. The coordinator
must be reachable from all nodes; `127.0.0.1` is appropriate only for tests on
one host. MPI is not required by the script's numerical implementation.

All processes must execute the same run settings and collective sequence.
The script checks agreement on run settings before creating numerical
collectives. Output paths must refer to the same shared filesystem on all
processes. Proxy environment variables can interfere with JAX distributed
startup; consult the
[initialization documentation](https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html)
if discovery times out.

Each process initializes only its addressable slabs using their global
coordinates. During stepping, `lax.ppermute` exchanges the three outgoing
populations on each face; the host does not stage population messages.
Compilation, warmup, diagnostics, and output are excluded from solve timing.
`--repetitions` repeats the same initial-value problem and reports the median
of the slowest-process times. CPU timings validate the execution path and do
not measure multi-GPU scaling.

Process zero writes a JSON summary to stdout or `--output`; application logs
go to stderr. Some backends also print native startup diagnostics to stdout,
so use `--output` when a file must contain strictly JSON.
Existing summaries require `--overwrite`. The checkpoint directory must be
new, even with `--overwrite`, and its parent must exist. Every GPU's final
population slab is stored as `slab-XXXXXX.npy`; `manifest.json` records the
global shape, direction ordering, x ranges, completed steps, and run settings.
A manifest is written only after all shard writes succeed. A directory without
a manifest is incomplete. The script writes a final snapshot rather than
periodic recovery checkpoints.

For a small snapshot, a separate analysis process can reconstruct the global
state from the manifest's ordered list of slabs:

```python
import json
from pathlib import Path
import numpy as np

directory = Path("lbm-state")
manifest = json.loads((directory / "manifest.json").read_text())
f = np.concatenate([
    np.load(directory / shard["file"], allow_pickle=False)
    for shard in manifest["shards"]
], axis=0)
```

This reconstruction needs enough host memory for the entire state. For large
cases, read and analyze individual slabs or downsample before assembly.

## Validation

Each notebook is intended to run from top to bottom in a fresh kernel.  Its
final cell checks numerical accuracy and the main scientific claims made in
the tutorial.  The examples are intentionally small so that they also run on a
CPU; a GPU is not automatically faster at this scale.

Compact text and figure outputs are intentionally retained so that the
notebooks remain useful in static previews.  Regenerate them by running all
cells from a fresh kernel.
