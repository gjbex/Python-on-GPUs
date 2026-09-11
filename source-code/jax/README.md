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

The multi-device wave-equation notebook needs at least two visible devices.
For correctness testing on a CPU-only machine, create four virtual CPU devices
before starting Jupyter:

```bash
JAX_MULTI_DEVICE_CPU_EMULATION=1 jupyter lab
```

This mode validates sharding and collective operations, but its timings say
nothing about multi-GPU performance.  The notebook keeps its real-GPU scaling
experiment disabled unless `RUN_JAX_MULTI_GPU_BENCHMARK=1` is set before
Jupyter starts.

## Batch script

The command-line solver writes human-readable diagnostics to standard error
and one JSON summary to standard output or `--output`.  Output files are never
replaced unless `--overwrite` is passed.  `--validation full` also repeats the
solve with an unsharded reference implementation, so use `standard` for large
production-sized grids after the full validation path has passed at a smaller
size.

Test the distributed code without GPUs:

```bash
python multi_device_wave_equation.py \
    --emulate-cpu-devices 4 \
    --nx 128 --ny 128 \
    --validation full
```

A single-node Slurm job can expose all allocated GPUs to one Python process:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00

python multi_device_wave_equation.py \
    --backend gpu \
    --nx 4096 --ny 4096 \
    --precision float32 \
    --validation standard \
    --output "wave-${SLURM_JOB_ID}.json" \
    --field-output "wave-${SLURM_JOB_ID}.npy"
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
    --backend gpu \
    --distributed auto \
    --nx 4096 --ny 4096 \
    --precision float32 \
    --validation standard \
    --output "wave-${SLURM_JOB_ID}.json"
```

The resource directives shown here are common Slurm spellings, not a portable
cluster standard; adapt the GPU request, modules, environment activation, and
account or partition settings to the target site.

Every process must execute the same command.  Distributed initialization must
occur before any JAX device query; the script handles that ordering.  The
simple `--field-output` path gathers a fully addressable array and is therefore
limited to single-process runs.  A multi-node production code should use a
sharded checkpoint format instead of gathering the global field on rank 0.

JAX accelerator support changes more rapidly than its array API.  Consult the
[official installation guide](https://docs.jax.dev/en/latest/installation.html)
if the solver cannot find the expected GPU, or when targeting AMD, Intel, Apple,
or TPU hardware.

## Validation

Each notebook is intended to run from top to bottom in a fresh kernel.  Its
final cell checks numerical accuracy and the main scientific claims made in
the tutorial.  The examples are intentionally small so that they also run on a
CPU; a GPU is not automatically faster at this scale.

Compact text and figure outputs are intentionally retained so that the
notebooks remain useful in static previews.  Regenerate them by running all
cells from a fresh kernel.
