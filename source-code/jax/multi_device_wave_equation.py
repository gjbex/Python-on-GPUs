#!/usr/bin/env python3
r"""Batch-oriented multi-device JAX solver for the 2-D wave equation.

The initial displacement is sin(kx*x)*sin(ky*y), with zero initial velocity,
where kx=2*pi*mode_x/length_x and ky=2*pi*mode_y/length_y.  A second-order
finite-difference solve advances this periodic standing wave to --final-time.
Diagnostics compare its final field with the continuum solution, check
discrete energy conservation, and optionally compare an unsharded solve.

The global periodic field is partitioned into x-slabs.  A ``shard_map``
kernel exchanges halo rows with ``lax.ppermute`` before applying a five-point
Laplacian.  The program supports a single process using multiple devices and
multi-process execution initialized by JAX (including Slurm auto-detection).

Human-readable diagnostics go to stderr.  Process 0 writes one JSON summary
to stdout or to ``--output``.  An optional final field can be written as a
NumPy ``.npy`` file for single-process runs.

Scaling experiments (run from this directory):

    # Validate a small case before timing large grids.
    python multi_device_wave_equation.py --backend gpu --num-gpus 2

    # Strong scaling: keep the global grid, step count, and precision fixed.
    for gpus in 1 2 4; do
        python multi_device_wave_equation.py --backend gpu --num-gpus "$gpus" \
            --nx 2048 --ny 2048 --steps 1000 --precision float32 \
            --warmup-runs 2 --repetitions 5 --validation none \
            --output "wave-strong-${gpus}.json"
    done

    # Weak scaling: keep the rows and columns per GPU and the step count fixed.
    for gpus in 1 2 4; do
        python multi_device_wave_equation.py --backend gpu --num-gpus "$gpus" \
            --nx-per-device 512 --ny 1024 --steps 1000 --precision float32 \
            --warmup-runs 2 --repetitions 5 --validation none \
            --output "wave-weak-${gpus}.json"
    done

``--num-gpus`` is an alias for the backend-independent ``--num-devices``.
Both choose a subset of the JAX-visible devices for the mesh; they do not change
the scheduler allocation or CUDA_VISIBLE_DEVICES. By default all visible devices
are used. ``--device-ids 0,2`` instead selects explicit indices in jax.devices(),
in the supplied order, on one process. In multi-process runs the global count
must divide equally across all launched processes, each using its first local
devices. With one process per GPU, change the launcher's process count to change
the GPU count. All ranks must pass the same simulation and timing options.

``--steps`` and ``--final-time`` are mutually exclusive. With ``--steps``, the
time step is CFL-based unless ``--dt`` is supplied; the physical final time is
steps*dt. With ``--final-time`` (default 0.5), the CFL-based step is shortened to
reach that time exactly. An explicit dt must satisfy the stencil's stability
condition. ``--nx-per-device`` and ``--nx`` are mutually exclusive. Weak scaling
at fixed domain lengths refines the x grid and can change dt and final time;
scale --length-x with the device count to keep the x spacing fixed as well.

Every timed repetition starts from the same initial state, includes all steps
(including the Taylor starting step), and waits for both output time levels.
Compilation, warmups, diagnostics, output, and timing barriers/reductions are
excluded from execution_seconds. Multi-process runs synchronize before each
repetition and report its maximum duration over ranks. JSON includes individual
durations, minimum/median/maximum and population standard deviation, update
throughput, actual mesh devices, global/local grid sizes, dt, and final time.
For strong scaling, speedup is the one-device median divided by the P-device
median; efficiency is speedup/P, where P is the selected device count. For weak
scaling, efficiency is the one-device median divided by the P-device median.

Initialization constructs only local slabs. ``--validation full`` still builds
the full small-grid reference on every process; use standard or none for large
benchmarks. Virtual CPU devices can test selection and halo exchange, but their
timings do not measure GPU scaling. See --help and README.md for model details.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from statistics import median, pstdev
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

AXIS_NAME = "devices"
LOGGER = logging.getLogger("multi_device_wave_equation")


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def parse_local_device_ids(value: str) -> tuple[int, ...]:
    try:
        device_ids = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of integer device IDs"
        ) from exc
    if not device_ids or any(device_id < 0 for device_id in device_ids):
        raise argparse.ArgumentTypeError(
            "must contain one or more non-negative device IDs"
        )
    if len(set(device_ids)) != len(device_ids):
        raise argparse.ArgumentTypeError("device IDs must be unique")
    return device_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a periodic 2-D wave equation with x-slab sharding and "
            "explicit JAX halo exchange."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Strong scaling: keep --nx, --ny, --steps and --precision fixed, "
            "vary --num-gpus. Weak scaling: use --nx-per-device instead of "
            "--nx, keeping --ny and --steps fixed. Use several --repetitions. "
            "See the module docstring for complete examples."
        ),
    )

    problem = parser.add_argument_group("scientific problem")
    x_size = problem.add_mutually_exclusive_group()
    x_size.add_argument(
        "--nx", type=positive_int, default=128, help="global grid rows along x"
    )
    x_size.add_argument(
        "--nx-per-device",
        type=positive_int,
        help="set global nx to this many rows times the selected device count",
    )
    problem.add_argument(
        "--ny", type=positive_int, default=128, help="global grid columns along y"
    )
    problem.add_argument(
        "--length-x",
        type=positive_float,
        default=2.0 * np.pi,
        help="periodic domain length along x",
    )
    problem.add_argument(
        "--length-y",
        type=positive_float,
        default=2.0 * np.pi,
        help="periodic domain length along y",
    )
    problem.add_argument(
        "--wave-speed",
        type=positive_float,
        default=1.0,
        help="wave speed in consistent length/time units",
    )
    problem.add_argument(
        "--mode-x", type=positive_int, default=1, help="positive period count along x"
    )
    problem.add_argument(
        "--mode-y", type=positive_int, default=2, help="positive period count along y"
    )
    duration = problem.add_mutually_exclusive_group()
    duration.add_argument(
        "--final-time",
        type=positive_float,
        default=0.5,
        help="physical final time; replaced by steps*dt when --steps is used",
    )
    duration.add_argument(
        "--steps",
        type=positive_int,
        help="fixed number of time steps, including the Taylor starting step",
    )
    problem.add_argument(
        "--dt",
        type=positive_float,
        help="time step with --steps; otherwise use the CFL-based step",
    )
    problem.add_argument(
        "--cfl",
        type=positive_float,
        default=0.5,
        help="c*dt/min(dx,dy); must not exceed 1/sqrt(2)",
    )
    problem.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float64",
        help="field and computation precision",
    )

    execution = parser.add_argument_group("execution and validation")
    execution.add_argument(
        "--backend",
        choices=("auto", "cpu", "gpu", "tpu"),
        default="auto",
        help="requested JAX backend; auto uses JAX detection",
    )
    execution.add_argument(
        "--emulate-cpu-devices",
        type=positive_int,
        metavar="COUNT",
        help="create COUNT virtual CPU devices for correctness testing",
    )
    selection = execution.add_mutually_exclusive_group()
    selection.add_argument(
        "--num-devices",
        "--num-gpus",
        dest="num_devices",
        type=positive_int,
        metavar="COUNT",
        help="global number of devices to use; default is all visible devices",
    )
    selection.add_argument(
        "--device-ids",
        type=parse_local_device_ids,
        help="ordered indices in jax.devices() to use; single-process only",
    )
    execution.add_argument(
        "--validation",
        choices=("full", "standard", "none"),
        default="full",
        help=(
            "full also runs an unsharded reference solve; standard checks "
            "halos and energy; none only reports diagnostics"
        ),
    )
    execution.add_argument(
        "--warmup-runs",
        type=nonnegative_int,
        default=1,
        help="untimed runs after compilation",
    )
    execution.add_argument(
        "--repetitions",
        type=positive_int,
        default=1,
        help="timed repetitions of the same initial-value problem",
    )

    distributed = parser.add_argument_group("multi-process execution")
    distributed.add_argument(
        "--distributed",
        choices=("off", "auto", "manual"),
        default="off",
        help=(
            "auto uses scheduler detection; manual uses the coordinator arguments below"
        ),
    )
    distributed.add_argument("--coordinator-address")
    distributed.add_argument("--num-processes", type=positive_int)
    distributed.add_argument("--process-id", type=nonnegative_int)
    distributed.add_argument(
        "--local-device-ids",
        type=parse_local_device_ids,
        help="comma-separated device IDs for this process",
    )
    distributed.add_argument(
        "--initialization-timeout",
        type=positive_int,
        default=300,
        help="distributed initialization timeout in seconds",
    )

    output = parser.add_argument_group("outputs")
    output.add_argument(
        "--output",
        type=Path,
        help="JSON summary path; default is process-0 stdout",
    )
    output.add_argument(
        "--field-output",
        type=Path,
        help="optional process-0 final field path in .npy format",
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing output files",
    )
    output.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def validate_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if (args.nx_per_device is None and args.nx < 3) or args.ny < 3:
        parser.error("--nx and --ny must both be at least 3")
    if args.nx_per_device is not None and args.nx_per_device < 2:
        parser.error("--nx-per-device must be at least 2")
    if args.dt is not None and args.steps is None:
        parser.error("--dt requires --steps")
    if args.device_ids is not None and args.distributed != "off":
        parser.error("--device-ids supports single-process execution only")
    if args.cfl > 1.0 / np.sqrt(2.0):
        parser.error("--cfl must not exceed 1/sqrt(2) for this scheme")

    manual_values = (
        args.coordinator_address,
        args.num_processes,
        args.process_id,
    )
    if args.distributed == "manual" and any(value is None for value in manual_values):
        parser.error(
            "manual distributed mode requires --coordinator-address, "
            "--num-processes, and --process-id"
        )
    if args.distributed != "manual" and any(
        value is not None for value in (*manual_values, args.local_device_ids)
    ):
        parser.error(
            "manual distributed options are accepted only with --distributed manual"
        )
    if (
        args.process_id is not None
        and args.num_processes is not None
        and args.process_id >= args.num_processes
    ):
        parser.error("--process-id must be smaller than --num-processes")
    if args.emulate_cpu_devices and args.distributed != "off":
        parser.error("--emulate-cpu-devices supports single-process execution only")
    if args.emulate_cpu_devices and args.backend not in ("auto", "cpu"):
        parser.error("CPU emulation is incompatible with the requested backend")

    for path, option in (
        (args.output, "--output"),
        (args.field_output, "--field-output"),
    ):
        if path is None:
            continue
        if not path.parent.exists():
            parser.error(f"parent directory for {option} does not exist: {path.parent}")
        if not os.access(path.parent, os.W_OK):
            parser.error(
                f"parent directory for {option} is not writable: {path.parent}"
            )
        if path.exists() and not args.overwrite:
            parser.error(
                f"{option} already exists: {path}; pass --overwrite to replace it"
            )
    if args.field_output and args.field_output.suffix != ".npy":
        parser.error("--field-output must end in .npy")
    if (
        args.output
        and args.field_output
        and args.output.resolve() == args.field_output.resolve()
    ):
        parser.error("--output and --field-output must be different paths")


def configure_jax(args: argparse.Namespace) -> bool:
    """Configure JAX and initialize distributed state before device discovery."""
    if args.emulate_cpu_devices:
        jax.config.update("jax_platform_name", "cpu")
        jax.config.update("jax_num_cpu_devices", args.emulate_cpu_devices)
    elif args.backend != "auto":
        jax.config.update("jax_platform_name", args.backend)

    jax.config.update("jax_enable_x64", args.precision == "float64")

    if args.distributed == "off":
        return False
    if args.distributed == "auto":
        jax.distributed.initialize(initialization_timeout=args.initialization_timeout)
    else:
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
            local_device_ids=args.local_device_ids,
            initialization_timeout=args.initialization_timeout,
        )
    return True


def gather_process_values(value: np.ndarray) -> np.ndarray:
    """Gather small host-side values; never gather simulation fields here."""
    if jax.process_count() == 1:
        return np.asarray(value)[None, ...]
    return np.asarray(multihost_utils.process_allgather(value, tiled=False))


def synchronize(label: str) -> None:
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices(label)


def slowest_process_seconds(local_seconds: float) -> float:
    return float(np.max(gather_process_values(np.asarray(local_seconds))))


def validate_process_configuration(args: argparse.Namespace) -> None:
    """Catch rank mismatches before entering differently shaped collectives."""
    if jax.process_count() == 1:
        return
    settings = {
        key: value
        for key, value in vars(args).items()
        if key
        not in {"process_id", "local_device_ids", "coordinator_address", "log_level"}
    }
    digest = hashlib.sha256(
        json.dumps(settings, sort_keys=True, default=str).encode()
    ).digest()
    digests = gather_process_values(np.frombuffer(digest, dtype=np.uint8))
    if np.any(digests != digests[0]):
        raise ValueError("processes have different run parameters or output paths")


def select_devices(args: argparse.Namespace) -> tuple[list[Any], list[Any]]:
    """Select a mesh subset without changing scheduler GPU visibility."""
    available = list(jax.devices())
    if args.device_ids is not None:
        if max(args.device_ids) >= len(available):
            raise ValueError(f"--device-ids indices must be below {len(available)}")
        return available, [available[index] for index in args.device_ids]
    count = args.num_devices or len(available)
    if count > len(available):
        raise ValueError(
            f"requested {count} devices, but only {len(available)} are visible"
        )
    if jax.process_count() == 1:
        return available, available[:count]
    # Every launched process participates. Do not silently drop some ranks.
    process_count = jax.process_count()
    if count % process_count or count < process_count:
        raise ValueError(
            "--num-devices must be a positive multiple of the process count; "
            "launch fewer processes to use fewer devices than processes"
        )
    per_process = count // process_count
    selected = []
    for process in range(process_count):
        local = sorted(
            (device for device in available if device.process_index == process),
            key=lambda device: device.id,
        )
        if len(local) < per_process:
            raise ValueError(
                f"process {process} has {len(local)} devices, needs {per_process}"
            )
        selected.extend(local[:per_process])
    return available, selected


def reference_laplacian(u: jax.Array, dx: float, dy: float) -> jax.Array:
    d2x = (jnp.roll(u, 1, axis=0) - 2.0 * u + jnp.roll(u, -1, axis=0)) / dx**2
    d2y = (jnp.roll(u, 1, axis=1) - 2.0 * u + jnp.roll(u, -1, axis=1)) / dy**2
    return d2x + d2y


def reference_solver(
    u_initial: jax.Array,
    u_after_one_step: jax.Array,
    n_steps: int,
    *,
    wave_speed: float,
    dt: float,
    dx: float,
    dy: float,
) -> tuple[jax.Array, jax.Array]:
    def body(_: int, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        previous, current = state
        following = (
            2.0 * current
            - previous
            + (wave_speed * dt) ** 2 * reference_laplacian(current, dx, dy)
        )
        return current, following

    return lax.fori_loop(1, n_steps, body, (u_initial, u_after_one_step))


def make_distributed_operators(
    mesh: Mesh,
    *,
    wave_speed: float,
    dt: float,
    dx: float,
    dy: float,
) -> tuple[Any, Any, Any, Any]:
    n_devices = mesh.size
    send_right = tuple(
        (source, (source + 1) % n_devices) for source in range(n_devices)
    )
    send_left = tuple((source, (source - 1) % n_devices) for source in range(n_devices))
    field_spec = P(AXIS_NAME, None)

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=field_spec,
        out_specs=field_spec,
        axis_names={AXIS_NAME},
    )
    def distributed_laplacian(u_local: jax.Array) -> jax.Array:
        left_halo = lax.ppermute(u_local[-1], AXIS_NAME, send_right)
        right_halo = lax.ppermute(u_local[0], AXIS_NAME, send_left)
        padded = jnp.concatenate(
            (left_halo[None, :], u_local, right_halo[None, :]), axis=0
        )
        d2x = (padded[:-2] - 2.0 * padded[1:-1] + padded[2:]) / dx**2
        d2y = (
            jnp.roll(u_local, 1, axis=1) - 2.0 * u_local + jnp.roll(u_local, -1, axis=1)
        ) / dy**2
        return d2x + d2y

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(field_spec, field_spec),
        out_specs=P(),
        axis_names={AXIS_NAME},
    )
    def distributed_energy(
        u_previous_local: jax.Array, u_current_local: jax.Array
    ) -> jax.Array:
        next_previous = lax.ppermute(u_previous_local[0], AXIS_NAME, send_left)
        next_current = lax.ppermute(u_current_local[0], AXIS_NAME, send_left)
        forward_previous = jnp.concatenate(
            (u_previous_local[1:], next_previous[None, :]), axis=0
        )
        forward_current = jnp.concatenate(
            (u_current_local[1:], next_current[None, :]), axis=0
        )
        velocity = (u_current_local - u_previous_local) / dt
        grad_x_previous = (forward_previous - u_previous_local) / dx
        grad_x_current = (forward_current - u_current_local) / dx
        grad_y_previous = (
            jnp.roll(u_previous_local, -1, axis=1) - u_previous_local
        ) / dy
        grad_y_current = (jnp.roll(u_current_local, -1, axis=1) - u_current_local) / dy
        local_energy = (
            0.5
            * jnp.sum(
                velocity**2
                + wave_speed**2
                * (grad_x_previous * grad_x_current + grad_y_previous * grad_y_current)
            )
            * dx
            * dy
        )
        return lax.psum(local_energy, AXIS_NAME)

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(field_spec, field_spec),
        out_specs=(P(), P()),
        axis_names={AXIS_NAME},
    )
    def distributed_error_sums(
        actual_local: jax.Array, expected_local: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        difference = jnp.abs(actual_local - expected_local)
        maximum = lax.pmax(jnp.max(difference), AXIS_NAME)
        sum_of_squares = lax.psum(jnp.sum(difference**2), AXIS_NAME)
        return maximum, sum_of_squares

    @jax.jit
    def solve(u_initial: jax.Array, n_steps: int) -> tuple[jax.Array, jax.Array]:
        # Include the first stencil evaluation in the timed workload.
        u_after_one_step = u_initial + (
            0.5 * (wave_speed * dt) ** 2 * distributed_laplacian(u_initial)
        )

        def body(
            _: int, state: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            previous, current = state
            following = (
                2.0 * current
                - previous
                + (wave_speed * dt) ** 2 * distributed_laplacian(current)
            )
            return current, following

        return lax.fori_loop(1, n_steps, body, (u_initial, u_after_one_step))

    return (
        distributed_laplacian,
        distributed_energy,
        distributed_error_sums,
        solve,
    )


def build_initial_fields(
    args: argparse.Namespace, *, dx: float, dy: float, dtype: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.arange(args.nx, dtype=dtype) * dx
    y = np.arange(args.ny, dtype=dtype) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    wave_number_x = 2.0 * np.pi * args.mode_x / args.length_x
    wave_number_y = 2.0 * np.pi * args.mode_y / args.length_y
    u_initial = (np.sin(wave_number_x * xx) * np.sin(wave_number_y * yy)).astype(dtype)
    laplacian_initial = (
        np.roll(u_initial, 1, axis=0) - 2.0 * u_initial + np.roll(u_initial, -1, axis=0)
    ) / dx**2 + (
        np.roll(u_initial, 1, axis=1) - 2.0 * u_initial + np.roll(u_initial, -1, axis=1)
    ) / dy**2
    angular_frequency = args.wave_speed * np.sqrt(wave_number_x**2 + wave_number_y**2)
    return u_initial, laplacian_initial, xx, angular_frequency


def build_initial_block(
    index: tuple[slice, ...],
    args: argparse.Namespace,
    *,
    dx: float,
    dy: float,
    dtype: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one x slab and its reference Laplacian, with periodic halos."""
    start, stop, _ = index[0].indices(args.nx)
    x_index = np.arange(start - 1, stop + 1) % args.nx
    x = x_index.astype(dtype) * dx
    y = np.arange(args.ny, dtype=dtype) * dy
    kx = 2.0 * np.pi * args.mode_x / args.length_x
    ky = 2.0 * np.pi * args.mode_y / args.length_y
    padded = (np.sin(kx * x[:, None]) * np.sin(ky * y[None, :])).astype(dtype)
    initial = padded[1:-1]
    laplacian = (padded[:-2] - 2.0 * initial + padded[2:]) / dx**2 + (
        np.roll(initial, 1, axis=1) - 2.0 * initial + np.roll(initial, -1, axis=1)
    ) / dy**2
    return initial, laplacian


def scalar(value: jax.Array) -> float:
    return float(np.asarray(value))


def error_metrics(
    error_sums: Any,
    actual: jax.Array,
    expected: jax.Array,
    *,
    point_count: int,
) -> tuple[float, float]:
    maximum, sum_of_squares = error_sums(actual, expected)
    return scalar(maximum), float(np.sqrt(scalar(sum_of_squares) / point_count))


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_process_configuration(args)
    available_devices, selected_devices = select_devices(args)
    global_devices = np.asarray(selected_devices)
    process_index = jax.process_index()
    process_count = jax.process_count()
    local_devices = [
        device for device in selected_devices if device.process_index == process_index
    ]
    if args.nx_per_device is not None:
        args.nx = args.nx_per_device * len(selected_devices)

    if args.field_output and process_count > 1:
        raise ValueError(
            "--field-output currently supports single-process runs only; "
            "multi-process output needs a sharded checkpoint format"
        )
    if args.nx % global_devices.size != 0:
        raise ValueError(
            f"--nx={args.nx} is not divisible by the global device count "
            f"({global_devices.size})"
        )
    if args.nx // global_devices.size < 2:
        raise ValueError("each device must own at least two x rows")
    if 2 * args.mode_x >= args.nx or 2 * args.mode_y >= args.ny:
        raise ValueError("mode counts must be below half their respective grid sizes")

    backend = jax.default_backend()
    dtype = np.float64 if args.precision == "float64" else np.float32
    dx = args.length_x / args.nx
    dy = args.length_y / args.ny
    dt_limit = args.cfl * min(dx, dy) / args.wave_speed
    if args.steps is None:
        final_time = args.final_time
        n_steps = int(np.ceil(final_time / dt_limit))
        dt = final_time / n_steps
    else:
        n_steps = args.steps
        dt = args.dt if args.dt is not None else dt_limit
        final_time = n_steps * dt
    if not np.isfinite(final_time) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("derived final time and time step must be finite and positive")
    stability_number = (args.wave_speed * dt / dx) ** 2 + (
        args.wave_speed * dt / dy
    ) ** 2
    if stability_number > 1.0:
        raise ValueError(
            f"unstable time step: 2-D stability number is {stability_number}"
        )

    mesh = Mesh(global_devices, (AXIS_NAME,))
    field_sharding = NamedSharding(mesh, P(AXIS_NAME, None))
    (
        distributed_laplacian,
        distributed_energy,
        distributed_error_sums,
        solve,
    ) = make_distributed_operators(
        mesh,
        wave_speed=args.wave_speed,
        dt=dt,
        dx=dx,
        dy=dy,
    )

    angular_frequency = args.wave_speed * np.sqrt(
        (2.0 * np.pi * args.mode_x / args.length_x) ** 2
        + (2.0 * np.pi * args.mode_y / args.length_y) ** 2
    )
    shape = (args.nx, args.ny)
    initial_block = partial(build_initial_block, args=args, dx=dx, dy=dy, dtype=dtype)
    u_initial = jax.make_array_from_callback(
        shape, field_sharding, lambda index: initial_block(index)[0]
    )
    # Only process-local slabs are constructed, including on multiple nodes.
    exact_final = jax.make_array_from_callback(
        shape,
        field_sharding,
        lambda index: np.cos(angular_frequency * final_time) * initial_block(index)[0],
    )
    start_step = jax.jit(
        lambda u: u + 0.5 * (args.wave_speed * dt) ** 2 * distributed_laplacian(u)
    )
    u_after_one_step = start_step(u_initial)
    jax.block_until_ready((u_initial, u_after_one_step, exact_final))

    LOGGER.info(
        "rank=%d/%d backend=%s global_devices=%d local_devices=%d grid=%dx%d steps=%d",
        process_index,
        process_count,
        backend,
        global_devices.size,
        len(local_devices),
        args.nx,
        args.ny,
        n_steps,
    )

    synchronize("wave-compile-start")
    compilation_started = time.perf_counter()
    executable = solve.lower(u_initial, n_steps).compile()
    compilation_seconds = slowest_process_seconds(
        time.perf_counter() - compilation_started
    )

    for _ in range(args.warmup_runs):
        jax.block_until_ready(executable(u_initial, n_steps))

    execution_seconds: list[float] = []
    u_previous = u_initial
    u_final = u_after_one_step
    for repetition in range(args.repetitions):
        synchronize(f"wave-run-start-{repetition}")
        started = time.perf_counter()
        u_previous, u_final = executable(u_initial, n_steps)
        jax.block_until_ready((u_previous, u_final))
        execution_seconds.append(slowest_process_seconds(time.perf_counter() - started))

    energy_initial = scalar(distributed_energy(u_initial, u_after_one_step))
    energy_final = scalar(distributed_energy(u_previous, u_final))
    if not np.isfinite(energy_initial) or energy_initial <= 0:
        raise RuntimeError("initial discrete energy must be finite and positive")
    relative_energy_drift = abs(energy_final - energy_initial) / abs(energy_initial)
    exact_max_error, exact_rms_error = error_metrics(
        distributed_error_sums,
        u_final,
        exact_final,
        point_count=args.nx * args.ny,
    )
    if not np.isfinite([relative_energy_drift, exact_max_error, exact_rms_error]).all():
        raise RuntimeError("simulation produced non-finite diagnostics")

    laplacian_max_error: float | None = None
    reference_max_error: float | None = None
    if args.validation != "none":
        laplacian_expected = jax.make_array_from_callback(
            shape, field_sharding, lambda index: initial_block(index)[1]
        )
        laplacian_actual = distributed_laplacian(u_initial)
        laplacian_max_error, _ = error_metrics(
            distributed_error_sums,
            laplacian_actual,
            laplacian_expected,
            point_count=args.nx * args.ny,
        )

    if args.validation == "full":
        u_initial_host, laplacian_initial_host, _, _ = build_initial_fields(
            args, dx=dx, dy=dy, dtype=dtype
        )
        u_after_one_step_host = u_initial_host + (
            0.5 * (args.wave_speed * dt) ** 2 * laplacian_initial_host
        )
        reference_solve = jax.jit(
            partial(
                reference_solver,
                wave_speed=args.wave_speed,
                dt=dt,
                dx=dx,
                dy=dy,
            )
        )
        _, reference_final = reference_solve(
            jnp.asarray(u_initial_host),
            jnp.asarray(u_after_one_step_host),
            n_steps,
        )
        reference_final_host = np.asarray(reference_final)
        reference_final_sharded = jax.device_put(reference_final_host, field_sharding)
        reference_max_error, _ = error_metrics(
            distributed_error_sums,
            u_final,
            reference_final_sharded,
            point_count=args.nx * args.ny,
        )

    laplacian_tolerance = 1.0e-11 if args.precision == "float64" else 2.0e-5
    solution_tolerance = 2.0e-11 if args.precision == "float64" else 2.0e-4
    energy_tolerance = 2.0e-11 if args.precision == "float64" else 5.0e-5
    if args.validation != "none":
        if laplacian_max_error is None or laplacian_max_error > laplacian_tolerance:
            raise RuntimeError(
                "halo-exchange validation failed: maximum Laplacian error "
                f"{laplacian_max_error!r} exceeds {laplacian_tolerance}"
            )
        if relative_energy_drift > energy_tolerance:
            raise RuntimeError(
                "energy validation failed: relative drift "
                f"{relative_energy_drift} exceeds {energy_tolerance}"
            )
    if args.validation == "full" and (
        reference_max_error is None or reference_max_error > solution_tolerance
    ):
        raise RuntimeError(
            "distributed/reference validation failed: maximum error "
            f"{reference_max_error!r} exceeds {solution_tolerance}"
        )

    if args.field_output:
        mode = "wb" if args.overwrite else "xb"
        with args.field_output.open(mode) as output_file:
            np.save(output_file, np.asarray(u_final), allow_pickle=False)

    scheduler_environment = {
        name: os.environ[name]
        for name in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_NTASKS",
            "SLURM_PROCID",
            "CUDA_VISIBLE_DEVICES",
        )
        if name in os.environ
    }
    median_execution = median(execution_seconds)
    return {
        "status": "ok",
        "configuration": {
            "nx": args.nx,
            "ny": args.ny,
            "length_x": args.length_x,
            "length_y": args.length_y,
            "wave_speed": args.wave_speed,
            "mode_x": args.mode_x,
            "mode_y": args.mode_y,
            "final_time": final_time,
            "requested_final_time": args.final_time if args.steps is None else None,
            "requested_steps": args.steps,
            "requested_dt": args.dt,
            "nx_per_device": args.nx_per_device,
            "requested_device_count": args.num_devices,
            "requested_device_ids": args.device_ids,
            "cfl": args.cfl,
            "precision": args.precision,
            "validation": args.validation,
            "warmup_runs": args.warmup_runs,
            "repetitions": args.repetitions,
            "requested_backend": args.backend,
            "emulated_cpu_device_count": args.emulate_cpu_devices,
            "distributed_mode": args.distributed,
        },
        "runtime": {
            "jax_version": jax.__version__,
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "backend": backend,
            "device_kinds": sorted(
                {getattr(device, "device_kind", backend) for device in global_devices}
            ),
            "process_count": process_count,
            "global_device_count": int(global_devices.size),
            "local_device_count": len(local_devices),
            "available_global_device_count": len(available_devices),
            "mesh_devices": [
                {
                    "id": device.id,
                    "process_index": device.process_index,
                    "description": str(device),
                }
                for device in selected_devices
            ],
            "scheduler_environment": scheduler_environment,
        },
        "derived": {
            "dx": dx,
            "dy": dy,
            "dt": dt,
            "n_steps": n_steps,
            "stability_number": stability_number,
            "local_nx": args.nx // global_devices.size,
            "grid_point_updates": args.nx * args.ny * n_steps,
        },
        "timing": {
            "compilation_seconds": compilation_seconds,
            "execution_seconds": execution_seconds,
            "median_execution_seconds": median_execution,
            "min_execution_seconds": min(execution_seconds),
            "max_execution_seconds": max(execution_seconds),
            "std_execution_seconds": pstdev(execution_seconds),
            "aggregation": "maximum_across_processes",
            "grid_point_updates_per_second": (
                args.nx * args.ny * n_steps / median_execution
            ),
        },
        "diagnostics": {
            "initial_discrete_energy": energy_initial,
            "final_discrete_energy": energy_final,
            "relative_energy_drift": relative_energy_drift,
            "continuous_exact_max_error": exact_max_error,
            "continuous_exact_rms_error": exact_rms_error,
            "laplacian_max_error": laplacian_max_error,
            "reference_max_error": reference_max_error,
        },
        "outputs": {
            "summary": str(args.output) if args.output else None,
            "field": str(args.field_output) if args.field_output else None,
        },
    }


def write_summary(
    summary: dict[str, Any], output: Path | None, overwrite: bool
) -> None:
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    mode = "w" if overwrite else "x"
    with output.open(mode, encoding="utf-8") as output_file:
        output_file.write(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(args, parser)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    distributed_initialized = False
    try:
        distributed_initialized = configure_jax(args)
        summary = run(args)
        if jax.process_index() == 0:
            write_summary(summary, args.output, args.overwrite)
        return 0
    except (ValueError, RuntimeError) as exc:
        LOGGER.error(
            "simulation failed: %s",
            exc,
            exc_info=LOGGER.isEnabledFor(logging.DEBUG),
        )
        return 1
    except Exception:
        LOGGER.exception("simulation failed unexpectedly")
        return 1
    finally:
        if distributed_initialized:
            jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
