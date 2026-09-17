#!/usr/bin/env python3
"""D2Q9 shear-wave LBM with explicit JAX neighbour exchanges across nodes.

All processes execute the same global-array program. Only locally addressable
slabs are initialized and written. No MPI calls or host population exchanges
are used. The optional full reference validation is restricted to small grids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from statistics import median

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

AXIS = "slabs"
C = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))
WEIGHTS = (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)
POS_X = (1, 5, 8)
NEG_X = (3, 6, 7)
LOGGER = logging.getLogger("lattice_boltzmann")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def device_ids(value: str) -> tuple[int, ...]:
    try:
        ids = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not ids or min(ids) < 0 or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError(
            "device IDs must be distinct non-negative integers"
        )
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--nx", type=positive_int, default=64)
    parser.add_argument("--ny", type=positive_int, default=96)
    parser.add_argument(
        "--steps", type=positive_int, default=600, help="lattice time steps"
    )
    parser.add_argument(
        "--tau", type=finite_float, default=0.8, help="BGK relaxation time"
    )
    parser.add_argument(
        "--amplitude", type=finite_float, default=0.03, help="initial lattice velocity"
    )
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--emulate-cpu-devices", type=positive_int)
    parser.add_argument(
        "--validation",
        choices=("full", "standard", "none"),
        default="full",
        help="full adds a small single-device reference; standard checks routing and physics",
    )
    parser.add_argument(
        "--repetitions",
        type=positive_int,
        default=1,
        help="timed repetitions, each restarting from the initial state",
    )
    parser.add_argument(
        "--distributed", choices=("off", "auto", "manual"), default="off"
    )
    parser.add_argument("--coordinator-address")
    parser.add_argument("--num-processes", type=positive_int)
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--local-device-ids", type=device_ids)
    parser.add_argument("--initialization-timeout", type=positive_int, default=300)
    parser.add_argument(
        "--output", type=Path, help="process-zero JSON summary; otherwise stdout"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="new shared directory for final population shards and manifest",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="allow replacing the JSON summary"
    )
    args = parser.parse_args()
    if min(args.nx, args.ny) < 2:
        parser.error("grid dimensions must be at least two")
    if args.tau <= 0.5:
        parser.error("--tau must exceed 0.5 for positive viscosity")
    if not 0 < args.amplitude * math.sqrt(3) <= 0.1:
        parser.error("--amplitude must be positive with Mach number <= 0.1")
    if args.validation == "full" and args.nx * args.ny > 65536:
        parser.error(
            "full validation is limited to 65536 sites; use --validation standard"
        )
    manual = (
        args.coordinator_address,
        args.num_processes,
        args.process_id,
        args.local_device_ids,
    )
    if args.distributed == "manual":
        if any(value is None for value in manual):
            parser.error(
                "manual mode requires coordinator, process count, process ID, and local device IDs"
            )
        if not 0 <= args.process_id < args.num_processes:
            parser.error("process ID must lie between zero and num-processes minus one")
    elif any(value is not None for value in manual):
        parser.error("manual distributed arguments require --distributed manual")
    if args.emulate_cpu_devices and args.backend == "gpu":
        parser.error("CPU emulation cannot be combined with --backend gpu")
    if (
        args.output
        and args.checkpoint_dir
        and args.output.resolve().is_relative_to(args.checkpoint_dir.resolve())
    ):
        parser.error("--output must be outside the checkpoint directory")
    return args


def configure_jax(args: argparse.Namespace) -> None:
    """No device queries or JAX array creation may precede this function."""
    if args.emulate_cpu_devices:
        jax.config.update("jax_platform_name", "cpu")
        jax.config.update("jax_num_cpu_devices", args.emulate_cpu_devices)
    elif args.backend != "auto":
        jax.config.update("jax_platform_name", args.backend)
    jax.config.update("jax_enable_x64", args.precision == "float64")
    if args.distributed == "auto":
        jax.distributed.initialize(initialization_timeout=args.initialization_timeout)
    elif args.distributed == "manual":
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
            local_device_ids=args.local_device_ids,
            initialization_timeout=args.initialization_timeout,
        )


def gather_small(value: np.ndarray) -> np.ndarray:
    if jax.process_count() == 1:
        return np.asarray(value)[None, ...]
    return np.asarray(multihost_utils.process_allgather(value, tiled=False))


def synchronize(label: str) -> None:
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices(label)


def validate_configuration(args: argparse.Namespace) -> None:
    """Reject divergent rank settings before constructing numerical collectives."""
    settings = {
        name: getattr(args, name)
        for name in (
            "nx",
            "ny",
            "steps",
            "tau",
            "amplitude",
            "precision",
            "validation",
            "repetitions",
            "overwrite",
            "backend",
            "emulate_cpu_devices",
        )
    }
    for name in ("output", "checkpoint_dir"):
        path = getattr(args, name)
        settings[name] = str(path.resolve()) if path else None
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).digest()
    digests = gather_small(np.frombuffer(digest, dtype=np.uint8))
    if np.any(digests != digests[0]):
        raise ValueError("processes have different run parameters or output paths")


def collective_io(action: Callable[[], None], description: str) -> None:
    """Propagate ordinary I/O failures before any rank enters the next collective."""
    error = None
    try:
        action()
    except OSError as exc:
        error = str(exc)
        LOGGER.error("rank=%d %s: %s", jax.process_index(), description, error)
    failed = gather_small(np.asarray(error is not None, dtype=np.int32))
    if np.any(failed):
        raise OSError(
            f"{description} failed on at least one process; inspect rank logs"
        )


def prepare_output(args: argparse.Namespace) -> None:
    def check() -> None:
        if jax.process_index() != 0:
            return
        if args.output:
            if not args.output.parent.is_dir():
                raise OSError(
                    f"summary parent directory does not exist: {args.output.parent}"
                )
            if args.output.exists() and not args.overwrite:
                raise FileExistsError(f"summary exists: {args.output}; use --overwrite")
            if args.output.exists() and not args.output.is_file():
                raise OSError(f"summary path is not a regular file: {args.output}")
            writable_path = args.output if args.output.exists() else args.output.parent
            if not os.access(writable_path, os.W_OK):
                raise OSError(f"summary destination is not writable: {writable_path}")
        if args.checkpoint_dir:
            # Never remove or replace a checkpoint directory, even with --overwrite.
            args.checkpoint_dir.mkdir()

    collective_io(check, "output preparation")


def shear_block(index: tuple[slice, ...], args: argparse.Namespace) -> np.ndarray:
    """Initialize only the requested global slice on this process's host."""
    dtype = np.dtype(args.precision)
    x = np.arange(*index[0].indices(args.nx), dtype=dtype)[:, None]
    ny = len(range(*index[1].indices(args.ny)))
    uy = np.broadcast_to(args.amplitude * np.sin(2 * np.pi * x / args.nx), (len(x), ny))
    cy = np.asarray([cy for _, cy in C], dtype=dtype)
    cu = uy[..., None] * cy
    return np.asarray(
        np.asarray(WEIGHTS, dtype=dtype)
        * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * uy[..., None] ** 2),
        dtype=dtype,
    )


def make_operators(mesh: Mesh, args: argparse.Namespace):
    """Build local collision, explicit ring streaming, and replicated reductions."""
    dtype = np.dtype(args.precision)
    cx = jnp.asarray([cx for cx, _ in C], dtype=dtype)
    cy = jnp.asarray([cy for _, cy in C], dtype=dtype)
    weights = jnp.asarray(WEIGHTS, dtype=dtype)
    spec = P(AXIS, None, None)
    send_right = tuple((p, (p + 1) % mesh.size) for p in range(mesh.size))
    send_left = tuple((p, (p - 1) % mesh.size) for p in range(mesh.size))

    def macroscopic(f):
        rho = jnp.sum(f, axis=-1)
        return rho, jnp.sum(f * cx, axis=-1) / rho, jnp.sum(f * cy, axis=-1) / rho

    def collision(f):
        rho, ux, uy = macroscopic(f)
        cu = ux[..., None] * cx + uy[..., None] * cy
        eq = (
            rho[..., None]
            * weights
            * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * (ux**2 + uy**2)[..., None])
        )
        return f + (eq - f) / args.tau

    @partial(jax.shard_map, mesh=mesh, in_specs=spec, out_specs=spec, axis_names={AXIS})
    def stream(f_post):
        from_left = lax.ppermute(
            jnp.take(f_post[-1], jnp.asarray(POS_X), axis=-1), AXIS, send_right
        )
        from_right = lax.ppermute(
            jnp.take(f_post[0], jnp.asarray(NEG_X), axis=-1), AXIS, send_left
        )
        populations = []
        for q, (vx, vy) in enumerate(C):
            shifted = jnp.roll(f_post[..., q], vx, axis=0)
            if vx == 1:
                shifted = shifted.at[0].set(from_left[:, POS_X.index(q)])
            elif vx == -1:
                shifted = shifted.at[-1].set(from_right[:, NEG_X.index(q)])
            populations.append(jnp.roll(shifted, vy, axis=1))
        return jnp.stack(populations, axis=-1)

    @jax.jit
    def solve(f, steps):
        return lax.fori_loop(0, steps, lambda _, state: stream(collision(state)), f)

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(spec, spec),
        out_specs=P(),
        axis_names={AXIS},
    )
    def max_difference(actual, expected):
        return lax.pmax(jnp.max(jnp.abs(actual - expected)), AXIS)

    @partial(
        jax.shard_map, mesh=mesh, in_specs=(spec, P()), out_specs=P(), axis_names={AXIS}
    )
    def diagnostics(f, steps):
        rho, ux, uy = macroscopic(f)
        sums = lax.psum(
            jnp.stack(
                (
                    jnp.sum(rho),
                    jnp.sum(rho * ux),
                    jnp.sum(rho * uy),
                    0.5 * jnp.sum(rho * (ux**2 + uy**2)),
                )
            ),
            AXIS,
        )
        # The mesh position, not the process ID, determines the global x offset.
        x = lax.axis_index(AXIS) * f.shape[0] + jnp.arange(f.shape[0], dtype=dtype)
        k = 2 * jnp.pi / args.nx
        nu = (args.tau - 0.5) / 3
        expected_uy = args.amplitude * jnp.sin(k * x) * jnp.exp(-nu * k**2 * steps)
        velocity_error = (
            lax.pmax(
                jnp.maximum(
                    jnp.max(jnp.abs(uy - expected_uy[:, None])), jnp.max(jnp.abs(ux))
                ),
                AXIS,
            )
            / args.amplitude
        )
        min_rho = lax.pmin(jnp.min(rho), AXIS)
        finite = lax.pmin(jnp.all(jnp.isfinite(f)).astype(jnp.int32), AXIS)
        return jnp.concatenate((sums, jnp.stack((velocity_error, min_rho, finite))))

    @jax.jit
    def reference_solve(f, steps):
        def step(_, state):
            post = collision(state)
            return jnp.stack(
                [
                    jnp.roll(post[..., q], (vx, vy), axis=(0, 1))
                    for q, (vx, vy) in enumerate(C)
                ],
                axis=-1,
            )

        return lax.fori_loop(0, steps, step, f)

    return (
        solve,
        jax.jit(stream),
        jax.jit(diagnostics),
        jax.jit(max_difference),
        reference_solve,
    )


def validate_routing(mesh: Mesh, stream, difference, dtype: np.dtype) -> None:
    """Independent upstream-index oracle, including one-row slabs and diagonals."""
    sharding = NamedSharding(mesh, P(AXIS, None, None))
    for local_nx in (1, 3):
        nx, ny = mesh.size * local_nx, 7

        def labels(index, *, upstream=False, nx=nx, ny=ny):
            x = np.arange(*index[0].indices(nx))[:, None]
            y = np.arange(*index[1].indices(ny))[None, :]
            arrays = []
            for q, (vx, vy) in enumerate(C):
                sx = (x - vx) % nx if upstream else x
                sy = (y - vy) % ny if upstream else y
                # Small integers remain exactly representable in float32.
                arrays.append(((sx * ny + sy) * 9 + q) % 100003)
            return np.stack(arrays, axis=-1).astype(dtype)

        tags = jax.make_array_from_callback((nx, ny, 9), sharding, labels, dtype=dtype)
        expected = jax.make_array_from_callback(
            (nx, ny, 9), sharding, partial(labels, upstream=True), dtype=dtype
        )
        if float(difference(stream(tags), expected)) != 0:
            raise ValueError(f"streaming routing check failed for {local_nx}-row slabs")


def write_checkpoint(
    f: jax.Array, mesh: Mesh, args: argparse.Namespace, summary: dict
) -> None:
    directory = args.checkpoint_dir
    if directory is None:
        return
    width = args.nx // mesh.size

    def write_local() -> None:
        for shard in f.addressable_shards:
            slab = shard.index[0].indices(args.nx)[0] // width
            with (directory / f"slab-{slab:06d}.npy").open("xb") as file:
                np.save(file, np.asarray(shard.data), allow_pickle=False)

    collective_io(write_local, "checkpoint shard write")

    def write_manifest() -> None:
        if jax.process_index() == 0:
            manifest = {
                "format": "jax-lbm-slabs-v1",
                "complete": True,
                "shape": list(f.shape),
                "dtype": str(f.dtype),
                "population_velocities": C,
                "run": summary,
                "shards": [
                    {
                        "file": f"slab-{p:06d}.npy",
                        "x_start": p * width,
                        "x_stop": (p + 1) * width,
                        "process": device.process_index,
                    }
                    for p, device in enumerate(mesh.devices.flat)
                ],
            }
            with (directory / "manifest.json").open("x", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2, allow_nan=False)
                file.write("\n")

    # A manifest is published only after every process has finished its shards.
    collective_io(write_manifest, "checkpoint manifest write")


def run(args: argparse.Namespace) -> dict:
    # Group by controller. Under a block-mapped one-process-per-GPU Slurm launch,
    # this also groups adjacent slabs on the same node. Verify site placement.
    validate_configuration(args)
    devices = sorted(
        jax.devices(), key=lambda device: (device.process_index, device.id)
    )
    if args.nx % len(devices):
        raise ValueError(
            f"nx={args.nx} must be divisible by global device count {len(devices)}"
        )
    mesh = Mesh(np.asarray(devices), (AXIS,))
    sharding = NamedSharding(mesh, P(AXIS, None, None))
    dtype = np.dtype(args.precision)
    prepare_output(args)
    solve, stream, diagnostics, difference, reference = make_operators(mesh, args)
    if args.validation != "none":
        validate_routing(mesh, stream, difference, dtype)
    initial = jax.make_array_from_callback(
        (args.nx, args.ny, 9), sharding, partial(shear_block, args=args), dtype=dtype
    )
    initial.block_until_ready()
    initial_totals = np.asarray(diagnostics(initial, 0), dtype=np.float64)
    LOGGER.info(
        "rank=%d/%d backend=%s global_devices=%d local_devices=%d grid=%dx%d steps=%d",
        jax.process_index(),
        jax.process_count(),
        jax.default_backend(),
        len(devices),
        jax.local_device_count(),
        args.nx,
        args.ny,
        args.steps,
    )
    started = time.perf_counter()
    executable = solve.lower(initial, args.steps).compile()
    compilation_seconds = float(
        np.max(gather_small(np.asarray(time.perf_counter() - started)))
    )
    executable(initial, args.steps).block_until_ready()  # Untimed warmup on every rank.
    elapsed = []
    final = initial
    for repetition in range(args.repetitions):
        synchronize(f"lbm-start-{repetition}")
        started = time.perf_counter()
        final = executable(initial, args.steps)
        final.block_until_ready()
        # Gathering timers is outside the measured solve. Use the slowest rank.
        elapsed.append(
            float(np.max(gather_small(np.asarray(time.perf_counter() - started))))
        )
    totals = np.asarray(diagnostics(final, args.steps), dtype=np.float64)
    if not np.isfinite(totals).all() or not totals[6] or totals[5] <= 0:
        raise ValueError(
            "simulation has non-finite populations or non-positive density"
        )
    nu = (args.tau - 0.5) / 3
    energy_ratio = totals[3] / initial_totals[3]
    expected_ratio = math.exp(-2 * nu * (2 * math.pi / args.nx) ** 2 * args.steps)
    mass_drift = abs(totals[0] / initial_totals[0] - 1)
    momentum_drift = float(
        np.max(np.abs(totals[1:3] - initial_totals[1:3])) / (args.nx * args.ny)
    )
    reference_error = None
    if args.validation == "full":
        # Intentionally small: each controller runs a local, unsharded oracle.
        whole_index = (slice(0, args.nx), slice(0, args.ny), slice(0, 9))
        ref_initial = jax.device_put(
            shear_block(whole_index, args), jax.local_devices()[0]
        )
        ref_final = np.asarray(reference(ref_initial, args.steps))
        expected = jax.make_array_from_callback(
            final.shape, sharding, lambda index: ref_final[index], dtype=dtype
        )
        reference_error = float(difference(final, expected))
    if args.validation != "none":
        mass_tol = 1e-9 if args.precision == "float64" else 1e-4
        momentum_tol = 1e-10 if args.precision == "float64" else 2e-6
        comparison_tol = 1e-11 if args.precision == "float64" else 2e-6
        if mass_drift > mass_tol or momentum_drift > momentum_tol:
            raise ValueError(
                "conservation check failed; inspect precision, step count, and parameters"
            )
        # Physical accuracy checks are calibrated for the tutorial regime. For
        # very narrow grids or extreme tau, enforce routing/conservation only.
        if (
            args.nx >= 64
            and 0.7 <= args.tau <= 1.0
            and (abs(energy_ratio - expected_ratio) > 0.005 or totals[4] > 0.005)
        ):
            raise ValueError("shear-wave continuum decay check failed")
        if reference_error is not None and reference_error > comparison_tol:
            raise ValueError("single-device reference check failed")
    seconds = median(elapsed)
    summary = {
        "model": "D2Q9-BGK-periodic-shear",
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "process_count": jax.process_count(),
        "device_count": len(devices),
        "mesh_order": [str(device) for device in devices],
        "parameters": {
            "nx": args.nx,
            "ny": args.ny,
            "steps": args.steps,
            "tau": args.tau,
            "viscosity": nu,
            "amplitude": args.amplitude,
            "precision": args.precision,
        },
        "validation": {
            "mode": args.validation,
            "routing_passed": args.validation != "none",
            "relative_mass_drift": float(mass_drift),
            "momentum_drift_per_site": momentum_drift,
            "energy_ratio": float(energy_ratio),
            "continuum_energy_ratio": expected_ratio,
            "velocity_error_over_amplitude": float(totals[4]),
            "reference_max_error": reference_error,
            "continuum_check_applied": args.validation != "none"
            and args.nx >= 64
            and 0.7 <= args.tau <= 1.0,
        },
        "timing": {
            "compilation_seconds_max_rank": compilation_seconds,
            "seconds_max_rank": elapsed,
            "median_seconds": seconds,
            "mlups": args.nx * args.ny * args.steps / (1e6 * seconds),
        },
        "sent_bytes_per_device_per_step": 6 * args.ny * dtype.itemsize,
    }
    write_checkpoint(final, mesh, args, summary)
    return summary


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr
    )
    initialized = False
    try:
        configure_jax(args)
        initialized = args.distributed != "off"
        summary = run(args)

        def write_summary() -> None:
            if jax.process_index() == 0:
                if args.output:
                    with args.output.open(
                        "w" if args.overwrite else "x", encoding="utf-8"
                    ) as file:
                        json.dump(summary, file, indent=2, allow_nan=False)
                        file.write("\n")
                else:
                    print(json.dumps(summary, indent=2, allow_nan=False))

        collective_io(write_summary, "summary write")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    finally:
        if initialized:
            jax.distributed.shutdown()


if __name__ == "__main__":
    sys.exit(main())
