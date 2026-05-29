#!/usr/bin/env python3
"""Print CUDA device information using PyCUDA."""

import sys

import pycuda.driver as drv


DEVICE_ATTRIBUTES = [
    ("Clock rate", drv.device_attribute.CLOCK_RATE, "MHz", lambda value: value / 1000),
    ("Warp size", drv.device_attribute.WARP_SIZE, None, None),
    ("Multiprocessors", drv.device_attribute.MULTIPROCESSOR_COUNT, None, None),
    (
        "Max threads per multiprocessor",
        drv.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR,
        None,
        None,
    ),
    ("Max threads per block", drv.device_attribute.MAX_THREADS_PER_BLOCK, None, None),
    ("Max block dim x", drv.device_attribute.MAX_BLOCK_DIM_X, None, None),
    ("Max block dim y", drv.device_attribute.MAX_BLOCK_DIM_Y, None, None),
    ("Max block dim z", drv.device_attribute.MAX_BLOCK_DIM_Z, None, None),
    ("Max grid dim x", drv.device_attribute.MAX_GRID_DIM_X, None, None),
    ("Max grid dim y", drv.device_attribute.MAX_GRID_DIM_Y, None, None),
    ("Max grid dim z", drv.device_attribute.MAX_GRID_DIM_Z, None, None),
    ("Max registers per block", drv.device_attribute.MAX_REGISTERS_PER_BLOCK, None, None),
    (
        "Max registers per multiprocessor",
        drv.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR,
        None,
        None,
    ),
    (
        "Max shared memory per block",
        drv.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK,
        "bytes",
        None,
    ),
    (
        "Max shared memory per multiprocessor",
        drv.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        "bytes",
        None,
    ),
    ("L2 cache size", drv.device_attribute.L2_CACHE_SIZE, "bytes", None),
]


def format_value(value, unit=None, transform=None):
    if transform is not None:
        value = transform(value)
    if isinstance(value, float):
        formatted = f"{value:.1f}"
    else:
        formatted = str(value)
    if unit:
        return f"{formatted} {unit}"
    return formatted


def print_device_info(device_id):
    device = drv.Device(device_id)
    major, minor = device.compute_capability()

    print(f"Device {device_id}: {device.name()}")
    print(f"  Compute capability: {major}.{minor}")
    print(f"  Total memory: {device.total_memory() / 1024**3:.2f} GiB")

    for label, attribute, unit, transform in DEVICE_ATTRIBUTES:
        value = device.get_attribute(attribute)
        print(f"  {label}: {format_value(value, unit, transform)}")


def main():
    try:
        drv.init()
    except drv.Error as error:
        print(f"Could not initialize CUDA driver: {error}", file=sys.stderr)
        return 1

    device_count = drv.Device.count()
    print(f"CUDA devices: {device_count}")

    if device_count == 0:
        return 1

    for device_id in range(device_count):
        if device_id > 0:
            print()
        print_device_info(device_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
