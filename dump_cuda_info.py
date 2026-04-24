import sys

#
# pip install pycuda
#

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # Automatically initializes CUDA driver
except ImportError:
    print("PyCUDA is not installed. Install it with: pip install pycuda")
    sys.exit(1)


def dump_cuda_info():
    """Prints detailed CUDA device information for all available GPUs."""
    device_count = cuda.Device.count()
    if device_count == 0:
        print("No CUDA-capable devices found.")
        return

    print(f"Found {device_count} CUDA device(s):\n")
    for i in range(device_count):
        device = cuda.Device(i)
        attrs = device.get_attributes()

        print(f"=== Device {i}: {device.name()} ===")
        print(f"  Compute Capability: {device.compute_capability()[0]}.{device.compute_capability()[1]}")
        print(f"  Total Memory: {device.total_memory() / (1024**2):.2f} MB")
        print(f"  Multiprocessors: {attrs.get(cuda.device_attribute.MULTIPROCESSOR_COUNT, 'N/A')}")
        print(f"  Max Threads per Block: {attrs.get(cuda.device_attribute.MAX_THREADS_PER_BLOCK, 'N/A')}")
        print(f"  Max Threads Dim: {attrs.get(cuda.device_attribute.MAX_BLOCK_DIM_X, 'N/A')}, "
              f"{attrs.get(cuda.device_attribute.MAX_BLOCK_DIM_Y, 'N/A')}, "
              f"{attrs.get(cuda.device_attribute.MAX_BLOCK_DIM_Z, 'N/A')}")
        print(f"  Max Grid Size: {attrs.get(cuda.device_attribute.MAX_GRID_DIM_X, 'N/A')}, "
              f"{attrs.get(cuda.device_attribute.MAX_GRID_DIM_Y, 'N/A')}, "
              f"{attrs.get(cuda.device_attribute.MAX_GRID_DIM_Z, 'N/A')}")
        print(f"  Clock Rate: {attrs.get(cuda.device_attribute.CLOCK_RATE, 0) / 1000:.2f} MHz")
        print(f"  Memory Clock Rate: {attrs.get(cuda.device_attribute.MEMORY_CLOCK_RATE, 0) / 1000:.2f} MHz")
        print(f"  Memory Bus Width: {attrs.get(cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH, 'N/A')} bits")
        print(f"  L2 Cache Size: {attrs.get(cuda.device_attribute.L2_CACHE_SIZE, 'N/A')} bytes")
        print(f"  Warp Size: {attrs.get(cuda.device_attribute.WARP_SIZE, 'N/A')}")
        print(f"  Unified Addressing: {bool(attrs.get(cuda.device_attribute.UNIFIED_ADDRESSING, 0))}")
        print()

if __name__ == "__main__":
    dump_cuda_info()
