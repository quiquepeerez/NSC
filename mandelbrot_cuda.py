"""
CUDA implementation of the Mandelbrot set using Numba.

Each CUDA thread computes one pixel of the output image.
"""

import math
import time

import numpy as np
from numba import cuda


X_MIN = -2.0
X_MAX = 1.0
Y_MIN = -1.5
Y_MAX = 1.5
ESCAPE_RADIUS = 2.0


@cuda.jit
def mandelbrot_cuda_kernel(
    result,
    width,
    height,
    max_iter,
    x_min,
    x_max,
    y_min,
    y_max,
    escape_radius,
):
    """
    CUDA kernel that computes the Mandelbrot escape value for each pixel.

    Parameters
    ----------
    result : numpy.ndarray
        2D output array stored on the GPU. Each thread writes one value.
    width : int
        Number of pixels along the real axis.
    height : int
        Number of pixels along the imaginary axis.
    max_iter : int
        Maximum number of Mandelbrot iterations.
    x_min, x_max : float
        Bounds of the real axis.
    y_min, y_max : float
        Bounds of the imaginary axis.
    escape_radius : float
        Escape radius used to decide whether a point leaves the set.

    Returns
    -------
    None
        The output is written directly into the result array.
    """
    col, row = cuda.grid(2)

    if row >= height or col >= width:
        return

    x = x_min + (col / (width - 1)) * (x_max - x_min)
    y = y_min + (row / (height - 1)) * (y_max - y_min)

    cr = x
    ci = y

    zr = 0.0
    zi = 0.0

    escape_iter = max_iter
    escape_radius_squared = escape_radius * escape_radius

    for iteration in range(max_iter):
        zr_new = zr * zr - zi * zi + cr
        zi_new = 2.0 * zr * zi + ci

        zr = zr_new
        zi = zi_new

        if zr * zr + zi * zi > escape_radius_squared:
            escape_iter = iteration + 1
            break

    result[row, col] = escape_iter / max_iter


def mandelbrot_cuda(
    width,
    height,
    max_iter,
    threads_per_block=(16, 16),
    x_min=X_MIN,
    x_max=X_MAX,
    y_min=Y_MIN,
    y_max=Y_MAX,
    escape_radius=ESCAPE_RADIUS,
):
    """
    Compute the Mandelbrot set using a CUDA kernel.

    Parameters
    ----------
    width : int
        Number of pixels along the real axis.
    height : int
        Number of pixels along the imaginary axis.
    max_iter : int
        Maximum number of iterations.
    threads_per_block : tuple of int, optional
        CUDA block shape, for example (16, 16). The product gives the total
        number of threads per block.
    x_min, x_max : float, optional
        Real-axis bounds.
    y_min, y_max : float, optional
        Imaginary-axis bounds.
    escape_radius : float, optional
        Escape radius.

    Returns
    -------
    numpy.ndarray
        2D array with shape (height, width), containing normalized escape
        values between 0.0 and 1.0.
    """
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. Check your NVIDIA GPU, driver, and CUDA setup.")

    result_device = cuda.device_array((height, width), dtype=np.float64)

    blocks_per_grid_x = math.ceil(width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    mandelbrot_cuda_kernel[blocks_per_grid, threads_per_block](
        result_device,
        width,
        height,
        max_iter,
        x_min,
        x_max,
        y_min,
        y_max,
        escape_radius,
    )

    cuda.synchronize()

    return result_device.copy_to_host()


def benchmark_block_sizes(
    width=2000,
    height=1500,
    max_iter=200,
    repetitions=5,
):
    """
    Benchmark different CUDA block sizes for the Mandelbrot kernel.

    Parameters
    ----------
    width : int
        Image width used in the benchmark.
    height : int
        Image height used in the benchmark.
    max_iter : int
        Maximum number of Mandelbrot iterations.
    repetitions : int
        Number of timed repetitions for each block size.

    Returns
    -------
    list of dict
        Benchmark results sorted by execution time.
    """
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. Check your NVIDIA GPU, driver, and CUDA setup.")

    block_sizes = [
        (8, 4),     # 32 threads
        (8, 8),     # 64 threads
        (16, 8),    # 128 threads
        (16, 16),   # 256 threads
        (32, 16),   # 512 threads
        (32, 32),   # 1024 threads
    ]

    results = []

    for threads_per_block in block_sizes:
        total_threads = threads_per_block[0] * threads_per_block[1]

        result_device = cuda.device_array((height, width), dtype=np.float64)

        blocks_per_grid_x = math.ceil(width / threads_per_block[0])
        blocks_per_grid_y = math.ceil(height / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Warm-up run: triggers compilation and avoids measuring JIT overhead.
        mandelbrot_cuda_kernel[blocks_per_grid, threads_per_block](
            result_device,
            width,
            height,
            max_iter,
            X_MIN,
            X_MAX,
            Y_MIN,
            Y_MAX,
            ESCAPE_RADIUS,
        )
        cuda.synchronize()

        timings = []

        for _ in range(repetitions):
            start = time.perf_counter()

            mandelbrot_cuda_kernel[blocks_per_grid, threads_per_block](
                result_device,
                width,
                height,
                max_iter,
                X_MIN,
                X_MAX,
                Y_MIN,
                Y_MAX,
                ESCAPE_RADIUS,
            )

            cuda.synchronize()
            end = time.perf_counter()

            timings.append(end - start)

        average_time = sum(timings) / len(timings)

        results.append(
            {
                "threads_per_block": threads_per_block,
                "total_threads": total_threads,
                "blocks_per_grid": blocks_per_grid,
                "average_time_seconds": average_time,
            }
        )

    results.sort(key=lambda item: item["average_time_seconds"])

    return results


if __name__ == "__main__":
    print("Running CUDA Mandelbrot benchmark...")

    benchmark_results = benchmark_block_sizes(
        width=2000,
        height=1500,
        max_iter=200,
        repetitions=5,
    )

    print("\nBlock size benchmark results:")
    print("-" * 80)
    print(f"{'Block size':>15} | {'Threads/block':>13} | {'Grid size':>18} | {'Avg time (s)':>12}")
    print("-" * 80)

    for row in benchmark_results:
        print(
            f"{str(row['threads_per_block']):>15} | "
            f"{row['total_threads']:>13} | "
            f"{str(row['blocks_per_grid']):>18} | "
            f"{row['average_time_seconds']:>12.6f}"
        )

    best = benchmark_results[0]

    print("\nBest configuration:")
    print(f"threads_per_block = {best['threads_per_block']}")
    print(f"total threads per block = {best['total_threads']}")
    print(f"average time = {best['average_time_seconds']:.6f} seconds")