"""
Benchmark for Mandelbrot implementations.

This script compares the NumPy, Numba CPU, and CUDA versions.
It measures execution time for different image sizes and computes speedups.
"""

import gc
import math
import statistics
import time

import numpy as np
from numba import njit, cuda


X_MIN = -2.0
X_MAX = 1.0
Y_MIN = -1.5
Y_MAX = 1.5
ESCAPE_RADIUS = 2.0


def mandelbrot_numpy(width, height, max_iter):
    """
    Compute the Mandelbrot set using NumPy.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        max_iter: Maximum number of iterations.
        x_min: Minimum real value.
        x_max: Maximum real value.
        y_min: Minimum imaginary value.
        y_max: Maximum imaginary value.
        escape_radius: Radius used to decide if a point escapes.

    Returns:
        A 2D NumPy array with normalized escape values between 0 and 1.
    """
    
    x = np.linspace(X_MIN, X_MAX, width)
    y = np.linspace(Y_MIN, Y_MAX, height)
    real_grid, imaginary_grid = np.meshgrid(x, y)
    complex_grid = real_grid + 1j * imaginary_grid

    z_values = np.zeros_like(complex_grid, dtype=np.complex128)
    result = np.ones(complex_grid.shape, dtype=np.float64)
    active = np.ones(complex_grid.shape, dtype=bool)

    for iteration in range(max_iter):
        z_values[active] = z_values[active] * z_values[active] + complex_grid[active]

        escaped_now = np.abs(z_values) > ESCAPE_RADIUS
        newly_escaped = escaped_now & active

        result[newly_escaped] = (iteration + 1) / max_iter
        active[newly_escaped] = False

        if not np.any(active):
            break

    result[active] = 1.0

    return result


@njit
def mandelbrot_numba_cpu(
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
    Compute the Mandelbrot set using Numba on the CPU.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        max_iter: Maximum number of iterations.
        x_min: Minimum real value.
        x_max: Maximum real value.
        y_min: Minimum imaginary value.
        y_max: Maximum imaginary value.
        escape_radius: Radius used to decide if a point escapes.

    Returns:
        A 2D NumPy array with normalized escape values.
    """
    result = np.zeros((height, width), dtype=np.float64)

    for row in range(height):
        for col in range(width):
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

    return result


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
    CUDA kernel for the Mandelbrot computation.

    Args:
        result: Output array stored on the GPU.
        width: Image width in pixels.
        height: Image height in pixels.
        max_iter: Maximum number of iterations.
        x_min: Minimum real value.
        x_max: Maximum real value.
        y_min: Minimum imaginary value.
        y_max: Maximum imaginary value.
        escape_radius: Radius used to decide if a point escapes.

    Returns:
        None. The result is written directly into the GPU output array.
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


def mandelbrot_cuda(width, height, max_iter, threads_per_block=(8, 4)):
    """
    Compute the Mandelbrot set using CUDA.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        max_iter: Maximum number of iterations.
        threads_per_block: CUDA block size.

    Returns:
        A 2D NumPy array copied from GPU memory to CPU memory.
    """
    result_device = cuda.device_array((height, width), dtype=np.float64)

    blocks_per_grid_x = math.ceil(width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

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

    return result_device.copy_to_host()


def measure_time(function, repetitions=3):
    """
    Measure the execution time of a function.

    Args:
        function: Function to execute.
        repetitions: Number of times to repeat the measurement.

    Returns:
        Median execution time in seconds.
    """
    times = []

    for _ in range(repetitions):
        gc.collect()

        start = time.perf_counter()
        function()
        end = time.perf_counter()

        times.append(end - start)

    return statistics.median(times)


def main():
    """
    Run the benchmark for all Mandelbrot implementations.

    Args:
        None.

    Returns:
        None.
    """
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")

    max_iter = 200
    threads_per_block = (8, 4)

    sizes = [
        (500, 375),
        (1000, 750),
        (2000, 1500),
        (3000, 2250),
        (4000, 3000),
    ]

    print("CUDA device:", cuda.get_current_device().name)
    print("CUDA warp size: 32")
    print("Selected CUDA block size:", threads_per_block)
    print("Threads per block:", threads_per_block[0] * threads_per_block[1])
    print("Max iterations:", max_iter)
    print()

    print("Warming up Numba CPU and CUDA compilation...")
    mandelbrot_numba_cpu(
        64,
        64,
        10,
        X_MIN,
        X_MAX,
        Y_MIN,
        Y_MAX,
        ESCAPE_RADIUS,
    )
    mandelbrot_cuda(64, 64, 10, threads_per_block=threads_per_block)
    print("Warm-up complete.")
    print()

    print("Correctness check on small input...")
    numpy_result = mandelbrot_numpy(128, 96, 50)
    numba_result = mandelbrot_numba_cpu(
        128,
        96,
        50,
        X_MIN,
        X_MAX,
        Y_MIN,
        Y_MAX,
        ESCAPE_RADIUS,
    )
    cuda_result = mandelbrot_cuda(128, 96, 50, threads_per_block=threads_per_block)

    np.testing.assert_allclose(numba_result, numpy_result, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cuda_result, numpy_result, rtol=1e-12, atol=1e-12)
    print("Correctness check passed.")
    print()

    header = (
        f"{'Size':>14} | "
        f"{'Pixels':>12} | "
        f"{'NumPy (s)':>12} | "
        f"{'Numba CPU (s)':>14} | "
        f"{'CUDA (s)':>12} | "
        f"{'Speedup NumPy/CUDA':>20} | "
        f"{'Speedup Numba/CUDA':>21}"
    )

    print(header)
    print("-" * len(header))

    for width, height in sizes:
        pixels = width * height

        numpy_time = measure_time(
            lambda: mandelbrot_numpy(width, height, max_iter),
            repetitions=3,
        )

        numba_cpu_time = measure_time(
            lambda: mandelbrot_numba_cpu(
                width,
                height,
                max_iter,
                X_MIN,
                X_MAX,
                Y_MIN,
                Y_MAX,
                ESCAPE_RADIUS,
            ),
            repetitions=3,
        )

        cuda_time = measure_time(
            lambda: mandelbrot_cuda(
                width,
                height,
                max_iter,
                threads_per_block=threads_per_block,
            ),
            repetitions=3,
        )

        speedup_numpy_cuda = numpy_time / cuda_time
        speedup_numba_cuda = numba_cpu_time / cuda_time

        print(
            f"{width}x{height: <7} | "
            f"{pixels:12d} | "
            f"{numpy_time:12.6f} | "
            f"{numba_cpu_time:14.6f} | "
            f"{cuda_time:12.6f} | "
            f"{speedup_numpy_cuda:20.2f} | "
            f"{speedup_numba_cuda:21.2f}"
        )


if __name__ == "__main__":
    main()