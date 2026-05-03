"""
NumPy implementation of the Mandelbrot set.

This module computes a normalized Mandelbrot escape-time image using vectorized
NumPy operations. The output is a 2D array where each value represents the
normalized iteration at which a complex point escapes. Points that do not escape
within the maximum number of iterations are assigned the value 1.0.
"""

import numpy as np


X_MIN = -2.0
X_MAX = 1.0
Y_MIN = -1.5
Y_MAX = 1.5
ESCAPE_RADIUS = 2.0


def validate_mandelbrot_inputs(width, height, max_iter, escape_radius=ESCAPE_RADIUS):
    """
    Validate the input parameters used to compute the Mandelbrot set.

    Parameters
    ----------
    width : int
        Number of pixels along the real axis. Must be a positive integer.
    height : int
        Number of pixels along the imaginary axis. Must be a positive integer.
    max_iter : int
        Maximum number of iterations used to test whether a point escapes.
        Must be a positive integer.
    escape_radius : float, optional
        Radius used to decide whether a point has escaped. Must be positive.

    Returns
    -------
    tuple
        A tuple containing validated versions of width, height, max_iter,
        and escape_radius.

    Raises
    ------
    ValueError
        If width, height, or max_iter are not positive integers, or if
        escape_radius is not positive.

    Examples
    --------
    >>> validate_mandelbrot_inputs(100, 80, 50)
    (100, 80, 50, 2.0)

    >>> validate_mandelbrot_inputs(0, 80, 50)
    Traceback (most recent call last):
        ...
    ValueError: width must be a positive integer
    """
    if not isinstance(width, (int, np.integer)) or width <= 0:
        raise ValueError("width must be a positive integer")

    if not isinstance(height, (int, np.integer)) or height <= 0:
        raise ValueError("height must be a positive integer")

    if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    if escape_radius <= 0:
        raise ValueError("escape_radius must be positive")

    return int(width), int(height), int(max_iter), float(escape_radius)


def create_complex_grid(
    width,
    height,
    x_min=X_MIN,
    x_max=X_MAX,
    y_min=Y_MIN,
    y_max=Y_MAX,
):
    """
    Create a 2D grid of complex numbers for the Mandelbrot computation.

    Parameters
    ----------
    width : int
        Number of points along the real axis.
    height : int
        Number of points along the imaginary axis.
    x_min : float, optional
        Minimum value of the real axis.
    x_max : float, optional
        Maximum value of the real axis.
    y_min : float, optional
        Minimum value of the imaginary axis.
    y_max : float, optional
        Maximum value of the imaginary axis.

    Returns
    -------
    numpy.ndarray
        A 2D complex-valued array with shape ``(height, width)``.

    Raises
    ------
    ValueError
        If x_min is greater than x_max, or if y_min is greater than y_max.

    Examples
    --------
    >>> grid = create_complex_grid(2, 2, x_min=0, x_max=1, y_min=0, y_max=1)
    >>> grid.shape
    (2, 2)

    >>> grid[0, 0]
    0j
    """
    if x_min > x_max:
        raise ValueError("x_min must be less than or equal to x_max")

    if y_min > y_max:
        raise ValueError("y_min must be less than or equal to y_max")

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)

    real_grid, imaginary_grid = np.meshgrid(x, y)

    return real_grid + 1j * imaginary_grid


def mandelbrot_numpy(
    width,
    height,
    max_iter,
    x_min=X_MIN,
    x_max=X_MAX,
    y_min=Y_MIN,
    y_max=Y_MAX,
    escape_radius=ESCAPE_RADIUS,
):
    """
    Compute the Mandelbrot set using a vectorized NumPy implementation.

    The function creates a complex grid and repeatedly applies the Mandelbrot
    recurrence relation:

    ``Z = Z**2 + C``

    Points whose absolute value becomes larger than ``escape_radius`` are marked
    as escaped. The returned value for each point is the normalized escape
    iteration. Points that do not escape are assigned the value 1.0.

    Parameters
    ----------
    width : int
        Number of pixels along the real axis.
    height : int
        Number of pixels along the imaginary axis.
    max_iter : int
        Maximum number of Mandelbrot iterations.
    x_min : float, optional
        Minimum value of the real axis.
    x_max : float, optional
        Maximum value of the real axis.
    y_min : float, optional
        Minimum value of the imaginary axis.
    y_max : float, optional
        Maximum value of the imaginary axis.
    escape_radius : float, optional
        Radius used to decide whether a point has escaped.

    Returns
    -------
    numpy.ndarray
        A 2D array with shape ``(height, width)`` containing normalized escape
        values between 0.0 and 1.0.

    Raises
    ------
    ValueError
        If width, height, max_iter, escape_radius, or the plotting bounds are
        invalid.

    Examples
    --------
    >>> result = mandelbrot_numpy(10, 5, 20)
    >>> result.shape
    (5, 10)

    >>> result = mandelbrot_numpy(1, 1, 10, x_min=0, x_max=0, y_min=0, y_max=0)
    >>> result[0, 0]
    1.0
    """
    width, height, max_iter, escape_radius = validate_mandelbrot_inputs(
        width,
        height,
        max_iter,
        escape_radius,
    )

    complex_grid = create_complex_grid(
        width,
        height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    z_values = np.zeros_like(complex_grid, dtype=np.complex128)
    result = np.ones(complex_grid.shape, dtype=np.float64)
    active = np.ones(complex_grid.shape, dtype=bool)

    for iteration in range(max_iter):
        z_values[active] = (
            z_values[active] * z_values[active] + complex_grid[active]
        )

        escaped_now = np.abs(z_values) > escape_radius
        newly_escaped = escaped_now & active

        result[newly_escaped] = (iteration + 1) / max_iter
        active[newly_escaped] = False

        if not np.any(active):
            break

    result[active] = 1.0

    return result