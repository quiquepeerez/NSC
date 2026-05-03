"""
Unit tests for the NumPy Mandelbrot implementation.
"""

import pytest
import numpy as np
import numpy.testing as npt

from mandelbrot import (
    create_complex_grid,
    mandelbrot_numpy,
    validate_mandelbrot_inputs,
)


def test_validate_mandelbrot_inputs_valid_case():
    result = validate_mandelbrot_inputs(100, 80, 50)

    assert result == (100, 80, 50, 2.0)


def test_validate_mandelbrot_inputs_invalid_width():
    with pytest.raises(ValueError):
        validate_mandelbrot_inputs(0, 80, 50)


def test_create_complex_grid_shape():
    grid = create_complex_grid(4, 3)

    assert grid.shape == (3, 4)
    assert np.iscomplexobj(grid)


def test_mandelbrot_output_shape():
    result = mandelbrot_numpy(100, 50, 20)

    assert result.shape == (50, 100)


def test_mandelbrot_values_are_normalized():
    result = mandelbrot_numpy(100, 50, 20)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_zero_point_does_not_escape():
    result = mandelbrot_numpy(
        1,
        1,
        20,
        x_min=0.0,
        x_max=0.0,
        y_min=0.0,
        y_max=0.0,
    )

    assert result[0, 0] == pytest.approx(1.0)


def test_far_point_escapes_first_iteration():
    result = mandelbrot_numpy(
        1,
        1,
        10,
        x_min=2.0,
        x_max=2.0,
        y_min=2.0,
        y_max=2.0,
    )

    npt.assert_allclose(result[0, 0], 0.1)