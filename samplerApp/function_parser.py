"""
Numerical gradient utility for Langevin sampling.

Provides central finite-difference approximation of ∇E(x) so users
only need to supply the energy function E(x), not its gradient.
"""

import numpy as np


def numerical_gradient(energy_func, x, h=1e-5):
    """
    Compute ∇E(x) via central finite differences.

    Parameters
    ----------
    energy_func : callable
        Energy function E(x).
        - 1-D: scalar → scalar.
        - d-D: np.ndarray(d,) → scalar.
    x : float or np.ndarray
        Point at which to evaluate the gradient.
    h : float
        Step size for finite differences.

    Returns
    -------
    grad : float or np.ndarray
        Approximation of ∇E(x), same shape as x.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 0:
        # Scalar case
        return (energy_func(float(x) + h) - energy_func(float(x) - h)) / (2.0 * h)

    # Vector case
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[i] += h
        x_bwd[i] -= h
        grad[i] = (energy_func(x_fwd) - energy_func(x_bwd)) / (2.0 * h)
    return grad
