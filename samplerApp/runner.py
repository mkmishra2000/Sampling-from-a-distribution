"""
Background sampling worker using QThread.

Runs sampling in a separate thread so the GUI stays responsive.
"""

import time
import numpy as np
from PySide6.QtCore import QThread, Signal

from .sampler import sampler
from .function_parser import numerical_gradient


class SamplingWorker(QThread):
    """
    QThread worker that runs a sampling method and emits results.

    Signals
    -------
    finished(samples, extra_info, elapsed)
        samples: np.ndarray
        extra_info: dict (may contain 'acceptance_rate')
        elapsed: float (seconds)
    error(message)
    progress(message)
    """

    finished = Signal(object, object, float)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, method, p_tilde, energy_func, params, parent=None):
        """
        Parameters
        ----------
        method : str
            "Rejection Sampling", "Metropolis-Hastings", or "Langevin Sampling"
        p_tilde : callable
            Unnormalized target density.
        energy_func : callable or None
            Energy function (used for Langevin only).
        params : dict
            Must include 'n_samples', 'x_range' (list of [lo, hi]).
            Method-specific: 'eta', 'step_size', 'safety_margin', 'grid_points'.
        """
        super().__init__(parent)
        self.method = method
        self.p_tilde = p_tilde
        self.energy_func = energy_func
        self.params = params

    def run(self):
        try:
            sam = sampler()
            n_samples = self.params["n_samples"]
            x_range_list = self.params["x_range"]  # list of [lo, hi]
            dim = len(x_range_list)

            # Build x_range in the format the sampler expects
            if dim == 1:
                x_range = (x_range_list[0][0], x_range_list[0][1])
            else:
                x_range = np.array(x_range_list, dtype=float)

            # Adapt p_tilde for 1-D: sampler expects scalar→scalar
            # but RejectionSampling also passes arrays for vectorised k estimation
            if dim == 1:
                _p = self.p_tilde

                def p_tilde_1d(z):
                    z_arr = np.asarray(z, dtype=float)
                    if z_arr.ndim == 0:
                        # scalar input (MH / Langevin loop, RS accept step)
                        return _p(np.array([z_arr.item()]))
                    else:
                        # vectorised input (RS grid for k estimation)
                        return np.array([_p(np.array([float(zi)]))
                                         for zi in z_arr.ravel()])
                p_func = p_tilde_1d
            else:
                p_func = self.p_tilde

            t0 = time.perf_counter()
            extra = {}

            if self.method == "Rejection Sampling":
                self.progress.emit("Running rejection sampling…")
                samples, acc_rate = sam.RejectionSampling(
                    p_tilde=p_func,
                    n_samples=n_samples,
                    x_range=x_range,
                    safety_margin=self.params.get("safety_margin", 1.1),
                    grid_points=self.params.get("grid_points", 100_000),
                )
                extra["acceptance_rate"] = acc_rate

            elif self.method == "Metropolis-Hastings":
                self.progress.emit("Running Metropolis-Hastings…")
                samples = sam.MetropolisHastings(
                    p_tilde=p_func,
                    step_size=self.params.get("step_size", 0.5),
                    n_samples=n_samples,
                    x_range=x_range,
                )

            elif self.method == "Langevin Sampling":
                self.progress.emit("Running Langevin sampling…")
                energy = self.energy_func

                # Build grad_energy via numerical gradient
                if dim == 1:
                    _e = energy

                    def energy_1d(z):
                        z_val = np.asarray(z, dtype=float)
                        if z_val.ndim == 0:
                            return _e(np.array([z_val.item()]))
                        return _e(np.array([float(z_val.ravel()[0])]))

                    def grad_energy(z):
                        return numerical_gradient(energy_1d, z)
                else:
                    def grad_energy(z):
                        return numerical_gradient(energy, z)

                samples = sam.LangevinSampling(
                    grad_energy=grad_energy,
                    eta=self.params.get("eta", 0.01),
                    n_samples=n_samples,
                    x_range=x_range,
                )
            else:
                self.error.emit(f"Unknown method: {self.method}")
                return

            elapsed = time.perf_counter() - t0
            self.finished.emit(samples, extra, elapsed)

        except Exception as e:
            self.error.emit(str(e))
