import numpy as np


class sampler:
    """A collection of sampling methods for probability distributions."""

    def LangevinSampling(self, grad_energy, eta, n_samples, x_range):
        """
        Unadjusted Langevin Algorithm (ULA) for sampling from
        p(x) ∝ exp(-E(x)) using the Langevin update:

            x_{t+1} = x_t - eta * ∇E(x_t) + sqrt(2*eta) * noise

        Supports both scalar (1-D) and multidimensional targets.

        Parameters
        ----------
        grad_energy : callable
            Gradient of the energy function, ∇E(x).
            - 1-D case: accepts and returns a scalar.
            - d-D case: accepts and returns a np.ndarray of shape (d,).
        eta : float
            Step size (learning rate) for the Langevin update.
        n_samples : int
            Number of iterations (samples) to generate.
        x_range : tuple or np.ndarray
            - 1-D case: (low, high) — initial point drawn from
              Uniform(low, high).
            - d-D case: array of shape (d, 2) where each row is
              [low_i, high_i] for dimension i.

        Returns
        -------
        samples : np.ndarray
            - 1-D case: shape (n_samples,)
            - d-D case: shape (n_samples, d)
        """
        x_range = np.asarray(x_range)

        if x_range.ndim == 1:
            # ── Scalar (1-D) case ──
            x = np.random.uniform(x_range[0], x_range[1])
            samples = np.zeros(n_samples)
            for t in range(n_samples):
                noise = np.sqrt(2 * eta) * np.random.randn()
                x = x - eta * grad_energy(x) + noise
                samples[t] = x
        else:
            # ── Multidimensional case ──
            d = x_range.shape[0]
            x = np.random.uniform(x_range[:, 0], x_range[:, 1])
            samples = np.zeros((n_samples, d))
            for t in range(n_samples):
                noise = np.sqrt(2 * eta) * np.random.randn(d)
                x = x - eta * grad_energy(x) + noise
                samples[t] = x

        return samples
