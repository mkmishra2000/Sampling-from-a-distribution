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

    def MetropolisHastings(self, p_tilde, step_size, n_samples, x_range):
        """
        Metropolis–Hastings sampler with a symmetric random-walk
        proposal:  z* = z + N(0, step_size^2 · I).

        Supports both scalar (1-D) and multidimensional targets.

        Parameters
        ----------
        p_tilde : callable
            Unnormalized target density.
            - 1-D case: accepts and returns a scalar.
            - d-D case: accepts a np.ndarray of shape (d,) and
              returns a scalar.
        step_size : float
            Standard deviation of the Gaussian random-walk proposal.
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
            z = np.random.uniform(x_range[0], x_range[1])
            samples = np.zeros(n_samples)
            for i in range(n_samples):
                z_star = z + np.random.normal(0, step_size)
                acceptance_ratio = p_tilde(z_star) / p_tilde(z)
                A = min(1.0, acceptance_ratio)
                if np.random.rand() < A:
                    z = z_star
                samples[i] = z
        else:
            # ── Multidimensional case ──
            d = x_range.shape[0]
            z = np.random.uniform(x_range[:, 0], x_range[:, 1])
            samples = np.zeros((n_samples, d))
            for i in range(n_samples):
                z_star = z + np.random.normal(0, step_size, size=d)
                acceptance_ratio = p_tilde(z_star) / p_tilde(z)
                A = min(1.0, acceptance_ratio)
                if np.random.rand() < A:
                    z = z_star
                samples[i] = z

        return samples

    def RejectionSampling(self, p_tilde, n_samples, x_range,
                          q_pdf=None, sample_q=None, k=None,
                          safety_margin=1.1, grid_points=100_000):
        """
        Rejection sampling from an unnormalized target p_tilde(z).

        Supports both 1-D and multidimensional targets.
        By default, uses a wide Gaussian proposal fitted to x_range.
        Optionally, the user can supply their own proposal.

        Parameters
        ----------
        p_tilde : callable
            Unnormalized target density.
            - 1-D: scalar in → scalar out.
            - d-D: np.ndarray of shape (d,) in → scalar out.
        n_samples : int
            Number of accepted samples to return.
        x_range : tuple or np.ndarray
            - 1-D: (low, high).
            - d-D: array of shape (d, 2) where each row is
              [low_i, high_i].
        q_pdf : callable, optional
            Proposal PDF q(z).  If None, a wide Gaussian is used:
            - 1-D: N(midpoint, ((high-low)/4)^2).
            - d-D: product of independent Gaussians per dimension.
        sample_q : callable, optional
            A zero-argument function returning one draw from q(z).
            Required when q_pdf is provided; ignored otherwise.
        k : float, optional
            Bounding constant such that k·q(z) >= p_tilde(z).
            If None, estimated automatically.
        safety_margin : float, default 1.1
            Multiplicative safety factor when auto-computing k.
        grid_points : int, default 100_000
            Number of points for auto-computing k (grid for 1-D,
            random samples for d-D).

        Returns
        -------
        samples : np.ndarray
            - 1-D: shape (n_samples,)
            - d-D: shape (n_samples, d)
        acceptance_rate : float
            Fraction of proposals that were accepted.
        """
        x_range = np.asarray(x_range, dtype=float)

        if x_range.ndim == 1:
            # ══════════════════════════════════
            #  1-D case
            # ══════════════════════════════════
            low, high = x_range

            if q_pdf is None:
                mu_q = (low + high) / 2.0
                sigma_q = (high - low) / 4.0

                def q_pdf(z):
                    return (1.0 / (np.sqrt(2 * np.pi) * sigma_q)) * \
                           np.exp(-0.5 * ((z - mu_q) / sigma_q) ** 2)

                def sample_q():
                    return np.random.normal(mu_q, sigma_q)

            elif sample_q is None:
                raise ValueError(
                    "When q_pdf is provided, sample_q must also be provided."
                )

            if k is None:
                z_grid = np.linspace(low, high, grid_points)
                q_vals = np.maximum(q_pdf(z_grid), 1e-300)
                k = np.max(p_tilde(z_grid) / q_vals) * safety_margin

            samples = []
            attempts = 0
            while len(samples) < n_samples:
                z0 = sample_q()
                u0 = np.random.uniform(0, k * q_pdf(z0))
                attempts += 1
                if u0 <= p_tilde(z0):
                    samples.append(z0)

            acceptance_rate = n_samples / attempts
            return np.array(samples), acceptance_rate

        else:
            # ══════════════════════════════════
            #  Multidimensional case
            # ══════════════════════════════════
            d = x_range.shape[0]
            lows = x_range[:, 0]
            highs = x_range[:, 1]

            if q_pdf is None:
                mu_q = (lows + highs) / 2.0
                sigma_q = (highs - lows) / 4.0

                def q_pdf(z):
                    # Product of independent Gaussians
                    return np.prod(
                        (1.0 / (np.sqrt(2 * np.pi) * sigma_q)) *
                        np.exp(-0.5 * ((z - mu_q) / sigma_q) ** 2)
                    )

                def sample_q():
                    return np.random.normal(mu_q, sigma_q)

            elif sample_q is None:
                raise ValueError(
                    "When q_pdf is provided, sample_q must also be provided."
                )

            if k is None:
                # Estimate k by sampling from q (grid is infeasible in d-D)
                max_ratio = 0.0
                for _ in range(grid_points):
                    z_try = sample_q()
                    q_val = max(q_pdf(z_try), 1e-300)
                    ratio = p_tilde(z_try) / q_val
                    if ratio > max_ratio:
                        max_ratio = ratio
                k = max_ratio * safety_margin

            samples = []
            attempts = 0
            while len(samples) < n_samples:
                z0 = sample_q()
                u0 = np.random.uniform(0, k * q_pdf(z0))
                attempts += 1
                if u0 <= p_tilde(z0):
                    samples.append(z0)

            acceptance_rate = n_samples / attempts
            return np.array(samples), acceptance_rate
