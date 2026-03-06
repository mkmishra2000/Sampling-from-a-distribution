# Rejection Sampling — Notebook Explanation

## Overview

This notebook implements the **Rejection Sampling** algorithm to draw samples from a complex, unnormalized target distribution using a simpler proposal distribution.

---

## Cell-by-Cell Breakdown

### Cell 1 — Imports

```python
import numpy as np
import matplotlib.pyplot as plt
```

Standard scientific computing and plotting libraries.

---

### Cell 2 — Core Algorithm

#### 1. Target Distribution (unnormalized)

$$\tilde{p}(z) = \exp\!\Bigl(-\tfrac{1}{2}(z-2)^2\Bigr) + 0.5\,\exp\!\Bigl(-0.2\,(z+3)^2\Bigr)\,\sin^2(3z)$$

This is a **multimodal, unnormalized** density composed of:

| Component | Description |
|---|---|
| $\exp(-\frac{1}{2}(z-2)^2)$ | Gaussian-like bump centered at $z = 2$ |
| $0.5\,\exp(-0.2(z+3)^2)\sin^2(3z)$ | Oscillatory, modulated Gaussian centered at $z = -3$ — creates a "wavy" region |

Because $\tilde{p}(z)$ is **not normalized** (i.e. $\int \tilde{p}(z)\,dz \neq 1$), we cannot sample from it directly — making it a perfect candidate for rejection sampling.

#### 2. Proposal Distribution

$$q(z) = \mathcal{N}(z \mid 0, 16) \quad (\mu=0,\;\sigma=4)$$

A wide Gaussian that covers the support of $\tilde{p}(z)$. Sampling from $q$ is trivial via `np.random.normal(0, 4)`.

#### 3. Bounding Constant $k$

$$k = 1.1 \times \max_z \frac{\tilde{p}(z)}{q(z)}$$

Computed numerically over a fine grid of 100,000 points in $[-15, 15]$, with a **10% safety margin** to guarantee $k\,q(z) \geq \tilde{p}(z)$ everywhere.

#### 4. Rejection Sampling Loop

```
Repeat until N accepted samples:
    1. Draw z₀ ~ q(z)
    2. Draw u₀ ~ Uniform(0, k·q(z₀))
    3. If u₀ ≤ p̃(z₀) → ACCEPT z₀
       Else           → REJECT
```

The algorithm accepts a candidate $z_0$ with probability:

$$\alpha(z_0) = \frac{\tilde{p}(z_0)}{k\,q(z_0)}$$

The overall **acceptance rate** is $\frac{1}{k} \cdot \frac{\int \tilde{p}(z)\,dz}{1}$. The notebook prints this after generating 10,000 samples.

---

### Cell 3 — Visualization

- **Histogram** of the 10,000 accepted samples (normalized to density).
- **Red curve**: the target density $\tilde{p}(z)$ normalized via numerical integration (`np.trapz`) for visual comparison.
- The close match between histogram and curve confirms the sampler is working correctly.

---

## Key Concepts

| Concept | Role in this notebook |
|---|---|
| **Unnormalized target** $\tilde{p}(z)$ | The distribution we want to sample from but can only evaluate point-wise |
| **Proposal** $q(z)$ | Easy-to-sample distribution that envelopes $\tilde{p}$ when scaled by $k$ |
| **Bounding constant** $k$ | Ensures $k\,q(z) \geq \tilde{p}(z)$ for all $z$ — critical for correctness |
| **Acceptance rate** | Efficiency metric; higher $k$ → lower acceptance rate → more wasted samples |

## Limitations & Notes

- **Curse of dimensionality**: Rejection sampling becomes extremely inefficient in high dimensions because the acceptance rate drops exponentially.
- The numerical computation of $k$ relies on a dense grid; a coarser grid or a distribution with sharp spikes could underestimate $k$ and produce biased samples.
- The 10% safety margin (`k *= 1.1`) is a practical heuristic, not a theoretical guarantee.
- For production use, methods like **MCMC** (Metropolis-Hastings, Hamiltonian MC) or **Importance Sampling** scale better to higher dimensions.
