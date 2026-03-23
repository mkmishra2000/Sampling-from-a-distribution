"""
Separate pop-up window for visualization (1-D and 2-D only).

Non-modal so the main GUI stays interactive.
Multiple instances can coexist for side-by-side comparison.
"""

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QHBoxLayout, QLabel,
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


class PlotWindow(QMainWindow):
    """
    Non-modal visualization window.

    Parameters
    ----------
    samples : np.ndarray
        1-D shape (n,) or 2-D shape (n, 2).
    p_tilde : callable
        Unnormalized target density.
    x_range : list of [lo, hi]
    method_name : str
    extra_info : dict  (may contain 'acceptance_rate')
    parent : QWidget or None
    """

    _instance_count = 0

    def __init__(self, samples, p_tilde, x_range, method_name,
                 extra_info=None, parent=None):
        super().__init__(parent)
        PlotWindow._instance_count += 1
        self.setAttribute(Qt.WA_DeleteOnClose)

        dim = 1 if samples.ndim == 1 else samples.shape[1]
        self.setWindowTitle(
            f"Visualization — {method_name} ({dim}-D) "
            f"[#{PlotWindow._instance_count}]"
        )

        self._samples = samples
        self._p_tilde = p_tilde
        self._x_range = x_range
        self._method = method_name
        self._extra = extra_info or {}

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Matplotlib canvas
        if dim == 1:
            self._fig = Figure(figsize=(12, 9), dpi=100)
            self._plot_1d()
        else:
            self._fig = Figure(figsize=(15, 5), dpi=100)
            self._plot_2d()

        self._canvas = FigureCanvas(self._fig)
        toolbar = NavigationToolbar(self._canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self._canvas)

        # Info + save button
        bottom = QHBoxLayout()
        info_parts = [f"Method: {method_name}", f"Samples: {len(samples):,}"]
        if "acceptance_rate" in self._extra:
            info_parts.append(
                f"Acceptance: {self._extra['acceptance_rate']:.2%}"
            )
        info_label = QLabel("  |  ".join(info_parts))
        bottom.addWidget(info_label)
        bottom.addStretch()

        save_btn = QPushButton("Save Figure")
        save_btn.clicked.connect(self._save_figure)
        bottom.addWidget(save_btn)
        layout.addLayout(bottom)

        self.resize(
            self._fig.get_size_inches()[0] * 100 + 40,
            self._fig.get_size_inches()[1] * 100 + 120,
        )

    # ── 1-D plots ──

    def _plot_1d(self):
        samples = self._samples
        lo, hi = self._x_range[0]

        axes = self._fig.subplots(2, 2)

        # (a) Histogram + true density
        ax = axes[0, 0]
        ax.hist(samples, bins=120, density=True, alpha=0.6,
                color="steelblue", edgecolor="white", label="Samples")
        z = np.linspace(lo, hi, 1000)
        pz = np.array([self._p_tilde(np.array([zi])) for zi in z])
        if np.max(pz) > 0:
            pz = pz / np.trapezoid(pz, z)
        ax.plot(z, pz, "r-", lw=2, label="p(x) (normalized)")
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        ax.set_title("Histogram vs True Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (b) Trace plot
        ax = axes[0, 1]
        n_trace = min(5000, len(samples))
        ax.plot(samples[:n_trace], lw=0.3, color="steelblue")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("x")
        ax.set_title(f"Trace Plot (first {n_trace:,})")
        ax.grid(True, alpha=0.3)

        # (c) Running mean
        ax = axes[1, 0]
        cum_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
        ax.plot(cum_mean, lw=0.8, color="darkorange")
        ax.axhline(np.mean(samples), ls="--", color="red", lw=1,
                    label=f"Final mean = {np.mean(samples):.4f}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Running mean")
        ax.set_title("Running Mean")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (d) Autocorrelation
        ax = axes[1, 1]
        # max_lag = min(200, len(samples) // 2)
        max_lag = min(2000, len(samples) // 2)
        s_centered = samples - np.mean(samples)
        var = np.var(samples)
        if var > 0:
            acf = np.correlate(s_centered[:max_lag * 5], s_centered[:max_lag * 5], "full")
            acf = acf[len(acf) // 2:]
            acf = acf[:max_lag] / acf[0]
            ax.bar(range(max_lag), acf, width=1.0, color="steelblue", alpha=0.7)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title("Autocorrelation")
        ax.grid(True, alpha=0.3)

        self._fig.suptitle(
            f"1-D {self._method}", fontsize=14, y=0.98
        )
        self._fig.tight_layout()

    # ── 2-D plots ──

    def _plot_2d(self):
        samples = self._samples
        x0 = samples[:, 0]
        x1 = samples[:, 1]

        axes = self._fig.subplots(1, 3)

        # (a) Joint scatter
        ax = axes[0]
        ax.scatter(x0, x1, s=1, alpha=0.15, color="steelblue")

        # Try to draw 2-sigma ellipse from sample covariance
        try:
            cov = np.cov(x0, x1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            theta = np.linspace(0, 2 * np.pi, 200)
            ell = (eigvecs @ np.diag(np.sqrt(eigvals))
                   @ np.array([np.cos(theta), np.sin(theta)])) * 2
            mean = [np.mean(x0), np.mean(x1)]
            ax.plot(ell[0] + mean[0], ell[1] + mean[1], "r-", lw=2,
                    label="Sample 2σ ellipse")
        except Exception:
            pass

        ax.set_xlabel("x₀")
        ax.set_ylabel("x₁")
        ax.set_title("Joint Scatter")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (b) Marginal x₀
        ax = axes[1]
        ax.hist(x0, bins=80, density=True, alpha=0.6,
                color="steelblue", edgecolor="white")
        ax.set_xlabel("x₀")
        ax.set_ylabel("Density")
        ax.set_title(f"Marginal x₀  (μ={np.mean(x0):.3f}, σ={np.std(x0):.3f})")
        ax.grid(True, alpha=0.3)

        # (c) Marginal x₁
        ax = axes[2]
        ax.hist(x1, bins=80, density=True, alpha=0.6,
                color="coral", edgecolor="white")
        ax.set_xlabel("x₁")
        ax.set_ylabel("Density")
        ax.set_title(f"Marginal x₁  (μ={np.mean(x1):.3f}, σ={np.std(x1):.3f})")
        ax.grid(True, alpha=0.3)

        self._fig.suptitle(
            f"2-D {self._method}", fontsize=14, y=0.98
        )
        self._fig.tight_layout()

    # ── Save ──

    def _save_figure(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", "",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
