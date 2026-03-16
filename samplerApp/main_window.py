"""
Main application window.

Compact QMainWindow with a scrollable sidebar as the central widget,
a status bar, and logic to wire sampling runs to plot pop-ups.
"""

import traceback
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QScrollArea, QStatusBar, QLabel, QMessageBox,
)
from PySide6.QtCore import Qt

from .sidebar import Sidebar
from .runner import SamplingWorker
from .plot_window import PlotWindow


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sampler App — Distribution Sampling Toolkit")
        self.setMinimumSize(420, 700)
        self.resize(460, 860)

        # ── Sidebar inside a scroll area ──
        self._sidebar = Sidebar()
        scroll = QScrollArea()
        scroll.setWidget(self._sidebar)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setCentralWidget(scroll)

        # ── Status bar ──
        self._status_label = QLabel("Ready")
        self.statusBar().addPermanentWidget(self._status_label, stretch=1)

        # ── Connect sidebar run signal ──
        self._sidebar.run_requested.connect(self._start_sampling)

        # Track worker and plot windows
        self._worker = None
        self._plot_windows = []

    # ──────────────────────────────────────────
    #  Sampling orchestration
    # ──────────────────────────────────────────

    def _start_sampling(self, method, p_tilde, energy, params, visualize):
        """Launch sampling in a background thread."""
        self._sidebar.set_running(True)
        self._status_label.setText(f"Running {method}…")

        # Store for later use in visualization
        self._vis_request = visualize
        self._vis_p_tilde = p_tilde
        self._vis_x_range = params["x_range"]
        self._vis_method = method

        self._worker = SamplingWorker(method, p_tilde, energy, params)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self._on_progress)
        self._worker.start()

    def _on_finished(self, samples, extra, elapsed):
        """Handle completed sampling."""
        self._sidebar.set_running(False)

        n = len(samples)
        dim = 1 if samples.ndim == 1 else samples.shape[1]

        parts = [
            f"✓ {self._vis_method}",
            f"{n:,} samples",
            f"{dim}-D",
            f"{elapsed:.2f}s",
        ]
        if "acceptance_rate" in extra:
            parts.append(f"accept={extra['acceptance_rate']:.2%}")

        status_text = "  |  ".join(parts)
        self._status_label.setText(status_text)

        # Give samples to sidebar for export
        self._sidebar.set_samples(samples)

        # ── Notification message box ──
        msg_lines = [
            f"<b>{self._vis_method}</b> completed successfully.",
            f"",
            f"Samples: <b>{n:,}</b>  |  Dimensions: <b>{dim}</b>  |  Time: <b>{elapsed:.2f}s</b>",
        ]
        if "acceptance_rate" in extra:
            msg_lines.append(
                f"Acceptance rate: <b>{extra['acceptance_rate']:.2%}</b>"
            )

        # Visualization pop-up
        show_plot = self._vis_request and dim <= 2
        if self._vis_request and dim > 2:
            msg_lines.append("")
            msg_lines.append(
                '<span style="color:#b07000;">⚠ Visualization is only '
                'available for 1-D and 2-D. Samples are ready for export.</span>'
            )
        elif not self._vis_request:
            msg_lines.append("")
            msg_lines.append(
                "Visualization is disabled. Enable it before running "
                "to see plots."
            )

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Sampling Complete")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText("<br>".join(msg_lines))
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

        if show_plot:
            try:
                pw = PlotWindow(
                    samples=samples,
                    p_tilde=self._vis_p_tilde,
                    x_range=self._vis_x_range,
                    method_name=self._vis_method,
                    extra_info=extra,
                    parent=None,
                )
                pw.show()
                self._plot_windows.append(pw)
            except Exception as e:
                QMessageBox.critical(
                    self, "Plot Error",
                    f"Failed to create visualization:\n{e}"
                )

    def _on_error(self, message):
        """Handle sampling error."""
        self._sidebar.set_running(False)
        self._status_label.setText(f"✗ Error: {message}")

        QMessageBox.critical(
            self, "Sampling Error",
            f"Sampling failed with error:\n\n{message}\n\n"
            "Check your distribution parameters and try again."
        )

    def _on_progress(self, message):
        """Handle progress updates."""
        self._status_label.setText(message)
