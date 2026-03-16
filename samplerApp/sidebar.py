"""
Sidebar widget — the main control panel for the sampling app.

Contains: method selector, dimensionality, preset loader,
distribution builder, x_range table, algorithm parameters,
visualization toggle, run button, and export section.
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QCheckBox,
    QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QFrame,
)
from PySide6.QtCore import Signal, Qt

from .distribution_builder import DistributionBuilder
from .presets import get_preset_names, resolve_preset


class Sidebar(QWidget):
    """
    Main control panel widget.

    Signals
    -------
    run_requested(method, p_tilde, energy, params, visualize)
    """

    run_requested = Signal(str, object, object, dict, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_samples = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ═══════════════════════════════════════
        #  1. Method selector
        # ═══════════════════════════════════════
        method_group = QGroupBox("Sampling Method")
        mg_layout = QVBoxLayout()
        self._method_combo = QComboBox()
        self._method_combo.addItems([
            "Rejection Sampling",
            "Metropolis-Hastings",
            "Langevin Sampling",
        ])
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        mg_layout.addWidget(self._method_combo)
        method_group.setLayout(mg_layout)
        layout.addWidget(method_group)

        # ═══════════════════════════════════════
        #  2. Dimensionality
        # ═══════════════════════════════════════
        dim_group = QGroupBox("Dimensionality")
        dg_layout = QHBoxLayout()
        dg_layout.addWidget(QLabel("d ="))
        self._dim_spin = QSpinBox()
        self._dim_spin.setRange(1, 10)
        self._dim_spin.setValue(1)
        self._dim_spin.valueChanged.connect(self._on_dim_changed)
        dg_layout.addWidget(self._dim_spin)
        dg_layout.addStretch()
        dim_group.setLayout(dg_layout)
        layout.addWidget(dim_group)

        # ═══════════════════════════════════════
        #  3. Preset loader
        # ═══════════════════════════════════════
        preset_group = QGroupBox("Preset")
        pg_layout = QHBoxLayout()
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("Custom")
        self._preset_combo.addItems(get_preset_names())
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        pg_layout.addWidget(self._preset_combo)
        preset_group.setLayout(pg_layout)
        layout.addWidget(preset_group)

        # ═══════════════════════════════════════
        #  4. Distribution builder
        # ═══════════════════════════════════════
        self._builder = DistributionBuilder(dim=1)
        layout.addWidget(self._builder)

        # ═══════════════════════════════════════
        #  5. x_range table
        # ═══════════════════════════════════════
        xr_group = QGroupBox("Domain  x_range  [low, high] per dimension")
        xr_layout = QVBoxLayout()
        self._xrange_table = QTableWidget(1, 2)
        self._xrange_table.setHorizontalHeaderLabels(["Low", "High"])
        self._xrange_table.verticalHeader().setVisible(True)
        self._setup_xrange_table(1)
        xr_layout.addWidget(self._xrange_table)
        xr_group.setLayout(xr_layout)
        layout.addWidget(xr_group)

        # ═══════════════════════════════════════
        #  6. Common parameters
        # ═══════════════════════════════════════
        common_group = QGroupBox("Parameters")
        cg_layout = QGridLayout()

        cg_layout.addWidget(QLabel("n_samples"), 0, 0)
        self._n_samples_spin = QSpinBox()
        self._n_samples_spin.setRange(100, 10_000_000)
        self._n_samples_spin.setSingleStep(1000)
        self._n_samples_spin.setValue(10_000)
        cg_layout.addWidget(self._n_samples_spin, 0, 1)

        # Method-specific parameters
        # -- Langevin: η
        self._eta_label = QLabel("η (step size)")
        self._eta_spin = QDoubleSpinBox()
        self._eta_spin.setRange(1e-6, 10.0)
        self._eta_spin.setDecimals(5)
        self._eta_spin.setSingleStep(0.001)
        self._eta_spin.setValue(0.01)
        cg_layout.addWidget(self._eta_label, 1, 0)
        cg_layout.addWidget(self._eta_spin, 1, 1)

        # -- MH: step_size
        self._step_label = QLabel("step_size (σ proposal)")
        self._step_spin = QDoubleSpinBox()
        self._step_spin.setRange(0.001, 100.0)
        self._step_spin.setDecimals(4)
        self._step_spin.setSingleStep(0.1)
        self._step_spin.setValue(0.5)
        cg_layout.addWidget(self._step_label, 2, 0)
        cg_layout.addWidget(self._step_spin, 2, 1)

        # -- Rejection: safety_margin, grid_points
        self._safety_label = QLabel("safety_margin")
        self._safety_spin = QDoubleSpinBox()
        self._safety_spin.setRange(1.0, 10.0)
        self._safety_spin.setDecimals(2)
        self._safety_spin.setValue(1.1)
        cg_layout.addWidget(self._safety_label, 3, 0)
        cg_layout.addWidget(self._safety_spin, 3, 1)

        self._grid_label = QLabel("grid_points (k estimate)")
        self._grid_spin = QSpinBox()
        self._grid_spin.setRange(1000, 10_000_000)
        self._grid_spin.setSingleStep(10000)
        self._grid_spin.setValue(100_000)
        cg_layout.addWidget(self._grid_label, 4, 0)
        cg_layout.addWidget(self._grid_spin, 4, 1)

        common_group.setLayout(cg_layout)
        layout.addWidget(common_group)

        # ═══════════════════════════════════════
        #  7. Visualization toggle
        # ═══════════════════════════════════════
        self._viz_cb = QCheckBox("Enable Visualization (1-D / 2-D only)")
        self._viz_cb.setChecked(True)
        layout.addWidget(self._viz_cb)

        # ═══════════════════════════════════════
        #  8. Run button
        # ═══════════════════════════════════════
        self._run_btn = QPushButton("▶  Run Sampling")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setStyleSheet(
            "QPushButton { background-color: #2d8cf0; color: white; "
            "font-size: 14px; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1a6fd1; }"
            "QPushButton:disabled { background-color: #aaa; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        # ═══════════════════════════════════════
        #  9. Export section
        # ═══════════════════════════════════════
        export_group = QGroupBox("Export Samples")
        eg_layout = QGridLayout()

        eg_layout.addWidget(QLabel("Samples to save"), 0, 0)
        self._save_count_spin = QSpinBox()
        self._save_count_spin.setRange(1, 10_000_000)
        self._save_count_spin.setValue(10_000)
        eg_layout.addWidget(self._save_count_spin, 0, 1)

        self._save_btn = QPushButton("💾  Save Samples")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        eg_layout.addWidget(self._save_btn, 1, 0, 1, 2)

        export_group.setLayout(eg_layout)
        layout.addWidget(export_group)

        # Bottom padding so the export section isn't clipped by the taskbar
        layout.addSpacing(40)
        layout.addStretch()

        # Initial visibility
        self._on_method_changed(self._method_combo.currentText())

    # ═══════════════════════════════════════════════════════════════
    #  Slot: method changed → show/hide relevant param widgets
    # ═══════════════════════════════════════════════════════════════

    def _on_method_changed(self, method):
        is_langevin = method == "Langevin Sampling"
        is_mh = method == "Metropolis-Hastings"
        is_rejection = method == "Rejection Sampling"

        self._eta_label.setVisible(is_langevin)
        self._eta_spin.setVisible(is_langevin)
        self._step_label.setVisible(is_mh)
        self._step_spin.setVisible(is_mh)
        self._safety_label.setVisible(is_rejection)
        self._safety_spin.setVisible(is_rejection)
        self._grid_label.setVisible(is_rejection)
        self._grid_spin.setVisible(is_rejection)

    # ═══════════════════════════════════════════════════════════════
    #  Slot: dimension changed
    # ═══════════════════════════════════════════════════════════════

    def _on_dim_changed(self, dim):
        self._builder.set_dim(dim)
        self._setup_xrange_table(dim)
        # Visualization only for dim ≤ 2
        self._viz_cb.setEnabled(dim <= 2)
        if dim > 2:
            self._viz_cb.setChecked(False)

    # ═══════════════════════════════════════════════════════════════
    #  Slot: preset changed
    # ═══════════════════════════════════════════════════════════════

    def _on_preset_changed(self, name):
        if name == "Custom":
            return

        dim = self._dim_spin.value()
        resolved = resolve_preset(name, dim)

        if resolved is None:
            # Preset doesn't match current dim — try to set the right dim
            from .presets import PRESETS
            preset = PRESETS.get(name)
            if preset and preset["dim"] is not None:
                self._dim_spin.setValue(preset["dim"])
                dim = preset["dim"]
                resolved = resolve_preset(name, dim)
            if resolved is None:
                QMessageBox.warning(
                    self, "Preset",
                    f"Preset '{name}' is not compatible with d={dim}."
                )
                self._preset_combo.setCurrentText("Custom")
                return

        # Load into builder
        self._builder.load_preset(
            resolved["components"], resolved["combinators"]
        )

        # Load x_range
        xr = resolved["x_range"]
        self._setup_xrange_table(dim, xr)

        # Load suggested params
        p = resolved.get("params", {})
        if "n_samples" in p:
            self._n_samples_spin.setValue(p["n_samples"])
        if "eta" in p:
            self._eta_spin.setValue(p["eta"])
        if "step_size" in p:
            self._step_spin.setValue(p["step_size"])
        if "safety_margin" in p:
            self._safety_spin.setValue(p["safety_margin"])
        if "grid_points" in p:
            self._grid_spin.setValue(p["grid_points"])

    # ═══════════════════════════════════════════════════════════════
    #  x_range table helpers
    # ═══════════════════════════════════════════════════════════════

    def _setup_xrange_table(self, dim, values=None):
        """Rebuild the x_range table with `dim` rows."""
        self._xrange_table.setRowCount(dim)
        subs = "₀₁₂₃₄₅₆₇₈₉"
        for i in range(dim):
            sub = subs[i] if i < 10 else str(i)
            self._xrange_table.setVerticalHeaderItem(
                i, QTableWidgetItem(f"x{sub}")
            )
            lo_val = values[i][0] if values else -5.0
            hi_val = values[i][1] if values else 5.0

            lo_item = QTableWidgetItem(f"{lo_val:.1f}")
            lo_item.setTextAlignment(Qt.AlignCenter)
            hi_item = QTableWidgetItem(f"{hi_val:.1f}")
            hi_item.setTextAlignment(Qt.AlignCenter)
            self._xrange_table.setItem(i, 0, lo_item)
            self._xrange_table.setItem(i, 1, hi_item)

        self._xrange_table.setMaximumHeight(30 + 30 * dim)

    def _read_xrange(self):
        """Read x_range as list of [lo, hi]."""
        dim = self._xrange_table.rowCount()
        xr = []
        for i in range(dim):
            try:
                lo = float(self._xrange_table.item(i, 0).text())
                hi = float(self._xrange_table.item(i, 1).text())
            except (ValueError, AttributeError):
                lo, hi = -5.0, 5.0
            xr.append([lo, hi])
        return xr

    # ═══════════════════════════════════════════════════════════════
    #  Run
    # ═══════════════════════════════════════════════════════════════

    def _on_run(self):
        method = self._method_combo.currentText()

        try:
            p_tilde = self._builder.build_function()
            energy = self._builder.build_energy()
        except Exception as e:
            QMessageBox.critical(
                self, "Distribution Error",
                f"Failed to build distribution:\n{e}"
            )
            return

        params = {
            "n_samples": self._n_samples_spin.value(),
            "x_range": self._read_xrange(),
            "eta": self._eta_spin.value(),
            "step_size": self._step_spin.value(),
            "safety_margin": self._safety_spin.value(),
            "grid_points": self._grid_spin.value(),
        }

        visualize = self._viz_cb.isChecked() and self._viz_cb.isEnabled()

        self.run_requested.emit(method, p_tilde, energy, params, visualize)

    # ═══════════════════════════════════════════════════════════════
    #  Export
    # ═══════════════════════════════════════════════════════════════

    def set_samples(self, samples):
        """Called by MainWindow after sampling completes."""
        self._last_samples = samples
        self._save_count_spin.setMaximum(len(samples))
        self._save_count_spin.setValue(min(self._save_count_spin.value(), len(samples)))
        self._save_btn.setEnabled(True)

    def _on_save(self):
        if self._last_samples is None:
            return

        n = self._save_count_spin.value()
        data = self._last_samples[:n]

        path, filt = QFileDialog.getSaveFileName(
            self, "Save Samples", "",
            "NumPy binary (*.npy);;CSV (*.csv)"
        )
        if not path:
            return

        try:
            if path.endswith(".csv") or "CSV" in filt:
                if not path.endswith(".csv"):
                    path += ".csv"
                np.savetxt(path, data, delimiter=",",
                           header=",".join(
                               f"x{i}" for i in range(data.shape[-1])
                               ) if data.ndim > 1 else "x")
            else:
                if not path.endswith(".npy"):
                    path += ".npy"
                np.save(path, data)

            QMessageBox.information(
                self, "Saved",
                f"Saved {n:,} samples to\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # ═══════════════════════════════════════════════════════════════
    #  Public helpers
    # ═══════════════════════════════════════════════════════════════

    def set_running(self, running):
        """Enable/disable controls during sampling."""
        self._run_btn.setEnabled(not running)
        self._run_btn.setText(
            "⏳  Sampling…" if running else "▶  Run Sampling"
        )
