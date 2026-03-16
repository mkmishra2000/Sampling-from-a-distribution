"""
Editable d×d covariance matrix dialog with positive-definiteness validation.
"""

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QPushButton, QDoubleSpinBox, QMessageBox,
    QGroupBox, QGridLayout,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor


class CovarianceEditor(QDialog):
    """
    Dialog for editing a mean vector μ and covariance matrix Σ.

    Parameters
    ----------
    dim : int
        Dimensionality.
    mean : np.ndarray or None
        Initial mean vector (zeros if None).
    covariance : np.ndarray or None
        Initial covariance matrix (identity if None).
    parent : QWidget or None
    """

    def __init__(self, dim, mean=None, covariance=None, parent=None):
        super().__init__(parent)
        self.dim = dim
        self.setWindowTitle(f"Gaussian Parameters ({dim}-D)")
        self.setMinimumWidth(max(350, 80 * dim + 120))

        if mean is None:
            mean = np.zeros(dim)
        if covariance is None:
            covariance = np.eye(dim)

        self._mean = mean.copy()
        self._cov = covariance.copy()

        layout = QVBoxLayout(self)

        # ── Mean vector ──
        mean_group = QGroupBox("Mean vector  μ")
        mean_layout = QGridLayout()
        self._mean_spins = []
        for i in range(dim):
            lbl = QLabel(f"μ{self._sub(i)}")
            sp = QDoubleSpinBox()
            sp.setRange(-1000.0, 1000.0)
            sp.setDecimals(4)
            sp.setSingleStep(0.1)
            sp.setValue(float(mean[i]))
            mean_layout.addWidget(lbl, 0, 2 * i)
            mean_layout.addWidget(sp, 0, 2 * i + 1)
            self._mean_spins.append(sp)
        mean_group.setLayout(mean_layout)
        layout.addWidget(mean_group)

        # ── Covariance matrix ──
        cov_group = QGroupBox("Covariance matrix  Σ  (edit upper triangle)")
        cov_layout = QVBoxLayout()

        self._table = QTableWidget(dim, dim)
        headers = [f"x{self._sub(i)}" for i in range(dim)]
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setVerticalHeaderLabels(headers)

        for r in range(dim):
            for c in range(dim):
                item = QTableWidgetItem(f"{covariance[r, c]:.4f}")
                item.setTextAlignment(Qt.AlignCenter)
                if c < r:
                    # Lower triangle: read-only mirror
                    item.setFlags(Qt.ItemIsEnabled)
                    item.setBackground(QColor(240, 240, 240))
                self._table.setItem(r, c, item)

        self._table.cellChanged.connect(self._on_cell_changed)
        cov_layout.addWidget(self._table)

        self._pd_label = QLabel("")
        cov_layout.addWidget(self._pd_label)
        cov_group.setLayout(cov_layout)
        layout.addWidget(cov_group)

        # ── Buttons ──
        btn_layout = QHBoxLayout()
        self._ok_btn = QPushButton("OK")
        self._ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self._ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self._validate_pd()

    # ── Helpers ──

    @staticmethod
    def _sub(i):
        """Unicode subscript digit."""
        subs = "₀₁₂₃₄₅₆₇₈₉"
        if i < 10:
            return subs[i]
        return str(i)

    def _on_cell_changed(self, row, col):
        """Mirror upper triangle to lower, then validate."""
        if col < row:
            return  # ignore lower-triangle edits
        item = self._table.item(row, col)
        if item is None:
            return
        try:
            val = float(item.text())
        except ValueError:
            val = 0.0
            item.setText("0.0000")

        # Mirror to lower triangle
        if row != col:
            mirror = self._table.item(col, row)
            if mirror is not None:
                self._table.blockSignals(True)
                mirror.setText(f"{val:.4f}")
                self._table.blockSignals(False)

        self._validate_pd()

    def _read_cov_from_table(self):
        """Read current covariance matrix from the table widget."""
        cov = np.zeros((self.dim, self.dim))
        for r in range(self.dim):
            for c in range(self.dim):
                item = self._table.item(r, c)
                try:
                    cov[r, c] = float(item.text())
                except (ValueError, AttributeError):
                    cov[r, c] = 0.0
        return cov

    def _validate_pd(self):
        """Check positive-definiteness and update UI."""
        cov = self._read_cov_from_table()
        eigvals = np.linalg.eigvalsh(cov)
        if np.all(eigvals > 0):
            self._pd_label.setText(
                '<span style="color:green;">✓ Positive definite</span>'
            )
            self._pd_label.setStyleSheet("")
            self._ok_btn.setEnabled(True)
        else:
            self._pd_label.setText(
                '<span style="color:red;">✗ NOT positive definite — '
                f'eigenvalues: {np.round(eigvals, 6).tolist()}</span>'
            )
            self._ok_btn.setEnabled(False)

    # ── Public interface ──

    def get_mean(self):
        """Return the edited mean vector."""
        return np.array([sp.value() for sp in self._mean_spins])

    def get_covariance(self):
        """Return the edited covariance matrix."""
        return self._read_cov_from_table()
