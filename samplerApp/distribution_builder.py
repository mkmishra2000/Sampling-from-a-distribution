"""
Component-based distribution builder widget.

Supports two component types (v1):
  - Gaussian: N(μ, Σ) with editable mean + covariance
  - Polynomial: P(x) with monomial terms, optional exp(-P) wrapping

Components are combined with + (mixture) or × (product) combinators.
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QFrame, QDoubleSpinBox, QSpinBox,
    QCheckBox, QDialog, QGridLayout, QScrollArea, QMessageBox,
    QSizePolicy,
)
from PySide6.QtCore import Signal

from .covariance_editor import CovarianceEditor


# ═══════════════════════════════════════════════════════════════════
#  Individual component data holders
# ═══════════════════════════════════════════════════════════════════

class GaussianComponent:
    """Stores parameters for a Gaussian component."""

    def __init__(self, dim):
        self.dim = dim
        self.mean = np.zeros(dim)
        self.covariance = np.eye(dim)

    def evaluate(self, x):
        """Evaluate exp(-0.5 (x-μ)ᵀ Σ⁻¹ (x-μ))."""
        x = np.asarray(x, dtype=float)
        diff = x - self.mean
        Linv = np.linalg.inv(self.covariance)
        if self.dim == 1:
            return float(np.exp(-0.5 * diff[0] ** 2 * Linv[0, 0]))
        return float(np.exp(-0.5 * diff @ Linv @ diff))

    def energy(self, x):
        """Return 0.5 (x-μ)ᵀ Σ⁻¹ (x-μ)."""
        x = np.asarray(x, dtype=float)
        diff = x - self.mean
        Linv = np.linalg.inv(self.covariance)
        if self.dim == 1:
            return float(0.5 * diff[0] ** 2 * Linv[0, 0])
        return float(0.5 * diff @ Linv @ diff)

    def summary(self):
        if self.dim == 1:
            return f"N(μ={self.mean[0]:.2f}, σ²={self.covariance[0,0]:.2f})"
        return f"N(μ={np.round(self.mean, 2).tolist()}, Σ={self.dim}×{self.dim})"


class PolynomialComponent:
    """
    Stores a polynomial as a list of monomial terms.

    Each term: (coefficient, exponents) where exponents is a tuple
    of length dim. E.g. in 2-D, (0.25, (4, 0)) means 0.25·x₀⁴.

    If wrap_exp is True, the component evaluates as exp(-P(x))
    and energy returns P(x). Otherwise evaluates as P(x) directly.
    """

    def __init__(self, dim):
        self.dim = dim
        self.terms = []       # list of (coeff, exponents_tuple)
        self.wrap_exp = True  # default: exp(-P(x))

    def _poly_val(self, x):
        """Evaluate P(x) = Σ coeff · Π xᵢ^eᵢ."""
        x = np.asarray(x, dtype=float).ravel()
        val = 0.0
        for coeff, exps in self.terms:
            term = coeff
            for i, e in enumerate(exps):
                if e != 0:
                    term *= x[i] ** e
            val += term
        return val

    def evaluate(self, x):
        p = self._poly_val(x)
        if self.wrap_exp:
            return float(np.exp(-p))
        return float(p)

    def energy(self, x):
        """Energy is P(x) when wrap_exp else -log(max(P(x), 1e-300))."""
        if self.wrap_exp:
            return self._poly_val(x)
        return -np.log(max(self._poly_val(x), 1e-300))

    def summary(self):
        if not self.terms:
            return "P(x) = 0"
        parts = []
        for coeff, exps in self.terms:
            monomial = "·".join(
                f"x{i}^{e}" for i, e in enumerate(exps) if e != 0
            ) or "1"
            parts.append(f"{coeff:+.3g}·{monomial}")
        s = " ".join(parts)
        if self.wrap_exp:
            return f"exp(−({s}))"
        return s


# ═══════════════════════════════════════════════════════════════════
#  Polynomial editor dialog
# ═══════════════════════════════════════════════════════════════════

class PolynomialEditor(QDialog):
    """Dialog for editing polynomial monomial terms."""

    def __init__(self, dim, component=None, parent=None):
        super().__init__(parent)
        self.dim = dim
        self.setWindowTitle(f"Polynomial Terms ({dim}-D)")
        self.setMinimumWidth(400)

        self._component = component or PolynomialComponent(dim)

        layout = QVBoxLayout(self)

        # Wrap in exp checkbox
        self._wrap_cb = QCheckBox("Wrap in exp(−P(x))  →  density = exp(−P(x))")
        self._wrap_cb.setChecked(self._component.wrap_exp)
        layout.addWidget(self._wrap_cb)

        layout.addWidget(QLabel("Each row: coefficient × x₀^e₀ · x₁^e₁ · …"))

        # Scrollable term list
        self._term_area = QVBoxLayout()
        self._term_rows = []
        scroll_w = QWidget()
        scroll_w.setLayout(self._term_area)
        scroll = QScrollArea()
        scroll.setWidget(scroll_w)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)
        layout.addWidget(scroll)

        # Populate existing terms
        for coeff, exps in self._component.terms:
            self._add_term_row(coeff, exps)

        # Add / remove buttons
        btn_row = QHBoxLayout()
        add_btn = QPushButton("➕ Add Term")
        add_btn.clicked.connect(lambda: self._add_term_row())
        btn_row.addWidget(add_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # OK / Cancel
        ok_cancel = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        ok_cancel.addStretch()
        ok_cancel.addWidget(ok_btn)
        ok_cancel.addWidget(cancel_btn)
        layout.addLayout(ok_cancel)

    def _add_term_row(self, coeff=1.0, exps=None):
        if exps is None:
            exps = tuple([2] + [0] * (self.dim - 1))

        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 2, 0, 2)

        coeff_sp = QDoubleSpinBox()
        coeff_sp.setRange(-1e6, 1e6)
        coeff_sp.setDecimals(4)
        coeff_sp.setValue(coeff)
        coeff_sp.setPrefix("c = ")
        row_l.addWidget(coeff_sp)

        exp_spins = []
        for i in range(self.dim):
            lbl = QLabel(f"x{self._sub(i)}^")
            sp = QSpinBox()
            sp.setRange(0, 20)
            sp.setValue(int(exps[i]))
            row_l.addWidget(lbl)
            row_l.addWidget(sp)
            exp_spins.append(sp)

        rm_btn = QPushButton("✕")
        rm_btn.setFixedWidth(30)
        rm_btn.clicked.connect(lambda: self._remove_term_row(row_w))
        row_l.addWidget(rm_btn)

        self._term_rows.append((row_w, coeff_sp, exp_spins))
        self._term_area.addWidget(row_w)

    def _remove_term_row(self, row_w):
        for i, (w, _, _) in enumerate(self._term_rows):
            if w is row_w:
                self._term_area.removeWidget(w)
                w.deleteLater()
                self._term_rows.pop(i)
                break

    @staticmethod
    def _sub(i):
        subs = "₀₁₂₃₄₅₆₇₈₉"
        return subs[i] if i < 10 else str(i)

    def get_component(self):
        """Return the edited PolynomialComponent."""
        comp = PolynomialComponent(self.dim)
        comp.wrap_exp = self._wrap_cb.isChecked()
        for _, coeff_sp, exp_spins in self._term_rows:
            coeff = coeff_sp.value()
            exps = tuple(sp.value() for sp in exp_spins)
            if abs(coeff) > 1e-12:
                comp.terms.append((coeff, exps))
        return comp


# ═══════════════════════════════════════════════════════════════════
#  Component row widget (one row in the builder)
# ═══════════════════════════════════════════════════════════════════

class ComponentRow(QFrame):
    """One component row: [Type ▼] [summary label] [Edit] [Remove]."""

    removed = Signal(object)
    changed = Signal()

    def __init__(self, dim, comp_data=None, parent=None):
        super().__init__(parent)
        self.dim = dim
        self.setFrameShape(QFrame.StyledPanel)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._type_combo = QComboBox()
        self._type_combo.addItems(["Gaussian", "Polynomial"])
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        layout.addWidget(self._type_combo)

        self._summary_label = QLabel("—")
        self._summary_label.setMinimumWidth(180)
        layout.addWidget(self._summary_label, stretch=1)

        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._edit)
        layout.addWidget(edit_btn)

        rm_btn = QPushButton("✕")
        rm_btn.setFixedWidth(30)
        rm_btn.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(rm_btn)

        # Initialize component data
        if comp_data is not None:
            self._comp = comp_data
            self._type_combo.setCurrentText(
                "Gaussian" if isinstance(comp_data, GaussianComponent)
                else "Polynomial"
            )
        else:
            self._comp = GaussianComponent(dim)

        self._update_summary()

    def _on_type_changed(self, text):
        if text == "Gaussian" and not isinstance(self._comp, GaussianComponent):
            self._comp = GaussianComponent(self.dim)
        elif text == "Polynomial" and not isinstance(self._comp, PolynomialComponent):
            self._comp = PolynomialComponent(self.dim)
        self._update_summary()
        self.changed.emit()

    def _edit(self):
        if isinstance(self._comp, GaussianComponent):
            dlg = CovarianceEditor(
                self.dim, self._comp.mean, self._comp.covariance, self
            )
            if dlg.exec() == QDialog.Accepted:
                self._comp.mean = dlg.get_mean()
                self._comp.covariance = dlg.get_covariance()
                self._update_summary()
                self.changed.emit()
        elif isinstance(self._comp, PolynomialComponent):
            dlg = PolynomialEditor(self.dim, self._comp, self)
            if dlg.exec() == QDialog.Accepted:
                self._comp = dlg.get_component()
                self._update_summary()
                self.changed.emit()

    def _update_summary(self):
        self._summary_label.setText(self._comp.summary())

    @property
    def component(self):
        return self._comp

    def set_dim(self, dim):
        """Update dimensionality — resets component to defaults."""
        self.dim = dim
        if isinstance(self._comp, GaussianComponent):
            self._comp = GaussianComponent(dim)
        else:
            self._comp = PolynomialComponent(dim)
        self._update_summary()


# ═══════════════════════════════════════════════════════════════════
#  Distribution Builder widget
# ═══════════════════════════════════════════════════════════════════

class DistributionBuilder(QGroupBox):
    """
    Widget for building a distribution from Gaussian + Polynomial
    components combined with + or × operators.

    Signals
    -------
    distribution_changed : emitted whenever any component is edited.
    """

    distribution_changed = Signal()

    def __init__(self, dim=1, parent=None):
        super().__init__("Distribution Builder", parent)
        self._dim = dim
        self._rows = []          # list of ComponentRow
        self._combinators = []   # list of QComboBox (between rows)

        self._layout = QVBoxLayout(self)

        # Component area
        self._comp_area = QVBoxLayout()
        self._layout.addLayout(self._comp_area)

        # Add component button
        add_btn = QPushButton("➕ Add Component")
        add_btn.clicked.connect(lambda: self.add_component())
        self._layout.addWidget(add_btn)

        # Start with one Gaussian
        self.add_component(GaussianComponent(dim))

    # ── Public API ──

    @property
    def dim(self):
        return self._dim

    def set_dim(self, dim):
        """Update dimensionality for all components."""
        self._dim = dim
        for row in self._rows:
            row.set_dim(dim)
        self.distribution_changed.emit()

    def clear(self):
        """Remove all components."""
        for row in list(self._rows):
            self._remove_component(row)

    def add_component(self, comp_data=None):
        """Add a new component row, optionally with pre-set data."""
        # If not the first, add a combinator
        if self._rows:
            combo = QComboBox()
            combo.addItems(["+", "×"])
            combo.setFixedWidth(50)
            combo.currentTextChanged.connect(
                lambda _: self.distribution_changed.emit()
            )
            self._combinators.append(combo)
            h = QHBoxLayout()
            h.addStretch()
            h.addWidget(combo)
            h.addStretch()
            w = QWidget()
            w.setLayout(h)
            self._comp_area.addWidget(w)
            combo._wrapper = w  # stash for removal

        row = ComponentRow(self._dim, comp_data, self)
        row.removed.connect(self._remove_component)
        row.changed.connect(lambda: self.distribution_changed.emit())
        self._rows.append(row)
        self._comp_area.addWidget(row)
        self.distribution_changed.emit()

    def load_preset(self, components, combinators):
        """
        Load a preset configuration.

        Parameters
        ----------
        components : list of dicts with "type", "mean", "covariance" or "terms"
        combinators : list of "+" or "×"
        """
        self.clear()
        for i, cdef in enumerate(components):
            if cdef["type"] == "Gaussian":
                c = GaussianComponent(self._dim)
                if cdef.get("mean") is not None:
                    c.mean = np.asarray(cdef["mean"], dtype=float)
                if cdef.get("covariance") is not None:
                    c.covariance = np.asarray(cdef["covariance"], dtype=float)
            else:
                c = PolynomialComponent(self._dim)
                if cdef.get("terms"):
                    c.terms = cdef["terms"]
                c.wrap_exp = cdef.get("wrap_exp", True)
            self.add_component(c)

        # Set combinators
        for i, op in enumerate(combinators):
            if i < len(self._combinators):
                idx = self._combinators[i].findText(op)
                if idx >= 0:
                    self._combinators[i].setCurrentIndex(idx)

    # ── Build callables ──

    def build_function(self):
        """
        Build and return p_tilde(x) as a callable.

        Returns a function that accepts a scalar (1-D) or np.ndarray (d-D)
        and returns a positive scalar.
        """
        components = [r.component for r in self._rows]
        ops = [c.currentText() for c in self._combinators]

        def p_tilde(x):
            x_arr = np.atleast_1d(np.asarray(x, dtype=float))
            val = components[0].evaluate(x_arr)
            for i, comp in enumerate(components[1:]):
                cv = comp.evaluate(x_arr)
                if ops[i] == "+":
                    val = val + cv
                else:
                    val = val * cv
            return max(val, 1e-300)

        return p_tilde

    def build_energy(self):
        """
        Build and return E(x) for Langevin sampling.

        For a single component, returns its energy directly.
        For mixtures (+), returns -log(p_tilde(x)).
        For products (×), returns sum of energies.
        """
        components = [r.component for r in self._rows]
        ops = [c.currentText() for c in self._combinators]

        # Check if all combinators are ×
        all_product = all(op == "×" for op in ops)

        if len(components) == 1 or all_product:
            # Sum of individual energies
            def energy(x):
                x_arr = np.atleast_1d(np.asarray(x, dtype=float))
                return sum(c.energy(x_arr) for c in components)
            return energy
        else:
            # General case: -log(p_tilde)
            p_tilde = self.build_function()

            def energy(x):
                return -np.log(max(p_tilde(x), 1e-300))
            return energy

    # ── Internal ──

    def _remove_component(self, row):
        if len(self._rows) <= 1:
            return  # keep at least one

        idx = self._rows.index(row)

        # Remove the combinator
        if idx > 0:
            combo = self._combinators[idx - 1]
            combo._wrapper.deleteLater()
            self._combinators.pop(idx - 1)
        elif idx == 0 and self._combinators:
            combo = self._combinators[0]
            combo._wrapper.deleteLater()
            self._combinators.pop(0)

        self._rows.remove(row)
        self._comp_area.removeWidget(row)
        row.deleteLater()
        self.distribution_changed.emit()
