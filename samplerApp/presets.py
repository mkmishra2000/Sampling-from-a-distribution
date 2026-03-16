"""
Preset distribution configurations for the distribution builder.

Each preset is a dict describing builder components, default x_range,
and suggested algorithm parameters.
"""

import numpy as np

# ──────────────────────────────────────────────────────────────────
#  Preset registry
# ──────────────────────────────────────────────────────────────────
# Each preset is a dict with:
#   "label"       : display name
#   "dim"         : None (any) or int (fixed dimensionality)
#   "components"  : list of component dicts
#   "combinators" : list of "+" or "×" between adjacent components
#   "x_range"     : callable(dim) → list of [lo, hi] per dimension
#   "params"      : dict of suggested algorithm parameters

PRESETS = {
    "Standard Gaussian": {
        "label": "Standard Gaussian",
        "dim": None,  # works in any dimension
        "components": [
            {
                "type": "Gaussian",
                "mean": None,       # filled dynamically: zeros(d)
                "covariance": None,  # filled dynamically: eye(d)
            }
        ],
        "combinators": [],
        "x_range": lambda d: [[-4.0, 4.0]] * d,
        "params": {
            "n_samples": 50_000,
            "eta": 0.01,
            "step_size": 0.5,
            "safety_margin": 1.1,
            "grid_points": 100_000,
        },
    },

    "Bimodal Gaussian": {
        "label": "Bimodal Gaussian",
        "dim": 1,
        "components": [
            {
                "type": "Gaussian",
                "mean": np.array([3.0]),
                "covariance": np.array([[1.0]]),
            },
            {
                "type": "Gaussian",
                "mean": np.array([-3.0]),
                "covariance": np.array([[1.0]]),
            },
        ],
        "combinators": ["+"],
        "x_range": lambda d: [[-8.0, 8.0]],
        "params": {
            "n_samples": 50_000,
            "eta": 0.04,
            "step_size": 0.8,
            "safety_margin": 1.1,
            "grid_points": 100_000,
        },
    },

    "Correlated Gaussian": {
        "label": "Correlated Gaussian (ρ = 0.6)",
        "dim": 2,
        "components": [
            {
                "type": "Gaussian",
                "mean": np.array([0.0, 0.0]),
                "covariance": np.array([[1.0, 0.6],
                                        [0.6, 1.0]]),
            }
        ],
        "combinators": [],
        "x_range": lambda d: [[-4.0, 4.0]] * d,
        "params": {
            "n_samples": 100_000,
            "eta": 0.01,
            "step_size": 0.5,
            "safety_margin": 1.1,
            "grid_points": 200_000,
        },
    },

    "Isotropic Gaussian": {
        "label": "Isotropic Gaussian",
        "dim": None,
        "components": [
            {
                "type": "Gaussian",
                "mean": None,
                "covariance": None,
            }
        ],
        "combinators": [],
        "x_range": lambda d: [[-4.0, 4.0]] * d,
        "params": {
            "n_samples": 50_000,
            "eta": 0.01,
            "step_size": 0.5,
            "safety_margin": 1.1,
            "grid_points": 100_000,
        },
    },
}


def get_preset_names():
    """Return ordered list of preset display names."""
    return list(PRESETS.keys())


def get_preset(name):
    """Return a deep copy of the preset dict, with dynamic fields resolved."""
    return PRESETS.get(name)


def resolve_preset(name, dim):
    """
    Resolve a preset for a given dimensionality.

    Fills in None mean/covariance with zeros(d) / eye(d),
    and evaluates the x_range callable.

    Returns
    -------
    dict with keys: components, combinators, x_range, params
    """
    preset = PRESETS.get(name)
    if preset is None:
        return None

    # Check dimension compatibility
    if preset["dim"] is not None and preset["dim"] != dim:
        return None

    import copy
    resolved = copy.deepcopy(preset)

    for comp in resolved["components"]:
        if comp["type"] == "Gaussian":
            if comp["mean"] is None:
                comp["mean"] = np.zeros(dim)
            if comp["covariance"] is None:
                comp["covariance"] = np.eye(dim)

    resolved["x_range"] = preset["x_range"](dim)
    return resolved
