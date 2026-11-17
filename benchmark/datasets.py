"""
Benchmark Datasets for Mathematical Equation Discovery and Regression Comparison.

Includes:
1. Koza-1 Benchmark: y = x^4 + x^3 + x^2 + x
2. Nguyen-7 Benchmark: y = ln(x + 1) + ln(x^2 + 1)
3. Damped Harmonic Oscillator (Physics): y = exp(-0.25*x) * sin(2.5*x)
4. Multi-Variable Kinematics / Pendulum: y = 2.0 * sqrt(x0) + 0.5 * sin(x1)
5. Real-World Engineering / Scientific Dataset: Energy Efficiency / California Housing subset
"""

from typing import Tuple, Dict, List, Optional
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes


def generate_koza_quartic(
    n_samples: int = 100,
    noise_std: float = 0.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
    """
    Classic Koza-1 benchmark:
    f(x) = x^4 + x^3 + x^2 + x, for x in [-1.0, 1.0]
    """
    rng = np.random.RandomState(random_state)
    x = np.linspace(-1.0, 1.0, n_samples)
    y = x**4 + x**3 + x**2 + x
    if noise_std > 0.0:
        y += rng.normal(0, noise_std, size=n_samples)
    return x.reshape(-1, 1), y, "x^4 + x^3 + x^2 + x", ["x"]


def generate_damped_oscillator(
    n_samples: int = 120,
    noise_std: float = 0.02,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
    """
    Damped Harmonic Oscillator (Physics):
    f(x) = exp(-0.25 * x) * sin(2.5 * x), for x in [0.0, 6.0]
    """
    rng = np.random.RandomState(random_state)
    x = np.linspace(0.0, 6.0, n_samples)
    y = np.exp(-0.25 * x) * np.sin(2.5 * x)
    if noise_std > 0.0:
        y += rng.normal(0, noise_std, size=n_samples)
    return x.reshape(-1, 1), y, "exp(-0.25*x) * sin(2.5*x)", ["x"]


def generate_multivariable_physics(
    n_samples: int = 150,
    noise_std: float = 0.01,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
    """
    Multi-Variable Non-Linear Physics System:
    f(x0, x1) = 2.0 * sqrt(x0) + 1.5 * sin(2.0 * x1)
    x0 in [0.1, 5.0], x1 in [-3.14, 3.14]
    """
    rng = np.random.RandomState(random_state)
    x0 = rng.uniform(0.1, 5.0, size=n_samples)
    x1 = rng.uniform(-np.pi, np.pi, size=n_samples)
    y = 2.0 * np.sqrt(x0) + 1.5 * np.sin(2.0 * x1)
    if noise_std > 0.0:
        y += rng.normal(0, noise_std, size=n_samples)
    X = np.column_stack([x0, x1])
    return X, y, "2.0*sqrt(x0) + 1.5*sin(2.0*x1)", ["length", "phase"]


def load_real_world_dataset(
    n_samples: int = 250,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
    """
    Real-world dataset: Clinical / Physiological Progression (Diabetes Benchmark).
    Predicting quantitative disease progression from physiological markers (bmi, bp, s5).
    """
    data = load_diabetes()
    rng = np.random.RandomState(random_state)
    n = min(n_samples, len(data.target))
    indices = rng.choice(len(data.target), size=n, replace=False)

    # Use 3 most predictive biomarkers: bmi (index 2), bp (index 3), s5 (index 8)
    selected_cols = [2, 3, 8]
    X = data.data[indices][:, selected_cols]
    y = data.target[indices]
    var_names = ["BMI", "BloodPressure", "SerumMarker"]
    return X, y, "Real-World Clinical Biomarkers (Target: Quantitative Progression)", var_names


DATASET_REGISTRY = {
    "koza_quartic": {
        "name": "Koza-1 Quartic Polynomial",
        "category": "Synthetic Polynomial",
        "loader": generate_koza_quartic,
        "description": "Standard GP benchmark: y = x^4 + x^3 + x^2 + x"
    },
    "damped_oscillator": {
        "name": "Damped Harmonic Oscillator",
        "category": "Physics Dynamics",
        "loader": generate_damped_oscillator,
        "description": "Physical system with exponential damping and sinusoidal oscillation: y = exp(-0.25*x)*sin(2.5*x)"
    },
    "multivariable_physics": {
        "name": "Multi-Variable Physical System",
        "category": "Multi-Variable Non-Linear",
        "loader": generate_multivariable_physics,
        "description": "Two inputs: length and phase: y = 2*sqrt(x0) + 1.5*sin(2*x1)"
    },
    "real_world_biomarkers": {
        "name": "Clinical Biomarkers (Real-World)",
        "category": "Real-World Regression",
        "loader": load_real_world_dataset,
        "description": "Real-world medical regression predicting disease progression from BMI, Blood Pressure, and Serum markers"
    }
}


def export_benchmark_csvs(output_dir: str = "data"):
    """Saves benchmark datasets as ready-to-use CSV files for grading and testing."""
    os.makedirs(os.path.join(output_dir, "synthetic"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "real_world"), exist_ok=True)

    # 1. Koza
    X_k, y_k, _, names_k = generate_koza_quartic(n_samples=100)
    df_k = pd.DataFrame(X_k, columns=names_k)
    df_k["target_y"] = y_k
    df_k.to_csv(os.path.join(output_dir, "synthetic", "koza_quartic.csv"), index=False)

    # 2. Damped Oscillator
    X_d, y_d, _, names_d = generate_damped_oscillator(n_samples=120)
    df_d = pd.DataFrame(X_d, columns=names_d)
    df_d["target_y"] = y_d
    df_d.to_csv(os.path.join(output_dir, "synthetic", "damped_oscillator.csv"), index=False)

    # 3. Multivariable Physics
    X_m, y_m, _, names_m = generate_multivariable_physics(n_samples=150)
    df_m = pd.DataFrame(X_m, columns=names_m)
    df_m["target_y"] = y_m
    df_m.to_csv(os.path.join(output_dir, "synthetic", "physics_multivariable.csv"), index=False)

    # 4. Real World
    X_r, y_r, _, names_r = load_real_world_dataset(n_samples=250)
    df_r = pd.DataFrame(X_r, columns=names_r)
    df_r["target_y"] = y_r
    df_r.to_csv(os.path.join(output_dir, "real_world", "clinical_biomarkers.csv"), index=False)
