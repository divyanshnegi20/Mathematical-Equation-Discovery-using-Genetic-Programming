"""
Evaluation Metrics and Statistical Comparisons.

Measures:
1. Error Metrics: MSE, RMSE, MAE, R^2 Score
2. Model Complexity: Parameter count, AST nodes, model degrees of freedom
3. Runtime Efficiency: Train time and Inference latency
4. Generalization: Train-to-Test error ratio
"""

from typing import Dict, Any
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Computes standard regression error metrics."""
    # Ensure safe finite values
    y_pred = np.nan_to_num(y_pred, nan=1e9, posinf=1e9, neginf=-1e9)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


def measure_inference_speed(
    model: Any,
    X: np.ndarray,
    n_repeats: int = 5
) -> float:
    """Measures average inference time in microseconds (us) per sample."""
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = model.predict(X)
    elapsed = time.perf_counter() - start
    us_per_sample = (elapsed / (n_repeats * len(X))) * 1e6
    return float(us_per_sample)
