"""
Standard Regression Models for Comparative Benchmarking.

Includes:
1. Linear Regression (OLS)
2. Ridge Regression (L2 Regularization)
3. Polynomial Regression (Degree 2 & 3)
4. Decision Tree Regressor (CART)
5. Random Forest Regressor
6. Gradient Boosting Regressor
7. Multi-Layer Perceptron (MLP Neural Network)
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


def get_standard_baselines(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Returns standard regression baselines with complexity calculation hooks.
    """
    baselines = {
        "Linear Regression": {
            "model": LinearRegression(),
            "type": "Linear / White-Box",
            "interpretability": "High (Linear Coefficients)",
            "complexity_calc": lambda m: int(len(m.coef_.ravel()) + 1)
        },
        "Ridge Regression": {
            "model": Ridge(alpha=1.0, random_state=random_state),
            "type": "Regularized Linear",
            "interpretability": "High (Shrunk Coefficients)",
            "complexity_calc": lambda m: int(len(m.coef_.ravel()) + 1)
        },
        "Polynomial Reg (Deg 2)": {
            "model": Pipeline([
                ("poly", PolynomialFeatures(degree=2)),
                ("reg", LinearRegression())
            ]),
            "type": "Polynomial Expansion",
            "interpretability": "Medium (Polynomial Terms)",
            "complexity_calc": lambda m: int(len(m.named_steps["reg"].coef_.ravel()) + 1)
        },
        "Polynomial Reg (Deg 3)": {
            "model": Pipeline([
                ("poly", PolynomialFeatures(degree=3)),
                ("reg", LinearRegression())
            ]),
            "type": "Polynomial Expansion",
            "interpretability": "Medium-Low (Many Terms)",
            "complexity_calc": lambda m: int(len(m.named_steps["reg"].coef_.ravel()) + 1)
        },
        "Decision Tree (CART)": {
            "model": DecisionTreeRegressor(max_depth=5, random_state=random_state),
            "type": "Decision Tree",
            "interpretability": "Medium (Hierarchical Splits)",
            "complexity_calc": lambda m: int(m.tree_.node_count)
        },
        "Random Forest": {
            "model": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=random_state),
            "type": "Ensemble (Bagging)",
            "interpretability": "Low (Black-Box Forest)",
            "complexity_calc": lambda m: int(sum(est.tree_.node_count for est in m.estimators_))
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=random_state),
            "type": "Ensemble (Boosting)",
            "interpretability": "Low (Sequential Additive Trees)",
            "complexity_calc": lambda m: int(sum(tree[0].tree_.node_count for tree in m.estimators_))
        },
        "MLP Neural Network": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    max_iter=500,
                    random_state=random_state,
                    early_stopping=True
                ))
            ]),
            "type": "Neural Network",
            "interpretability": "Very Low (Opaque Weights)",
            "complexity_calc": lambda m: int(sum(
                w.size + b.size for w, b in zip(m.named_steps["mlp"].coefs_, m.named_steps["mlp"].intercepts_)
            ))
        }
    }
    return baselines
