"""
Individual (Expression Tree) Representation in Genetic Programming.

Wraps AST root node and manages fitness evaluation, complexity metrics,
and mathematical expression conversion.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import sympy as sp
from .node import Node


class Individual:
    """
    Candidate solution (mathematical equation) in the GP population.
    """

    def __init__(self, root: Node):
        self.root = root
        self.raw_fitness: float = float('inf')  # Base error (e.g. MSE)
        self.fitness: float = float('inf')      # Penalized error with parsimony pressure
        self.rmse: float = float('inf')
        self.mae: float = float('inf')
        self.r2: float = float('-inf')
        self._size: Optional[int] = None
        self._depth: Optional[int] = None
        self._simplified_expr: Optional[sp.Expr] = None

    @property
    def size(self) -> int:
        if self._size is None:
            self._size = self.root.size()
        return self._size

    @property
    def depth(self) -> int:
        if self._depth is None:
            self._depth = self.root.depth()
        return self._depth

    def invalidate_cache(self) -> None:
        """Call whenever tree structure is modified."""
        self._size = None
        self._depth = None
        self._simplified_expr = None
        self.raw_fitness = float('inf')
        self.fitness = float('inf')

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Computes model output predictions y_pred."""
        return self.root.evaluate(X)

    def calculate_fitness(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        parsimony_coefficient: float = 0.001
    ) -> float:
        """
        Calculates MSE, RMSE, MAE, R^2, and applies bloat penalty (parsimony pressure).
        Fitness = MSE + parsimony_coefficient * tree_size
        Minimization objective: lower fitness is better.
        """
        y_pred = self.evaluate(X)

        # Check for non-finite values
        if not np.all(np.isfinite(y_pred)):
            self.raw_fitness = 1e9
            self.fitness = 1e9
            self.rmse = 1e9
            self.mae = 1e9
            self.r2 = -1e9
            return self.fitness

        errors = y_pred - y_true
        mse = float(np.mean(errors ** 2))
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(mse))

        # R^2 Score
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum(errors ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 1e-12 else 0.0

        self.raw_fitness = mse
        self.rmse = rmse
        self.mae = mae
        self.r2 = float(r2)

        # Bloat penalty: parsimony pressure
        # Fitness penalizes excessively large trees to encourage minimal, elegant equations
        self.fitness = mse + (parsimony_coefficient * self.size)
        return self.fitness

    def clone(self) -> 'Individual':
        """Deep copies the individual."""
        ind = Individual(self.root.clone())
        ind.raw_fitness = self.raw_fitness
        ind.fitness = self.fitness
        ind.rmse = self.rmse
        ind.mae = self.mae
        ind.r2 = self.r2
        ind._size = self._size
        ind._depth = self._depth
        return ind

    def to_equation_str(self, variable_names: Optional[List[str]] = None) -> str:
        """Raw string representation of tree."""
        return self.root.to_str(variable_names)

    def to_sympy(self, variable_names: Optional[List[str]] = None, simplify: bool = True) -> sp.Expr:
        """Converts to SymPy symbolic algebraic expression and simplifies."""
        if self._simplified_expr is not None and simplify:
            return self._simplified_expr

        expr = self.root.to_sympy(variable_names)
        if simplify:
            try:
                # Round floats in sympy expression
                expr = sp.nsimplify(expr, tolerance=1e-3, rational=False)
                expr = sp.simplify(expr)
            except Exception:
                pass
            self._simplified_expr = expr
        return expr

    def to_latex(self, variable_names: Optional[List[str]] = None, simplify: bool = True) -> str:
        """Generates LaTeX mathematical formula representation."""
        try:
            expr = self.to_sympy(variable_names, simplify=simplify)
            return sp.latex(expr)
        except Exception:
            return self.to_equation_str(variable_names)

    def __repr__(self) -> str:
        return f"Individual(size={self.size}, depth={self.depth}, MSE={self.raw_fitness:.5g}, R2={self.r2:.4f}, eq='{self.to_equation_str()}')"
