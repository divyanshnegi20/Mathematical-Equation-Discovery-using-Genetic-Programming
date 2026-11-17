"""
SymPy Equation Simplification and LaTeX Formatter for Discovered Formulas.

Takes the raw Genetic Programming AST and produces:
- Mathematically simplified closed-form analytical formulas
- Clean LaTeX equations for academic papers, slides, and web rendering
- Complexity node comparison (pre-simplification vs post-simplification)
"""

from typing import List, Optional, Tuple, Dict, Any
import sympy as sp
from .tree import Individual


class EquationSimplifier:
    """
    Simplifies and formats mathematical equations evolved by GP.
    """

    @staticmethod
    def simplify_individual(
        individual: Individual,
        variable_names: Optional[List[str]] = None,
        round_decimals: int = 4
    ) -> Dict[str, Any]:
        """
        Extracts, simplifies, and formats an individual's symbolic formula.
        """
        raw_eq = individual.to_equation_str(variable_names)

        try:
            raw_sympy = individual.root.to_sympy(variable_names)
            # Simplify expression
            simplified_sympy = sp.simplify(raw_sympy)

            # Round any floating numbers inside the sympy expression
            def round_expr(expr, num_digits):
                return expr.xreplace({
                    n: round(n, num_digits)
                    for n in expr.atoms(sp.Number)
                    if not n.is_Integer
                })

            rounded_sympy = round_expr(simplified_sympy, round_decimals)
            latex_str = sp.latex(rounded_sympy)
            readable_str = str(rounded_sympy)

            # Calculate simplified complexity (number of SymPy AST atoms/operations)
            simplified_node_count = len(list(sp.preorder_traversal(rounded_sympy)))
        except Exception as e:
            readable_str = raw_eq
            latex_str = raw_eq
            simplified_node_count = individual.size

        return {
            "raw_equation": raw_eq,
            "simplified_equation": readable_str,
            "latex": latex_str,
            "original_tree_size": individual.size,
            "original_depth": individual.depth,
            "simplified_node_count": simplified_node_count,
            "raw_mse": individual.raw_fitness,
            "rmse": individual.rmse,
            "mae": individual.mae,
            "r2_score": individual.r2
        }
