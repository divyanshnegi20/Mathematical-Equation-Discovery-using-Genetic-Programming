"""
Tree Node Representation for Genetic Programming AST.

Represents operators (internal nodes), variables (leaf terminals),
and Ephemeral Random Constants (ERC leaf terminals).
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
import sympy as sp
from .functions import FUNCTION_REGISTRY, FunctionInfo


class Node:
    """
    Abstract Syntax Tree (AST) node for Symbolic Regression.
    """

    def __init__(
        self,
        node_type: str,  # 'function', 'variable', 'constant'
        value: Any,      # str (function name), int (variable index), float (constant)
        children: Optional[List['Node']] = None
    ):
        if node_type not in {'function', 'variable', 'constant'}:
            raise ValueError(f"Invalid node_type '{node_type}'. Must be 'function', 'variable', or 'constant'.")

        self.node_type = node_type
        self.value = value
        self.children = children if children is not None else []

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def arity(self) -> int:
        return len(self.children)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the subtree on input matrix X of shape (n_samples, n_features).
        Returns a 1D numpy array of shape (n_samples,).
        """
        n_samples = X.shape[0]

        if self.node_type == 'constant':
            return np.full(n_samples, float(self.value), dtype=np.float64)

        if self.node_type == 'variable':
            idx = int(self.value)
            if idx >= X.shape[1]:
                raise IndexError(f"Variable index X[{idx}] out of bounds for feature dim {X.shape[1]}")
            return X[:, idx].astype(np.float64)

        if self.node_type == 'function':
            fn_info = FUNCTION_REGISTRY.get(self.value)
            if fn_info is None:
                raise KeyError(f"Unknown function operator '{self.value}'")

            evaluated_children = [child.evaluate(X) for child in self.children]
            res = fn_info(*evaluated_children)
            # Ensure scalar or 1D array
            if np.isscalar(res):
                res = np.full(n_samples, res, dtype=np.float64)
            # Replace NaNs or Infs that slipped through with 0.0 or finite boundary
            res = np.nan_to_num(res, nan=0.0, posinf=1e9, neginf=-1e9)
            return res

        raise RuntimeError(f"Unknown node type: {self.node_type}")

    def size(self) -> int:
        """Total number of nodes in this subtree."""
        return 1 + sum(child.size() for child in self.children)

    def depth(self) -> int:
        """Maximum depth of this subtree (leaf node has depth 1)."""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def clone(self) -> 'Node':
        """Deep copy of the node and all its descendants."""
        return Node(
            node_type=self.node_type,
            value=self.value,
            children=[c.clone() for c in self.children]
        )

    def get_all_subtrees(self) -> List['Node']:
        """Returns a list of all nodes in the subtree (pre-order traversal)."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_subtrees())
        return nodes

    def replace_child(self, old_child: 'Node', new_child: 'Node') -> bool:
        """Replaces a direct child reference. Returns True if replaced."""
        for i, child in enumerate(self.children):
            if child is old_child:
                self.children[i] = new_child
                return True
        return False

    def to_str(self, variable_names: Optional[List[str]] = None) -> str:
        """Returns standard infix mathematical expression string."""
        if self.node_type == 'constant':
            val = float(self.value)
            return f"{val:.4g}"

        if self.node_type == 'variable':
            idx = int(self.value)
            if variable_names and idx < len(variable_names):
                return variable_names[idx]
            return f"x{idx}"

        if self.node_type == 'function':
            fn_info = FUNCTION_REGISTRY.get(self.value)
            sym = fn_info.symbol if fn_info else self.value

            if len(self.children) == 1:
                child_str = self.children[0].to_str(variable_names)
                return f"{sym}({child_str})"
            elif len(self.children) == 2:
                left = self.children[0].to_str(variable_names)
                right = self.children[1].to_str(variable_names)
                return f"({left} {sym} {right})"
            else:
                args = ", ".join(c.to_str(variable_names) for c in self.children)
                return f"{self.value}({args})"

        return str(self.value)

    def to_sympy(self, variable_names: Optional[List[str]] = None) -> sp.Expr:
        """Converts AST node recursively to a SymPy analytical symbolic expression."""
        if self.node_type == 'constant':
            return sp.Float(round(float(self.value), 5))

        if self.node_type == 'variable':
            idx = int(self.value)
            name = variable_names[idx] if (variable_names and idx < len(variable_names)) else f"x{idx}"
            return sp.Symbol(name)

        if self.node_type == 'function':
            fn_info = FUNCTION_REGISTRY.get(self.value)
            if not fn_info:
                raise KeyError(f"Unknown function {self.value}")

            child_exprs = [c.to_sympy(variable_names) for c in self.children]
            return fn_info.sympy_builder(*child_exprs)

        return sp.Integer(0)

    def __repr__(self) -> str:
        return self.to_str()
