"""
Protected Mathematical Functions and Operator Registry for Genetic Programming.

In Genetic Programming (GP) for Symbolic Regression, mathematical operations
must be protected to avoid runtime exceptions such as Division by Zero,
Overflow, or Invalid Domain errors (e.g. sqrt/log of negative numbers).
"""

from typing import Callable, Dict, Any, List
import numpy as np
import sympy as sp


def protected_div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Koza-style protected division: returns 1.0 when denominator is close to 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(y) < 1e-7, 1.0, x / y)


def protected_log(x: np.ndarray) -> np.ndarray:
    """Protected natural log: returns 0.0 when argument is close to 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_x = np.abs(x)
        return np.where(abs_x < 1e-7, 0.0, np.log(abs_x))


def protected_sqrt(x: np.ndarray) -> np.ndarray:
    """Protected square root: operates on absolute value to prevent complex outputs."""
    return np.sqrt(np.abs(x))


def protected_exp(x: np.ndarray) -> np.ndarray:
    """Protected exponential: clipped to prevent overflow in floating point calculations."""
    clipped = np.clip(x, -25.0, 25.0)
    return np.exp(clipped)


def protected_inv(x: np.ndarray) -> np.ndarray:
    """Protected reciprocal: 1 / x."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) < 1e-7, 0.0, 1.0 / x)


class FunctionInfo:
    """Metadata container for a GP function node."""
    def __init__(
        self,
        name: str,
        arity: int,
        func: Callable,
        symbol: str,
        sympy_builder: Callable
    ):
        self.name = name
        self.arity = arity
        self.func = func
        self.symbol = symbol
        self.sympy_builder = sympy_builder

    def __call__(self, *args):
        return self.func(*args)


# Function definitions with SymPy mappings
FUNCTION_REGISTRY: Dict[str, FunctionInfo] = {
    "add": FunctionInfo(
        name="add",
        arity=2,
        func=lambda x, y: x + y,
        symbol="+",
        sympy_builder=lambda x, y: x + y
    ),
    "sub": FunctionInfo(
        name="sub",
        arity=2,
        func=lambda x, y: x - y,
        symbol="-",
        sympy_builder=lambda x, y: x - y
    ),
    "mul": FunctionInfo(
        name="mul",
        arity=2,
        func=lambda x, y: x * y,
        symbol="*",
        sympy_builder=lambda x, y: x * y
    ),
    "div": FunctionInfo(
        name="div",
        arity=2,
        func=protected_div,
        symbol="/",
        sympy_builder=lambda x, y: x / y if y != 0 else x
    ),
    "sin": FunctionInfo(
        name="sin",
        arity=1,
        func=lambda x: np.sin(x),
        symbol="sin",
        sympy_builder=lambda x: sp.sin(x)
    ),
    "cos": FunctionInfo(
        name="cos",
        arity=1,
        func=lambda x: np.cos(x),
        symbol="cos",
        sympy_builder=lambda x: sp.cos(x)
    ),
    "exp": FunctionInfo(
        name="exp",
        arity=1,
        func=protected_exp,
        symbol="exp",
        sympy_builder=lambda x: sp.exp(x)
    ),
    "log": FunctionInfo(
        name="log",
        arity=1,
        func=protected_log,
        symbol="log",
        sympy_builder=lambda x: sp.log(sp.Abs(x) + 1e-9)
    ),
    "sqrt": FunctionInfo(
        name="sqrt",
        arity=1,
        func=protected_sqrt,
        symbol="sqrt",
        sympy_builder=lambda x: sp.sqrt(sp.Abs(x))
    ),
    "neg": FunctionInfo(
        name="neg",
        arity=1,
        func=lambda x: -x,
        symbol="-",
        sympy_builder=lambda x: -x
    ),
}


def get_default_functions(include_trig: bool = True, include_transcendental: bool = False) -> Dict[str, FunctionInfo]:
    """Returns standard function set tailored to the complexity desired."""
    base = ["add", "sub", "mul", "div"]
    if include_trig:
        base.extend(["sin", "cos"])
    if include_transcendental:
        base.extend(["exp", "log", "sqrt"])
    return {k: FUNCTION_REGISTRY[k] for k in base}
