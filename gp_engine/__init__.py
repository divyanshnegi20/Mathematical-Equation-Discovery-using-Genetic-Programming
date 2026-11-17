"""
Genetic Programming Symbolic Regression Engine.

A modular, textbook-faithful implementation of Genetic Programming
for Mathematical Equation Discovery.
"""

from .functions import FUNCTION_REGISTRY, FunctionInfo, get_default_functions
from .node import Node
from .tree import Individual
from .population import TreeGenerator, PopulationManager
from .operators import GeneticOperators
from .selection import Selector
from .bloat import BloatController
from .simplifier import EquationSimplifier
from .estimator import GeneticProgrammingRegressor

__all__ = [
    "FUNCTION_REGISTRY",
    "FunctionInfo",
    "get_default_functions",
    "Node",
    "Individual",
    "TreeGenerator",
    "PopulationManager",
    "GeneticOperators",
    "Selector",
    "BloatController",
    "EquationSimplifier",
    "GeneticProgrammingRegressor",
]
