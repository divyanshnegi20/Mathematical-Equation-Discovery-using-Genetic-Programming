"""
Tree Generation and Population Initialization for Genetic Programming.

Implements standard textbook initialization algorithms:
1. Full Method
2. Grow Method
3. Ramped Half-and-Half (RH&H)
"""

from typing import List, Dict, Optional, Tuple, Union
import random
import numpy as np
from .node import Node
from .tree import Individual
from .functions import FunctionInfo, FUNCTION_REGISTRY, get_default_functions


class TreeGenerator:
    """
    Factory for constructing GP expression trees.
    """

    def __init__(
        self,
        n_features: int,
        function_set: Optional[Dict[str, FunctionInfo]] = None,
        constant_range: Tuple[float, float] = (-5.0, 5.0),
        p_constant: float = 0.35,
        rng: Optional[np.random.RandomState] = None
    ):
        self.n_features = n_features
        self.function_set = function_set if function_set is not None else get_default_functions()
        self.function_names = list(self.function_set.keys())
        self.constant_range = constant_range
        self.p_constant = p_constant
        self.rng = rng if rng is not None else np.random.RandomState()

    def generate_random_terminal(self) -> Node:
        """Generates a random terminal leaf (Variable or Constant)."""
        # If random choice falls below p_constant or no features
        if self.rng.rand() < self.p_constant or self.n_features == 0:
            c_val = self.rng.uniform(self.constant_range[0], self.constant_range[1])
            # Round constant slightly for cleaner equations
            c_val = round(float(c_val), 3)
            return Node(node_type='constant', value=c_val)
        else:
            var_idx = int(self.rng.randint(0, self.n_features))
            return Node(node_type='variable', value=var_idx)

    def generate_random_function(self) -> Tuple[str, FunctionInfo]:
        """Selects a random function operator from function set."""
        fn_name = self.rng.choice(self.function_names)
        return fn_name, self.function_set[fn_name]

    def build_tree(self, method: str, max_depth: int, current_depth: int = 1) -> Node:
        """
        Builds a tree recursively using 'full' or 'grow' methodology.
        - 'full': every branch reaches max_depth.
        - 'grow': branches may terminate before max_depth.
        """
        if current_depth >= max_depth:
            return self.generate_random_terminal()

        if method == 'full':
            is_function = True
        elif method == 'grow':
            # At root (current_depth == 1), prefer function to prevent single-leaf individuals
            if current_depth == 1:
                is_function = True
            else:
                is_function = self.rng.rand() >= self.p_constant
        else:
            raise ValueError(f"Unknown tree building method: {method}")

        if is_function:
            fn_name, fn_info = self.generate_random_function()
            children = [
                self.build_tree(method, max_depth, current_depth + 1)
                for _ in range(fn_info.arity)
            ]
            return Node(node_type='function', value=fn_name, children=children)
        else:
            return self.generate_random_terminal()

    def create_individual(self, method: str, max_depth: int) -> Individual:
        """Creates a single individual with specified method and depth limit."""
        root = self.build_tree(method, max_depth)
        return Individual(root)


class PopulationManager:
    """
    Manages population creation, tracking, and evaluation.
    """

    def __init__(
        self,
        population_size: int,
        tree_generator: TreeGenerator,
        min_depth: int = 2,
        max_init_depth: int = 5,
        rng: Optional[np.random.RandomState] = None
    ):
        self.population_size = population_size
        self.tree_generator = tree_generator
        self.min_depth = min_depth
        self.max_init_depth = max_init_depth
        self.rng = rng if rng is not None else np.random.RandomState()
        self.individuals: List[Individual] = []

    def initialize_ramped_half_and_half(self) -> List[Individual]:
        """
        Ramped Half-and-Half (Koza 1992):
        Distributes population across depths [min_depth, max_init_depth].
        For each depth:
        - 50% individuals created via 'full'
        - 50% individuals created via 'grow'
        """
        depths = list(range(self.min_depth, self.max_init_depth + 1))
        n_depths = len(depths)
        per_depth = self.population_size // n_depths
        remainder = self.population_size % n_depths

        population: List[Individual] = []
        for i, depth in enumerate(depths):
            count = per_depth + (1 if i < remainder else 0)
            half = count // 2
            other_half = count - half

            for _ in range(half):
                population.append(self.tree_generator.create_individual('full', depth))
            for _ in range(other_half):
                population.append(self.tree_generator.create_individual('grow', depth))

        self.individuals = population
        return self.individuals

    def evaluate_population(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parsimony_coefficient: float = 0.001
    ) -> Individual:
        """Evaluates all individuals and returns the current generation champion."""
        best_ind = None
        best_fitness = float('inf')

        for ind in self.individuals:
            fitness = ind.calculate_fitness(X, y, parsimony_coefficient)
            if fitness < best_fitness:
                best_fitness = fitness
                best_ind = ind

        return best_ind
