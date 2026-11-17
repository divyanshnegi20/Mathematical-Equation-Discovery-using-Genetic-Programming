"""
Bloat Control Mechanisms for Genetic Programming.

Code Bloat is the uncontrolled growth of program size without proportional
fitness improvement, common in tree-based GP.

This module provides:
1. Parsimony Pressure (Fitness Penalty)
2. Strict Depth and Size Limiting
3. Tarpeian Selection Filter
4. Double Tournament Selection (Multi-Objective Fitness & Size)
"""

from typing import List, Optional
import numpy as np
from .tree import Individual


class BloatController:
    """
    Manages bloat mitigation techniques during GP evolution.
    """

    def __init__(
        self,
        parsimony_coefficient: float = 0.001,
        max_depth: int = 8,
        max_size: int = 40,
        tarpeian_prob: float = 0.0,
        rng: Optional[np.random.RandomState] = None
    ):
        self.parsimony_coefficient = parsimony_coefficient
        self.max_depth = max_depth
        self.max_size = max_size
        self.tarpeian_prob = tarpeian_prob
        self.rng = rng if rng is not None else np.random.RandomState()

    def apply_parsimony_pressure(self, raw_fitness: float, tree_size: int) -> float:
        """
        Adjusted Fitness = Raw MSE + (alpha * Size).
        Penalizes large expression trees to favor compact, Occam's razor formulas.
        """
        return raw_fitness + (self.parsimony_coefficient * tree_size)

    def apply_tarpeian_filter(self, population: List[Individual]) -> None:
        """
        Tarpeian Method:
        Individuals whose size exceeds the population average size are assigned
        an atrocious fitness with probability tarpeian_prob.
        """
        if self.tarpeian_prob <= 0.0 or not population:
            return

        avg_size = np.mean([ind.size for ind in population])
        for ind in population:
            if ind.size > avg_size and self.rng.rand() < self.tarpeian_prob:
                ind.fitness = 1e9

    def double_tournament_selection(
        self,
        population: List[Individual],
        sample_size: int = 6
    ) -> Individual:
        """
        Double Tournament:
        First tournament selects subset based on fitness.
        Second tournament selects the champion among them based on smaller tree size.
        Balances accuracy with model simplicity.
        """
        indices = self.rng.choice(len(population), size=sample_size, replace=True)
        candidates = [population[i] for i in indices]

        # Stage 1: Select top 2 individuals by fitness
        candidates.sort(key=lambda ind: ind.fitness)
        top_tier = candidates[:2]

        # Stage 2: Pick the more parsimonious (smaller) of the top candidates
        champion = min(top_tier, key=lambda ind: ind.size)
        return champion
