"""
Selection Mechanisms for Genetic Programming.

Implements:
1. Tournament Selection (primary standard in GP)
2. Roulette Wheel / Fitness-Proportionate Selection
3. Elitism Preservation
"""

from typing import List, Optional
import numpy as np
from .tree import Individual


class Selector:
    """
    Parent selection algorithms for evolutionary reproduction.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng if rng is not None else np.random.RandomState()

    def tournament_selection(
        self,
        population: List[Individual],
        tournament_size: int = 5
    ) -> Individual:
        """
        Tournament Selection:
        Samples k candidates at random with replacement and returns the champion
        with minimal fitness (since regression fitness is minimized error + bloat penalty).
        """
        indices = self.rng.choice(len(population), size=tournament_size, replace=True)
        candidates = [population[i] for i in indices]
        champion = min(candidates, key=lambda ind: ind.fitness)
        return champion

    def roulette_selection(
        self,
        population: List[Individual]
    ) -> Individual:
        """
        Fitness-proportionate selection adapted for minimization.
        Uses rank-based inversion or scaled inverted fitness.
        """
        # Sort ascending by fitness (best is index 0)
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        n = len(sorted_pop)
        # Linear ranking selection probabilities: rank 1 has highest probability
        ranks = np.arange(n, 0, -1, dtype=np.float64)
        probs = ranks / np.sum(ranks)
        selected_idx = self.rng.choice(n, p=probs)
        return sorted_pop[selected_idx]

    def select_elites(
        self,
        population: List[Individual],
        elite_count: int
    ) -> List[Individual]:
        """
        Elitism: Preserves the top 'elite_count' best individuals into the next generation.
        Clones them so crossover/mutation does not corrupt elite solutions.
        """
        if elite_count <= 0:
            return []
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        return [ind.clone() for ind in sorted_pop[:elite_count]]
