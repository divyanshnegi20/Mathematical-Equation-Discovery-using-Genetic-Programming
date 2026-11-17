"""
Scikit-Learn Compatible Symbolic Regression Estimator using Genetic Programming.

Provides standard fit/predict/score API with rich generational logging,
early stopping, and bloat control.
"""

from typing import Optional, List, Dict, Any, Callable, Union
import time
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .functions import get_default_functions, FunctionInfo
from .tree import Individual
from .population import TreeGenerator, PopulationManager
from .operators import GeneticOperators
from .selection import Selector
from .bloat import BloatController
from .simplifier import EquationSimplifier


class GeneticProgrammingRegressor(BaseEstimator, RegressorMixin):
    """
    Symbolic Regression using Genetic Programming.

    Parameters:
    -----------
    population_size : int, default=200
        Number of candidate programs in each generation.
    generations : int, default=40
        Maximum evolutionary cycles.
    tournament_size : int, default=5
        Number of candidates competing in parent selection tournament.
    p_crossover : float, default=0.70
        Probability of subtree crossover recombination.
    p_subtree_mutation : float, default=0.15
        Probability of replacing a subtree with a new random branch.
    p_point_mutation : float, default=0.08
        Probability of single node perturbation.
    p_hoist_mutation : float, default=0.04
        Probability of hoist mutation (bloat pruning operator).
    elite_count : int, default=2
        Number of top individuals preserved unconditionally.
    min_depth : int, default=2
        Minimum initial tree depth for Ramped Half-and-Half.
    max_init_depth : int, default=4
        Maximum initial tree depth.
    max_depth : int, default=6
        Hard ceiling on tree depth throughout evolution to curb bloat.
    parsimony_coefficient : float, default=0.001
        Penalty per tree node added to MSE fitness.
    constant_range : tuple, default=(-5.0, 5.0)
        Numeric bounds for ephemeral random constants.
    functions : Optional[Dict[str, FunctionInfo]], default=None
        Allowed mathematical operators.
    variable_names : Optional[List[str]], default=None
        Human-readable feature names (e.g. ['x0', 'x1', ...]).
    early_stopping_r2 : float, default=0.999
        Evolution stops early if R^2 exceeds this threshold.
    random_state : Optional[int], default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        population_size: int = 200,
        generations: int = 40,
        tournament_size: int = 5,
        p_crossover: float = 0.70,
        p_subtree_mutation: float = 0.15,
        p_point_mutation: float = 0.08,
        p_hoist_mutation: float = 0.04,
        elite_count: int = 2,
        min_depth: int = 2,
        max_init_depth: int = 4,
        max_depth: int = 6,
        parsimony_coefficient: float = 0.001,
        constant_range: tuple = (-5.0, 5.0),
        functions: Optional[Dict[str, FunctionInfo]] = None,
        variable_names: Optional[List[str]] = None,
        early_stopping_r2: float = 0.9995,
        random_state: Optional[int] = 42
    ):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_point_mutation = p_point_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.elite_count = elite_count
        self.min_depth = min_depth
        self.max_init_depth = max_init_depth
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.constant_range = constant_range
        self.functions = functions
        self.variable_names = variable_names
        self.early_stopping_r2 = early_stopping_r2
        self.random_state = random_state

        # Internal state
        self.best_individual_: Optional[Individual] = None
        self.history_: List[Dict[str, Any]] = []
        self.training_time_: float = 0.0
        self.n_features_in_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None
    ) -> 'GeneticProgrammingRegressor':
        """
        Evolves mathematical equations to fit X -> y.
        """
        start_time = time.time()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        if self.variable_names is None:
            self.variable_names = [f"x{i}" for i in range(self.n_features_in_)]

        rng = np.random.RandomState(self.random_state)
        func_set = self.functions if self.functions is not None else get_default_functions()

        tree_generator = TreeGenerator(
            n_features=self.n_features_in_,
            function_set=func_set,
            constant_range=self.constant_range,
            rng=rng
        )

        pop_mgr = PopulationManager(
            population_size=self.population_size,
            tree_generator=tree_generator,
            min_depth=self.min_depth,
            max_init_depth=self.max_init_depth,
            rng=rng
        )

        selector = Selector(rng=rng)
        operators = GeneticOperators(rng=rng)
        bloat_mgr = BloatController(
            parsimony_coefficient=self.parsimony_coefficient,
            max_depth=self.max_depth,
            rng=rng
        )

        # 1. Initialize with Ramped Half-and-Half
        pop_mgr.initialize_ramped_half_and_half()
        gen_champion = pop_mgr.evaluate_population(X, y, self.parsimony_coefficient)

        self.best_individual_ = gen_champion.clone()
        self.history_ = []

        # Record generation 0
        gen_stats = self._compute_generation_stats(0, pop_mgr.individuals, self.best_individual_)
        self.history_.append(gen_stats)
        if callback:
            callback(0, gen_stats)

        # 2. Main Evolutionary Loop
        for gen in range(1, self.generations + 1):
            next_generation: List[Individual] = []

            # Elitism preservation
            elites = selector.select_elites(pop_mgr.individuals, self.elite_count)
            next_generation.extend(elites)

            # Produce offspring until population is filled
            while len(next_generation) < self.population_size:
                op_rand = rng.rand()

                if op_rand < self.p_crossover:
                    # Subtree Crossover
                    p1 = selector.tournament_selection(pop_mgr.individuals, self.tournament_size)
                    p2 = selector.tournament_selection(pop_mgr.individuals, self.tournament_size)
                    c1, c2 = operators.subtree_crossover(p1, p2, max_depth=self.max_depth)
                    next_generation.append(c1)
                    if len(next_generation) < self.population_size:
                        next_generation.append(c2)

                elif op_rand < (self.p_crossover + self.p_subtree_mutation):
                    # Subtree Mutation
                    p = selector.tournament_selection(pop_mgr.individuals, self.tournament_size)
                    child = operators.subtree_mutation(
                        p,
                        tree_generator,
                        max_mutation_depth=3,
                        max_depth=self.max_depth
                    )
                    next_generation.append(child)

                elif op_rand < (self.p_crossover + self.p_subtree_mutation + self.p_point_mutation):
                    # Point Mutation
                    p = selector.tournament_selection(pop_mgr.individuals, self.tournament_size)
                    child = operators.point_mutation(p, tree_generator)
                    next_generation.append(child)

                elif op_rand < (self.p_crossover + self.p_subtree_mutation + self.p_point_mutation + self.p_hoist_mutation):
                    # Hoist Mutation (bloat pruning)
                    p = selector.tournament_selection(pop_mgr.individuals, self.tournament_size)
                    child = operators.hoist_mutation(p)
                    next_generation.append(child)

                else:
                    # Reproduction (Direct clone)
                    p = selector.tournament_selection(pop_mgr.individuals, self.tournament_size)
                    next_generation.append(p.clone())

            pop_mgr.individuals = next_generation[:self.population_size]

            # Evaluate offspring
            gen_champion = pop_mgr.evaluate_population(X, y, self.parsimony_coefficient)

            # Update all-time best individual
            if gen_champion.fitness < self.best_individual_.fitness:
                self.best_individual_ = gen_champion.clone()

            # Stats logging
            gen_stats = self._compute_generation_stats(gen, pop_mgr.individuals, self.best_individual_)
            self.history_.append(gen_stats)

            if callback:
                callback(gen, gen_stats)

            # Early stopping check
            if self.best_individual_.r2 >= self.early_stopping_r2:
                break

        self.training_time_ = time.time() - start_time
        return self

    def _compute_generation_stats(
        self,
        generation: int,
        population: List[Individual],
        all_time_best: Individual
    ) -> Dict[str, Any]:
        """Calculates statistical summary for the current generation."""
        fitnesses = [ind.fitness for ind in population if np.isfinite(ind.fitness)]
        raw_mses = [ind.raw_fitness for ind in population if np.isfinite(ind.raw_fitness)]
        sizes = [ind.size for ind in population]
        depths = [ind.depth for ind in population]

        return {
            "generation": generation,
            "best_fitness": float(all_time_best.fitness),
            "best_mse": float(all_time_best.raw_fitness),
            "best_rmse": float(all_time_best.rmse),
            "best_mae": float(all_time_best.mae),
            "best_r2": float(all_time_best.r2),
            "best_size": int(all_time_best.size),
            "best_depth": int(all_time_best.depth),
            "best_equation": all_time_best.to_equation_str(self.variable_names),
            "mean_fitness": float(np.mean(fitnesses)) if fitnesses else 1e9,
            "mean_mse": float(np.mean(raw_mses)) if raw_mses else 1e9,
            "mean_size": float(np.mean(sizes)),
            "mean_depth": float(np.mean(depths)),
            "size_std": float(np.std(sizes))
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions using the discovered symbolic mathematical formula."""
        if self.best_individual_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.best_individual_.evaluate(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns coefficient of determination R^2."""
        y_pred = self.predict(X)
        return float(r2_score(y, y_pred))

    def get_equation_summary(self) -> Dict[str, Any]:
        """Returns rich analytical simplification and LaTeX formatting."""
        if self.best_individual_ is None:
            raise RuntimeError("Model is not fitted yet.")
        summary = EquationSimplifier.simplify_individual(
            self.best_individual_,
            variable_names=self.variable_names
        )
        summary["training_time_sec"] = self.training_time_
        summary["generations_run"] = len(self.history_)
        return summary
