"""
Unit tests for Genetic Programming Engine components.
"""

import unittest
import numpy as np
import sympy as sp

from gp_engine.functions import protected_div, protected_log, protected_sqrt, get_default_functions
from gp_engine.node import Node
from gp_engine.tree import Individual
from gp_engine.population import TreeGenerator, PopulationManager
from gp_engine.operators import GeneticOperators
from gp_engine.selection import Selector
from gp_engine.bloat import BloatController
from gp_engine.simplifier import EquationSimplifier
from gp_engine.estimator import GeneticProgrammingRegressor


class TestGPEngine(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.generator = TreeGenerator(
            n_features=2,
            constant_range=(-2.0, 2.0),
            rng=self.rng
        )

    def test_protected_functions(self):
        # Protected division
        self.assertEqual(protected_div(np.array([5.0]), np.array([0.0]))[0], 1.0)
        self.assertEqual(protected_div(np.array([10.0]), np.array([2.0]))[0], 5.0)

        # Protected log
        self.assertEqual(protected_log(np.array([0.0]))[0], 0.0)
        self.assertAlmostEqual(protected_log(np.array([np.e]))[0], 1.0)

        # Protected sqrt
        self.assertEqual(protected_sqrt(np.array([-9.0]))[0], 3.0)

    def test_node_evaluation(self):
        # Build tree: (x0 + 3.0) * x1
        x0_node = Node('variable', 0)
        c_node = Node('constant', 3.0)
        add_node = Node('function', 'add', [x0_node, c_node])
        x1_node = Node('variable', 1)
        root = Node('function', 'mul', [add_node, x1_node])

        X = np.array([
            [1.0, 2.0],  # (1 + 3) * 2 = 8
            [2.0, 4.0],  # (2 + 3) * 4 = 20
        ])
        res = root.evaluate(X)
        np.testing.assert_allclose(res, [8.0, 20.0])

        self.assertEqual(root.size(), 5)
        self.assertEqual(root.depth(), 3)

    def test_ramped_half_and_half_initialization(self):
        pop_mgr = PopulationManager(
            population_size=50,
            tree_generator=self.generator,
            min_depth=2,
            max_init_depth=4,
            rng=self.rng
        )
        pop = pop_mgr.initialize_ramped_half_and_half()
        self.assertEqual(len(pop), 50)
        for ind in pop:
            self.assertGreaterEqual(ind.depth, 2)
            self.assertLessEqual(ind.depth, 4)

    def test_genetic_operators(self):
        operators = GeneticOperators(rng=self.rng)
        ind1 = self.generator.create_individual('full', 3)
        ind2 = self.generator.create_individual('full', 3)

        c1, c2 = operators.subtree_crossover(ind1, ind2, max_depth=6)
        self.assertIsInstance(c1, Individual)
        self.assertIsInstance(c2, Individual)
        self.assertLessEqual(c1.depth, 6)

        mutated = operators.subtree_mutation(ind1, self.generator, max_depth=6)
        self.assertIsInstance(mutated, Individual)

        hoisted = operators.hoist_mutation(ind1)
        self.assertIsInstance(hoisted, Individual)
        self.assertLessEqual(hoisted.size, ind1.size)

    def test_bloat_control(self):
        bloat = BloatController(parsimony_coefficient=0.01)
        penalized = bloat.apply_parsimony_pressure(raw_fitness=0.5, tree_size=10)
        self.assertAlmostEqual(penalized, 0.5 + 0.01 * 10)

    def test_equation_simplifier(self):
        # x0 + x0 -> 2*x0
        x0_a = Node('variable', 0)
        x0_b = Node('variable', 0)
        root = Node('function', 'add', [x0_a, x0_b])
        ind = Individual(root)
        ind.raw_fitness = 0.01

        summary = EquationSimplifier.simplify_individual(ind, variable_names=['x'])
        self.assertIn("2", summary["simplified_equation"])
        self.assertIn("x", summary["simplified_equation"])

    def test_regressor_fit_predict(self):
        # Simple test: y = 2*x + 1
        X = np.linspace(-1, 1, 30).reshape(-1, 1)
        y = 2.0 * X[:, 0] + 1.0

        model = GeneticProgrammingRegressor(
            population_size=60,
            generations=15,
            max_depth=4,
            random_state=42
        )
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), 30)
        self.assertGreater(model.score(X, y), 0.70)


if __name__ == '__main__':
    unittest.main()
