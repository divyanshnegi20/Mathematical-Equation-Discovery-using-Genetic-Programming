"""
Unit tests for Benchmarking datasets, baseline regression models, and evaluator.
"""

import unittest
import numpy as np

from benchmark.datasets import (
    generate_koza_quartic,
    generate_damped_oscillator,
    generate_multivariable_physics,
    load_real_world_dataset
)
from benchmark.baselines import get_standard_baselines
from benchmark.metrics import evaluate_predictions
from benchmark.evaluator import BenchmarkEvaluator


class TestBenchmark(unittest.TestCase):

    def test_dataset_generation(self):
        X, y, eq, names = generate_koza_quartic(n_samples=50)
        self.assertEqual(X.shape, (50, 1))
        self.assertEqual(len(y), 50)
        self.assertEqual(names, ["x"])

        X_d, y_d, _, _ = generate_damped_oscillator(n_samples=60)
        self.assertEqual(X_d.shape, (60, 1))
        self.assertEqual(len(y_d), 60)

        X_m, y_m, _, names_m = generate_multivariable_physics(n_samples=70)
        self.assertEqual(X_m.shape, (70, 2))
        self.assertEqual(len(y_m), 70)
        self.assertEqual(len(names_m), 2)

    def test_baselines_loading(self):
        baselines = get_standard_baselines(random_state=42)
        self.assertIn("Linear Regression", baselines)
        self.assertIn("Polynomial Reg (Deg 2)", baselines)
        self.assertIn("Decision Tree (CART)", baselines)
        self.assertIn("Random Forest", baselines)
        self.assertIn("MLP Neural Network", baselines)

    def test_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        metrics = evaluate_predictions(y_true, y_pred)
        self.assertIn("RMSE", metrics)
        self.assertIn("R2", metrics)
        self.assertGreater(metrics["R2"], 0.90)

    def test_evaluator_small_run(self):
        evaluator = BenchmarkEvaluator(random_state=42, output_dir="results")
        X, y, _, names = generate_koza_quartic(n_samples=40)
        res = evaluator.run_comparison(
            X, y,
            dataset_name="Test Koza Small",
            variable_names=names,
            gp_params={
                "population_size": 30,
                "generations": 5,
                "max_depth": 4
            }
        )
        self.assertIn("results_dataframe", res)
        self.assertIn("plot_paths", res)
        df = res["results_dataframe"]
        self.assertGreater(len(df), 5)


if __name__ == '__main__':
    unittest.main()
