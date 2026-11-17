"""
Benchmarking and Comparative Evaluation Suite.
"""

from .datasets import (
    DATASET_REGISTRY,
    generate_koza_quartic,
    generate_damped_oscillator,
    generate_multivariable_physics,
    load_real_world_dataset,
    export_benchmark_csvs
)
from .baselines import get_standard_baselines
from .metrics import evaluate_predictions, measure_inference_speed
from .evaluator import BenchmarkEvaluator

__all__ = [
    "DATASET_REGISTRY",
    "generate_koza_quartic",
    "generate_damped_oscillator",
    "generate_multivariable_physics",
    "load_real_world_dataset",
    "export_benchmark_csvs",
    "get_standard_baselines",
    "evaluate_predictions",
    "measure_inference_speed",
    "BenchmarkEvaluator"
]
