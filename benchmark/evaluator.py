"""
Comparative Evaluation Engine: Genetic Programming vs. Standard Regression Models.

Performs rigorous train/test split experiments, computes accuracy & complexity
metrics, and generates publication-grade comparison tables and plots.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
# Set non-interactive backend for headless execution
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from gp_engine.estimator import GeneticProgrammingRegressor
from .baselines import get_standard_baselines
from .metrics import evaluate_predictions, measure_inference_speed
from .datasets import DATASET_REGISTRY


class BenchmarkEvaluator:
    """
    Orchestrates comparative benchmarking between Genetic Programming
    and classical machine learning regression algorithms.
    """

    def __init__(
        self,
        random_state: int = 42,
        output_dir: str = "results"
    ):
        self.random_state = random_state
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_comparison(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "Benchmark",
        variable_names: Optional[List[str]] = None,
        gp_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.30
    ) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation comparing GP with standard regression models.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        results_list: List[Dict[str, Any]] = []
        predictions: Dict[str, np.ndarray] = {}

        # 1. Train Genetic Programming Symbolic Regressor
        default_gp_params = {
            "population_size": 250,
            "generations": 45,
            "tournament_size": 5,
            "p_crossover": 0.70,
            "p_subtree_mutation": 0.15,
            "p_point_mutation": 0.08,
            "p_hoist_mutation": 0.04,
            "max_depth": 6,
            "parsimony_coefficient": 0.001,
            "random_state": self.random_state,
            "variable_names": variable_names
        }
        if gp_params:
            default_gp_params.update(gp_params)

        gp_model = GeneticProgrammingRegressor(**default_gp_params)

        t0 = time.time()
        gp_model.fit(X_train, y_train)
        gp_train_time = time.time() - t0

        y_train_pred_gp = gp_model.predict(X_train)
        y_test_pred_gp = gp_model.predict(X_test)
        predictions["Genetic Programming"] = gp_model.predict(X)

        gp_eq_summary = gp_model.get_equation_summary()
        gp_train_metrics = evaluate_predictions(y_train, y_train_pred_gp)
        gp_test_metrics = evaluate_predictions(y_test, y_test_pred_gp)
        gp_inf_speed = measure_inference_speed(gp_model, X_test)

        results_list.append({
            "Model": "Genetic Programming (Ours)",
            "Model Type": "Symbolic Regression (AST)",
            "Train RMSE": gp_train_metrics["RMSE"],
            "Test RMSE": gp_test_metrics["RMSE"],
            "Test MAE": gp_test_metrics["MAE"],
            "Test R2": gp_test_metrics["R2"],
            "Complexity (Nodes/Params)": gp_eq_summary["simplified_node_count"],
            "Interpretability": "Analytical Equation",
            "Discovered Equation": gp_eq_summary["simplified_equation"],
            "LaTeX": gp_eq_summary["latex"],
            "Train Time (s)": round(gp_train_time, 3),
            "Inference (us/sample)": round(gp_inf_speed, 2)
        })

        # 2. Train Standard Regression Baselines
        baselines = get_standard_baselines(random_state=self.random_state)

        for name, config in baselines.items():
            model = config["model"]
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            y_tr_pred = model.predict(X_train)
            y_te_pred = model.predict(X_test)
            predictions[name] = model.predict(X)

            tr_metrics = evaluate_predictions(y_train, y_tr_pred)
            te_metrics = evaluate_predictions(y_test, y_te_pred)
            inf_speed = measure_inference_speed(model, X_test)
            complexity = config["complexity_calc"](model)

            results_list.append({
                "Model": name,
                "Model Type": config["type"],
                "Train RMSE": tr_metrics["RMSE"],
                "Test RMSE": te_metrics["RMSE"],
                "Test MAE": te_metrics["MAE"],
                "Test R2": te_metrics["R2"],
                "Complexity (Nodes/Params)": complexity,
                "Interpretability": config["interpretability"],
                "Discovered Equation": "N/A (Implicit/Black-Box)",
                "LaTeX": "N/A",
                "Train Time (s)": round(train_time, 4),
                "Inference (us/sample)": round(inf_speed, 2)
            })

        df_results = pd.DataFrame(results_list)

        # Save summary CSV
        csv_filename = os.path.join(self.output_dir, f"{dataset_name.replace(' ', '_').lower()}_comparison.csv")
        df_results.to_csv(csv_filename, index=False)

        # Generate Visualizations
        plot_paths = self._generate_visualizations(
            dataset_name=dataset_name,
            X=X,
            y=y,
            predictions=predictions,
            df_results=df_results,
            gp_model=gp_model
        )

        return {
            "dataset_name": dataset_name,
            "results_dataframe": df_results,
            "gp_model": gp_model,
            "gp_equation_summary": gp_eq_summary,
            "plot_paths": plot_paths,
            "csv_path": csv_filename
        }

    def _generate_visualizations(
        self,
        dataset_name: str,
        X: np.ndarray,
        y: np.ndarray,
        predictions: Dict[str, np.ndarray],
        df_results: pd.DataFrame,
        gp_model: GeneticProgrammingRegressor
    ) -> Dict[str, str]:
        """Generates comprehensive comparison plots and Pareto curves."""
        plot_paths = {}
        slug = dataset_name.replace(' ', '_').lower()

        # Set styling
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        # -------------------------------------------------------------
        # Plot 1: Model Prediction Curves vs. Ground Truth (for 1D features)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        if X.shape[1] == 1:
            sort_idx = np.argsort(X[:, 0])
            ax.scatter(X[sort_idx, 0], y[sort_idx], color='black', alpha=0.6, s=25, label='Actual Data Points', zorder=5)
            # Highlight GP
            ax.plot(X[sort_idx, 0], predictions["Genetic Programming"][sort_idx], color='#d9534f', linewidth=2.5, label='Genetic Programming (Discovered Eq)', zorder=6)
            # Highlight selected baselines
            key_baselines = ["Linear Regression", "Polynomial Reg (Deg 3)", "Random Forest"]
            colors = ['#0275d8', '#5cb85c', '#f0ad4e']
            for b_name, col in zip(key_baselines, colors):
                if b_name in predictions:
                    ax.plot(X[sort_idx, 0], predictions[b_name][sort_idx], linestyle='--', linewidth=1.8, color=col, label=b_name)
            ax.set_xlabel("Input Feature (x)", fontsize=12)
            ax.set_ylabel("Target (y)", fontsize=12)
        else:
            # Multi-dimensional: Parity plot (Predicted vs Actual for GP and RF)
            ax.scatter(y, predictions["Genetic Programming"], color='#d9534f', alpha=0.6, label='Genetic Programming', s=30)
            if "Random Forest" in predictions:
                ax.scatter(y, predictions["Random Forest"], color='#5cb85c', alpha=0.4, label='Random Forest', s=30)
            min_val = min(np.min(y), np.min(predictions["Genetic Programming"]))
            max_val = max(np.max(y), np.max(predictions["Genetic Programming"]))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Ideal 1:1 Parity')
            ax.set_xlabel("Actual Ground Truth", fontsize=12)
            ax.set_ylabel("Model Predictions", fontsize=12)

        ax.set_title(f"Model Fits: {dataset_name}", fontsize=14, fontweight='bold')
        ax.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        pred_path = os.path.join(self.output_dir, f"{slug}_fits.png")
        plt.savefig(pred_path)
        plt.close()
        plot_paths["fits"] = pred_path

        # -------------------------------------------------------------
        # Plot 2: Error & R2 Comparison Bar Charts
        # -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        models = df_results["Model"]
        y_pos = np.arange(len(models))

        # Test RMSE
        ax1.barh(y_pos, df_results["Test RMSE"], color=['#d9534f' if "Genetic" in m else '#428bca' for m in models])
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(models, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel("Test RMSE (Lower is Better)", fontsize=11)
        ax1.set_title("Test Root Mean Squared Error", fontsize=12, fontweight='bold')

        # Test R^2
        ax2.barh(y_pos, df_results["Test R2"], color=['#d9534f' if "Genetic" in m else '#5cb85c' for m in models])
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([])  # Hide duplicate labels
        ax2.invert_yaxis()
        ax2.set_xlabel("Test R^2 Score (Higher is Better)", fontsize=11)
        ax2.set_title("Coefficient of Determination (R^2)", fontsize=12, fontweight='bold')
        ax2.set_xlim([min(0.0, df_results["Test R2"].min()), 1.05])

        plt.suptitle(f"Error & Performance Benchmark: {dataset_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        metrics_path = os.path.join(self.output_dir, f"{slug}_metrics_bar.png")
        plt.savefig(metrics_path)
        plt.close()
        plot_paths["metrics_bar"] = metrics_path

        # -------------------------------------------------------------
        # Plot 3: Pareto Frontier (Error vs. Model Complexity)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

        for _, row in df_results.iterrows():
            m_name = row["Model"]
            comp = row["Complexity (Nodes/Params)"]
            rmse = row["Test RMSE"]
            is_gp = "Genetic" in m_name

            color = '#d9534f' if is_gp else '#0275d8'
            marker = '*' if is_gp else 'o'
            size = 200 if is_gp else 100

            ax.scatter(comp, rmse, color=color, marker=marker, s=size, zorder=5)
            offset = (5, 5) if not is_gp else (8, -10)
            ax.annotate(
                m_name,
                (comp, rmse),
                textcoords="offset points",
                xytext=offset,
                fontsize=9,
                fontweight='bold' if is_gp else 'normal'
            )

        ax.set_xscale('log')
        ax.set_xlabel("Model Complexity (Parameters / AST Nodes, Log Scale)", fontsize=11)
        ax.set_ylabel("Test RMSE (Error, Lower is Better)", fontsize=11)
        ax.set_title(f"Pareto Trade-off: Error vs. Complexity ({dataset_name})", fontsize=13, fontweight='bold')
        plt.tight_layout()
        pareto_path = os.path.join(self.output_dir, f"{slug}_pareto_frontier.png")
        plt.savefig(pareto_path)
        plt.close()
        plot_paths["pareto"] = pareto_path

        # -------------------------------------------------------------
        # Plot 4: GP Evolutionary Dynamics (Fitness & Bloat Tracking)
        # -------------------------------------------------------------
        if gp_model.history_:
            history = gp_model.history_
            gens = [h["generation"] for h in history]
            best_rmse = [h["best_rmse"] for h in history]
            mean_rmse = [np.sqrt(h["mean_mse"]) if h["mean_mse"] < 1e6 else np.nan for h in history]
            best_size = [h["best_size"] for h in history]
            mean_size = [h["mean_size"] for h in history]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, dpi=150)

            # Subplot 1: Fitness Convergence
            ax1.plot(gens, best_rmse, 'r-', linewidth=2.2, label='Best Individual RMSE')
            ax1.plot(gens, mean_rmse, 'b--', alpha=0.7, label='Population Mean RMSE')
            ax1.set_ylabel("RMSE Error", fontsize=11)
            ax1.set_title(f"Evolutionary Convergence & Bloat Dynamics ({dataset_name})", fontsize=13, fontweight='bold')
            ax1.legend(loc='upper right', frameon=True)

            # Subplot 2: Code Bloat Tracking (Size)
            ax2.plot(gens, best_size, 'g-', linewidth=2.0, label='Champion Tree Node Count')
            ax2.plot(gens, mean_size, 'm--', alpha=0.7, label='Population Average Tree Size')
            ax2.set_xlabel("Evolutionary Generation", fontsize=11)
            ax2.set_ylabel("Tree Size (Node Count)", fontsize=11)
            ax2.legend(loc='upper left', frameon=True)

            plt.tight_layout()
            dynamics_path = os.path.join(self.output_dir, f"{slug}_gp_dynamics.png")
            plt.savefig(dynamics_path)
            plt.close()
            plot_paths["gp_dynamics"] = dynamics_path

        return plot_paths
