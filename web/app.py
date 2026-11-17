"""
FastAPI Web Application and API Backend for Genetic Programming Equation Discovery.

Serves an interactive dashboard for running symbolic regression, viewing mathematical
equations rendered in LaTeX, and comparing performance with standard regression models.
"""

from typing import Optional, List, Dict, Any
import os
import io
import base64
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from benchmark.datasets import (
    DATASET_REGISTRY,
    generate_koza_quartic,
    generate_damped_oscillator,
    generate_multivariable_physics,
    load_real_world_dataset
)
from benchmark.evaluator import BenchmarkEvaluator
from gp_engine.functions import FUNCTION_REGISTRY, get_default_functions

app = FastAPI(title="Mathematical Equation Discovery Using Genetic Programming")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Mount results as static files for viewing generated plots
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


def load_dataset_by_key(key: str) -> tuple:
    if key == "koza_quartic":
        return generate_koza_quartic(n_samples=100)
    elif key == "damped_oscillator":
        return generate_damped_oscillator(n_samples=120)
    elif key == "multivariable_physics":
        return generate_multivariable_physics(n_samples=150)
    elif key in ("real_world_biomarkers", "real_world_housing"):
        return load_real_world_dataset(n_samples=250)
    else:
        return generate_koza_quartic(n_samples=100)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Index template not found</h1>", status_code=404)


@app.get("/api/datasets")
async def list_datasets():
    datasets = []
    for k, info in DATASET_REGISTRY.items():
        datasets.append({
            "id": k,
            "name": info["name"],
            "category": info["category"],
            "description": info["description"]
        })
    return {"datasets": datasets}


@app.post("/api/run")
async def run_gp_experiment(
    dataset: str = Form("koza_quartic"),
    population_size: int = Form(200),
    generations: int = Form(35),
    tournament_size: int = Form(5),
    p_crossover: float = Form(0.70),
    p_mutation: float = Form(0.20),
    max_depth: int = Form(6),
    parsimony_coefficient: float = Form(0.001)
):
    try:
        X, y, true_eq, var_names = load_dataset_by_key(dataset)

        evaluator = BenchmarkEvaluator(random_state=42, output_dir=RESULTS_DIR)

        gp_params = {
            "population_size": population_size,
            "generations": generations,
            "tournament_size": tournament_size,
            "p_crossover": p_crossover,
            "p_subtree_mutation": p_mutation * 0.75,
            "p_point_mutation": p_mutation * 0.25,
            "p_hoist_mutation": 0.04,
            "max_depth": max_depth,
            "parsimony_coefficient": parsimony_coefficient,
            "variable_names": var_names
        }

        eval_res = evaluator.run_comparison(
            X, y,
            dataset_name=DATASET_REGISTRY.get(dataset, {}).get("name", dataset),
            variable_names=var_names,
            gp_params=gp_params
        )

        df = eval_res["results_dataframe"]
        records = df.to_dict(orient="records")

        # History tracking
        gp_model = eval_res["gp_model"]
        history = [
            {
                "gen": h["generation"],
                "best_rmse": round(h["best_rmse"], 5),
                "best_size": h["best_size"],
                "mean_size": round(h["mean_size"], 2)
            }
            for h in gp_model.history_
        ]

        # Extract plot URLs
        plot_urls = {
            k: f"/results/{os.path.basename(p)}"
            for k, p in eval_res["plot_paths"].items()
        }

        return JSONResponse({
            "status": "success",
            "dataset_name": eval_res["dataset_name"],
            "true_equation": true_eq,
            "discovered_equation": eval_res["gp_equation_summary"]["simplified_equation"],
            "raw_equation": eval_res["gp_equation_summary"]["raw_equation"],
            "latex": eval_res["gp_equation_summary"]["latex"],
            "tree_size": eval_res["gp_equation_summary"]["original_tree_size"],
            "simplified_nodes": eval_res["gp_equation_summary"]["simplified_node_count"],
            "training_time": eval_res["gp_equation_summary"]["training_time_sec"],
            "history": history,
            "comparison_table": records,
            "plots": plot_urls
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/upload")
async def upload_custom_dataset(
    file: UploadFile = File(...),
    target_col: str = Form(""),
    population_size: int = Form(200),
    generations: int = Form(30)
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        if not target_col or target_col not in df.columns:
            # Default to last column
            target_col = df.columns[-1]

        y = df[target_col].values.astype(np.float64)
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].values.astype(np.float64)

        evaluator = BenchmarkEvaluator(random_state=42, output_dir=RESULTS_DIR)
        eval_res = evaluator.run_comparison(
            X, y,
            dataset_name=f"Custom: {file.filename}",
            variable_names=feature_cols,
            gp_params={
                "population_size": population_size,
                "generations": generations,
                "variable_names": feature_cols
            }
        )

        records = eval_res["results_dataframe"].to_dict(orient="records")
        plot_urls = {
            k: f"/results/{os.path.basename(p)}"
            for k, p in eval_res["plot_paths"].items()
        }

        return JSONResponse({
            "status": "success",
            "dataset_name": f"Custom CSV ({file.filename})",
            "discovered_equation": eval_res["gp_equation_summary"]["simplified_equation"],
            "latex": eval_res["gp_equation_summary"]["latex"],
            "comparison_table": records,
            "plots": plot_urls
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
