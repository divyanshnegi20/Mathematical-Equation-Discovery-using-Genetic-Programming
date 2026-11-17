"""
Unified Command-Line Interface (CLI) for Mathematical Equation Discovery Using Genetic Programming.

Commands:
- run-experiment : Runs equation discovery on a selected benchmark dataset
- run-all        : Runs comparative study across all benchmark datasets
- export-data    : Saves benchmark datasets to CSV files in data/
- serve          : Launches the interactive FastAPI web dashboard
- report         : Generates the formal Academic Mini-Project Report (.docx & .md)
"""

import sys
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))
import warnings
warnings.filterwarnings("ignore")
import argparse
import uvicorn

from benchmark.datasets import (
    DATASET_REGISTRY,
    generate_koza_quartic,
    generate_damped_oscillator,
    generate_multivariable_physics,
    load_real_world_dataset,
    export_benchmark_csvs
)
from benchmark.evaluator import BenchmarkEvaluator


def print_banner():
    banner = r"""
========================================================================================
   MATHEMATICAL EQUATION DISCOVERY USING GENETIC PROGRAMMING (SYMBOLIC REGRESSION)
   Course: CSN442 Evolutionary Computing | B.Tech CSE 7th Sem | Project ID: P11
========================================================================================
    """
    print(banner)


def cmd_export_data(args):
    print("[+] Exporting synthetic and real-world benchmark datasets to CSV...")
    export_benchmark_csvs("data")
    print("    -> Saved datasets in 'data/synthetic/' and 'data/real_world/'")


def cmd_run_experiment(args):
    print_banner()
    dataset_key = args.dataset
    if dataset_key not in DATASET_REGISTRY:
        print(f"[!] Error: Unknown dataset '{dataset_key}'. Choose from: {list(DATASET_REGISTRY.keys())}")
        sys.exit(1)

    info = DATASET_REGISTRY[dataset_key]
    print(f"[*] Benchmark: {info['name']}")
    print(f"[*] Description: {info['description']}")

    # Load data
    loader = info["loader"]
    X, y, true_eq, var_names = loader()

    print(f"[*] Dataset shape: X={X.shape}, y={y.shape}, Features={var_names}")
    print(f"[*] True Formula: {true_eq}\n")

    evaluator = BenchmarkEvaluator(random_state=args.seed, output_dir=args.output_dir)

    gp_params = {
        "population_size": args.pop_size,
        "generations": args.generations,
        "tournament_size": args.tournament_size,
        "p_crossover": args.p_crossover,
        "p_subtree_mutation": args.p_mutation * 0.75,
        "p_point_mutation": args.p_mutation * 0.25,
        "max_depth": args.max_depth,
        "parsimony_coefficient": args.parsimony,
        "variable_names": var_names,
        "random_state": args.seed
    }

    print("[*] Evolving equation & training baseline models...")
    res = evaluator.run_comparison(
        X, y,
        dataset_name=info["name"],
        variable_names=var_names,
        gp_params=gp_params
    )

    df = res["results_dataframe"]
    summary = res["gp_equation_summary"]

    print("\n" + "="*80)
    print("  EVOLUTION RESULTS & DISCOVERED SYMBOLIC FORMULA")
    print("="*80)
    print(f"  • Discovered Equation : {summary['simplified_equation']}")
    print(f"  • LaTeX Formula       : {summary['latex']}")
    print(f"  • Original AST Size   : {summary['original_tree_size']} nodes (Depth: {summary['original_depth']})")
    print(f"  • Simplified Nodes    : {summary['simplified_node_count']} atoms")
    print(f"  • Training Time       : {summary['training_time_sec']:.2f} seconds")
    print("="*80 + "\n")

    print("[*] Comparative Performance Table:")
    cols_to_print = ["Model", "Test RMSE", "Test R2", "Complexity (Nodes/Params)", "Interpretability", "Inference (us/sample)"]
    print(df[cols_to_print].to_string(index=False))

    print(f"\n[+] Results CSV saved to: {res['csv_path']}")
    print("[+] Generated Visualizations:")
    for name, p in res["plot_paths"].items():
        print(f"    - {name.upper()}: {p}")


def cmd_run_all(args):
    print_banner()
    print("[*] Executing full comparative study across all 4 benchmark datasets...\n")
    for key in DATASET_REGISTRY.keys():
        print(f"\n>>> Running Benchmark: {key} <<<")
        args.dataset = key
        cmd_run_experiment(args)


def cmd_serve(args):
    print_banner()
    print(f"[*] Starting Interactive Web Dashboard at http://{args.host}:{args.port}")
    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_report(args):
    print_banner()
    print("[*] Generating academic project documentation and Word report...")
    from generate_report import build_academic_report
    build_academic_report()
    print("[+] Reports successfully generated!")


def main():
    parser = argparse.ArgumentParser(
        description="Mathematical Equation Discovery Using Genetic Programming (CSN442)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run-experiment
    p_run = subparsers.add_parser("run-experiment", help="Run equation discovery on a single dataset")
    p_run.add_argument("--dataset", type=str, default="koza_quartic", choices=list(DATASET_REGISTRY.keys()), help="Dataset name")
    p_run.add_argument("--pop-size", type=int, default=250, help="GP population size")
    p_run.add_argument("--generations", type=int, default=40, help="Number of evolutionary generations")
    p_run.add_argument("--tournament-size", type=int, default=5, help="Tournament selection size")
    p_run.add_argument("--p-crossover", type=float, default=0.70, help="Crossover probability")
    p_run.add_argument("--p-mutation", type=float, default=0.20, help="Mutation probability")
    p_run.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth (bloat control)")
    p_run.add_argument("--parsimony", type=float, default=0.001, help="Parsimony pressure coefficient")
    p_run.add_argument("--seed", type=int, default=42, help="Random state seed")
    p_run.add_argument("--output-dir", type=str, default="results", help="Directory for plots and CSVs")
    p_run.set_defaults(func=cmd_run_experiment)

    # Command: run-all
    p_all = subparsers.add_parser("run-all", help="Run experiments on all benchmarks")
    p_all.add_argument("--pop-size", type=int, default=200)
    p_all.add_argument("--generations", type=int, default=35)
    p_all.add_argument("--tournament-size", type=int, default=5)
    p_all.add_argument("--p-crossover", type=float, default=0.70)
    p_all.add_argument("--p-mutation", type=float, default=0.20)
    p_all.add_argument("--max-depth", type=int, default=6)
    p_all.add_argument("--parsimony", type=float, default=0.001)
    p_all.add_argument("--seed", type=int, default=42)
    p_all.add_argument("--output-dir", type=str, default="results")
    p_all.set_defaults(func=cmd_run_all)

    # Command: export-data
    p_exp = subparsers.add_parser("export-data", help="Export benchmark datasets as CSV files")
    p_exp.set_defaults(func=cmd_export_data)

    # Command: serve
    p_srv = subparsers.add_parser("serve", help="Launch interactive web GUI")
    p_srv.add_argument("--host", type=str, default="127.0.0.1", help="Host binding")
    p_srv.add_argument("--port", type=int, default=8000, help="Port binding")
    p_srv.add_argument("--reload", action="store_true", help="Auto reload")
    p_srv.set_defaults(func=cmd_serve)

    # Command: report
    p_rep = subparsers.add_parser("report", help="Generate project report Word (.docx) & Markdown (.md) documents")
    p_rep.set_defaults(func=cmd_report)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
