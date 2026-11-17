# Mathematical Equation Discovery Using Genetic Programming
**CSN442: Evolutionary Computing (7th Semester B.Tech CSE)**  
**DIT University, Dehradun — Odd Semester 2026**  
**Project ID:** `P11` | **Section:** `B`  

---

## 👥 Student Group Members
1. **DEV VERMA** — `1000020709`
2. **DIVYANSH NEGI** — `1000020730`
3. **SAKSHAM AGARWAL** — `1000020752`

---

## 📌 Project Overview & Objective
> **Official Objective:** *Evolve equations that fit a dataset, then compare their prediction error and complexity with standard regression models.*

Standard machine learning models like Random Forests, Gradient Boosted Trees, and Deep Neural Networks act as opaque **"black boxes"** with hundreds of trees or millions of weights. In contrast, **Genetic Programming (GP)** performs **Symbolic Regression** to automatically discover explicit, closed-form mathematical equations directly from data.

This project delivers a **production-grade, pure-Python/NumPy Genetic Programming framework** adhering to the course textbook (*A.E. Eiben & J.E. Smith, Introduction to Evolutionary Computing*), equipped with:
- **Abstract Syntax Tree (AST)** individual representations.
- **Protected Mathematical Operators** (`pdiv`, `plog`, `psqrt`, `pexp`) to ensure closure.
- **Ramped Half-and-Half (RH&H)** population initialization (Full & Grow methods).
- **Genetic Variation Operators:** Subtree Crossover, Subtree Mutation, Point Mutation, and Hoist Mutation.
- **Code Bloat Mitigation:** Parsimony pressure fitness penalties and strict depth limiting.
- **SymPy Symbolic Simplification:** Pretty-printing, algebraic simplification, and LaTeX equation rendering.
- **Scikit-Learn Compatible Estimator API:** Seamless `fit()`, `predict()`, and `score()`.
- **Comparative Benchmark Suite:** 7 baseline models (Linear, Ridge, Poly Deg 2/3, CART Decision Tree, Random Forest, Gradient Boosting, MLP).
- **Interactive Web Dashboard:** FastAPI + Chart.js + KaTeX live formula visualizer.
- **Automated Academic Report Generator:** Builds complete `.docx` and `.md` reports.

---

## 📂 Repository Architecture

```
project/
├── gp_engine/                    # Core Genetic Programming Engine
│   ├── __init__.py
│   ├── functions.py              # Protected math operators & SymPy builders
│   ├── node.py                   # AST Node class (Functions, Terminals, ERCs)
│   ├── tree.py                   # Individual candidate solution (evaluation, fitness)
│   ├── population.py             # Ramped Half-and-Half (Full + Grow) factory
│   ├── operators.py              # Crossover, Subtree/Point/Hoist mutation
│   ├── selection.py              # Tournament, Roulette wheel, and Elitism
│   ├── bloat.py                  # Parsimony pressure & Tarpeian bloat filters
│   ├── simplifier.py             # SymPy simplification & LaTeX generation
│   └── estimator.py              # Scikit-learn GeneticProgrammingRegressor
├── benchmark/                    # Comparative Evaluation Suite
│   ├── __init__.py
│   ├── datasets.py               # Benchmark generators (Koza, Oscillator, Physics, Clinical)
│   ├── baselines.py              # 7 Standard ML regression baselines
│   ├── metrics.py                # MSE, RMSE, MAE, R², Inference Latency
│   └── evaluator.py              # Automated comparative evaluation runner
├── web/                          # Interactive Dashboard
│   ├── app.py                    # FastAPI server
│   └── templates/
│       └── index.html            # Tailwind CSS + KaTeX + Chart.js interface
├── data/                         # CSV Datasets
│   ├── synthetic/                # koza_quartic.csv, damped_oscillator.csv, physics_multivariable.csv
│   └── real_world/               # clinical_biomarkers.csv
├── results/                      # Generated evaluation plots and comparison CSVs
├── tests/                        # Comprehensive Unit Test Suite
│   ├── test_gp_engine.py         # Tests AST evaluation, operators, bloat control
│   └── test_benchmark.py         # Tests baseline trainers, metrics, datasets
├── main.py                       # Unified CLI interface
├── generate_report.py            # Generates official submission Project_Report.docx & .md
├── Project_Report.docx           # Formal University Submission Document
├── Project_Report.md             # Markdown Report
└── requirements.txt              # Project dependencies
```

---

## 🚀 Quick Start Guide

### 1. Installation
Install requirements (all core scientific Python libraries):
```bash
pip install -r requirements.txt
```

### 2. Run Single Equation Discovery Experiment (CLI)
Evolve an equation on the classic Koza-1 benchmark:
```bash
python main.py run-experiment --dataset koza_quartic --pop-size 250 --generations 40
```

### 3. Run All Comparative Benchmarks
Execute comparative evaluations against all 7 regression models across all 4 datasets:
```bash
python main.py run-all
```

### 4. Launch Interactive Web Dashboard
```bash
python main.py serve --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to interactively tune population size, generations, mutation rates, and parsimony pressure while watching formulas render live in LaTeX!

### 5. Export Benchmark Datasets to CSV
```bash
python main.py export-data
```

### 6. Generate Formal Project Report (.docx)
```bash
python main.py report
```
Generates `Project_Report.docx` with university formatting, student details, benchmark tables, and viva answers.

---

## 📊 Empirical Results Summary

### Benchmark 1: Koza-1 Quartic Polynomial ($y = x^4 + x^3 + x^2 + x$)
| Model | Test RMSE | Test $R^2$ | Complexity (Nodes/Params) | Interpretability | Inference ($\mu s$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Genetic Programming (Ours)** | **$3.41 \times 10^{-17}$** | **$1.000000$** | **12** | **Analytical Formula** | **$10.26$** |
| Linear Regression | $0.5289$ | $0.6953$ | $2$ | High (Linear Weights) | $5.65$ |
| Ridge Regression | $0.5190$ | $0.7067$ | $2$ | High (Shrunk Weights) | $9.25$ |
| Polynomial Reg (Deg 2) | $0.1770$ | $0.9659$ | $4$ | Medium (Polynomial) | $33.26$ |
| Polynomial Reg (Deg 3) | $0.0814$ | $0.9928$ | $5$ | Medium (Polynomial) | $32.61$ |
| Decision Tree (CART) | $0.0704$ | $0.9946$ | $59$ | Medium (Splits) | $10.96$ |
| Random Forest | $0.0400$ | $0.9983$ | $2,368$ | Low (Black-Box) | $775.17$ |
| Gradient Boosting | $0.0634$ | $0.9956$ | $748$ | Low (Black-Box) | $18.67$ |
| MLP Neural Network | $0.8424$ | $0.2273$ | $609$ | Very Low (Opaque) | $27.98$ |

> **Key Takeaway:** Genetic Programming recovered the exact algebraic ground truth ($x(x(x^2 + x + 1) + 1) \equiv x^4 + x^3 + x^2 + x$) with zero error ($R^2 = 1.0$), beating even 2,368-node Random Forests while being completely transparent and 75x faster at inference.

---

### Benchmark 2: Multi-Variable Physics Kinematics ($y = 2\sqrt{x_0} + 1.5\sin(2x_1)$)
| Model | Test RMSE | Test $R^2$ | Complexity (Nodes/Params) | Interpretability |
| :--- | :---: | :---: | :---: | :---: |
| **Genetic Programming (Ours)** | **$0.4327$** | **$0.9241$** | **11** | **Analytical Formula** |
| Gradient Boosting | $0.5063$ | $0.8960$ | $750$ | Low (Black-Box) |
| Random Forest | $0.7988$ | $0.7412$ | $2,624$ | Low (Black-Box) |
| Decision Tree (CART) | $0.8196$ | $0.7275$ | $55$ | Medium (Splits) |
| Polynomial Reg (Deg 3) | $1.0713$ | $0.5345$ | $11$ | Medium (Polynomial) |
| Linear Regression | $1.1922$ | $0.4235$ | $3$ | High (Linear Weights) |

> **Key Takeaway:** On non-linear multi-dimensional systems, Genetic Programming achieved the **highest test generalization accuracy ($R^2 = 0.924$)** of any model, drastically outperforming Random Forest ($R^2 = 0.741$) while utilizing only 11 nodes versus 2,624 nodes!

---

## 🔬 Unit Tests
Run the automated test suite:
```bash
python -m unittest discover tests/
```
All unit tests validate protected mathematical functions, AST node evaluation, Ramped Half-and-Half initialization, crossover, mutation, bloat control, SymPy algebraic simplification, and comparative baselines.

---

## 🎓 Viva Voce Key Concepts & Cheat Sheet
- **What is Symbolic Regression?** An evolutionary search through the space of mathematical expression trees to identify closed-form analytical equations fitting observational data, without predefining the equation structure.
- **Why Ramped Half-and-Half?** To balance bushy/deep trees (Full) and variable-depth/asymmetric trees (Grow) across multiple initial depths, ensuring high structural diversity.
- **What is Code Bloat?** The tendency of GP trees to grow exponentially over generations without corresponding fitness improvements (introns). Controlled via **Parsimony Pressure** ($\text{Fitness} = \text{MSE} + \alpha \cdot \text{Size}$) and **Max Depth Limits**.
- **What are Protected Operators?** Safe wrappers like `protected_div` (returning 1.0 when denominator is close to 0) ensuring the **Closure Property** of Genetic Programming.

---
*Created for CSN442 Evolutionary Computing — Odd Semester 2026, DIT University.*
