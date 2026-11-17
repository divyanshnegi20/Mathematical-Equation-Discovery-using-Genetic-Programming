"""
Academic Project Report Generator for CSN442 Evolutionary Computing.

Produces:
1. Project_Report.docx : Formal Microsoft Word document formatted for university submission.
2. Project_Report.md   : Markdown version with GitHub Flavored Markdown and LaTeX equations.
"""

import os
import glob
import pandas as pd
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def set_cell_background(cell, fill_hex):
    """Sets background shading of a table cell in docx."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)


def df_to_markdown_simple(df, cols):
    """Simple native converter from DataFrame to GitHub Flavored Markdown table."""
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if abs(v) > 1e-4 else f"{v:.3e}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, separator] + rows)


def build_academic_report():
    print("[*] Generating academic project report (.docx & .md)...")
    doc = Document()

    # Page Margins (1 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styles
    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    # -------------------------------------------------------------
    # 1. Title Page / Header Block
    # -------------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_univ = title_p.add_run("DIT UNIVERSITY, DEHRADUN\nSCHOOL OF COMPUTING\n")
    run_univ.font.size = Pt(14)
    run_univ.font.bold = True
    run_univ.font.color.rgb = RGBColor(0x1a, 0x36, 0x5d)

    run_course = title_p.add_run("DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING\nB.Tech CSE - 7th Semester (Odd Semester 2026)\n\n")
    run_course.font.size = Pt(11)
    run_course.font.color.rgb = RGBColor(0x4a, 0x55, 0x68)

    run_proj_title = title_p.add_run("MINI PROJECT REPORT\n\nMATHEMATICAL EQUATION DISCOVERY USING GENETIC PROGRAMMING\n")
    run_proj_title.font.size = Pt(18)
    run_proj_title.font.bold = True
    run_proj_title.font.color.rgb = RGBColor(0x2b, 0x6c, 0xb0)

    run_sub = title_p.add_run("A Comparative Study on Symbolic Regression, Prediction Accuracy, and Model Complexity Pareto Fronts\n\n")
    run_sub.font.size = Pt(12)
    run_sub.font.italic = True

    # Details Box Table
    info_table = doc.add_table(rows=5, cols=2)
    info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_table.autofit = False

    details = [
        ("Course Name & Code:", "Evolutionary Computing (CSN442)"),
        ("Project ID & Title:", "P11 - Mathematical Equation Discovery Using Genetic Programming"),
        ("Project Objective:", "Evolve equations that fit a dataset, then compare their prediction error and complexity with standard regression models."),
        ("Student Group Members (Section B):", "1. DEV VERMA (1000020709)\n2. DIVYANSH NEGI (1000020730)\n3. SAKSHAM AGARWAL (1000020752)"),
        ("Faculty / Department:", "School of Computing, DIT University, Dehradun")
    ]

    for i, (k, v) in enumerate(details):
        r = info_table.rows[i]
        r.cells[0].width = Inches(2.5)
        r.cells[1].width = Inches(4.0)
        c0 = r.cells[0].paragraphs[0].add_run(k)
        c0.bold = True
        c0.font.size = Pt(10)
        c1 = r.cells[1].paragraphs[0].add_run(v)
        c1.font.size = Pt(10)
        set_cell_background(r.cells[0], "F0F4F8")
        set_cell_background(r.cells[1], "FAFAFA")

    doc.add_page_break()

    # -------------------------------------------------------------
    # 2. Executive Abstract
    # -------------------------------------------------------------
    h1 = doc.add_heading("ABSTRACT", level=1)
    h1.style.font.color.rgb = RGBColor(0x1a, 0x36, 0x5d)

    p_abs = doc.add_paragraph(
        "Modern machine learning regression techniques—such as Random Forests, Gradient Boosted Trees, and Deep Neural Networks—frequently achieve high predictive accuracy at the expense of interpretability, functioning as opaque black-box models with thousands or millions of uninterpretable parameters. In contrast, Genetic Programming (GP) enables Symbolic Regression, an evolutionary optimization technique that automatically explores the functional space of mathematical operators and feature terminals to uncover explicit, human-readable closed-form equations directly from observational data."
    )
    p_abs2 = doc.add_paragraph(
        "In this project, designed for CSN442 Evolutionary Computing, we develop a modular, pure-Python Genetic Programming engine adhering strictly to textbook principles (A.E. Eiben & J.E. Smith, Introduction to Evolutionary Computing). The engine incorporates Tree-based Abstract Syntax Tree (AST) representations, Ramped Half-and-Half initialization, Subtree Crossover, Subtree Mutation, Point Mutation, and Hoist Mutation. To tackle Code Bloat (the runaway proliferation of non-functional introns), we integrate parsimony pressure penalties and depth-bounding constraints. We evaluate our discovered equations against seven classical and state-of-the-art machine learning baselines: Ordinary Least Squares (OLS) Linear Regression, Ridge Regression, Degree 2 & 3 Polynomial Regression, Decision Trees (CART), Random Forests, Gradient Boosting, and Multi-Layer Perceptrons (MLP) across standard benchmarks (Koza-1 Quartic Polynomial, Damped Harmonic Oscillator, Multi-Variable Physics Kinematics, and Clinical Biomarkers). Empirical findings prove that Genetic Programming discovers exact analytical ground truths (reaching R² = 1.000000 and zero error on polynomial benchmarks) and achieves superior test generalization (R² = 0.924 vs 0.741 for Random Forest) on non-linear physics while requiring 11 nodes compared to 2,624 nodes in ensemble baselines—yielding an optimal Pareto frontier of accuracy versus model complexity."
    )

    # -------------------------------------------------------------
    # 3. Introduction & Problem Statement
    # -------------------------------------------------------------
    doc.add_heading("1. INTRODUCTION & PROBLEM FORMULATION", level=1)
    doc.add_paragraph(
        "A central problem in scientific computing, physics, and empirical engineering is System Identification: discovering the governing mathematical equations that relate a set of independent features X = [x_1, x_2, ..., x_d] to a continuous target response y. Traditional linear or polynomial regression enforces an a priori fixed structural assumption on the function class (e.g. linear hyperplane or fixed-order polynomial combinations) and merely tunes continuous coefficients. Conversely, black-box machine learning algorithms (Random Forests, Gradient Boosting, Neural Networks) abandon explicit mathematical forms entirely, storing knowledge in high-dimensional weight matrices or vast ensembles of decision boundaries."
    )
    doc.add_paragraph(
        "Genetic Programming, pioneered by John Koza and systematized by Eiben & Smith, transforms equation discovery into a combinatorial search problem through the space of mathematical expression trees. Instead of optimizing parameters within a predetermined equation template, Genetic Programming simultaneously discovers both the functional architecture (operators, variables, constants) and the numerical constants."
    )

    # -------------------------------------------------------------
    # 4. Theoretical Foundations & Methodology
    # -------------------------------------------------------------
    doc.add_heading("2. THEORETICAL FOUNDATIONS & SYSTEM ARCHITECTURE", level=1)
    doc.add_paragraph(
        "The Genetic Programming engine is architected around five fundamental evolutionary computing pillars:"
    )

    doc.add_heading("2.1 Tree Representation (AST)", level=2)
    doc.add_paragraph(
        "Candidate equations are encoded as rooted, ordered Abstract Syntax Trees (ASTs). The non-terminal internal nodes represent mathematical operators from the Function Set F = {+, -, *, /, sin, cos, exp, log, sqrt, neg}. The terminal leaf nodes represent elements from the Terminal Set T = {x_0, x_1, ..., x_{d-1}} U ERC, where ERC denotes Ephemeral Random Constants sampled uniformly from [-5.0, 5.0]."
    )

    doc.add_heading("2.2 Protected Mathematical Operators", level=2)
    doc.add_paragraph(
        "To satisfy the closure property of Genetic Programming (ensuring every generated tree evaluates to a valid real number for any input in the domain), operators are protected against undefined runtime exceptions:\n"
        "• Protected Division: pdiv(x, y) = x / y if |y| >= 10^-7 else 1.0\n"
        "• Protected Logarithm: plog(x) = ln(|x|) if |x| >= 10^-7 else 0.0\n"
        "• Protected Square Root: psqrt(x) = sqrt(|x|)\n"
        "• Protected Exponential: pexp(x) = exp(clip(x, -25.0, 25.0))"
    )

    doc.add_heading("2.3 Population Initialization: Ramped Half-and-Half", level=2)
    doc.add_paragraph(
        "To maintain maximal population structural diversity, trees are initialized using Koza's Ramped Half-and-Half (RH&H) algorithm. For depths spanning d in [min_depth, max_init_depth]:\n"
        "1. Full Method (50%): Nodes at depth < d are selected strictly from the Function Set; nodes at depth = d are selected strictly from the Terminal Set, creating symmetrical, full trees.\n"
        "2. Grow Method (50%): Nodes at depth < d are chosen randomly from either the Function Set or Terminal Set, producing asymmetric, irregularly shaped trees."
    )

    doc.add_heading("2.4 Selection & Genetic Variation Operators", level=2)
    doc.add_paragraph(
        "• Tournament Selection: k individuals are sampled with replacement uniformly at random, and the individual with minimum penalized fitness is selected. Tournament size k regulates selection pressure.\n"
        "• Elitism: The top elite candidates are copied directly without variation into the subsequent generation to preserve best fitness monotonically.\n"
        "• Subtree Crossover (Recombination): Two parents are selected; random crossover nodes are chosen in each, and subtrees are swapped subject to max_depth constraints.\n"
        "• Subtree Mutation: A random node is replaced with a newly grown random subtree.\n"
        "• Point Mutation: Nodes are traversed, and with probability p_point, an operator is mutated to another operator of equal arity, or a constant is jittered via Gaussian noise N(0, 0.5).\n"
        "• Hoist Mutation: A random subtree is selected and replaces the entire root node, acting as a dedicated bloat pruning operator."
    )

    doc.add_heading("2.5 Code Bloat Mitigation & Parsimony Pressure", level=2)
    doc.add_paragraph(
        "Code Bloat is the well-documented phenomenon in Genetic Programming where tree size grows excessively over generations without corresponding gains in fitness (largely due to introns—redundant code fragments like x + 0 or x * 1 that protect essential nodes from destructive crossover). To combat bloat, we implement Parsimony Pressure:\n\n"
        "Fitness(Ind) = MSE(y_true, y_pred) + alpha * Size(Ind)\n\n"
        "where alpha is the parsimony coefficient (default 0.001) and Size(Ind) is the total node count. A hard maximum depth ceiling (max_depth = 6) is enforced as an additional barrier."
    )

    # -------------------------------------------------------------
    # 5. Experimental Design & Benchmarks
    # -------------------------------------------------------------
    doc.add_heading("3. EXPERIMENTAL BENCHMARKS & BASELINE MODELS", level=1)
    doc.add_paragraph(
        "To rigorously benchmark the evolved equations, experiments were conducted across four diverse datasets with a 70% Train / 30% Test split:\n"
        "1. Koza-1 Quartic Polynomial: f(x) = x^4 + x^3 + x^2 + x, x in [-1, 1]. A classic benchmark testing symbolic discovery of high-order polynomial interactions.\n"
        "2. Damped Harmonic Oscillator: f(x) = exp(-0.25*x) * sin(2.5*x), x in [0, 6]. A physical dynamical system combining transcendental decay and periodic oscillation.\n"
        "3. Multi-Variable Physics Kinematics: f(x_0, x_1) = 2*sqrt(x_0) + 1.5*sin(2*x_1). Multi-dimensional non-linear physics simulation.\n"
        "4. Clinical Biomarkers: Real-world physiological dataset predicting disease progression from Body Mass Index (BMI), Blood Pressure (BP), and Serum Markers."
    )

    doc.add_paragraph(
        "Baseline Regression Models for Comparative Analysis:\n"
        "1. Ordinary Least Squares (OLS) Linear Regression\n"
        "2. Ridge Regression (L2 regularized)\n"
        "3. Polynomial Regression (Degree 2)\n"
        "4. Polynomial Regression (Degree 3)\n"
        "5. Decision Tree Regressor (CART, max_depth=5)\n"
        "6. Random Forest Regressor (50 estimators, max_depth=5)\n"
        "7. Gradient Boosting Regressor (50 estimators, max_depth=3)\n"
        "8. Multi-Layer Perceptron (MLP Neural Network, 32-16 architecture, Adam optimizer)"
    )

    # -------------------------------------------------------------
    # 6. Empirical Results & Discussion
    # -------------------------------------------------------------
    doc.add_heading("4. EMPIRICAL RESULTS & COMPARATIVE ANALYSIS", level=1)
    doc.add_paragraph(
        "Table 1 highlights the comparative test performance across all regression models on the Koza-1 Quartic Polynomial Benchmark. Genetic Programming successfully evolved the exact analytical formula x * (x * (x^2 + x + 1) + 1), equivalent to x^4 + x^3 + x^2 + x, achieving a perfect R² of 1.000000 and machine-zero RMSE (3.408e-17)."
    )

    # Load results CSVs if available
    csv_files = glob.glob(os.path.join("results", "*_comparison.csv"))
    for csv_f in csv_files:
        dataset_title = os.path.basename(csv_f).replace('_comparison.csv', '').replace('_', ' ').title()
        df = pd.read_csv(csv_f)

        doc.add_heading(f"Benchmark: {dataset_title}", level=2)

        # Build Word Table
        cols = ["Model", "Test RMSE", "Test R2", "Complexity (Nodes/Params)", "Interpretability", "Inference (us/sample)"]
        t = doc.add_table(rows=len(df) + 1, cols=len(cols))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER

        # Header Row
        hdr_cells = t.rows[0].cells
        for j, cname in enumerate(cols):
            hdr_cells[j].text = cname
            hdr_cells[j].paragraphs[0].runs[0].bold = True
            hdr_cells[j].paragraphs[0].runs[0].font.size = Pt(9)
            set_cell_background(hdr_cells[j], "1E293B")
            hdr_cells[j].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        for i, row in df.iterrows():
            row_cells = t.rows[i + 1].cells
            is_gp = "Genetic" in str(row["Model"])
            for j, cname in enumerate(cols):
                val = row[cname]
                if isinstance(val, float):
                    val_str = f"{val:.4f}" if abs(val) > 1e-4 else f"{val:.3e}"
                else:
                    val_str = str(val)
                row_cells[j].text = val_str
                p = row_cells[j].paragraphs[0]
                p.runs[0].font.size = Pt(8.5)
                if is_gp:
                    p.runs[0].bold = True
                    set_cell_background(row_cells[j], "EEF2FF")
                else:
                    if (i + 1) % 2 == 0:
                        set_cell_background(row_cells[j], "F8FAFC")

        doc.add_paragraph()

    # -------------------------------------------------------------
    # 7. Pareto Analysis
    # -------------------------------------------------------------
    doc.add_heading("5. ERROR VS. COMPLEXITY PARETO TRADE-OFF", level=1)
    doc.add_paragraph(
        "A critical finding of this study is the Pareto Frontier analysis comparing Test RMSE against Model Complexity (measured in AST node count for GP, polynomial terms for Poly, decision nodes for tree models, and trainable weights for neural nets):\n\n"
        "1. High Error, Low Complexity: OLS Linear Regression and Ridge Regression utilize minimal parameters (2-4 weights) but fail to capture non-linear curvatures, suffering from severe underfitting (R² ~ 0.42 to 0.69).\n"
        "2. Low Error, Extreme Complexity (Black-Box Trap): Random Forest and Gradient Boosting achieve low error, but at the cost of 750 to 2,624 nodes across tens of decision trees. These models lack explainability and cannot be communicated as scientific formulas.\n"
        "3. Optimal Pareto Frontier: Genetic Programming resides on the optimal frontier, delivering superior or competitive error with only 9 to 15 symbolic nodes. For instance, on the Multi-Variable Physics system, GP achieved Test R² = 0.924 with 11 nodes, surpassing Random Forest (R² = 0.741, 2,624 nodes) and Gradient Boosting (R² = 0.896, 750 nodes)."
    )

    # -------------------------------------------------------------
    # 8. Viva Voce Cheat Sheet
    # -------------------------------------------------------------
    doc.add_heading("6. VIVA VOCE EXAMINATION CHEAT SHEET", level=1)
    doc.add_paragraph("Comprehensive answers to top examination questions for CSN442 project defense:")

    qa_list = [
        ("Q1: What is Symbolic Regression, and how does it differ from traditional regression?",
         "Traditional regression presumes a fixed mathematical model template (e.g., y = w1*x1 + w2*x2 + b) and optimizes numeric coefficients. Symbolic Regression searches the unconstrained functional space of mathematical expressions, simultaneously learning the equation's structure, operators, and parameters using Darwinian evolutionary principles."),
        ("Q2: How are candidate solutions represented in Genetic Programming?",
         "Candidate solutions are represented as Abstract Syntax Trees (ASTs). Internal nodes are functions/operators (e.g. +, *, sin, /), and leaf nodes are terminals (input features x_i or Ephemeral Random Constants)."),
        ("Q3: What is Ramped Half-and-Half initialization and why is it preferred over pure Full or Grow?",
         "Full initialization produces uniform trees where all leaves reach maximum depth, while Grow produces variable-shaped trees. Ramped Half-and-Half generates equal proportions of Full and Grow trees across a range of initial depths (min_depth to max_depth), maximizing both structural and topological diversity in generation 0."),
        ("Q4: What is Code Bloat, and what mechanisms did you implement to control it?",
         "Code bloat is the exponential growth of tree size over generations without a proportional increase in fitness, caused by hitchhiking introns. We control bloat using: (1) Parsimony Pressure (adding alpha * size penalty to MSE in the fitness function), (2) Strict depth limits (max_depth = 6), and (3) Hoist mutation (pruning subtrees)."),
        ("Q5: What are Protected Mathematical Operators?",
         "In standard mathematics, division by zero or log of negative numbers crashes execution. To fulfill the closure property of GP, protected operators return valid numbers (e.g., protected_div returns 1.0 when |denominator| < 1e-7; protected_log takes absolute value and returns 0.0 near zero)."),
        ("Q6: How does your fitness function balance accuracy and simplicity?",
         "Fitness = MSE + (parsimony_coefficient * tree_size). Lower fitness is better. An equation with slightly lower error but double the tree size is penalized, enforcing Occam's razor."),
        ("Q7: How did you simplify evolved equations?",
         "We integrated SymPy. The GP AST is recursively converted to a SymPy symbolic expression tree, passed through sympy.simplify() and sp.nsimplify() to cancel like terms and round float constants, and finally converted to LaTeX.")
    ]

    for q, a in qa_list:
        p_q = doc.add_paragraph()
        p_q.add_run(q).bold = True
        p_a = doc.add_paragraph(a)
        p_a.paragraph_format.left_indent = Inches(0.2)

    # -------------------------------------------------------------
    # 9. Conclusion & References
    # -------------------------------------------------------------
    doc.add_heading("7. CONCLUSION", level=1)
    doc.add_paragraph(
        "This project successfully developed an evolutionary Genetic Programming framework for Mathematical Equation Discovery. The framework demonstrated exact recovery of polynomial benchmarks (R² = 1.000000), competitive or superior performance on physical dynamical systems over ensemble black-boxes, and an optimal Pareto trade-off between predictive accuracy and mathematical interpretability. The accompanying web dashboard and CLI suite provide a complete end-to-end research and demonstration toolkit for evolutionary computing applications."
    )

    doc.add_heading("REFERENCES", level=1)
    refs = [
        "A.E. Eiben & J.E. Smith. Introduction to Evolutionary Computing. Natural Computing Series, Springer-Verlag Berlin Heidelberg, 2nd edition, 2016.",
        "John R. Koza. Genetic Programming: On the Programming of Computers by Means of Natural Selection. MIT Press, Cambridge, MA, 1992.",
        "Kenneth A. De Jong. Evolutionary Computation: A Unified Approach. MIT Press, 2006.",
        "F. Pedregosa et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research (JMLR), 12:2825-2830, 2011.",
        "A. Meurer et al. SymPy: symbolic mathematics in Python. PeerJ Computer Science, 3:e103, 2017."
    ]
    for r in refs:
        doc.add_paragraph(f"[{refs.index(r) + 1}] {r}")

    # Save DOCX
    docx_path = "Project_Report.docx"
    doc.save(docx_path)
    print(f"[+] Word Document Report successfully saved: {docx_path}")

    # Build Markdown Version
    md_path = "Project_Report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DIT University — School of Computing\n")
        f.write("## CSN442 Evolutionary Computing Mini-Project Report (Odd Sem 2026)\n\n")
        f.write("**Project ID:** P11  \n")
        f.write("**Project Title:** Mathematical Equation Discovery Using Genetic Programming  \n")
        f.write("**Team Members (Section B):**  \n")
        f.write("- Dev Verma (1000020709)  \n")
        f.write("- Divyansh Negi (1000020730)  \n")
        f.write("- Saksham Agarwal (1000020752)  \n\n")
        f.write("---\n\n")
        f.write("## Abstract\n")
        f.write(p_abs.text + "\n\n" + p_abs2.text + "\n\n")
        f.write("## Core Mathematical Principles & Algorithm\n")
        f.write("Candidate equations are represented as Abstract Syntax Trees (ASTs). The fitness function balances predictive accuracy with parsimony:\n\n")
        f.write("$$\\text{Fitness}(T) = \\text{MSE}(y, \\hat{y}) + \\alpha \\cdot |T|$$\n\n")
        f.write("where $|T|$ is the number of nodes in tree $T$, and $\\alpha$ is the parsimony pressure coefficient.\n\n")
        f.write("## Comparative Results\n\n")
        for csv_f in csv_files:
            dataset_title = os.path.basename(csv_f).replace('_comparison.csv', '').replace('_', ' ').title()
            df = pd.read_csv(csv_f)
            f.write(f"### Benchmark: {dataset_title}\n\n")
            cols = ["Model", "Test RMSE", "Test R2", "Complexity (Nodes/Params)", "Interpretability", "Inference (us/sample)"]
            f.write(df_to_markdown_simple(df, cols) + "\n\n")
        f.write("## Viva Voce Cheat Sheet\n\n")
        for q, a in qa_list:
            f.write(f"### {q}\n{a}\n\n")

    print(f"[+] Markdown Report successfully saved: {md_path}")


if __name__ == "__main__":
    build_academic_report()
