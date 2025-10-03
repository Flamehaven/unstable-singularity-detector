#!/usr/bin/env python3
"""
Lambda Comparison and Validation Script
Compares experimental results against DeepMind reference values
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI
import matplotlib.pyplot as plt


def load_results(path: str) -> Dict:
    """Load results from JSON file"""
    with open(path, "r") as f:
        return json.load(f)


def compare_lambda(exp: List[float], ref: List[float], rtol: float = 1e-3) -> Tuple[bool, List[float], List[float]]:
    """
    Compare lambda estimates between experiment and reference

    Args:
        exp: Experimental lambda values
        ref: Reference lambda values
        rtol: Relative tolerance

    Returns:
        (all_passed, absolute_differences, relative_errors)
    """
    if len(exp) != len(ref):
        print(f"[!] Length mismatch: exp={len(exp)}, ref={len(ref)}")
        return False, [], []

    diffs = [abs(e - r) for e, r in zip(exp, ref)]
    rel_errors = [d / abs(r) if r != 0 else d for d, r in zip(diffs, ref)]
    passed = [re < rtol for re in rel_errors]

    return all(passed), diffs, rel_errors


def plot_lambda_comparison(ref: List[float], exp: List[float], output_path: str):
    """Generate lambda comparison plot"""
    plt.figure(figsize=(8, 5))
    x = range(len(ref))

    plt.plot(x, ref, label="DeepMind Reference", marker="o", linestyle="-", linewidth=2)
    plt.plot(x, exp, label="Our Implementation", marker="x", linestyle="--", linewidth=2)

    plt.xlabel("Case Index", fontsize=12)
    plt.ylabel("λ (Blowup Rate)", fontsize=12)
    plt.title("Lambda Estimates Comparison", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"[+] Saved plot: {output_path}")


def plot_residual_history(residuals: List[float], output_path: str):
    """Generate residual convergence plot"""
    plt.figure(figsize=(8, 5))

    plt.semilogy(residuals, linewidth=2, color="blue")
    plt.axhline(y=1e-12, color="red", linestyle="--", label="Target: 1e-12")

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Residual (log scale)", fontsize=12)
    plt.title("Convergence History", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"[+] Saved plot: {output_path}")


def generate_summary_table(ref: List[float], exp: List[float], diffs: List[float], rel_errors: List[float], rtol: float) -> str:
    """Generate markdown summary table"""
    lines = ["| Case | Reference λ | Experimental λ | |Δ| | Rel. Error | Status |"]
    lines.append("|------|-------------|----------------|------|------------|--------|")

    for i, (r, e, d, re) in enumerate(zip(ref, exp, diffs, rel_errors)):
        status = "[+]" if re < rtol else "[-]"
        lines.append(f"| {i+1:4d} | {r:11.6f} | {e:14.6f} | {d:.6f} | {re:.6f} | {status:6s} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare lambda estimates against reference")
    parser.add_argument("--ref", required=True, help="Reference results JSON")
    parser.add_argument("--exp", required=True, help="Experiment results JSON")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument("--output-dir", default="results/plots", help="Output directory for plots")
    args = parser.parse_args()

    # Load data
    print(f"[*] Loading reference: {args.ref}")
    ref_data = load_results(args.ref)

    print(f"[*] Loading experiment: {args.exp}")
    exp_data = load_results(args.exp)

    # Extract lambda values
    ref_lambda = ref_data.get("lambda_estimates", [])
    exp_lambda = exp_data.get("lambda_estimates", [])

    if not ref_lambda or not exp_lambda:
        print("[-] Error: Missing lambda_estimates in JSON files")
        sys.exit(1)

    # Compare
    print("\n" + "="*60)
    print("Lambda Comparison Results")
    print("="*60)

    passed, diffs, rel_errors = compare_lambda(exp_lambda, ref_lambda, rtol=args.rtol)

    print(f"\nReference:     {ref_lambda}")
    print(f"Experiment:    {exp_lambda}")
    print(f"Abs. Diffs:    {[f'{d:.6f}' for d in diffs]}")
    print(f"Rel. Errors:   {[f'{re:.6f}' for re in rel_errors]}")
    print(f"Tolerance:     {args.rtol}")
    print(f"\nResult: {'[+] PASS' if passed else '[-] FAIL'}")

    # Generate table
    table = generate_summary_table(ref_lambda, exp_lambda, diffs, rel_errors, args.rtol)
    print("\n" + table)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_lambda_comparison(ref_lambda, exp_lambda, str(output_dir / "lambda_comparison.png"))

    # Plot residual history if available
    if "residual_history" in exp_data:
        plot_residual_history(exp_data["residual_history"], str(output_dir / "residual_history.png"))

    # Save summary
    summary_path = output_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("# Validation Summary\n\n")
        f.write(f"**Tolerance**: {args.rtol}\n\n")
        f.write(f"**Status**: {'PASS' if passed else 'FAIL'}\n\n")
        f.write("## Lambda Comparison\n\n")
        f.write(table + "\n\n")

        if "final_residual" in exp_data:
            f.write(f"**Final Residual**: {exp_data['final_residual']:.2e}\n\n")

        if "precision_mode" in exp_data:
            f.write(f"**Precision Mode**: {exp_data['precision_mode']}\n\n")

    print(f"\n[+] Summary saved: {summary_path}")

    # Exit code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
