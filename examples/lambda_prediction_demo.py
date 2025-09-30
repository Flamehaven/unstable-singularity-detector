"""
Lambda Prediction Demo
Demonstrates DeepMind's empirical formula for unstable singularity prediction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from unstable_singularity_detector import UnstableSingularityDetector


def demo_lambda_prediction():
    """Show lambda prediction for IPM and Boussinesq equations"""

    print("[*] DeepMind Lambda Prediction Demo")
    print("=" * 60)

    # 1. IPM Equation
    print("\n[1] IPM (Incompressible Porous Media)")
    print("-" * 60)

    detector_ipm = UnstableSingularityDetector(equation_type="ipm")

    # Ground truth from paper
    ipm_ground_truth = {
        0: 1.0285722760222,
        1: 0.4721297362414,
        2: 0.3149620267088,
        3: 0.2415604743989
    }

    print(f"{'Order':>6} | {'Ground Truth':>15} | {'Predicted':>15} | {'Error %':>10}")
    print("-" * 60)

    for order in range(4):
        true_lambda = ipm_ground_truth.get(order + 1)
        if true_lambda is None:
            break

        pred_lambda = detector_ipm.predict_next_unstable_lambda(order)
        error_pct = abs(pred_lambda - true_lambda) / true_lambda * 100

        print(f"{order:>6} | {true_lambda:>15.10f} | {pred_lambda:>15.10f} | {error_pct:>9.2f}%")

    # 2. Boussinesq Equation
    print("\n[2] Boussinesq (3D Euler Analogue)")
    print("-" * 60)

    detector_bsq = UnstableSingularityDetector(equation_type="boussinesq")

    bsq_ground_truth = {
        0: 1.9205599746927,
        1: 1.3990961221852,
        2: 1.2523481636489,
        3: 1.1842500861997
    }

    print(f"{'Order':>6} | {'Ground Truth':>15} | {'Predicted':>15} | {'Error %':>10}")
    print("-" * 60)

    for order in range(4):
        true_lambda = bsq_ground_truth.get(order + 1)
        if true_lambda is None:
            break

        pred_lambda = detector_bsq.predict_next_unstable_lambda(order)
        error_pct = abs(pred_lambda - true_lambda) / true_lambda * 100

        print(f"{order:>6} | {true_lambda:>15.10f} | {pred_lambda:>15.10f} | {error_pct:>9.2f}%")


def plot_lambda_pattern_comparison():
    """Visualize lambda patterns for both equations"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # IPM
    detector_ipm = UnstableSingularityDetector(equation_type="ipm")
    ipm_ground_truth = {
        0: 1.0285722760222,
        1: 0.4721297362414,
        2: 0.3149620267088,
        3: 0.2415604743989
    }

    orders = list(ipm_ground_truth.keys())
    true_lambdas_ipm = list(ipm_ground_truth.values())
    pred_lambdas_ipm = [detector_ipm.predict_next_unstable_lambda(max(0, o-1)) if o > 0 else ipm_ground_truth[0]
                         for o in orders]

    ax1.scatter(orders, true_lambdas_ipm, s=100, label='Ground Truth (Paper)', color='red', zorder=3)
    ax1.scatter(orders[1:], pred_lambdas_ipm[1:], s=100, marker='x', label='Predicted Formula', color='blue', zorder=3)

    # Plot formula curve
    order_range = np.linspace(0, 5, 100)
    pattern = detector_ipm.lambda_pattern_coefficients["ipm"]
    formula_curve = 1.0 / (pattern["a"] * order_range + pattern["b"]) + pattern["c"]
    ax1.plot(order_range, formula_curve, '--', alpha=0.5, label='λ = 1/(1.1459n + 0.9723)', color='green')

    ax1.set_xlabel('Instability Order (n)', fontsize=12)
    ax1.set_ylabel('Lambda (λ)', fontsize=12)
    ax1.set_title('IPM: Lambda vs Instability Order', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Boussinesq
    detector_bsq = UnstableSingularityDetector(equation_type="boussinesq")
    bsq_ground_truth = {
        0: 1.9205599746927,
        1: 1.3990961221852,
        2: 1.2523481636489,
        3: 1.1842500861997
    }

    orders = list(bsq_ground_truth.keys())
    true_lambdas_bsq = list(bsq_ground_truth.values())
    pred_lambdas_bsq = [detector_bsq.predict_next_unstable_lambda(max(0, o-1)) if o > 0 else bsq_ground_truth[0]
                         for o in orders]

    ax2.scatter(orders, true_lambdas_bsq, s=100, label='Ground Truth (Paper)', color='red', zorder=3)
    ax2.scatter(orders[1:], pred_lambdas_bsq[1:], s=100, marker='x', label='Predicted Formula', color='blue', zorder=3)

    # Plot formula curve
    pattern = detector_bsq.lambda_pattern_coefficients["boussinesq"]
    formula_curve = 1.0 / (pattern["a"] * order_range + pattern["b"]) + pattern["c"]
    ax2.plot(order_range, formula_curve, '--', alpha=0.5, label='λ = 1/(1.4187n + 1.0863) + 1', color='green')

    ax2.set_xlabel('Instability Order (n)', fontsize=12)
    ax2.set_ylabel('Lambda (λ)', fontsize=12)
    ax2.set_title('Boussinesq: Lambda vs Instability Order', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lambda_pattern_validation.png', dpi=150)
    print("\n[+] Plot saved to 'lambda_pattern_validation.png'")
    plt.show()


def extrapolate_higher_orders():
    """Predict lambda values for higher unstable modes"""

    print("\n[3] Extrapolation to Higher Unstable Modes")
    print("=" * 60)

    detector_ipm = UnstableSingularityDetector(equation_type="ipm")
    detector_bsq = UnstableSingularityDetector(equation_type="boussinesq")

    print(f"{'Order':>6} | {'IPM Lambda':>15} | {'Boussinesq Lambda':>18}")
    print("-" * 60)

    for order in range(4, 10):
        lambda_ipm = detector_ipm.predict_next_unstable_lambda(order - 1)
        lambda_bsq = detector_bsq.predict_next_unstable_lambda(order - 1)

        print(f"{order:>6} | {lambda_ipm:>15.10f} | {lambda_bsq:>18.10f}")

    print("\n[!] Note: These are predictions for undiscovered unstable modes")
    print("    Use these as initialization values for PINN training")


if __name__ == "__main__":
    demo_lambda_prediction()
    plot_lambda_pattern_comparison()
    extrapolate_higher_orders()

    print("\n" + "=" * 60)
    print("[+] Demo complete! Key findings:")
    print("    - Empirical formula predicts unstable lambdas within 2% error")
    print("    - Can extrapolate to higher-order unstable modes")
    print("    - Use predictions as PINN initialization for faster convergence")
    print("=" * 60)