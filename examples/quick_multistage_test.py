"""
Quick Multi-stage Test
Reduced epochs for validation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn

from multistage_training import MultiStageTrainer, MultiStageConfig, FourierFeatureNetwork


class SimplePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, xy):
        return self.net(xy)


class PoissonPDE:
    def __init__(self):
        self.pi = 3.14159265359

    def compute_residual(self, u_pred, x, y):
        u_x = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u_pred.sum(), y, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        laplacian = u_xx + u_yy
        return -laplacian


def main():
    print("[*] Quick Multi-stage Validation Test")
    print("=" * 60)

    # Create training data
    n_points = 20
    x_vals = torch.linspace(0, 1, n_points, dtype=torch.float64)
    y_vals = torch.linspace(0, 1, n_points, dtype=torch.float64)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    x = X.flatten().reshape(-1, 1).requires_grad_(True)
    y = Y.flatten().reshape(-1, 1).requires_grad_(True)

    pde = PoissonPDE()

    # Stage 1: Quick training
    print("\n[+] Stage 1: Coarse solution (200 epochs)")
    stage1_net = SimplePINN().to(torch.float64)
    optimizer1 = torch.optim.Adam(stage1_net.parameters(), lr=1e-3)

    for epoch in range(200):
        optimizer1.zero_grad()
        xy = torch.cat([x, y], dim=1)
        u_pred = stage1_net(xy)
        residual = pde.compute_residual(u_pred, x, y)
        loss = torch.mean(residual ** 2)
        loss.backward()
        optimizer1.step()

        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: loss = {loss.item():.6e}")

    # Evaluate stage 1
    xy = torch.cat([x, y], dim=1)
    u1 = stage1_net(xy)
    res1 = pde.compute_residual(u1, x, y)
    max_res1 = torch.max(torch.abs(res1)).item()
    print(f"[+] Stage 1 final residual: {max_res1:.6e}")

    # Stage 2: Fourier features
    print("\n[+] Stage 2: Fourier refinement (200 epochs)")
    stage2_net = FourierFeatureNetwork(
        input_dim=2,
        hidden_dim=32,
        output_dim=1,
        fourier_sigma=10.0,  # Fixed sigma for quick test
        num_fourier_features=32
    ).to(torch.float64)

    optimizer2 = torch.optim.Adam(stage2_net.parameters(), lr=1e-4)
    epsilon = 1.0

    for epoch in range(200):
        optimizer2.zero_grad()
        xy = torch.cat([x, y], dim=1)

        with torch.no_grad():
            u1 = stage1_net(xy)

        u2 = stage2_net(xy)
        u_combined = u1 + epsilon * u2

        residual = pde.compute_residual(u_combined, x, y)
        loss = torch.mean(residual ** 2)
        loss.backward()
        optimizer2.step()

        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: loss = {loss.item():.6e}")

    # Evaluate combined
    xy = torch.cat([x, y], dim=1)
    u1 = stage1_net(xy)
    u2 = stage2_net(xy)
    u_combined = u1 + epsilon * u2
    res_combined = pde.compute_residual(u_combined, x, y)
    max_res_combined = torch.max(torch.abs(res_combined)).item()

    print(f"\n[+] Stage 2 combined residual: {max_res_combined:.6e}")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Stage 1 residual:    {max_res1:.6e}")
    print(f"Stage 2 residual:    {max_res_combined:.6e}")

    if max_res_combined < max_res1:
        improvement = max_res1 / max_res_combined
        print(f"Improvement:         {improvement:.1f}x")
        print("\n[+] Multi-stage training shows improvement!")

        if max_res_combined < 1e-10:
            print("[+] Near machine precision achieved (< 10^-10)")
        elif max_res_combined < 1e-8:
            print("[+] High precision achieved (< 10^-8)")
    else:
        print("\n[!] Stage 2 did not improve (may need more epochs or tuning)")

    print("\n[+] Test complete!")


if __name__ == "__main__":
    main()