"""
generate_data.py
=================
Eurocode-constrained synthetic data generator for CFST Moment-Curvature
(M-phi) analysis.

Runs the custom Fiber-Section Analysis physics engine across a randomized
design space [D, t, fy, fc], solving the neutral-axis equilibrium at every
curvature step with Brent's method (scipy.optimize.brentq), and produces a
dual-output target per sample:

    Channel 0 -> Moment capacity M(phi)
    Channel 1 -> Neutral axis depth c(phi)   (the latent physical state)

Samples are generated in parallel across all CPU cores via joblib.

Output:
    data/cfst_dataset.npz   -> X_train, X_test, y_train, y_test
    data/scaler_X.pkl       -> fitted MinMaxScaler for [D, t, fy, fc]
    data/scaler_y.pkl       -> fitted MinMaxScaler for [M, c] (flattened)

Usage:
    python src/generate_data.py
    python src/generate_data.py --n-samples 2000 --out-dir data --plot
"""

import argparse
import os
import time

import joblib
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import brentq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Eurocode 2 / Eurocode 4 constrained design-parameter bounds.
# NOTE: these bounds alone do not enforce a local-buckling / slenderness
# (width-to-thickness) check. If that check is required to match the
# project description, add a rejection rule on D/t inside
# generate_one_sample() before accepting a sample.
PARAM_BOUNDS = {
    "D": (200, 800),    # outer diameter, mm
    "t": (4, 20),       # wall thickness, mm
    "fy": (235, 460),   # steel yield strength, MPa
    "fc": (30, 80),     # concrete compressive strength, MPa
}


# ---------------------------------------------------------------------------
# 1. Material constitutive models
# ---------------------------------------------------------------------------
def material_steel(strain, fy, Es=200000):
    """Bi-linear steel stress-strain law (elastic-plastic with 1% hardening)."""
    ey = fy / Es
    if abs(strain) <= ey:
        return strain * Es
    return np.sign(strain) * (fy + (abs(strain) - ey) * Es * 0.01)


def material_concrete(strain, fc, epsilon_c0=0.002, epsilon_cu=0.01):
    """Non-linear concrete stress-strain law with parabolic rise and linear softening."""
    if strain > 0:
        return 0
    strain = abs(strain)
    if strain <= epsilon_c0:
        return fc * (2 * (strain / epsilon_c0) - (strain / epsilon_c0) ** 2)
    if strain <= epsilon_cu:
        return fc * (1 - 0.15 * (strain - epsilon_c0) / (epsilon_cu - epsilon_c0))
    return 0


# ---------------------------------------------------------------------------
# 2. Fiber-section solver
# ---------------------------------------------------------------------------
def calculate_force_moment(na_depth, phi, D, t, fy, fc, n_layers=100):
    """Discretizes the circular CFST section into n_layers fibers and
    integrates axial force / moment for a given neutral-axis depth and
    curvature."""
    R_outer = D / 2
    R_inner = R_outer - t
    dy = D / n_layers
    y_coords = np.linspace(-R_outer + dy / 2, R_outer - dy / 2, n_layers)

    net_force = 0
    net_moment = 0

    for y in y_coords:
        strain = -phi * (y - na_depth)

        width_total = 2 * np.sqrt(R_outer**2 - y**2) if abs(y) < R_outer else 0
        width_inner = 2 * np.sqrt(R_inner**2 - y**2) if abs(y) < R_inner else 0

        area_steel = (width_total - width_inner) * dy
        area_concrete = width_inner * dy

        sig_s = material_steel(strain, fy)
        sig_c = material_concrete(strain, fc)

        force_layer = (sig_s * area_steel) + (sig_c * area_concrete)
        net_force += force_layer
        net_moment += force_layer * y

    return net_force, -net_moment / 1e6


def get_moment_curvature_robust(D, t, fy, fc, n_points=100):
    """Sweeps curvature from 0 to a diameter-scaled limit, solving for the
    equilibrium neutral axis at each step via Brent's method, and returns
    both the Moment and Neutral Axis trajectories."""
    max_phi = (0.005 / (D / 2)) * 8
    curvatures = np.linspace(0, max_phi, n_points)

    moments = []
    neutral_axes = []
    current_na = 0

    for phi in curvatures:
        if phi == 0:
            moments.append(0)
            neutral_axes.append(0)
            continue

        def force_imbalance(na, phi=phi):
            F, _ = calculate_force_moment(na, phi, D, t, fy, fc)
            return F

        try:
            root_na = brentq(force_imbalance, -D / 2 * 0.99, D / 2 * 0.99)
            current_na = root_na
        except ValueError:
            root_na = current_na

        _, M = calculate_force_moment(root_na, phi, D, t, fy, fc)
        moments.append(M)
        neutral_axes.append(root_na)

    return moments, neutral_axes


# ---------------------------------------------------------------------------
# 3. Parallel sample generation
# ---------------------------------------------------------------------------
def generate_one_sample(i, param_bounds):
    """Draws one random design point and solves its full M-phi / c-phi curve.
    Returns None on numerical failure so the caller can discard it."""
    np.random.seed(i)
    D = np.random.uniform(*param_bounds["D"])
    t = np.random.uniform(*param_bounds["t"])
    fy = np.random.uniform(*param_bounds["fy"])
    fc = np.random.uniform(*param_bounds["fc"])

    try:
        M_curve, NA_curve = get_moment_curvature_robust(D, t, fy, fc, n_points=100)
        if np.max(M_curve) < 1 or np.isnan(M_curve).any():
            return None
        return [D, t, fy, fc], M_curve, NA_curve
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. Pipeline entry point
# ---------------------------------------------------------------------------
def run(n_samples=2000, out_dir="data", show_plot=False):
    os.makedirs(out_dir, exist_ok=True)

    print("--- STARTING MULTI-TASK PHYSICS ENGINE ---")
    start_time = time.time()

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(generate_one_sample)(i, PARAM_BOUNDS) for i in range(n_samples)
    )

    valid_results = [r for r in results if r is not None]
    X_raw = [r[0] for r in valid_results]
    y_M_raw = [r[1] for r in valid_results]
    y_NA_raw = [r[2] for r in valid_results]

    print(f"\nGeneration complete in {time.time() - start_time:.2f}s")
    print(f"Valid samples: {len(X_raw)}/{n_samples}")

    # Stack Moment + Neutral Axis into (Samples, Steps, 2)
    X_data = np.array(X_raw)
    y_combined = np.stack((y_M_raw, y_NA_raw), axis=2)

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_data)

    scaler_y = MinMaxScaler()
    y_flat = y_combined.reshape(-1, 2)
    y_flat_scaled = scaler_y.fit_transform(y_flat)
    y_scaled = y_flat_scaled.reshape(len(X_raw), 100, 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    print("Dataset shape:", y_train.shape, "(expected: (N, 100, 2))")

    np.savez(
        os.path.join(out_dir, "cfst_dataset.npz"),
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    )
    joblib.dump(scaler_X, os.path.join(out_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(out_dir, "scaler_y.pkl"))
    print(f"Saved dataset + scalers to '{out_dir}/'")

    if show_plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for i in range(3):
            plt.plot(y_scaled[i, :, 0], label=f"Sample {i}")
        plt.title("Target 1: Scaled Moment (Capacity)")
        plt.xlabel("Step")
        plt.ylabel("Moment (0-1)")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        for i in range(3):
            plt.plot(y_scaled[i, :, 1], label=f"Sample {i}")
        plt.title("Target 2: Scaled Neutral Axis (Latent Physics)")
        plt.xlabel("Step")
        plt.ylabel("NA Depth (0-1)")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "multitask_targets_preview.png"), dpi=150)
        plt.show()

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of CFST designs to sample")
    parser.add_argument("--out-dir", type=str, default="data", help="Directory to write the dataset + scalers to")
    parser.add_argument("--plot", action="store_true", help="Show/save the sanity-check target plot")
    args = parser.parse_args()

    run(n_samples=args.n_samples, out_dir=args.out_dir, show_plot=args.plot)
