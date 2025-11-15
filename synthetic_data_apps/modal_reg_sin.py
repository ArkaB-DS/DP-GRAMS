# modal_reg_sin.py
#
# Modal regression experiments (sinusoidal 2-mixture):
# - Baseline PMS vs LOWESS (non-private)
# - DP-PMS behavior (synthetic_data_apps.dp_pms)
# - Privacy–utility tradeoff across n and ε

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys
import csv
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.nonparametric.smoothers_lowess import lowess

# Ensure local imports work when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synthetic_data_apps.dp_pms import dp_pms
from pms import partial_mean_shift
from main_scripts.mode_matching_mse import mode_matching_mse

# ---------------------------------------------------------------------
# Plot styling (cosmetic, consistent with other figures)
# ---------------------------------------------------------------------
sns.set_context("talk")
sns.set_style("white")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11
})

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(text)

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------------------------------------
# Synthetic data generator: sinusoidal 2-mixture
# ---------------------------------------------------------------------

sigma_noise = 0.15

def generate_sin_data(n, sigma=sigma_noise, seed=None):
    """
    Sinusoidal 2-component mixture:
      - Half samples: Y ~ N(1.5 + 0.5 sin(3πX), sigma^2)
      - Half samples: Y ~ N(0.5 sin(3πX), sigma^2)
      with X ~ Unif(0,1) independently for each component.
    """
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1

    X1 = rng.uniform(0, 1, n1)
    X2 = rng.uniform(0, 1, n2)
    X = np.concatenate([X1, X2])

    Y1 = rng.normal(1.5 + 0.5 * np.sin(3 * np.pi * X1), sigma, n1)
    Y2 = rng.normal(0.5 * np.sin(3 * np.pi * X2), sigma, n2)
    Y = np.concatenate([Y1, Y2])

    return X, Y

# ---------------------------------------------------------------------
# Global experiment settings
# ---------------------------------------------------------------------

results_dir = "results/modal_regression_sin"
ensure_dir(results_dir)

# Match structure of modal_reg_3.py
n_values = [200, 500, 1000, 2000]
eps_values = [0.1, 0.2, 0.5, 1.0]
n_runs = 20
MAX_WORKERS = min(8, os.cpu_count() or 8)

# Per-run CSV for DP-PMS
perrun_csv = os.path.join(results_dir, "dp_per_run_results.csv")
if not os.path.exists(perrun_csv):
    with open(perrun_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_samples", "epsilon", "run_idx",
            "seed", "dp_mse", "dp_runtime_s", "timestamp"
        ])

# ---------------------------------------------------------------------
# Worker for DP-PMS (for parallel runs)
# ---------------------------------------------------------------------

def dp_pms_worker(X, Y, epsilon, delta, seed, T_iter=None, verbose=False):
    """
    Run DP-PMS once with given seed, return (seed, runtime, Y_dp, X_copy).
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    n = len(X)
    T_use = T_iter if T_iter is not None else int(np.ceil(np.log(max(2, n))))

    Y_dp = dp_pms(
        X, Y,
        mesh_points=Y.copy(),
        epsilon=float(epsilon),
        delta=float(delta),
        T=int(T_use),
        rng=rng,
        verbose=verbose
    )
    runtime = time.perf_counter() - t0
    return int(seed), float(runtime), Y_dp, X.copy()

# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():
    # ------------------------------------------------------------
    # Baseline visualization on a fixed dataset
    # ------------------------------------------------------------
    Xb, Yb = generate_sin_data(500, sigma_noise, seed=42)

    # Non-private PMS
    T_base = int(np.log(500))
    Y_pms_baseline = partial_mean_shift(Xb, Yb, mesh_points=None, bandwidth=None, T=T_base)

    # DP-PMS example (ε=1)
    Y_dp_baseline = dp_pms(
        Xb, Yb,
        mesh_points=Yb.copy(),
        epsilon=1.0,
        delta=1e-5,
        T=T_base,
        rng=np.random.default_rng(42),
        verbose=False
    )

    # LOWESS smooth (non-modal baseline)
    Y_lowess = lowess(Yb, Xb, frac=0.2, return_sorted=False)

    # ----- Figure 1: PMS vs LOWESS (separate) -----
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(Xb, Yb, color='skyblue', alpha=0.5, s=15, label='Data')
    ax1.scatter(Xb, Y_pms_baseline, color='#1f77b4', s=25, marker='D', label='PMS')
    ax1.plot(
        np.sort(Xb),
        Y_lowess[np.argsort(Xb)],
        color='orange', linewidth=2,
        label='LOWESS'
    )
    ax1.set_title('PMS vs LOWESS (Sinusoidal Mixture)')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_xlim(0, 1)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()
    plt.tight_layout()
    plt.show()

    outpath1 = os.path.join(results_dir, "baseline_pms_vs_lowess.png")
    fig1.savefig(outpath1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"[saved] PMS vs LOWESS -> {outpath1}")

    # ----- Figure 2: DP-PMS baseline (ε=1) -----
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(Xb, Yb, color='skyblue', alpha=0.5, s=15, label='Data')
    ax2.scatter(
        Xb, Y_dp_baseline,
        color='#d95f02', s=35, marker='X',
        label='DP-PMS (ε=1)'
    )
    ax2.set_title('DP-PMS Modal Regression (Sinusoidal Mixture)')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_xlim(0, 1)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()
    plt.tight_layout()
    plt.show()

    outpath2 = os.path.join(results_dir, "baseline_dp_pms_eps1.png")
    fig2.savefig(outpath2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[saved] DP-PMS baseline -> {outpath2}")

    # ------------------------------------------------------------
    # Privacy–utility experiments across (n, ε)
    # ------------------------------------------------------------
    print("\n[PU] Starting privacy–utility experiments (sinusoidal mixture)...")
    results = []
    stats_summary = "=== Privacy–Utility Experiments (DP-PMS vs PMS, Sinusoidal Mixture) ===\n"

    for n in n_values:
        print(f"\n[PU] --- n = {n} ---")
        # Fixed dataset for given n
        X_data, Y_data = generate_sin_data(n, sigma_noise, seed=123)

        # PMS baseline at this n
        T_pms = int(np.log(n))
        Y_pms_est = partial_mean_shift(X_data, Y_data, mesh_points=None, bandwidth=None, T=T_pms)

        # Treat PMS estimate as reference modal regression curve
        pms_ref = np.column_stack([X_data, Y_pms_est])
        pms_mse = mode_matching_mse(pms_ref, pms_ref)  # = 0.0; kept for table consistency

        for eps in eps_values:
            print(f"[PU] n={n}, eps={eps}: launching {n_runs} DP-PMS runs...")
            seeds = [1000 * n + 10 * int(eps * 10) + run for run in range(n_runs)]
            dp_mses, dp_times = [], []

            # Parallel DP-PMS runs
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        dp_pms_worker,
                        X_data, Y_data,
                        float(eps), 1e-5,
                        seed
                    ): seed
                    for seed in seeds
                }

                for fut in as_completed(futures):
                    try:
                        seed_ret, runtime_ret, Y_dp_out, mesh_ret = fut.result()
                    except Exception as e:
                        print(f"[PU][warn] n={n}, eps={eps}: a worker failed: {e}")
                        continue

                    est_modes = np.column_stack([mesh_ret, Y_dp_out])
                    dp_mse_val = mode_matching_mse(pms_ref, est_modes)
                    dp_mses.append(dp_mse_val)
                    dp_times.append(runtime_ret)

                    # Log per-run result
                    with open(perrun_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            n,
                            eps,
                            seeds.index(seed_ret),
                            seed_ret,
                            dp_mse_val,
                            runtime_ret,
                            now_str()
                        ])

            if len(dp_mses) == 0:
                print(f"[PU][warn] n={n}, eps={eps}: no successful DP-PMS runs.")
                continue

            dp_mean = float(np.mean(dp_mses))
            dp_se = float(np.std(dp_mses, ddof=1) / np.sqrt(len(dp_mses)))
            dp_time_mean = float(np.mean(dp_times))
            dp_time_std = float(np.std(dp_times, ddof=1) if len(dp_times) > 1 else 0.0)

            results.append([
                n, eps,
                dp_mean, dp_se,
                pms_mse,
                dp_time_mean, dp_time_std
            ])

            msg = (
                f"[PU] n={n}, eps={eps}: "
                f"DP-MSE={dp_mean:.4f}±{dp_se:.4f}, "
                f"PMS-MSE={pms_mse:.4f}, "
                f"DP-time={dp_time_mean:.2f}±{dp_time_std:.2f}s"
            )
            print(msg)
            stats_summary += msg + "\n"

    # Save numeric summary
    csv_path = os.path.join(results_dir, "privacy_utility_results_sin_simplified.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_samples",
            "epsilon",
            "DP_mse_mean",
            "DP_mse_se",
            "PMS_mse",
            "DP_runtime_mean",
            "DP_runtime_std"
        ])
        writer.writerows(results)
    save_text(os.path.join(results_dir, "stats_summary.txt"), stats_summary)
    print(f"\n[saved] Summary CSV -> {csv_path}")
    print("[saved] Stats summary -> stats_summary.txt")

    # ------------------------------------------------------------
    # Privacy–utility plot (separate figure)
    # ------------------------------------------------------------
    if len(results) > 0:
        print("[PU] Generating privacy–utility plot (sinusoidal mixture)...")
        fig, ax = plt.subplots(figsize=(8, 6))
        palette = sns.color_palette("Set2", len(n_values))

        for i, n in enumerate(n_values):
            eps_subset = [r[1] for r in results if r[0] == n]
            mse_subset = [r[2] for r in results if r[0] == n]
            mse_errs = [r[3] for r in results if r[0] == n]
            pms_mse_val = next((r[4] for r in results if r[0] == n), np.nan)

            if len(eps_subset) == 0:
                continue

            # DP-PMS curve
            ax.errorbar(
                eps_subset, mse_subset,
                yerr=mse_errs,
                marker='o', linestyle='-',
                linewidth=2, markersize=7,
                capsize=3,
                label=f"DP-PMS (n={n})",
                color=palette[i]
            )

            # PMS baseline (horizontal line)
            if not np.isnan(pms_mse_val):
                ax.hlines(
                    pms_mse_val,
                    min(eps_subset), max(eps_subset),
                    colors=palette[i],
                    linestyles='dashed',
                    linewidth=1.5,
                    label=f"PMS baseline (n={n})"
                )

        ax.set_xlabel("Privacy budget ε")
        ax.set_xscale('log')
        ax.set_ylabel("MSE")
        ax.set_title("Privacy–Utility Tradeoff in Sinusoidal 2-Mixture Data")
        ax.grid(axis='y', linestyle="--", alpha=0.4)
        ax.legend(ncol=1, loc="upper right")
        plt.tight_layout()
        plt.show()

        pu_path = os.path.join(results_dir, "privacy_utility_modal_sin_simplified.png")
        fig.savefig(pu_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] Privacy–utility plot -> {pu_path}")

    print("\n[done] Modal regression DP-PMS sinusoidal experiments complete.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
