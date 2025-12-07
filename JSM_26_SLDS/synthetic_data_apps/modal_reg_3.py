# modal_reg_3.py
#
# Modal regression experiments (3-component mixture):
# - Baseline PMS vs LOWESS (non-private)
# - DP-PMS behavior (synthetic_data_apps.dp_pms)
# - Privacy-utility tradeoff across n and epsilon
# - Hyperparameter sweeps for DP-PMS: MSE vs C^* and MSE vs m

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys
import csv
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure local imports work when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synthetic_data_apps.dp_pms import dp_pms
from pms import partial_mean_shift
from main_scripts.mode_matching_mse import mode_matching_mse
from statsmodels.nonparametric.smoothers_lowess import lowess

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

# True modal levels for the 3-component mixture
true_modes_list = [3.0, 2.0, 1.0]

# ---------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------

def generate_modal_data(n, sd=0.2, seed=None):
    """
    3-component mixture with distinct X ranges and modal Y levels:
      - Component 1: X in [0, 0.5],    Y ~ N(3, sd^2)
      - Component 2: X in [0.4, 0.7],  Y ~ N(2, sd^2)
      - Component 3: X in [0.6, 1.0],  Y ~ N(1, sd^2)
    """
    rng = np.random.default_rng(seed)
    n1, n2 = n // 3, n // 3
    n3 = n - n1 - n2

    x1 = rng.uniform(0.0, 0.5, n1)
    x2 = rng.uniform(0.4, 0.7, n2)
    x3 = rng.uniform(0.6, 1.0, n3)

    y1 = rng.normal(true_modes_list[0], sd, n1)
    y2 = rng.normal(true_modes_list[1], sd, n2)
    y3 = rng.normal(true_modes_list[2], sd, n3)

    X = np.concatenate([x1, x2, x3])
    Y = np.concatenate([y1, y2, y3])
    return X, Y

def make_modal_data_with_pms(n, sd, seed):
    """
    Helper for hyperparameter sweeps:
    generate (X, Y), compute PMS reference curve, and return (X, Y, pms_ref, T_pms).
    """
    X, Y = generate_modal_data(n, sd=sd, seed=seed)
    T_pms = int(np.ceil(np.log(max(2, n))))
    Y_pms = partial_mean_shift(X, Y, mesh_points=None, bandwidth=None, T=T_pms)
    pms_ref = np.column_stack([X, Y_pms])
    return X, Y, pms_ref, T_pms

# ---------------------------------------------------------------------
# Global experiment settings
# ---------------------------------------------------------------------

results_dir = "results/modal_regression_3"
ensure_dir(results_dir)

n_values = [300, 600, 1200, 2100]
eps_values = [0.1, 0.2, 0.5, 1.0]
n_runs = 20
MAX_WORKERS = min(8, os.cpu_count() or 8)
DELTA_DEFAULT = 1e-5

# Hyperparam sweep settings (analogous to bivariate_4mix)
clip_grid = [0.01, 0.02, 0.05, 0.1, 0.25]         # C^* grid
m_grid_frac = [0.01, 0.05, 0.1, 0.2, 1.0]       # fractions of n
epsilon_hparam = 1.0                            # eps used for sweeps
n_reps_hparam = 20                              # repetitions per grid point
sd_default = 0.2                                # noise level in generator

# Per-run CSV for DP-PMS (privacy-utility section)
perrun_csv = os.path.join(results_dir, "dp_per_run_results.csv")
if not os.path.exists(perrun_csv):
    with open(perrun_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_samples", "epsilon", "run_idx",
            "seed", "dp_mse", "dp_runtime_s", "timestamp"
        ])

# ---------------------------------------------------------------------
# Workers / helpers
# ---------------------------------------------------------------------

def dp_pms_worker(X, Y, epsilon, delta, seed, T_iter=None, verbose=False):
    """
    Run DP-PMS once with given seed, return (seed, runtime, Y_dp, X_copy).
    Uses default m (n/log n) and clip_multiplier from dp_pms.
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

def _run_dp_pms_mse_single(
    X,
    Y,
    pms_ref,
    epsilon,
    delta,
    T,
    m,
    clip_multiplier,
    seed,
):
    """
    Single DP-PMS run for hyperparameter sweeps, returning (MSE, runtime).
    - X, Y: data
    - pms_ref: PMS reference curve as array of shape (n, 2) with columns (X, Y_pms)
    - epsilon, delta: privacy parameters
    - T: iterations
    - m: minibatch size
    - clip_multiplier: C^* scaling
    """
    rng = np.random.default_rng(int(seed))
    t0 = time.perf_counter()

    Y_dp = dp_pms(
        X, Y,
        mesh_points=Y.copy(),
        epsilon=float(epsilon),
        delta=float(delta),
        T=int(T),
        m=int(m),
        clip_multiplier=float(clip_multiplier),
        rng=rng,
        verbose=False
    )
    runtime = time.perf_counter() - t0

    est_modes = np.column_stack([X, Y_dp])
    mse = mode_matching_mse(pms_ref, est_modes)
    return float(mse), float(runtime)

def sweep_clip_multiplier_modal(
    n_list,
    clip_values,
    epsilon,
    delta,
    sd=0.2,
    n_reps=20,
    base_seed=12345,
):
    """
    Hyperparameter sweep over clip_multiplier (C^*) for DP-PMS in the
    3-component modal regression setting, across n in n_list.

    Returns
    -------
    clip_results_by_n : dict
        Maps n -> list of dicts, each with keys:
          "clip_multiplier", "mean_mse", "std_mse",
          "min_mse", "max_mse", "mean_time", "std_time"
    """
    rng = np.random.default_rng(base_seed)
    clip_results_by_n = {}

    for idx, n in enumerate(n_list):
        seed_n = int(rng.integers(0, 2**31 - 1))
        X, Y, pms_ref, T_pms = make_modal_data_with_pms(n, sd=sd, seed=seed_n)
        T_use = int(np.ceil(np.log(max(2, n))))  # DP-PMS iterations
        m_default = max(1, int(n / max(np.log(max(2, n)), 1.0)))  # baseline m ~ n/log n

        print(
            f"[hyperparam][clip] n={n}, T={T_use}, m_default={m_default}, "
            f"seed_data={seed_n}"
        )

        results_n = []
        for cm in clip_values:
            mses, times = [], []
            for rep in range(n_reps):
                seed_run = int(rng.integers(0, 2**31 - 1))
                mse, rt = _run_dp_pms_mse_single(
                    X=X,
                    Y=Y,
                    pms_ref=pms_ref,
                    epsilon=epsilon,
                    delta=delta,
                    T=T_use,
                    m=m_default,
                    clip_multiplier=cm,
                    seed=seed_run,
                )
                mses.append(mse)
                times.append(rt)

            mses = np.asarray(mses, dtype=float)
            times = np.asarray(times, dtype=float)
            results_n.append({
                "clip_multiplier": float(cm),
                "mean_mse": float(np.nanmean(mses)),
                "std_mse": float(np.nanstd(mses)),
                "min_mse": float(np.nanmin(mses)),
                "max_mse": float(np.nanmax(mses)),
                "mean_time": float(np.nanmean(times)),
                "std_time": float(np.nanstd(times)),
            })

        results_n.sort(key=lambda r: r["clip_multiplier"])
        clip_results_by_n[n] = results_n

    return clip_results_by_n

def sweep_minibatch_size_modal(
    n_list,
    m_frac_grid,
    epsilon,
    delta,
    sd=0.2,
    n_reps=20,
    base_seed=54321,
):
    """
    Hyperparameter sweep over minibatch size m for DP-PMS in the
    3-component modal regression setting, across n in n_list.

    Returns
    -------
    m_results_by_n : dict
        Maps n -> list of dicts, each with keys:
          "m", "mean_mse", "std_mse",
          "min_mse", "max_mse", "mean_time", "std_time"
    """
    rng = np.random.default_rng(base_seed)
    m_results_by_n = {}

    for idx, n in enumerate(n_list):
        seed_n = int(rng.integers(0, 2**31 - 1))
        X, Y, pms_ref, T_pms = make_modal_data_with_pms(n, sd=sd, seed=seed_n)
        T_use = int(np.ceil(np.log(max(2, n))))
        m_grid = sorted(set(max(1, int(frac * n)) for frac in m_frac_grid))

        print(
            f"[hyperparam][minibatch] n={n}, T={T_use}, "
            f"m_grid={m_grid}, seed_data={seed_n}"
        )

        results_n = []
        for m_val in m_grid:
            mses, times = [], []
            for rep in range(n_reps):
                seed_run = int(rng.integers(0, 2**31 - 1))
                mse, rt = _run_dp_pms_mse_single(
                    X=X,
                    Y=Y,
                    pms_ref=pms_ref,
                    epsilon=epsilon,
                    delta=delta,
                    T=T_use,
                    m=m_val,
                    clip_multiplier=0.01,  # baseline C^* for m-sweep
                    seed=seed_run,
                )
                mses.append(mse)
                times.append(rt)

            mses = np.asarray(mses, dtype=float)
            times = np.asarray(times, dtype=float)
            results_n.append({
                "m": int(m_val),
                "mean_mse": float(np.nanmean(mses)),
                "std_mse": float(np.nanstd(mses)),
                "min_mse": float(np.nanmin(mses)),
                "max_mse": float(np.nanmax(mses)),
                "mean_time": float(np.nanmean(times)),
                "std_time": float(np.nanstd(times)),
            })

        results_n.sort(key=lambda r: r["m"])
        m_results_by_n[n] = results_n

    return m_results_by_n

def save_clip_multiplier_results_txt(results_by_n, clip_values, path):
    ensure_dir(os.path.dirname(path))
    lines = []
    for n in sorted(results_by_n.keys()):
        lines.append(f"=== n = {n} ===")
        header = (
            f"{'clip_mult':>10} | {'mean_mse':>10} | {'std_mse':>10} | "
            f"{'min_mse':>10} | {'max_mse':>10} | {'mean_t':>10} | {'std_t':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for r in results_by_n[n]:
            line = (
                f"{r['clip_multiplier']:10.3f} | "
                f"{r['mean_mse']:10.4f} | "
                f"{r['std_mse']:10.4f} | "
                f"{r['min_mse']:10.4f} | "
                f"{r['max_mse']:10.4f} | "
                f"{r['mean_time']:10.4f} | "
                f"{r['std_time']:10.4f}"
            )
            lines.append(line)
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] Clip-multiplier sweep results -> {path}")

def save_minibatch_results_txt(results_by_n, path):
    ensure_dir(os.path.dirname(path))
    lines = []
    for n in sorted(results_by_n.keys()):
        lines.append(f"=== n = {n} ===")
        header = (
            f"{'m':>10} | {'mean_mse':>10} | {'std_mse':>10} | "
            f"{'min_mse':>10} | {'max_mse':>10} | {'mean_t':>10} | {'std_t':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for r in results_by_n[n]:
            line = (
                f"{r['m']:10d} | "
                f"{r['mean_mse']:10.4f} | "
                f"{r['std_mse']:10.4f} | "
                f"{r['min_mse']:10.4f} | "
                f"{r['max_mse']:10.4f} | "
                f"{r['mean_time']:10.4f} | "
                f"{r['std_time']:10.4f}"
            )
            lines.append(line)
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] Minibatch sweep results -> {path}")

def plot_clip_multiplier_results_multi_n_modal(clip_results_by_n, clip_values, n_list, out_path):
    """
    Plot MSE vs C^* across n (single panel), mirroring bivariate_4mix style.
    """
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("Set2", len(n_list))

    for idx, n in enumerate(n_list):
        results = clip_results_by_n.get(n, [])
        stats_by_cm = {r["clip_multiplier"]: r for r in results}

        xs, ys, yerr = [], [], []
        for cm in clip_values:
            if cm in stats_by_cm:
                xs.append(cm)
                ys.append(stats_by_cm[cm]["mean_mse"])
                yerr.append(stats_by_cm[cm]["std_mse"])

        if not xs:
            continue

        plt.errorbar(
            xs, ys, yerr=yerr,
            marker="o", linestyle="-",
            label=f"n={n}",
            color=palette[idx],
            capsize=3
        )

    plt.xlabel("Clip Multiplier ($C^*$ scaling)")
    plt.ylabel("MSE")
    plt.title("MSE vs $C^*$ across n for 3-Component Modal Regression")
    plt.grid(True, alpha=0.4)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[saved] Multi-n clip-multiplier MSE plot -> {out_path}")
    plt.show()
    plt.close()

def plot_minibatch_results_grid_modal(m_results_by_n, n_list, out_path):
    """
    2x2 grid: MSE vs minibatch size m for different n (mirrors bivariate_4mix).
    """
    ensure_dir(os.path.dirname(out_path))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()
    palette = sns.color_palette("Set2", len(n_list))

    for idx, n in enumerate(n_list):
        if idx >= len(axes):
            break
        ax = axes[idx]
        results = m_results_by_n.get(n, [])
        m_vals = [r["m"] for r in results]
        mean_mses = [r["mean_mse"] for r in results]
        std_mses = [r["std_mse"] for r in results]

        if not m_vals:
            ax.set_visible(False)
            continue

        ax.errorbar(
            m_vals, mean_mses, yerr=std_mses,
            marker="o", linestyle="-",
            color=palette[idx],
            capsize=3
        )
        ax.set_title(f"n = {n}")
        ax.grid(True, alpha=0.4)
        ax.set_xscale("log")

    axes[0].set_ylabel("MSE")
    axes[2].set_ylabel("MSE")
    axes[2].set_xlabel("Minibatch size m")
    axes[3].set_xlabel("Minibatch size m")

    fig.suptitle("MSE vs m across n for 3-Component DP Modal Regression", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[saved] Minibatch-size MSE grid plot -> {out_path}")
    plt.show()
    plt.close()

# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():
    # ------------------------------------------------------------
    # Baseline visualization on a fixed dataset
    # ------------------------------------------------------------
    Xb, Yb = generate_modal_data(500, sd=0.2, seed=42)

    # Non-private PMS
    T_base = int(np.ceil(np.log(500)))
    Y_pms_truth = partial_mean_shift(Xb, Yb, mesh_points=None, bandwidth=None, T=T_base)

    # DP-PMS example (epsilon=1), with m = n for the single-run visualization
    Y_dp_baseline = dp_pms(
        Xb, Yb,
        mesh_points=Yb.copy(),
        epsilon=1.0,
        delta=DELTA_DEFAULT,
        T=T_base,
        # m=len(Xb),                 # full-batch DP-PMS for the demo plot
        rng=np.random.default_rng(42),
        verbose=False
    )

    # LOWESS smooth (non-modal baseline)
    Y_lowess = lowess(Yb, Xb, frac=0.2, return_sorted=False)

    # ----- Figure 1: PMS vs LOWESS (non-private modal regression) -----
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(Xb, Yb, color='skyblue', alpha=0.5, s=15, label='Data')
    ax1.scatter(Xb, Y_pms_truth, color='#1f77b4', s=25, marker='D', label='PMS')
    ax1.plot(
        np.sort(Xb),
        Y_lowess[np.argsort(Xb)],
        color='orange', linewidth=2,
        label='LOWESS'
    )
    ax1.set_title("Modal Regression for 3-Component Mixture")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 4)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()
    plt.tight_layout()
    plt.show()

    outpath1 = os.path.join(results_dir, "pms_vs_lowess_modal_regression.pdf")
    fig1.savefig(outpath1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"[saved] PMS vs LOWESS -> {outpath1}")

    # ----- Figure 2: DP-PMS baseline (epsilon=1, m=n) -----
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(Xb, Yb, color='skyblue', alpha=0.5, s=15, label='Data')
    ax2.scatter(
        Xb, Y_dp_baseline,
        color='#d95f02', s=35, marker='X',
        label='DP-PMS ($\epsilon$=1)'
    )
    ax2.set_title("Differentially Private Modal Regression for 3-Component Mixture")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 4)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()
    plt.tight_layout()
    plt.show()

    outpath2 = os.path.join(results_dir, "dp_pms_modal_regression_eps1_full_batch.pdf")
    fig2.savefig(outpath2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[saved] DP-PMS baseline -> {outpath2}")

    # ------------------------------------------------------------
    # Privacy-utility experiments across (n, epsilon)
    # ------------------------------------------------------------
    print("\n[PU] Starting privacy-utility experiments...")
    results = []
    stats_summary = "=== Privacy-Utility Experiments (DP-PMS vs PMS, 3-Component Mixture) ===\n"

    for n in n_values:
        print(f"\n[PU] --- n = {n} ---")
        # Fixed dataset for given n
        X_data, Y_data = generate_modal_data(n, sd=sd_default, seed=123)

        # PMS baseline at this n
        T_pms = int(np.ceil(np.log(n)))
        Y_pms_est = partial_mean_shift(X_data, Y_data, mesh_points=None, bandwidth=None, T=T_pms)

        # Treat PMS estimate as "reference" modal regression curve
        pms_ref = np.column_stack([X_data, Y_pms_est])
        pms_mse = mode_matching_mse(pms_ref, pms_ref)  # = 0.0; kept for table consistency

        for eps in eps_values:
            print(f"[PU] n={n}, eps={eps}: launching {n_runs} DP-PMS runs...")
            seeds = [1000 * n + 10 * int(eps * 10) + run for run in range(n_runs)]
            dp_mses, dp_times = [], []

            # Parallel DP-PMS runs (default m ~ n/log n, clip_multiplier=0.01)
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        dp_pms_worker,
                        X_data, Y_data,
                        float(eps), DELTA_DEFAULT,
                        seed
                    ): seed
                    for seed in seeds
                }

                for fut in as_completed(futures):
                    try:
                        seed_ret, runtime_ret, Y_refined, mesh_ret = fut.result()
                    except Exception as e:
                        print(f"[PU][warn] n={n}, eps={eps}: a worker failed: {e}")
                        continue

                    # Compare DP-PMS outputs to PMS reference (same X grid)
                    est_modes = np.column_stack([mesh_ret, Y_refined])
                    dp_mse = mode_matching_mse(pms_ref, est_modes)
                    dp_mses.append(dp_mse)
                    dp_times.append(runtime_ret)

                    # Log per-run result
                    with open(perrun_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            n,
                            eps,
                            seeds.index(seed_ret),
                            seed_ret,
                            dp_mse,
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
    csv_path = os.path.join(results_dir, "privacy_utility_results_modal_regression.csv")
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
    save_text(os.path.join(results_dir, "stats_summary_modal_regression.txt"), stats_summary)
    print(f"\n[saved] Summary CSV -> {csv_path}")
    print("[saved] Stats summary -> stats_summary_modal_regression.txt")

    # ------------------------------------------------------------
    # Privacy-utility plot (separate figure)
    # ------------------------------------------------------------
    if len(results) > 0:
        print("[PU] Generating privacy-utility plot...")
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
                label=f"DP-PMS n={n}",
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
                    label=f"PMS n={n}"
                )

        ax.set_xlabel("Privacy budget $\epsilon$")
        ax.set_xscale('log')
        ax.set_ylabel("MSE")
        ax.set_title("Privacy-Utility Tradeoff for 3-Component DP Modal Regression")
        ax.grid(axis='y', linestyle="--", alpha=0.4)
        ax.legend(ncol=1, loc="upper right")
        plt.tight_layout()
        plt.show()

        pu_path = os.path.join(results_dir, "privacy_utility_pretty_modal_regression.pdf")
        fig.savefig(pu_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] Privacy-utility plot -> {pu_path}")

    # ------------------------------------------------------------
    # Hyperparameter sweeps: MSE vs C^* and vs m
    # ------------------------------------------------------------
    print("\n[hyperparam] Starting DP-PMS hyperparameter sweeps (C^*, m)...")

    # --- MSE vs C^* (clip_multiplier) across n ---
    clip_results_by_n = sweep_clip_multiplier_modal(
        n_list=n_values,
        clip_values=clip_grid,
        epsilon=epsilon_hparam,
        delta=DELTA_DEFAULT,
        sd=sd_default,
        n_reps=n_reps_hparam,
        base_seed=12345,
    )
    clip_txt_path = os.path.join(results_dir, "mse_vs_clip_multiplier_multi_n_modal.txt")
    save_clip_multiplier_results_txt(clip_results_by_n, clip_grid, clip_txt_path)

    clip_plot_path = os.path.join(results_dir, "mse_vs_clip_multiplier_multi_n_modal.pdf")
    plot_clip_multiplier_results_multi_n_modal(
        clip_results_by_n,
        clip_grid,
        n_values,
        clip_plot_path
    )

    # --- MSE vs minibatch size m across n ---
    m_results_by_n = sweep_minibatch_size_modal(
        n_list=n_values,
        m_frac_grid=m_grid_frac,
        epsilon=epsilon_hparam,
        delta=DELTA_DEFAULT,
        sd=sd_default,
        n_reps=n_reps_hparam,
        base_seed=54321,
    )
    m_txt_path = os.path.join(results_dir, "mse_vs_minibatch_grid_modal.txt")
    save_minibatch_results_txt(m_results_by_n, m_txt_path)

    m_plot_path = os.path.join(results_dir, "mse_vs_minibatch_grid_modal.pdf")
    plot_minibatch_results_grid_modal(
        m_results_by_n,
        n_values,
        m_plot_path
    )

    print("\n[done] Modal regression DP-PMS experiments (including C^* and m sweeps) complete.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
