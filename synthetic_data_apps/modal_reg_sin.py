# # modal_reg_sin.py

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synthetic_data_apps.dp_pms import dp_pms
from pms import partial_mean_shift
from main_scripts.mode_matching_mse import mode_matching_mse

# ---------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------

sns.set_context("talk")
sns.set_style("white")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
    }
)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def ensure_dir(path: str):
    """
    Create directory path if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)


def save_text(path: str, text: str):
    """
    Save plain text to disk, creating parent directory if needed.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(text)


def now_str():
    """
    Human-readable timestamp for logs / CSV rows.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_std(arr):
    """
    Standard deviation with ddof=1 when possible; otherwise 0.
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def safe_se(arr):
    """
    Standard error using sample SD / sqrt(n), with ddof=1 when possible.
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def sort_xy_for_line_plot(x, y):
    """
    Return x and y sorted by x, useful for clean line plotting.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    idx = np.argsort(x)
    return x[idx], y[idx]


# ---------------------------------------------------------------------
# Synthetic data generator: sinusoidal 2-component mixture
# ---------------------------------------------------------------------

sigma_noise = 0.15


def generate_sin_data(n, sigma=sigma_noise, seed=None):
    """
    Sinusoidal 2-component mixture:
      - Half samples: Y ~ N(1.5 + 0.5 sin(3πX), sigma^2)
      - Half samples: Y ~ N(0.5 sin(3πX), sigma^2)
      with X ~ Unif(0,1) independently for each component.

    Parameters
    ----------
    n : int
        Number of total samples.
    sigma : float
        Gaussian noise level.
    seed : int or None
        RNG seed.

    Returns
    -------
    X : ndarray shape (n,)
    Y : ndarray shape (n,)
    """
    rng = np.random.default_rng(seed)

    n1 = n // 2
    n2 = n - n1

    X1 = rng.uniform(0.0, 1.0, n1)
    X2 = rng.uniform(0.0, 1.0, n2)
    X = np.concatenate([X1, X2])

    Y1 = rng.normal(1.5 + 0.5 * np.sin(3 * np.pi * X1), sigma, n1)
    Y2 = rng.normal(0.5 * np.sin(3 * np.pi * X2), sigma, n2)
    Y = np.concatenate([Y1, Y2])

    return X, Y


def make_sin_data_with_pms(n, sigma, seed):
    """
    Helper used in hyperparameter sweeps:
      1) generate sinusoidal data
      2) compute PMS reference curve
      3) return (X, Y, pms_ref, T_pms)

    Parameters
    ----------
    n : int
    sigma : float
    seed : int

    Returns
    -------
    X : ndarray
    Y : ndarray
    pms_ref : ndarray of shape (n, 2)
        Columns are (X, Y_pms)
    T_pms : int
        PMS iteration count
    """
    X, Y = generate_sin_data(n, sigma=sigma, seed=seed)
    T_pms = int(np.ceil(np.log(max(2, n))))
    Y_pms = partial_mean_shift(
        X,
        Y,
        mesh_points=None,
        bandwidth=None,
        T=T_pms,
    )
    pms_ref = np.column_stack([X, Y_pms])
    return X, Y, pms_ref, T_pms


# ---------------------------------------------------------------------
# Global experiment settings
# ---------------------------------------------------------------------

results_dir = "results/modal_regression_sin"
ensure_dir(results_dir)

n_values = [200, 500, 1000, 2000]
eps_values = [0.1, 0.2, 0.5, 1.0]
n_runs = 20
MAX_WORKERS = min(8, os.cpu_count() or 8)
DELTA_DEFAULT = 1e-5

# Hyperparameter sweep settings
clip_grid = [0.01, 0.02, 0.05, 0.1, 0.25]
m_grid_frac = [0.01, 0.05, 0.1, 0.2, 1.0]
epsilon_hparam = 1.0
n_reps_hparam = 20
sigma_default = sigma_noise

# ---------------------------------------------------------------------
# Per-run CSVs
# ---------------------------------------------------------------------
dp_perrun_csv = os.path.join(results_dir, "dp_per_run_results.csv")
if not os.path.exists(dp_perrun_csv):
    with open(dp_perrun_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_samples",
                "epsilon",
                "run_idx",
                "data_seed",
                "dp_seed",
                "dp_mse",
                "dp_runtime_s",
                "timestamp",
            ]
        )

# PMS per-run CSV
pms_perrun_csv = os.path.join(results_dir, "pms_per_run_results.csv")
if not os.path.exists(pms_perrun_csv):
    with open(pms_perrun_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_samples",
                "epsilon",
                "run_idx",
                "data_seed",
                "pms_mse",
                "pms_runtime_s",
                "timestamp",
            ]
        )

paired_perrun_csv = os.path.join(results_dir, "paired_pms_dp_per_run_results.csv")
if not os.path.exists(paired_perrun_csv):
    with open(paired_perrun_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_samples",
                "epsilon",
                "run_idx",
                "data_seed",
                "dp_seed",
                "pms_mse",
                "pms_runtime_s",
                "dp_mse",
                "dp_runtime_s",
                "timestamp",
            ]
        )

# ---------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------


def pms_worker_on_dataset(X, Y, T_iter=None):
    """
    Run non-private PMS once on a provided dataset and return:
        (runtime, Y_pms, X_copy)

    Parameters
    ----------
    X, Y : ndarrays
    T_iter : int or None

    Returns
    -------
    runtime : float
    Y_pms : ndarray
    X_copy : ndarray
    """
    n = len(X)
    T_use = T_iter if T_iter is not None else int(np.ceil(np.log(max(2, n))))

    t0 = time.perf_counter()
    Y_pms = partial_mean_shift(
        X,
        Y,
        mesh_points=None,
        bandwidth=None,
        T=int(T_use),
    )
    runtime = time.perf_counter() - t0
    return float(runtime), Y_pms, X.copy()


def dp_pms_worker_on_dataset(
    X,
    Y,
    epsilon,
    delta,
    seed,
    T_iter=None,
    m=None,
    clip_multiplier=None,
    verbose=False,
):
    """
    Run DP-PMS once on a provided dataset and return:
        (seed, runtime, Y_dp, X_copy)

    Parameters
    ----------
    X, Y : ndarrays
    epsilon, delta : float
    seed : int
    T_iter : int or None
    m : int or None
    clip_multiplier : float or None
    verbose : bool

    Returns
    -------
    seed : int
    runtime : float
    Y_dp : ndarray
    X_copy : ndarray
    """
    rng = np.random.default_rng(int(seed))
    n = len(X)
    T_use = T_iter if T_iter is not None else int(np.ceil(np.log(max(2, n))))

    kwargs = dict(
        epsilon=float(epsilon),
        delta=float(delta),
        T=int(T_use),
        rng=rng,
        verbose=verbose,
    )
    if m is not None:
        kwargs["m"] = int(m)
    if clip_multiplier is not None:
        kwargs["clip_multiplier"] = float(clip_multiplier)

    t0 = time.perf_counter()
    Y_dp = dp_pms(
        X,
        Y,
        **kwargs,
    )
    runtime = time.perf_counter() - t0

    return int(seed), float(runtime), Y_dp, X.copy()


def paired_pms_dp_worker(
    n,
    sigma,
    epsilon,
    delta,
    data_seed,
    dp_seed,
    T_iter=None,
    dp_m=None,
    dp_clip_multiplier=None,
):
    """
    Paired worker for one privacy-utility run:
      1) generate a fresh dataset using data_seed
      2) run PMS on that dataset
      3) run DP-PMS on that same dataset
      4) compute DP-MSE against the PMS reference curve
      5) return runtimes and MSEs

    Returns
    -------
    out : dict
        Keys:
          - n
          - epsilon
          - data_seed
          - dp_seed
          - T_use
          - pms_mse
          - pms_runtime
          - dp_mse
          - dp_runtime
    """
    # Generate data for this run
    X_run, Y_run = generate_sin_data(
        n=n,
        sigma=sigma,
        seed=int(data_seed),
    )

    T_use = T_iter if T_iter is not None else int(np.ceil(np.log(max(2, n))))

    # PMS
    pms_t0 = time.perf_counter()
    Y_pms = partial_mean_shift(
        X_run,
        Y_run,
        mesh_points=None,
        bandwidth=None,
        T=int(T_use),
    )
    pms_runtime = time.perf_counter() - pms_t0
    pms_ref = np.column_stack([X_run, Y_pms])

    # PMS vs itself = 0 by construction, included for table consistency
    pms_mse = mode_matching_mse(pms_ref, pms_ref)

    # DP-PMS
    rng_dp = np.random.default_rng(int(dp_seed))
    dp_kwargs = dict(
        epsilon=float(epsilon),
        delta=float(delta),
        T=int(T_use),
        rng=rng_dp,
        verbose=False,
    )
    if dp_m is not None:
        dp_kwargs["m"] = int(dp_m)
    if dp_clip_multiplier is not None:
        dp_kwargs["clip_multiplier"] = float(dp_clip_multiplier)

    dp_t0 = time.perf_counter()
    Y_dp = dp_pms(
        X_run,
        Y_run,
        **dp_kwargs,
    )
    dp_runtime = time.perf_counter() - dp_t0

    est_modes = np.column_stack([X_run, Y_dp])
    dp_mse = mode_matching_mse(pms_ref, est_modes)

    return {
        "n": int(n),
        "epsilon": float(epsilon),
        "data_seed": int(data_seed),
        "dp_seed": int(dp_seed),
        "T_use": int(T_use),
        "pms_mse": float(pms_mse),
        "pms_runtime": float(pms_runtime),
        "dp_mse": float(dp_mse),
        "dp_runtime": float(dp_runtime),
    }


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

    Parameters
    ----------
    X, Y : ndarrays
        Data for a fixed experimental dataset.
    pms_ref : ndarray
        PMS reference curve for this fixed dataset.
    epsilon, delta : float
        Privacy parameters.
    T : int
        Number of iterations.
    m : int
        Minibatch size.
    clip_multiplier : float
        Clip multiplier.
    seed : int
        DP RNG seed.

    Returns
    -------
    mse : float
    runtime : float
    """
    rng = np.random.default_rng(int(seed))

    t0 = time.perf_counter()
    Y_dp = dp_pms(
        X,
        Y,
        epsilon=float(epsilon),
        delta=float(delta),
        T=int(T),
        m=int(m),
        clip_multiplier=float(clip_multiplier),
        rng=rng,
        verbose=False,
    )
    runtime = time.perf_counter() - t0

    est_modes = np.column_stack([X, Y_dp])
    mse = mode_matching_mse(pms_ref, est_modes)
    return float(mse), float(runtime)


# ---------------------------------------------------------------------
# Hyperparameter sweeps
# ---------------------------------------------------------------------


def sweep_clip_multiplier_modal(
    n_list,
    clip_values,
    epsilon,
    delta,
    sigma=sigma_default,
    n_reps=20,
    base_seed=12345,
):
    """
    Sweep over clip_multiplier (C^*) for DP-PMS in the sinusoidal
    2-component modal regression setting, across n in n_list.

    Returns
    -------
    clip_results_by_n : dict
        Maps n -> list of dicts.
    """
    rng = np.random.default_rng(base_seed)
    clip_results_by_n = {}

    for idx, n in enumerate(n_list):
        seed_n = int(rng.integers(0, 2**31 - 1))
        X, Y, pms_ref, T_pms = make_sin_data_with_pms(
            n=n,
            sigma=sigma,
            seed=seed_n,
        )
        T_use = int(np.ceil(np.log(max(2, n))))
        m_default = max(1, int(n / max(np.log(max(2, n)), 1.0)))

        print(
            f"[hyperparam][clip-sin] n={n}, T={T_use}, m_default={m_default}, "
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

            results_n.append(
                {
                    "clip_multiplier": float(cm),
                    "mean_mse": float(np.nanmean(mses)),
                    "std_mse": float(np.nanstd(mses)),
                    "min_mse": float(np.nanmin(mses)),
                    "max_mse": float(np.nanmax(mses)),
                    "mean_time": float(np.nanmean(times)),
                    "std_time": float(np.nanstd(times)),
                }
            )

        results_n.sort(key=lambda r: r["clip_multiplier"])
        clip_results_by_n[n] = results_n

    return clip_results_by_n


def sweep_minibatch_size_modal(
    n_list,
    m_frac_grid,
    epsilon,
    delta,
    sigma=sigma_default,
    n_reps=20,
    base_seed=54321,
):
    """
    Sweep over minibatch size m for DP-PMS in the sinusoidal
    2-component modal regression setting, across n in n_list.

    Returns
    -------
    m_results_by_n : dict
        Maps n -> list of dicts.
    """
    rng = np.random.default_rng(base_seed)
    m_results_by_n = {}

    for idx, n in enumerate(n_list):
        seed_n = int(rng.integers(0, 2**31 - 1))
        X, Y, pms_ref, T_pms = make_sin_data_with_pms(
            n=n,
            sigma=sigma,
            seed=seed_n,
        )
        T_use = int(np.ceil(np.log(max(2, n))))
        m_grid = sorted(set(max(1, int(frac * n)) for frac in m_frac_grid))

        print(
            f"[hyperparam][minibatch-sin] n={n}, T={T_use}, "
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
                    clip_multiplier=1.0,  # baseline C^* for m-sweep
                    seed=seed_run,
                )
                mses.append(mse)
                times.append(rt)

            mses = np.asarray(mses, dtype=float)
            times = np.asarray(times, dtype=float)

            results_n.append(
                {
                    "m": int(m_val),
                    "mean_mse": float(np.nanmean(mses)),
                    "std_mse": float(np.nanstd(mses)),
                    "min_mse": float(np.nanmin(mses)),
                    "max_mse": float(np.nanmax(mses)),
                    "mean_time": float(np.nanmean(times)),
                    "std_time": float(np.nanstd(times)),
                }
            )

        results_n.sort(key=lambda r: r["m"])
        m_results_by_n[n] = results_n

    return m_results_by_n


# ---------------------------------------------------------------------
# Hyperparameter result writers
# ---------------------------------------------------------------------


def save_clip_multiplier_results_txt(results_by_n, clip_values, path):
    """
    Save clip-multiplier sweep results as a plain-text table.
    """
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
    """
    Save minibatch-size sweep results as a plain-text table.
    """
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


# ---------------------------------------------------------------------
# Hyperparameter plots
# ---------------------------------------------------------------------


def plot_clip_multiplier_results_multi_n_modal(
    clip_results_by_n,
    clip_values,
    n_list,
    out_path,
):
    """
    Plot MSE vs C^* across n (single panel) for sinusoidal 2-component modal regression.
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
            xs,
            ys,
            yerr=yerr,
            marker="o",
            linestyle="-",
            label=f"n={n}",
            color=palette[idx],
            capsize=3,
        )

    plt.xlabel("Clip Multiplier ($C^*$ scaling)")
    plt.ylabel("MSE")
    plt.title("MSE vs $C^*$ across n for Sinusoidal 2-Component DP Modal Regression")
    plt.grid(True, alpha=0.4)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"[saved] Multi-n clip-multiplier MSE plot -> {out_path}")
    plt.show()
    plt.close()


def plot_minibatch_results_grid_modal(
    m_results_by_n,
    n_list,
    out_path,
):
    """
    2x2 grid: MSE vs minibatch size m for different n
    in sinusoidal 2-component modal regression.
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
            m_vals,
            mean_mses,
            yerr=std_mses,
            marker="o",
            linestyle="-",
            color=palette[idx],
            capsize=3,
        )
        ax.set_title(f"n = {n}")
        ax.grid(True, alpha=0.4)
        ax.set_xscale("log")

    axes[0].set_ylabel("MSE")
    axes[2].set_ylabel("MSE")
    axes[2].set_xlabel("Minibatch size m")
    axes[3].set_xlabel("Minibatch size m")

    fig.suptitle(
        "MSE vs m across n for Sinusoidal 2-Component DP Modal Regression",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120)
    print(f"[saved] Minibatch-size MSE grid plot -> {out_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------
# Baseline plots
# ---------------------------------------------------------------------


def make_baseline_visualizations():
    """
    Create the two baseline sinusoidal figures:
      1) PMS vs LOWESS
      2) DP-PMS example at epsilon=1, full batch
    """
    # Fixed dataset for illustration
    Xb, Yb = generate_sin_data(500, sigma_noise, seed=42)

    # Non-private PMS
    T_base = int(np.ceil(np.log(500)))
    Y_pms_baseline = partial_mean_shift(
        Xb,
        Yb,
        mesh_points=None,
        bandwidth=None,
        T=T_base,
    )

    # DP-PMS example (epsilon=1, m=n)
    Y_dp_baseline = dp_pms(
        Xb,
        Yb,
        epsilon=1.0,
        delta=DELTA_DEFAULT,
        T=T_base,
        m=len(Xb),
        rng=np.random.default_rng(42),
        verbose=False,
    )

    # LOWESS smooth
    Y_lowess = lowess(Yb, Xb, frac=0.2, return_sorted=False)

    # Figure 1: PMS vs LOWESS
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(Xb, Yb, color="skyblue", alpha=0.5, s=15, label="Data")
    ax1.scatter(Xb, Y_pms_baseline, color="#1f77b4", s=25, marker="D", label="PMS")

    x_sorted, y_lowess_sorted = sort_xy_for_line_plot(Xb, Y_lowess)
    ax1.plot(
        x_sorted,
        y_lowess_sorted,
        color="orange",
        linewidth=2,
        label="LOWESS",
    )

    ax1.set_title("Modal Regression for Sinusoidal 2-Component Mixture")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_xlim(0, 1)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()
    plt.tight_layout()
    plt.show()

    outpath1 = os.path.join(results_dir, "pms_vs_lowess_modal_regression_sin.pdf")
    fig1.savefig(outpath1, dpi=120)
    plt.close(fig1)
    print(f"[saved] PMS vs LOWESS -> {outpath1}")

    # Figure 2: DP-PMS baseline
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(Xb, Yb, color="skyblue", alpha=0.5, s=15, label="Data")
    ax2.scatter(
        Xb,
        Y_dp_baseline,
        color="#d95f02",
        s=35,
        marker="X",
        label="DP-PMS ($\\epsilon$=1, m=n)",
    )
    ax2.set_title("DP Modal Regression for Sinusoidal 2-Component Mixture")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_xlim(0, 1)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()
    plt.tight_layout()
    plt.show()

    outpath2 = os.path.join(results_dir, "dp_pms_modal_regression_sin_eps1_full_batch.pdf")
    fig2.savefig(outpath2, dpi=120)
    plt.close(fig2)
    print(f"[saved] DP-PMS baseline -> {outpath2}")


# ---------------------------------------------------------------------
# Privacy-utility experiment (paired PMS / DP-PMS runs)
# ---------------------------------------------------------------------


def run_privacy_utility_experiments():
    """
    Run the main privacy-utility experiments for the sinusoidal design.

    Returns
    -------
    results : list of rows
        Rows for the aggregate CSV.
    stats_summary : str
        Human-readable summary text.
    """
    print("\n[PU] Starting privacy-utility experiments (sinusoidal mixture)...")

    results = []
    stats_summary = (
        "=== Privacy-Utility Experiments "
        "(DP-PMS vs PMS, Sinusoidal 2-Component Mixture) ===\n"
    )

    for n in n_values:
        print(f"\n[PU] --- n = {n} ---")

        for eps in eps_values:
            print(f"[PU] n={n}, eps={eps}: launching {n_runs} paired PMS/DP-PMS runs...")

            # Paired seeds: one dataset seed and one DP seed per run
            data_seeds = [50_000_000 + 100_000 * n + 1_000 * int(10 * eps) + run for run in range(n_runs)]
            dp_seeds = [70_000_000 + 100_000 * n + 1_000 * int(10 * eps) + run for run in range(n_runs)]

            pms_mses = []
            pms_times = []
            dp_mses = []
            dp_times = []

            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        paired_pms_dp_worker,
                        n,
                        sigma_default,
                        float(eps),
                        DELTA_DEFAULT,
                        int(data_seeds[run]),
                        int(dp_seeds[run]),
                    ): run
                    for run in range(n_runs)
                }

                for fut in as_completed(futures):
                    run_idx = futures[fut]
                    try:
                        out = fut.result()
                    except Exception as e:
                        print(f"[PU][warn] n={n}, eps={eps}, run={run_idx}: worker failed: {e}")
                        continue

                    pms_mses.append(out["pms_mse"])
                    pms_times.append(out["pms_runtime"])
                    dp_mses.append(out["dp_mse"])
                    dp_times.append(out["dp_runtime"])

                    # Log DP-only row
                    with open(dp_perrun_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                n,
                                eps,
                                run_idx,
                                out["data_seed"],
                                out["dp_seed"],
                                out["dp_mse"],
                                out["dp_runtime"],
                                now_str(),
                            ]
                        )

                    # Log PMS-only row
                    with open(pms_perrun_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                n,
                                eps,
                                run_idx,
                                out["data_seed"],
                                out["pms_mse"],
                                out["pms_runtime"],
                                now_str(),
                            ]
                        )

                    # Log paired row
                    with open(paired_perrun_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                n,
                                eps,
                                run_idx,
                                out["data_seed"],
                                out["dp_seed"],
                                out["pms_mse"],
                                out["pms_runtime"],
                                out["dp_mse"],
                                out["dp_runtime"],
                                now_str(),
                            ]
                        )

            if len(dp_mses) == 0:
                print(f"[PU][warn] n={n}, eps={eps}: no successful paired runs.")
                continue

            # Aggregate
            dp_mean = float(np.mean(dp_mses))
            dp_se = safe_se(dp_mses)

            pms_mse_mean = float(np.mean(pms_mses)) if len(pms_mses) > 0 else np.nan
            # PMS MSE will be exactly zero by construction here
            # because PMS is compared to itself on each run.

            pms_time_mean = float(np.mean(pms_times)) if len(pms_times) > 0 else np.nan
            pms_time_std = safe_std(pms_times)

            dp_time_mean = float(np.mean(dp_times))
            dp_time_std = safe_std(dp_times)

            results.append(
                [
                    n,
                    eps,
                    dp_mean,
                    dp_se,
                    pms_mse_mean,
                    pms_time_mean,
                    pms_time_std,
                    dp_time_mean,
                    dp_time_std,
                ]
            )

            msg = (
                f"[PU] n={n}, eps={eps}: "
                f"DP-MSE={dp_mean:.4f}+-{dp_se:.4f}, "
                f"PMS-MSE={pms_mse_mean:.4f}, "
                f"PMS-time={pms_time_mean:.4f}+-{pms_time_std:.4f}s, "
                f"DP-time={dp_time_mean:.4f}+-{dp_time_std:.4f}s"
            )
            print(msg)
            stats_summary += msg + "\n"

    return results, stats_summary


def save_privacy_utility_results(results, stats_summary):
    """
    Save aggregate privacy-utility CSV and text summary.
    """
    csv_path = os.path.join(results_dir, "privacy_utility_results_modal_regression_sin.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_samples",
                "epsilon",
                "DP_mse_mean",
                "DP_mse_se",
                "PMS_mse",
                "PMS_runtime_mean",
                "PMS_runtime_std",
                "DP_runtime_mean",
                "DP_runtime_std",
            ]
        )
        writer.writerows(results)

    txt_path = os.path.join(results_dir, "stats_summary_modal_regression_sin.txt")
    save_text(txt_path, stats_summary)

    print(f"\n[saved] Summary CSV -> {csv_path}")
    print(f"[saved] Stats summary -> {txt_path}")


def plot_privacy_utility_results(results):
    """
    Plot privacy-utility MSE vs epsilon for sinusoidal mixture.

    PMS baseline is a horizontal line at zero MSE for each n, since PMS is
    compared to itself. That remains true even though PMS is rerun each time.
    """
    if len(results) == 0:
        print("[PU] No results available; skipping privacy-utility plot.")
        return

    print("[PU] Generating privacy-utility plot (sinusoidal mixture)...")

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("Set2", len(n_values))

    for i, n in enumerate(n_values):
        eps_subset = [r[1] for r in results if r[0] == n]
        mse_subset = [r[2] for r in results if r[0] == n]
        mse_errs = [r[3] for r in results if r[0] == n]
        pms_mse_val = next((r[4] for r in results if r[0] == n), np.nan)

        if len(eps_subset) == 0:
            continue

        sort_idx = np.argsort(np.asarray(eps_subset, dtype=float))
        eps_subset = list(np.asarray(eps_subset)[sort_idx])
        mse_subset = list(np.asarray(mse_subset)[sort_idx])
        mse_errs = list(np.asarray(mse_errs)[sort_idx])

        ax.errorbar(
            eps_subset,
            mse_subset,
            yerr=mse_errs,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=7,
            capsize=3,
            label=f"DP-PMS n={n}",
            color=palette[i],
        )

        if not np.isnan(pms_mse_val):
            ax.hlines(
                pms_mse_val,
                min(eps_subset),
                max(eps_subset),
                colors=palette[i],
                linestyles="dashed",
                linewidth=1.5,
                label=f"PMS n={n}",
            )

    ax.set_xlabel("Privacy budget $\\epsilon$")
    ax.set_xscale("log")
    ax.set_ylabel("MSE")
    ax.set_title("Privacy-Utility Tradeoff for Sinusoidal 2-Component DP Modal Regression")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=1, loc="upper right")
    plt.tight_layout()
    plt.show()

    pu_path = os.path.join(results_dir, "privacy_utility_pretty_modal_regression_sin.pdf")
    fig.savefig(pu_path, dpi=120)
    plt.close(fig)

    print(f"[saved] Privacy-utility plot -> {pu_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    # ------------------------------------------------------------
    # Baseline visualization on a fixed illustrative dataset
    # ------------------------------------------------------------
    make_baseline_visualizations()

    results, stats_summary = run_privacy_utility_experiments()
    save_privacy_utility_results(results, stats_summary)
    plot_privacy_utility_results(results)

    # ------------------------------------------------------------
    # Hyperparameter sweeps: MSE vs C^* and vs m
    # ------------------------------------------------------------
    print("\n[hyperparam] Starting DP-PMS hyperparameter sweeps (C^*, m) for sinusoidal mixture...")

    # --- MSE vs C^* across n ---
    clip_results_by_n = sweep_clip_multiplier_modal(
        n_list=n_values,
        clip_values=clip_grid,
        epsilon=epsilon_hparam,
        delta=DELTA_DEFAULT,
        sigma=sigma_default,
        n_reps=n_reps_hparam,
        base_seed=12345,
    )

    clip_txt_path = os.path.join(results_dir, "mse_vs_clip_multiplier_multi_n_modal_sin.txt")
    save_clip_multiplier_results_txt(
        clip_results_by_n,
        clip_grid,
        clip_txt_path,
    )

    clip_plot_path = os.path.join(results_dir, "mse_vs_clip_multiplier_multi_n_modal_sin.pdf")
    plot_clip_multiplier_results_multi_n_modal(
        clip_results_by_n,
        clip_grid,
        n_values,
        clip_plot_path,
    )

    # --- MSE vs minibatch size m across n ---
    m_results_by_n = sweep_minibatch_size_modal(
        n_list=n_values,
        m_frac_grid=m_grid_frac,
        epsilon=epsilon_hparam,
        delta=DELTA_DEFAULT,
        sigma=sigma_default,
        n_reps=n_reps_hparam,
        base_seed=54321,
    )

    m_txt_path = os.path.join(results_dir, "mse_vs_minibatch_grid_modal_sin.txt")
    save_minibatch_results_txt(
        m_results_by_n,
        m_txt_path,
    )

    m_plot_path = os.path.join(results_dir, "mse_vs_minibatch_grid_modal_sin.pdf")
    plot_minibatch_results_grid_modal(
        m_results_by_n,
        n_values,
        m_plot_path,
    )

    print("\n[done] Modal regression DP-PMS sinusoidal experiments (including C^* and m sweeps) complete.")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()