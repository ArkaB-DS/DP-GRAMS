# bivariate_4mix_kernel_order4_compare.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import seaborn as sns
import time
import sys
import os
import math
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main_scripts.dp_grams import dp_grams
from main_scripts.dp_grams_beta import dp_grams_beta
from main_scripts.merge import merge_modes
from main_scripts.mode_matching_mse import mode_matching_mse


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------
# Data: 4-corners Gaussian mixture
# ------------------------------------------------------------
def generate_4corners(n_samples, seed=None):
    rng = np.random.default_rng(seed)
    means = np.array([[3, 3], [3, -3], [-3, 3], [-3, -3]])
    cov = np.eye(2)
    pts = [rng.multivariate_normal(m, cov, n_samples // 4) for m in means]
    return np.vstack(pts), means


# ------------------------------------------------------------
# Bandwidth chooser
# ------------------------------------------------------------
def beta_rule_bandwidth(n, d, beta, epsilon, delta, T, m):
    logn = math.log(max(3, n))
    polylog = math.log(2.0 / delta) * math.log(2.5 * m * max(T, 1) / (n * delta))
    Kbundle = max(1.0, float(T * d * polylog))

    h_non_dp = (logn / n) ** (1.0 / (d + 2.0 * beta))
    h_dp = (Kbundle / (n * n * epsilon * epsilon)) ** (1.0 / (2.0 * d + 2.0 * beta))

    return float(h_dp if h_dp > h_non_dp else h_non_dp)


# ------------------------------------------------------------
# Run one DP method many times and aggregate MMMSE
# (bandwidth is fixed and shared; seeds are paired)
# ------------------------------------------------------------
def _run_dp_many(
    method_name,
    X,
    true_modes,
    eps,
    delta,
    beta,
    n_runs,
    p0,
    clip_mult,
    run_seeds,
    h_shared,
):
    mses = []
    times = []
    phat_nonpos_rates = []
    phat_mins = []

    n, _ = X.shape
    T = int(np.ceil(np.log(max(3, n))))
    m = max(1, int(np.ceil(n / np.log(max(3, n)))))

    for r in range(n_runs):
        rng = np.random.default_rng(run_seeds[r])
        t0 = time.perf_counter()

        if method_name == "gaussian":
            _, raw = dp_grams(
                X=X,
                epsilon=eps,
                delta=delta,
                T=T,
                m=m,
                h=h_shared,
                p0=p0,
                rng=rng,
                clip_multiplier=clip_mult,
            )
            est = merge_modes(raw, bandwidth=h_shared, k=1)
            rt = time.perf_counter() - t0
            mse = mode_matching_mse(true_modes, est)

            mses.append(mse)
            times.append(rt)

        elif method_name == "order4":
            _, raw, diag = dp_grams_beta(
                X=X,
                epsilon=eps,
                delta=delta,
                beta=beta,
                T=T,
                m=m,
                h=h_shared,
                p0=p0,
                rng=rng,
                clip_multiplier=clip_mult,
                return_diagnostics=True,
            )

            est = merge_modes(raw, bandwidth=h_shared, k=1)
            rt = time.perf_counter() - t0
            mse = mode_matching_mse(true_modes, est)

            mses.append(mse)
            times.append(rt)
            phat_nonpos_rates.append(diag.get("p_hat_nonpos_rate", np.nan))
            phat_mins.append(diag.get("p_hat_min", np.nan))

        else:
            raise ValueError("method_name must be 'gaussian' or 'order4'")

    mses = np.asarray(mses, dtype=float)
    times = np.asarray(times, dtype=float)

    mse_mean = float(np.nanmean(mses))
    mse_sd = float(np.nanstd(mses, ddof=1)) if len(mses) > 1 else 0.0
    t_mean = float(np.nanmean(times))
    t_sd = float(np.nanstd(times, ddof=1)) if len(times) > 1 else 0.0

    diag_out = None
    if method_name == "order4":
        diag_out = {
            "phat_nonpos_rate_mean": float(np.nanmean(phat_nonpos_rates)) if phat_nonpos_rates else np.nan,
            "phat_min_min": float(np.nanmin(phat_mins)) if phat_mins else np.nan,
        }

    return mse_mean, mse_sd, t_mean, t_sd, diag_out


# ------------------------------------------------------------
# Main experiment: 2x2 grid over eps
# ------------------------------------------------------------
def run_experiment(
    n_list,
    eps_list,
    delta=1e-6,
    beta=4,
    n_runs=20,
    p0=0.1,
    clip_mult=1.0,
    base_seed=123,
    results_dir="results/bivariate_4mix_kernel_order_dp",
):
    ensure_dir(results_dir)

    out_csv = os.path.join(
        results_dir,
        "dp_kernel_compare_gauss_vs_order4_SHARED_beta_rule_pairedseeds.csv",
    )
    out_pdf = os.path.join(
        results_dir,
        "dp_kernel_compare_2x2_eps_grid_SHARED_beta_rule_pairedseeds.pdf",
    )

    series = {
        eps: {"n": [], "g_mean": [], "g_sd": [], "o_mean": [], "o_sd": []}
        for eps in eps_list
    }

    rows = []
    for eps_idx, eps in enumerate(eps_list):
        for idx, n in enumerate(n_list):
            X, true_modes = generate_4corners(n, seed=base_seed + idx)

            n0, d = X.shape
            T = int(np.ceil(np.log(max(3, n0))))
            m = max(1, int(np.ceil(n0 / np.log(max(3, n0)))))

            # Compute dp_grams_beta-optimal h ONCE, then force into both
            h_shared = beta_rule_bandwidth(
                n=n0,
                d=d,
                beta=beta,
                epsilon=eps,
                delta=delta,
                T=T,
                m=m,
            )

            # Paired seeds across methods for this (eps, n)
            run_seeds = [
                base_seed + 1_000_000 * eps_idx + 10_000 * idx + r
                for r in range(n_runs)
            ]

            # Gaussian DP (forced h_shared)
            g_mean, g_sd, gt_mean, gt_sd, _ = _run_dp_many(
                method_name="gaussian",
                X=X,
                true_modes=true_modes,
                eps=eps,
                delta=delta,
                beta=beta,
                n_runs=n_runs,
                p0=p0,
                clip_mult=clip_mult,
                run_seeds=run_seeds,
                h_shared=h_shared,
            )

            # Order-4 DP (forced h_shared)
            o_mean, o_sd, ot_mean, ot_sd, o_diag = _run_dp_many(
                method_name="order4",
                X=X,
                true_modes=true_modes,
                eps=eps,
                delta=delta,
                beta=beta,
                n_runs=n_runs,
                p0=p0,
                clip_mult=clip_mult,
                run_seeds=run_seeds,
                h_shared=h_shared,
            )

            series[eps]["n"].append(n)
            series[eps]["g_mean"].append(g_mean)
            series[eps]["g_sd"].append(g_sd)
            series[eps]["o_mean"].append(o_mean)
            series[eps]["o_sd"].append(o_sd)

            rows.append(
                [
                    n,
                    eps,
                    delta,
                    beta,
                    n_runs,
                    p0,
                    clip_mult,
                    h_shared,
                    g_mean,
                    g_sd,
                    gt_mean,
                    gt_sd,
                    o_mean,
                    o_sd,
                    ot_mean,
                    ot_sd,
                    o_diag["phat_nonpos_rate_mean"] if o_diag else np.nan,
                    o_diag["phat_min_min"] if o_diag else np.nan,
                ]
            )

            msg = (
                f"[eps={eps:.3g}][n={n}] h_shared(beta-rule)={h_shared:.4g} :: "
                f"Gauss DP MSE={g_mean:.4e} (SD={g_sd:.2e}) | "
                f"Order4 DP MSE={o_mean:.4e} (SD={o_sd:.2e})"
            )
            if o_diag:
                msg += f" | p_hat<=0 rate~{(o_diag['phat_nonpos_rate_mean'] * 100):.3f}%"
            print(msg)

    # Save CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n",
                "epsilon",
                "delta",
                "beta",
                "n_runs",
                "p0",
                "clip_mult",
                "h_shared_beta_rule",
                "gauss_mse_mean",
                "gauss_mse_sd",
                "gauss_time_mean",
                "gauss_time_sd",
                "order4_mse_mean",
                "order4_mse_sd",
                "order4_time_mean",
                "order4_time_sd",
                "order4_phat_nonpos_rate_mean",
                "order4_phat_min_min",
            ]
        )
        for r in rows:
            w.writerow(r)

    print(f"[saved] CSV -> {out_csv}")

    # Plot 2x2 grid with visible SD error bars
    _plot_2x2(series, eps_list, out_pdf, n_list)

    print(f"[saved] Plot -> {out_pdf}")
    return out_csv, out_pdf


def _plot_2x2(series, eps_list, out_pdf, n_list):
    ensure_dir(os.path.dirname(out_pdf))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    g_kwargs = dict(
        marker="o",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        capsize=4,
        elinewidth=1.3,
        capthick=1.1,
    )
    o_kwargs = dict(
        marker="s",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        capsize=4,
        elinewidth=1.3,
        capthick=1.1,
    )

    for k, eps in enumerate(eps_list):
        ax = axes[k]
        n_vals = np.asarray(series[eps]["n"], dtype=float)

        g_mean = np.asarray(series[eps]["g_mean"], dtype=float)
        g_sd = np.asarray(series[eps]["g_sd"], dtype=float)

        o_mean = np.asarray(series[eps]["o_mean"], dtype=float)
        o_sd = np.asarray(series[eps]["o_sd"], dtype=float)

        label_g = "Gaussian DP-GRAMS" if k == 0 else None
        label_o = r"Order-4 DP-GRAMS-$\beta$" if k == 0 else None

        ax.errorbar(n_vals, g_mean, yerr=g_sd, label=label_g, **g_kwargs)
        ax.errorbar(n_vals, o_mean, yerr=o_sd, label=label_o, **o_kwargs)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min(n_list) * 0.9, max(n_list) * 1.1)
        ax.set_xticks([1000, 2000, 5000], minor=False)
        ax.set_xticklabels([r"$10^3$", r"$2\times 10^3$", r"$5\times 10^3$"])
        ax.xaxis.set_minor_formatter(NullFormatter())

        ax.grid(True, which="both", alpha=0.35)
        ax.set_title(rf"$\varepsilon = {eps}$, $\delta = 10^{{-6}}$")

        if k in (2, 3):
            ax.set_xlabel("Sample size $n$")
        if k in (0, 2):
            ax.set_ylabel("MSE")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True)

    fig.suptitle("Gaussian vs Order-4 DP-GRAMS on 4-Modal Bivariate Gaussian Mixture", y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(out_pdf, dpi=120)
    plt.show()
    plt.close(fig)


def main():
    n_list = [700, 1000, 2000, 5000]
    eps_list = [0.2, 0.5, 1.0, 2.0]

    run_experiment(
        n_list=n_list,
        eps_list=eps_list,
        delta=1e-6,
        beta=4,
        n_runs=20,
        p0=0.1,
        clip_mult=1.0,
        base_seed=123,
        results_dir="results/bivariate_4mix_kernel_order_dp",
    )


if __name__ == "__main__":
    main()