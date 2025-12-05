# mnist_compare_centroids_subsampled.py

import numpy as np
import os
import sys
import time
import csv
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor, as_completed
from diffprivlib.models import KMeans as DPKMeans

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from real_data_scripts.dp_grams_c import dpms_private
from main_scripts.mode_matching_mse import mode_matching_mse
from main_scripts.ms import mean_shift
from main_scripts.merge import merge_modes_agglomerative
from main_scripts.bandwidth import silverman_bandwidth

# ---------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------
merge_n_clusters = 10     # digits 0-9
ms_T = 10                 # MS iterations (fixed small number)
ms_p = 0.1                # fraction of seeds for MS
delta = 1e-5
p_seed = 0.1              # seed fraction for DP-GRAMS-C
base_rng_seed = 41
n_runs = 20               # DP runs per epsilon for PU curves
subsample_n = 3000        # base subsample size for main experiments

results_dir = "results/mnist_subsample_centroid_comparison"
os.makedirs(results_dir, exist_ok=True)

sns.set(style="whitegrid", context="talk")

eps_modes_list = [0.001, 0.01, 0.1, 1.0]  # epsilon grid for PU curves
MAX_WORKERS = min(8, os.cpu_count() or 8)

# Hyperparameter sweep settings for SUBSAMPLING EFFECT (DP-GRAMS-C only)
subsample_sizes_hparam = [1000, 3000, 6000, 10000]
clip_grid_subsample = [0.01, 0.05, 0.1, 0.25]
m_frac_grid_subsample = [0.01, 0.05, 0.1, 0.2]
n_runs_subsample_effect = 10
epsilon_subsample_effect = 1.0  # epsilon_modes used for C^* / m subsampling-effect sweeps

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def compute_metrics(y_true, labels, X):
    """
    Clustering quality metrics on given feature space X.
    Returns ARI and NMI.
    """
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    return ari, nmi


def timed(func, *args, **kwargs):
    start = time.time()
    out = func(*args, **kwargs)
    return out, time.time() - start


def relabel_clusters(y_true, y_pred, n_clusters=10):
    """
    Relabel cluster indices to best match true labels via Hungarian algorithm.
    Ensures consistency when comparing ARI/NMI/etc across runs.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(n_clusters))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in y_pred])


def stratified_subsample(X, y, n_total=20000, rng=None):
    """
    Stratified subsample of size n_total from (X, y).
    Per-class counts as even as possible.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    X = np.asarray(X)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    n_per_class = n_total // n_classes
    indices = []

    for c in classes:
        idx_c = np.where(y == c)[0]
        if len(idx_c) <= n_per_class:
            chosen = idx_c
        else:
            chosen = rng.choice(idx_c, size=n_per_class, replace=False)
        indices.append(chosen)

    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]

# ---------------------------------------------------------------------
# Workers for parallel privacy-utility experiments (fixed subsample_n)
# ---------------------------------------------------------------------

def dpms_single_run(run, X, y_true, eps_modes, merge_n_clusters,
                    pca_50, scaler, true_means_scaled, base_seed=41):
    seed = base_seed + run
    rng_run = np.random.default_rng(seed)

    print(f"[{now_str()}] [DPMS worker] start run={run} eps={eps_modes} seed={seed}")
    t0 = time.time()

    modes_dpms_run, labels_dpms_run = dpms_private(
        data=X,
        epsilon_modes=eps_modes,
        delta=delta,
        p_seed=p_seed,
        rng=rng_run,
        k_est=merge_n_clusters,
        bandwidth_multiplier=1.0,
        clip_multiplier=0.01
    )
    runtime = time.time() - t0

    labels_dpms_run = relabel_clusters(y_true, labels_dpms_run, merge_n_clusters)
    ari, nmi = compute_metrics(y_true, labels_dpms_run, X)

    # Back to raw 784D via PCA fitted on subsample, then standardize using full-data scaler
    modes_orig = pca_50.inverse_transform(modes_dpms_run)
    modes_scaled = scaler.transform(modes_orig)
    mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

    print(f"[{now_str()}] [DPMS worker] end   run={run} eps={eps_modes} "
          f"ARI={ari:.4f} MSE={mse:.4f} time={runtime:.2f}s")

    return modes_dpms_run, labels_dpms_run, (ari, nmi), float(mse), float(runtime), int(seed)


def dpkm_single_run(run, X, y_true, eps_modes, merge_n_clusters,
                    pca_50, scaler, true_means_scaled, base_seed=41):
    seed = base_seed + 1000 + run
    t0 = time.time()
    print(f"[{now_str()}] [DP-KMeans worker] start run={run} eps={eps_modes} seed={seed}")

    dp_kmeans_run = DPKMeans(
        n_clusters=merge_n_clusters,
        epsilon=eps_modes,
        random_state=seed
    )
    try:
        dp_kmeans_run.fit(X)
    except ValueError:
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        dp_kmeans_run.fit(X, bounds=(mins, maxs))

    runtime = time.time() - t0

    labels_dpkm_run = (
        dp_kmeans_run.labels_.astype(int)
        if hasattr(dp_kmeans_run, "labels_")
        else dp_kmeans_run.predict(X)
    )
    labels_dpkm_run = relabel_clusters(y_true, labels_dpkm_run, merge_n_clusters)
    ari, nmi = compute_metrics(y_true, labels_dpkm_run, X)

    modes_dpkm_run = dp_kmeans_run.cluster_centers_
    modes_orig = pca_50.inverse_transform(modes_dpkm_run)
    modes_scaled = scaler.transform(modes_orig)
    mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

    print(f"[{now_str()}] [DP-KMeans worker] end   run={run} eps={eps_modes} "
          f"ARI={ari:.4f} MSE={mse:.4f} time={runtime:.2f}s")

    return modes_dpkm_run, labels_dpkm_run, (ari, nmi), float(mse), float(runtime), int(seed)

# ---------------------------------------------------------------------
# Helpers for SUBSAMPLING EFFECT on hyperparameters (DP-GRAMS-C only)
# ---------------------------------------------------------------------

def dpms_hparam_single_subsample(
    X_raw_full,
    y_full,
    subsample_n,
    epsilon_modes,
    clip_multiplier,
    m,
    scaler,
    true_means_scaled,
    run,
    base_seed=12345,
):
    """
    Single DP-GRAMS-C run for subsampling-effect hyperparameter sweeps.

    - Stratified subsample of size subsample_n from full MNIST.
    - PCA-50 on that subsample for clustering space.
    - DP-GRAMS-C on PCA-50.
    - ARI/NMI (in PCA-50 space) and MSE of centroids in standardized 784D.
    """
    rng = np.random.default_rng(base_seed + run)

    # Stratified subsample
    X_raw_sub, y_sub = stratified_subsample(
        X_raw_full,
        y_full,
        n_total=subsample_n,
        rng=rng
    )

    # PCA-50 on this subsample
    pca_50 = PCA(n_components=50, random_state=base_seed + 1000 + run)
    X_sub = pca_50.fit_transform(X_raw_sub)

    # DP-GRAMS-C on PCA-50 subsample
    rng_dp = np.random.default_rng(base_seed + 2000 + run)

    t0 = time.time()
    modes_dpms_run, labels_dpms_run = dpms_private(
        data=X_sub,
        epsilon_modes=epsilon_modes,
        delta=delta,
        p_seed=p_seed,
        rng=rng_dp,
        k_est=merge_n_clusters,
        bandwidth_multiplier=1.0,
        clip_multiplier=clip_multiplier,
        m=m,
    )
    runtime = time.time() - t0

    if modes_dpms_run.size == 0:
        return np.nan, np.nan, np.nan, float(runtime)

    labels_dpms_run = relabel_clusters(y_sub, labels_dpms_run, merge_n_clusters)
    ari, nmi = compute_metrics(y_sub, labels_dpms_run, X_sub)

    # Centroid MSE in full standardized 784D
    modes_orig = pca_50.inverse_transform(modes_dpms_run)
    modes_scaled = scaler.transform(modes_orig)
    mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

    return float(ari), float(nmi), float(mse), float(runtime)


def subsampling_effect_clip_grid(
    X_raw_full,
    y_full,
    subsample_sizes,
    clip_values,
    epsilon_modes,
    n_reps,
    scaler,
    true_means_scaled,
    base_seed=12345,
):
    """
    Subsampling-effect for DP-GRAMS-C: ARI/NMI/MSE vs C^* across subsample sizes.
    Returns dict: n_sub -> list of stats dicts keyed by clip_multiplier.
    """
    results_by_n = {}

    for n_sub in subsample_sizes:
        print(f"\n[{now_str()}] [subsample-effect][C*] n_sub={n_sub}")
        stats_list = []
        for cm in clip_values:
            ari_vals, nmi_vals, mse_vals, time_vals = [], [], [], []
            for run in range(n_reps):
                ari, nmi, mse, rt = dpms_hparam_single_subsample(
                    X_raw_full=X_raw_full,
                    y_full=y_full,
                    subsample_n=n_sub,
                    epsilon_modes=epsilon_modes,
                    clip_multiplier=cm,
                    m=None,  # default minibatch (n/log n) inside dp_grams
                    scaler=scaler,
                    true_means_scaled=true_means_scaled,
                    run=run + int(1000 * cm),
                    base_seed=base_seed,
                )
                ari_vals.append(ari)
                nmi_vals.append(nmi)
                mse_vals.append(mse)
                time_vals.append(rt)

            ari_vals = np.array(ari_vals, dtype=float)
            nmi_vals = np.array(nmi_vals, dtype=float)
            mse_vals = np.array(mse_vals, dtype=float)
            time_vals = np.array(time_vals, dtype=float)

            stats_list.append({
                "clip_multiplier": float(cm),
                "mean_ari": float(np.nanmean(ari_vals)),
                "std_ari": float(np.nanstd(ari_vals, ddof=1)),
                "mean_nmi": float(np.nanmean(nmi_vals)),
                "std_nmi": float(np.nanstd(nmi_vals, ddof=1)),
                "mean_mse": float(np.nanmean(mse_vals)),
                "std_mse": float(np.nanstd(mse_vals, ddof=1)),
                "mean_time": float(np.nanmean(time_vals)),
                "std_time": float(np.nanstd(time_vals, ddof=1)),
            })

        stats_list.sort(key=lambda r: r["clip_multiplier"])
        results_by_n[n_sub] = stats_list

    return results_by_n


def subsampling_effect_m_grid(
    X_raw_full,
    y_full,
    subsample_sizes,
    m_frac_grid,
    epsilon_modes,
    clip_multiplier_fixed,
    n_reps,
    scaler,
    true_means_scaled,
    base_seed=22345,
):
    """
    Subsampling-effect for DP-GRAMS-C: ARI/NMI/MSE vs m across subsample sizes.
    Returns dict: n_sub -> list of stats dicts keyed by m.
    """
    results_by_n = {}

    for n_sub in subsample_sizes:
        print(f"\n[{now_str()}] [subsample-effect][m] n_sub={n_sub}")
        # Build m-grid as fractions of n_sub
        m_grid = sorted(set(max(1, int(frac * n_sub)) for frac in m_frac_grid))
        stats_list = []

        for m_val in m_grid:
            ari_vals, nmi_vals, mse_vals, time_vals = [], [], [], []
            for run in range(n_reps):
                ari, nmi, mse, rt = dpms_hparam_single_subsample(
                    X_raw_full=X_raw_full,
                    y_full=y_full,
                    subsample_n=n_sub,
                    epsilon_modes=epsilon_modes,
                    clip_multiplier=clip_multiplier_fixed,
                    m=int(m_val),
                    scaler=scaler,
                    true_means_scaled=true_means_scaled,
                    run=run + int(10 * m_val),
                    base_seed=base_seed,
                )
                ari_vals.append(ari)
                nmi_vals.append(nmi)
                mse_vals.append(mse)
                time_vals.append(rt)

            ari_vals = np.array(ari_vals, dtype=float)
            nmi_vals = np.array(nmi_vals, dtype=float)
            mse_vals = np.array(mse_vals, dtype=float)
            time_vals = np.array(time_vals, dtype=float)

            stats_list.append({
                "m": int(m_val),
                "mean_ari": float(np.nanmean(ari_vals)),
                "std_ari": float(np.nanstd(ari_vals, ddof=1)),
                "mean_nmi": float(np.nanmean(nmi_vals)),
                "std_nmi": float(np.nanstd(nmi_vals, ddof=1)),
                "mean_mse": float(np.nanmean(mse_vals)),
                "std_mse": float(np.nanstd(mse_vals, ddof=1)),
                "mean_time": float(np.nanmean(time_vals)),
                "std_time": float(np.nanstd(time_vals, ddof=1)),
            })

        stats_list.sort(key=lambda r: r["m"])
        results_by_n[n_sub] = stats_list

    return results_by_n


def plot_subsample_effect_grid(
    results_by_n,
    subsample_sizes,
    x_key,
    metric_key,
    x_label,
    y_label,
    title_prefix,
    out_path,
    log_x=True,
):
    """
    Generic 2x2 grid plot for subsampling-effect results.
    metric_key in {"ari", "nmi", "mse"} corresponding to
    mean_* and std_* keys in result dicts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()

    for ax, n_sub in zip(axes, subsample_sizes):
        stats_list = results_by_n.get(n_sub, [])
        if not stats_list:
            ax.set_visible(False)
            continue

        xs = [r[x_key] for r in stats_list]
        ys = [r[f"mean_{metric_key}"] for r in stats_list]
        yerr = [r[f"std_{metric_key}"] for r in stats_list]

        ax.errorbar(
            xs, ys, yerr=yerr,
            marker="o", linestyle="-",
            linewidth=2, markersize=6, capsize=3,
        )
        if log_x:
            ax.set_xscale("log")
        ax.set_title(f"Subsample n = {n_sub}")
        ax.grid(True, alpha=0.4)

    axes[0].set_ylabel(y_label)
    axes[2].set_ylabel(y_label)
    axes[2].set_xlabel(x_label)
    axes[3].set_xlabel(x_label)

    fig.suptitle(title_prefix, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[{now_str()}] [saved] {title_prefix} -> {out_path}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # ------------------------------------------------------------
    # Load full MNIST
    # ------------------------------------------------------------
    print(f"[{now_str()}] Loading FULL MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X_raw_full = mnist.data.astype(np.float32)
    y_full = mnist.target.astype(int)
    print(f"[{now_str()}] Full MNIST shape: {X_raw_full.shape}")

    # ------------------------------------------------------------
    # Standardize full 784D, compute true means for MSE space
    # ------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(X_raw_full)
    true_means_scaled = np.array([
        X_scaled_full[y_full == c].mean(axis=0)
        for c in np.unique(y_full)
    ])

    # ------------------------------------------------------------
    # Stratified subsample for main clustering experiments
    # ------------------------------------------------------------
    print(f"[{now_str()}] Stratified subsampling to {subsample_n} points...")
    rng_sub = np.random.default_rng(base_rng_seed)
    X_raw, y_true = stratified_subsample(
        X_raw_full, y_full,
        n_total=subsample_n,
        rng=rng_sub
    )
    print(f"[{now_str()}] Subsampled shape: {X_raw.shape}")

    # PCA-50 for clustering on subsample (fit on subsample raw pixels)
    print(f"[{now_str()}] Computing PCA-50 embedding on subsample...")
    pca_50 = PCA(n_components=50, random_state=base_rng_seed)
    X = pca_50.fit_transform(X_raw)
    print(f"[{now_str()}] PCA-50 shape: {X.shape}")

    true_means = np.array([
        X[y_true == c].mean(axis=0)
        for c in np.unique(y_true)
    ])

    # ------------------------------------------------------------
    # Single-run baselines on subsample (MS, KMeans, DP-KMeans, DP-GRAMS-C)
    # ------------------------------------------------------------
    print(f"[{now_str()}] Running single-run baselines on subsample...")

    rng_ms = np.random.default_rng(base_rng_seed)
    h_ms = silverman_bandwidth(X)
    print(f"[{now_str()}] [DEBUG] MS bandwidth h = {h_ms:.6e}")

    (raw_ms_modes, ms_time) = timed(
        mean_shift, X, None, ms_T, h_ms, ms_p, rng_ms
    )
    ms_merged_modes = merge_modes_agglomerative(
        raw_ms_modes,
        n_clusters=merge_n_clusters,
        random_state=base_rng_seed
    )
    dists_ms = np.linalg.norm(
        X[:, None, :] - ms_merged_modes[None, :, :],
        axis=2
    )
    labels_ms = np.argmin(dists_ms, axis=1)
    labels_ms = relabel_clusters(y_true, labels_ms, merge_n_clusters)

    kmeans = KMeans(
        n_clusters=merge_n_clusters,
        random_state=base_rng_seed,
        n_init=10
    )
    (labels_km, km_time) = timed(kmeans.fit_predict, X)
    modes_km = kmeans.cluster_centers_
    labels_km = relabel_clusters(y_true, labels_km, merge_n_clusters)

    dp_kmeans = DPKMeans(
        n_clusters=merge_n_clusters,
        epsilon=1.0,
        random_state=base_rng_seed
    )
    try:
        (_, dpkm_time) = timed(dp_kmeans.fit, X)
    except ValueError:
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        (_, dpkm_time) = timed(dp_kmeans.fit, X, bounds=(mins, maxs))
    labels_dpkm = (
        dp_kmeans.labels_.astype(int)
        if hasattr(dp_kmeans, "labels_")
        else dp_kmeans.predict(X)
    )
    modes_dpkm = dp_kmeans.cluster_centers_
    labels_dpkm = relabel_clusters(y_true, labels_dpkm, merge_n_clusters)

    rng_dpms = np.random.default_rng(base_rng_seed)
    (modes_dpms, labels_dpms), dpms_time = timed(
        dpms_private,
        data=X,
        epsilon_modes=1.0,
        delta=delta,
        p_seed=p_seed,
        rng=rng_dpms,
        k_est=merge_n_clusters,
        bandwidth_multiplier=1.0,
        clip_multiplier=0.01
    )
    labels_dpms = relabel_clusters(y_true, labels_dpms, merge_n_clusters)

    algorithms = ["MS Clustering", "DP-GRAMS-C", "KMeans", "DP-KMeans"]
    labels_list = [labels_ms, labels_dpms, labels_km, labels_dpkm]
    modes_list = [ms_merged_modes, modes_dpms, modes_km, modes_dpkm]
    runtimes = [ms_time, dpms_time, km_time, dpkm_time]

    metrics = {}
    mse_centroids = {}
    for alg, labels, modes in zip(algorithms, labels_list, modes_list):
        ari, nmi = compute_metrics(y_true, labels, X)
        metrics[alg] = (ari, nmi)

        modes_orig = pca_50.inverse_transform(modes)
        modes_scaled = scaler.transform(modes_orig)
        mse_centroids[alg] = mode_matching_mse(
            true_means_scaled.copy(),
            modes_scaled.copy()
        )

    single_csv = os.path.join(
        results_dir, "clustering_metrics_single_run.csv"
    )
    with open(single_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Algorithm", "ARI", "NMI", "MSE", "Runtime(s)"]
        )
        for alg, rt in zip(algorithms, runtimes):
            ari, nmi = metrics[alg]
            writer.writerow([alg, ari, nmi, mse_centroids[alg], rt])
    print(f"[{now_str()}] Single-run clustering metrics saved to {single_csv}")

    single_txt = os.path.join(
        results_dir, "clustering_metrics_single_run.txt"
    )
    with open(single_txt, "w") as f:
        for alg, rt in zip(algorithms, runtimes):
            ari, nmi = metrics[alg]
            f.write(
                f"{alg}: ARI={ari:.4f}, NMI={nmi:.4f}, "
                f"MSE={mse_centroids[alg]:.6f}, "
                f"Runtime={rt:.4f}s\n"
            )
    print(f"[{now_str()}] Single-run clustering metrics (text) saved to {single_txt}")

    print("\nSingle-run metrics (PCA-50 clustering space; MSE in standardized 784D):")
    print(f"{'Alg':<18} {'ARI':>6} {'NMI':>6} "
          f"{'MSE_centroids':>14} {'Runtime(s)':>12}")
    for alg, rt in zip(algorithms, runtimes):
        ari, nmi = metrics[alg]
        print(
            f"{alg:<18} {ari:6.3f} {nmi:6.3f} "
            f"{mse_centroids[alg]:14.6f} {rt:12.4f}"
        )

    # ------------------------------------------------------------
    # Clustering comparison: 1 row x 4 columns (PCA-2 of PCA-50)
    # ------------------------------------------------------------
    print(f"[{now_str()}] Generating clustering comparison plot (1x4)...")

    pca_2d = PCA(n_components=2, random_state=base_rng_seed)
    X_2d = pca_2d.fit_transform(X)
    true_means_2d = pca_2d.transform(true_means)
    modes_2d_list = [pca_2d.transform(m) for m in modes_list]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    palette = sns.color_palette("tab10", merge_n_clusters)
    global_handles, global_labels = [], []

    for ax, alg, labels_pred, modes_2d in zip(
        axes, algorithms, labels_list, modes_2d_list
    ):
        for i, color in enumerate(palette):
            sc = ax.scatter(
                X_2d[labels_pred == i, 0],
                X_2d[labels_pred == i, 1],
                c=[color],
                s=5,
                alpha=0.6
            )
            if len(global_handles) < merge_n_clusters:
                global_handles.append(sc)
                global_labels.append(f"Cluster {i}")

        true_sc = ax.scatter(
            true_means_2d[:, 0],
            true_means_2d[:, 1],
            marker="X",
            c="magenta",
            s=120,
            linewidths=2
        )
        if "True means" not in global_labels:
            global_handles.append(true_sc)
            global_labels.append("True means")

        modes_sc = ax.scatter(
            modes_2d[:, 0],
            modes_2d[:, 1],
            marker="X",
            c="blue",
            s=80,
            linewidths=2
        )
        if "Estimated modes" not in global_labels:
            global_handles.append(modes_sc)
            global_labels.append("Estimated modes")

        ax.set_title(alg, fontsize=14)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    legend = fig.legend(
        global_handles,
        global_labels,
        fontsize=9,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.02),
        title="Cluster Assignments & Centroids"
    )
    plt.setp(legend.get_title(), fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Clustering Comparison on MNIST (PCA-50, subsample {subsample_n/1000:.1f}k)",
        fontsize=18,
        y=1.02
    )

    save_path_clusters = os.path.join(
        results_dir, "mnist_subsample_clustering_comparison.png"
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(save_path_clusters, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[{now_str()}] Clustering comparison saved to: {save_path_clusters}")

    # ------------------------------------------------------------
    # Privacy-utility experiments (fixed subsample_n, various epsilon)
    # ------------------------------------------------------------
    print(
        f"[{now_str()}] Starting privacy-utility experiments "
        f"({len(eps_modes_list)} eps x {n_runs} runs) on subsample..."
    )

    dpms_perrun_csv = os.path.join(
        results_dir, "dpms_per_run_results.csv"
    )
    dpkm_perrun_csv = os.path.join(
        results_dir, "dpkm_per_run_results.csv"
    )

    if not os.path.exists(dpms_perrun_csv):
        with open(dpms_perrun_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Algorithm", "Eps_modes", "Run", "Seed",
                "ARI", "NMI", "MSE",
                "Runtime_s", "timestamp"
            ])
    if not os.path.exists(dpkm_perrun_csv):
        with open(dpkm_perrun_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Algorithm", "Eps_modes", "Run", "Seed",
                "ARI", "NMI", "MSE",
                "Runtime_s", "timestamp"
            ])

    dpms_metrics = {m: [] for m in ["ARI", "NMI", "MSE"]}
    dpkm_metrics = {m: [] for m in ["ARI", "NMI", "MSE"]}
    dpms_err = {m: [] for m in ["ARI", "NMI", "MSE"]}
    dpkm_err = {m: [] for m in ["ARI", "NMI", "MSE"]}
    dpms_times_mean, dpms_times_std = [], []
    dpkm_times_mean, dpkm_times_std = [], []

    for eps in eps_modes_list:
        print(
            f"\n[{now_str()}] === epsilon_modes = {eps} "
            f"(subsample {subsample_n/1000:.1f}k) ==="
        )

        # DP-GRAMS-C
        ari_list, nmi_list, mse_list, runtimes_list = [], [], [], []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    dpms_single_run,
                    run, X, y_true, eps,
                    merge_n_clusters,
                    pca_50, scaler, true_means_scaled,
                    base_rng_seed
                ): run
                for run in range(n_runs)
            }
            for fut in as_completed(futures):
                run_id = futures[fut]
                try:
                    _, _, (ari, nmi), mse, runtime, seed = fut.result()
                except Exception as e:
                    print(f"[{now_str()}] [ERROR] DPMS run {run_id}, eps={eps}: {e}")
                    continue

                ari_list.append(ari)
                nmi_list.append(nmi)
                mse_list.append(mse)
                runtimes_list.append(runtime)

                with open(dpms_perrun_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "DP-GRAMS-C", eps, run_id, seed,
                        ari, nmi, mse, runtime, now_str()
                    ])

        for key, arr in zip(
            ["ARI", "NMI", "MSE"],
            [ari_list, nmi_list, mse_list]
        ):
            if arr:
                dpms_metrics[key].append(float(np.mean(arr)))
                dpms_err[key].append(float(np.std(arr, ddof=1)))
            else:
                dpms_metrics[key].append(np.nan)
                dpms_err[key].append(np.nan)
        dpms_times_mean.append(
            float(np.mean(runtimes_list)) if runtimes_list else np.nan
        )
        dpms_times_std.append(
            float(np.std(runtimes_list, ddof=1))
            if len(runtimes_list) > 1 else 0.0
        )

        # DP-KMeans
        ari_list, nmi_list, mse_list, runtimes_list = [], [], [], []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    dpkm_single_run,
                    run, X, y_true, eps,
                    merge_n_clusters,
                    pca_50, scaler, true_means_scaled,
                    base_rng_seed
                ): run
                for run in range(n_runs)
            }
            for fut in as_completed(futures):
                run_id = futures[fut]
                try:
                    _, _, (ari, nmi), mse, runtime, seed = fut.result()
                except Exception as e:
                    print(f"[{now_str()}] [ERROR] DP-KMeans run {run_id}, eps={eps}: {e}")
                    continue

                ari_list.append(ari)
                nmi_list.append(nmi)
                mse_list.append(mse)
                runtimes_list.append(runtime)

                with open(dpkm_perrun_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "DP-KMeans", eps, run_id, seed,
                        ari, nmi, mse, runtime, now_str()
                    ])

        for key, arr in zip(
            ["ARI", "NMI", "MSE"],
            [ari_list, nmi_list, mse_list]
        ):
            if arr:
                dpkm_metrics[key].append(float(np.mean(arr)))
                dpkm_err[key].append(float(np.std(arr, ddof=1)))
            else:
                dpkm_metrics[key].append(np.nan)
                dpkm_err[key].append(np.nan)
        dpkm_times_mean.append(
            float(np.mean(runtimes_list)) if runtimes_list else np.nan
        )
        dpkm_times_std.append(
            float(np.std(runtimes_list, ddof=1))
            if len(runtimes_list) > 1 else 0.0
        )

    # Aggregate CSV for PU
    agg_csv = os.path.join(
        results_dir, "privacy_utility_metrics_subsample.csv"
    )
    with open(agg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Eps_modes", "Algorithm",
            "ARI", "NMI", "MSE",
            "Std_ARI", "Std_NMI", "Std_MSE",
            "Mean_Runtime(s)", "Std_Runtime(s)"
        ])
        for i, eps in enumerate(eps_modes_list):
            writer.writerow([
                eps, "DP-GRAMS-C",
                dpms_metrics["ARI"][i], dpms_metrics["NMI"][i],
                dpms_metrics["MSE"][i],
                dpms_err["ARI"][i], dpms_err["NMI"][i],
                dpms_err["MSE"][i],
                dpms_times_mean[i], dpms_times_std[i]
            ])
            writer.writerow([
                eps, "DP-KMeans",
                dpkm_metrics["ARI"][i], dpkm_metrics["NMI"][i],
                dpkm_metrics["MSE"][i],
                dpkm_err["ARI"][i], dpkm_err["NMI"][i],
                dpkm_err["MSE"][i],
                dpkm_times_mean[i], dpkm_times_std[i]
            ])
    print(f"[{now_str()}] Aggregated privacy-utility metrics saved to: {agg_csv}")

    # 3 separate PU plots vs epsilon (ARI, NMI, MSE)
    for metric in ["ARI", "NMI", "MSE"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(
            eps_modes_list,
            dpms_metrics[metric],
            yerr=dpms_err[metric],
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=7,
            capsize=3,
            label="DP-GRAMS-C"
        )
        ax.errorbar(
            eps_modes_list,
            dpkm_metrics[metric],
            yerr=dpkm_err[metric],
            marker='s',
            linestyle='-',
            linewidth=2,
            markersize=7,
            capsize=3,
            label="DP-KMeans"
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\epsilon_{\mathrm{modes}}$")
        ax.set_ylabel(metric)
        ax.set_title(
            f"Privacy-Utility: {metric} "
            f"(MNIST, subsample {subsample_n/1000:.1f}k)"
        )
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        plt.tight_layout()
        out_path = os.path.join(
            results_dir,
            f"mnist_subsample_privacy_utility_{metric.lower()}.png"
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(
            f"[{now_str()}] Privacy-utility ({metric}) saved to: {out_path}"
        )

    # ------------------------------------------------------------
    # SUBSAMPLING EFFECT for DP-GRAMS-C: C^* and m
    # ------------------------------------------------------------
    print(
        f"\n[{now_str()}] Starting subsampling-effect experiments for "
        f"DP-GRAMS-C across C^* and m..."
    )
    print(
        f"Subsample sizes: {subsample_sizes_hparam}, "
        f"n_runs per configuration: {n_runs_subsample_effect}"
    )

    # --- C^* sweep across subsample sizes ---
    clip_results_by_n = subsampling_effect_clip_grid(
        X_raw_full=X_raw_full,
        y_full=y_full,
        subsample_sizes=subsample_sizes_hparam,
        clip_values=clip_grid_subsample,
        epsilon_modes=epsilon_subsample_effect,
        n_reps=n_runs_subsample_effect,
        scaler=scaler,
        true_means_scaled=true_means_scaled,
        base_seed=34567,
    )

    # Save textual C^* subsampling-effect results
    clip_txt_path = os.path.join(
        results_dir,
        "mnist_dpgrams_c_subsample_effect_clip_stats.txt"
    )
    with open(clip_txt_path, "w") as f:
        for n_sub in subsample_sizes_hparam:
            f.write(f"=== Subsample n = {n_sub} ===\n")
            header = (
                f"{'clip_mult':>10} | {'mean_ari':>9} | {'std_ari':>8} | "
                f"{'mean_nmi':>9} | {'std_nmi':>8} | "
                f"{'mean_mse':>10} | {'std_mse':>9} | "
                f"{'mean_t':>9} | {'std_t':>8}\n"
            )
            f.write(header)
            f.write("-" * len(header) + "\n")
            for r in clip_results_by_n.get(n_sub, []):
                line = (
                    f"{r['clip_multiplier']:10.3f} | "
                    f"{r['mean_ari']:9.4f} | {r['std_ari']:8.4f} | "
                    f"{r['mean_nmi']:9.4f} | {r['std_nmi']:8.4f} | "
                    f"{r['mean_mse']:10.4f} | {r['std_mse']:9.4f} | "
                    f"{r['mean_time']:9.4f} | {r['std_time']:8.4f}\n"
                )
                f.write(line)
            f.write("\n")
    print(f"[{now_str()}] C^* subsampling-effect stats saved to {clip_txt_path}")

    # 2x2 grids for C^* subsampling-effect: ARI, NMI, MSE
    clip_ari_grid_path = os.path.join(
        results_dir,
        "mnist_dpgrams_c_subsample_effect_clip_ari_grid.pdf"
    )
    plot_subsample_effect_grid(
        results_by_n=clip_results_by_n,
        subsample_sizes=subsample_sizes_hparam,
        x_key="clip_multiplier",
        metric_key="ari",
        x_label=r"Clip Multiplier ($C^*$)",
        y_label="ARI",
        title_prefix="DP-GRAMS-C on MNIST: ARI vs $C^*$ across subsamples",
        out_path=clip_ari_grid_path,
        log_x=True,
    )

    clip_nmi_grid_path = os.path.join(
        results_dir,
        "mnist_dpgrams_c_subsample_effect_clip_nmi_grid.pdf"
    )
    plot_subsample_effect_grid(
        results_by_n=clip_results_by_n,
        subsample_sizes=subsample_sizes_hparam,
        x_key="clip_multiplier",
        metric_key="nmi",
        x_label=r"Clip Multiplier ($C^*$)",
        y_label="NMI",
        title_prefix="DP-GRAMS-C on MNIST: NMI vs $C^*$ across subsamples",
        out_path=clip_nmi_grid_path,
        log_x=True,
    )

    clip_mse_grid_path = os.path.join(
        results_dir,
        "mnist_dpgrams_c_subsample_effect_clip_mse_grid.pdf"
    )
    plot_subsample_effect_grid(
        results_by_n=clip_results_by_n,
        subsample_sizes=subsample_sizes_hparam,
        x_key="clip_multiplier",
        metric_key="mse",
        x_label=r"Clip Multiplier ($C^*$)",
        y_label="Centroid MSE",
        title_prefix="DP-GRAMS-C on MNIST: MSE vs $C^*$ across subsamples",
        out_path=clip_mse_grid_path,
        log_x=True,
    )

    # --- m sweep across subsample sizes ---
    m_results_by_n = subsampling_effect_m_grid(
        X_raw_full=X_raw_full,
        y_full=y_full,
        subsample_sizes=subsample_sizes_hparam,
        m_frac_grid=m_frac_grid_subsample,
        epsilon_modes=epsilon_subsample_effect,
        clip_multiplier_fixed=0.01,
        n_reps=n_runs_subsample_effect,
        scaler=scaler,
        true_means_scaled=true_means_scaled,
        base_seed=45678,
    )

    # Save textual m subsampling-effect results
    m_txt_path = os.path.join(
        results_dir,
        "mnist_dpgrams_c_subsample_effect_minibatch_stats.txt"
    )
    with open(m_txt_path, "w") as f:
        for n_sub in subsample_sizes_hparam:
            f.write(f"=== Subsample n = {n_sub} ===\n")
            header = (
                f"{'m':>8} | {'mean_ari':>9} | {'std_ari':>8} | "
                f"{'mean_nmi':>9} | {'std_nmi':>8} | "
                f"{'mean_mse':>10} | {'std_mse':>9} | "
                f"{'mean_t':>9} | {'std_t':>8}\n"
            )
            f.write(header)
            f.write("-" * len(header) + "\n")
            for r in m_results_by_n.get(n_sub, []):
                line = (
                    f"{r['m']:8d} | "
                    f"{r['mean_ari']:9.4f} | {r['std_ari']:8.4f} | "
                    f"{r['mean_nmi']:9.4f} | {r['std_nmi']:8.4f} | "
                    f"{r['mean_mse']:10.4f} | {r['std_mse']:9.4f} | "
                    f"{r['mean_time']:9.4f} | {r['std_time']:8.4f}\n"
                )
                f.write(line)
            f.write("\n")
    print(f"[{now_str()}] Minibatch subsampling-effect stats saved to {m_txt_path}")

    # 2x2 grids for m subsampling-effect: ARI, NMI, MSE
    m_ari_grid_path = os.path.join(
        results_dir,
        "mnist_dpgrams_c_subsample_effect_m_ari_grid.pdf"
    )
    plot_subsample_effect_grid(
        results_by_n=m_results_by_n,
        subsample_sizes=subsample_sizes_hparam,
        x_key="m",
        metric_key="ari",
        x_label="Minibatch size m",
        y_label="ARI",
        title_prefix="DP-GRAMS-C on MNIST: ARI vs m across subsamples",
        out_path=m_ari_grid_path,
        log_x=True,
    )

    m_nmi_grid_path = os.path.join(
        results_dir,
        "mnist_dpgrams_c_subsample_effect_m_nmi_grid.pdf"
    )
    plot_subsample_effect_grid(
        results_by_n=m_results_by_n,
        subsample_sizes=subsample_sizes_hparam,
        x_key="m",
        metric_key="nmi",
        x_label="Minibatch size m",
        y_label="NMI",
        title_prefix="DP-GRAMS-C on MNIST: NMI vs m across subsamples",
        out_path=m_nmi_grid_path,
        log_x=True,
    )

    m_mse_grid_path = os.path.join(
        results_dir,
        "mnist_dpgrams_c_subsample_effect_m_mse_grid.pdf"
    )
    plot_subsample_effect_grid(
        results_by_n=m_results_by_n,
        subsample_sizes=subsample_sizes_hparam,
        x_key="m",
        metric_key="mse",
        x_label="Minibatch size m",
        y_label="Centroid MSE",
        title_prefix="DP-GRAMS-C on MNIST: MSE vs m across subsamples",
        out_path=m_mse_grid_path,
        log_x=True,
    )

    print(
        f"\n[{now_str()}] All MNIST subsample experiments completed. "
        f"Results in: {results_dir}"
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
