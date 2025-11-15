# mnist_compare_centroids_parallel.py
#
# Clustering comparison on MNIST (PCA-50):
# - MS + merge (non-private)
# - DP-GRAMS-C style (dpms_private): private centroids + nearest-center labels
# - KMeans (non-private)
# - DP-KMeans (diffprivlib)
#
# Also: privacy–utility curves vs ε for DP-GRAMS-C vs DP-KMeans.
# All DP-GRAMS-C code here uses ONLY private modes + deterministic

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
    silhouette_score,
    confusion_matrix
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
ms_T = 10                 # MS iterations
ms_p = 0.1                # fraction of seeds for MS
delta = 1e-5
p_seed = 0.1              # seed fraction for DP-GRAMS-C
base_rng_seed = 41
n_runs = 20

results_dir = "results/mnist_centroid_comparison"
os.makedirs(results_dir, exist_ok=True)

sns.set(style="whitegrid", context="talk")

eps_modes_list = [0.1, 0.5, 1, 2, 5, 10]
MAX_WORKERS = min(8, os.cpu_count() or 8)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def compute_metrics(y_true, labels, X):
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
    return ari, nmi, sil

def timed(func, *args, **kwargs):
    start = time.time()
    out = func(*args, **kwargs)
    return out, time.time() - start

def relabel_clusters(y_true, y_pred, n_clusters=10):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_clusters))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in y_pred])

# ---------------------------------------------------------------------
# Workers for parallel privacy–utility experiments
# ---------------------------------------------------------------------

def dpms_single_run(run, X, y_true, eps_modes, merge_n_clusters, pca_50, scaler, true_means_scaled, base_seed=41):
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
    ari, nmi, sil = compute_metrics(y_true, labels_dpms_run, X)

    modes_orig = pca_50.inverse_transform(modes_dpms_run)
    modes_scaled = scaler.transform(modes_orig)
    mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

    print(f"[{now_str()}] [DPMS worker] end   run={run} eps={eps_modes} "
          f"ARI={ari:.4f} MSE={mse:.4f} time={runtime:.2f}s")

    return modes_dpms_run, labels_dpms_run, (ari, nmi, sil), float(mse), float(runtime), int(seed)

def dpkm_single_run(run, X, y_true, eps_modes, merge_n_clusters, pca_50, scaler, true_means_scaled, base_seed=41):
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

    labels_dpkm_run = dp_kmeans_run.labels_.astype(int) if hasattr(dp_kmeans_run, "labels_") else dp_kmeans_run.predict(X)
    labels_dpkm_run = relabel_clusters(y_true, labels_dpkm_run, merge_n_clusters)
    ari, nmi, sil = compute_metrics(y_true, labels_dpkm_run, X)

    modes_dpkm_run = dp_kmeans_run.cluster_centers_
    modes_orig = pca_50.inverse_transform(modes_dpkm_run)
    modes_scaled = scaler.transform(modes_orig)
    mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

    print(f"[{now_str()}] [DP-KMeans worker] end   run={run} eps={eps_modes} "
          f"ARI={ari:.4f} MSE={mse:.4f} time={runtime:.2f}s")

    return modes_dpkm_run, labels_dpkm_run, (ari, nmi, sil), float(mse), float(runtime), int(seed)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # ------------------------------------------------------------
    # Load MNIST
    # ------------------------------------------------------------
    print(f"[{now_str()}] Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X_raw = mnist.data.astype(np.float32)
    y_true = mnist.target.astype(int)
    print(f"[{now_str()}] MNIST data shape: {X_raw.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    true_means_scaled = np.array([X_scaled[y_true == c].mean(axis=0) for c in np.unique(y_true)])

    # PCA-50 for clustering
    pca_cache = os.path.join(results_dir, "mnist_pca_50d.npy")
    if os.path.exists(pca_cache):
        print(f"[{now_str()}] Loading cached PCA-50 embedding...")
        X = np.load(pca_cache)
        pca_50 = PCA(n_components=50, random_state=base_rng_seed)
        pca_50.fit(X_raw)
    else:
        print(f"[{now_str()}] Computing PCA-50 embedding...")
        pca_50 = PCA(n_components=50, random_state=base_rng_seed)
        X = pca_50.fit_transform(X_raw)
        np.save(pca_cache, X)
        print(f"[{now_str()}] PCA-50 embedding saved to: {pca_cache}")

    true_means = np.array([X[y_true == c].mean(axis=0) for c in np.unique(y_true)])

    # ------------------------------------------------------------
    # Single-run baselines
    # ------------------------------------------------------------
    print(f"[{now_str()}] Running single-run baselines...")

    rng_ms = np.random.default_rng(base_rng_seed)
    h_ms = silverman_bandwidth(X)
    (raw_ms_modes, ms_time) = timed(mean_shift, X, None, ms_T, h_ms, ms_p, rng_ms)
    ms_merged_modes = merge_modes_agglomerative(raw_ms_modes, n_clusters=merge_n_clusters, random_state=base_rng_seed)
    dists_ms = np.linalg.norm(X[:, None, :] - ms_merged_modes[None, :, :], axis=2)
    labels_ms = np.argmin(dists_ms, axis=1)
    labels_ms = relabel_clusters(y_true, labels_ms, merge_n_clusters)

    kmeans = KMeans(n_clusters=merge_n_clusters, random_state=base_rng_seed, n_init=10)
    (labels_km, km_time) = timed(kmeans.fit_predict, X)
    modes_km = kmeans.cluster_centers_
    labels_km = relabel_clusters(y_true, labels_km, merge_n_clusters)

    dp_kmeans = DPKMeans(n_clusters=merge_n_clusters, epsilon=1.0, random_state=base_rng_seed)
    try:
        (_, dpkm_time) = timed(dp_kmeans.fit, X)
    except ValueError:
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        (_, dpkm_time) = timed(dp_kmeans.fit, X, bounds=(mins, maxs))
    labels_dpkm = dp_kmeans.labels_.astype(int) if hasattr(dp_kmeans, "labels_") else dp_kmeans.predict(X)
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
        ari, nmi, sil = compute_metrics(y_true, labels, X)
        metrics[alg] = (ari, nmi, sil)

        modes_orig = pca_50.inverse_transform(modes)
        modes_scaled = scaler.transform(modes_orig)
        mse_centroids[alg] = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

    single_csv = os.path.join(results_dir, "clustering_metrics_single_run.csv")
    with open(single_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "ARI", "NMI", "Silhouette", "MSE", "Runtime(s)"])
        for alg, rt in zip(algorithms, runtimes):
            ari, nmi, sil = metrics[alg]
            writer.writerow([alg, ari, nmi, sil, mse_centroids[alg], rt])
    print(f"[{now_str()}] Single-run clustering metrics saved to {single_csv}")

    single_txt = os.path.join(results_dir, "clustering_metrics_single_run.txt")
    with open(single_txt, "w") as f:
        for alg, rt in zip(algorithms, runtimes):
            ari, nmi, sil = metrics[alg]
            f.write(f"{alg}: ARI={ari:.4f}, NMI={nmi:.4f}, Silhouette={sil:.4f}, "
                    f"MSE={mse_centroids[alg]:.6f}, Runtime={rt:.4f}s\n")
    print(f"[{now_str()}] Single-run clustering metrics (text) saved to {single_txt}")

    # ------------------------------------------------------------
    # Clustering comparison: 1 row x 4 columns
    # ------------------------------------------------------------
    print(f"[{now_str()}] Generating clustering comparison plot (1x4)...")

    pca_2d = PCA(n_components=2, random_state=base_rng_seed)
    X_2d = pca_2d.fit_transform(X)
    true_means_2d = pca_2d.transform(true_means)
    modes_2d_list = [pca_2d.transform(m) for m in modes_list]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    palette = sns.color_palette("tab10", merge_n_clusters)
    global_handles, global_labels = [], []

    for ax, alg, labels_pred, modes_2d in zip(axes, algorithms, labels_list, modes_2d_list):
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

    fig.suptitle("Clustering Comparison on MNIST (PCA-50)", fontsize=18, y=1.02)

    save_path_clusters = os.path.join(results_dir, "mnist_clustering_comparison.png")
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(save_path_clusters, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[{now_str()}] Clustering comparison saved to: {save_path_clusters}")

    # ------------------------------------------------------------
    # Privacy–utility experiments
    # ------------------------------------------------------------
    print(f"[{now_str()}] Starting privacy–utility experiments "
          f"({len(eps_modes_list)} eps × {n_runs} runs)...")

    dpms_perrun_csv = os.path.join(results_dir, "dpms_per_run_results.csv")
    dpkm_perrun_csv = os.path.join(results_dir, "dpkm_per_run_results.csv")

    if not os.path.exists(dpms_perrun_csv):
        with open(dpms_perrun_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Algorithm","Eps_modes","Run","Seed","ARI","NMI","Silhouette","MSE","Runtime_s","timestamp"])
    if not os.path.exists(dpkm_perrun_csv):
        with open(dpkm_perrun_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Algorithm","Eps_modes","Run","Seed","ARI","NMI","Silhouette","MSE","Runtime_s","timestamp"])

    dpms_metrics = {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}
    dpkm_metrics = {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}
    dpms_err = {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}
    dpkm_err = {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}
    dpms_times_mean, dpms_times_std = [], []
    dpkm_times_mean, dpkm_times_std = [], []

    for eps in eps_modes_list:
        print(f"\n[{now_str()}] === ε_modes = {eps} ===")

        # DPMS
        ari_list, nmi_list, sil_list, mse_list, runtimes_list = [], [], [], [], []
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
                    _, _, (ari, nmi, sil), mse, runtime, seed = fut.result()
                except Exception as e:
                    print(f"[{now_str()}] [ERROR] DPMS run {run_id}, eps={eps}: {e}")
                    continue

                ari_list.append(ari); nmi_list.append(nmi)
                sil_list.append(sil); mse_list.append(mse)
                runtimes_list.append(runtime)

                with open(dpms_perrun_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["DP-GRAMS-C", eps, run_id, seed, ari, nmi, sil, mse, runtime, now_str()])

        for key, arr in zip(["ARI","NMI","Silhouette","MSE"], [ari_list,nmi_list,sil_list,mse_list]):
            if arr:
                dpms_metrics[key].append(float(np.mean(arr)))
                dpms_err[key].append(float(np.std(arr, ddof=1)))
            else:
                dpms_metrics[key].append(np.nan)
                dpms_err[key].append(np.nan)
        dpms_times_mean.append(float(np.mean(runtimes_list)) if runtimes_list else np.nan)
        dpms_times_std.append(float(np.std(runtimes_list, ddof=1)) if len(runtimes_list) > 1 else 0.0)

        # DP-KMeans
        ari_list, nmi_list, sil_list, mse_list, runtimes_list = [], [], [], [], []
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
                    _, _, (ari, nmi, sil), mse, runtime, seed = fut.result()
                except Exception as e:
                    print(f"[{now_str()}] [ERROR] DP-KMeans run {run_id}, eps={eps}: {e}")
                    continue

                ari_list.append(ari); nmi_list.append(nmi)
                sil_list.append(sil); mse_list.append(mse)
                runtimes_list.append(runtime)

                with open(dpkm_perrun_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["DP-KMeans", eps, run_id, seed, ari, nmi, sil, mse, runtime, now_str()])

        for key, arr in zip(["ARI","NMI","Silhouette","MSE"], [ari_list,nmi_list,sil_list,mse_list]):
            if arr:
                dpkm_metrics[key].append(float(np.mean(arr)))
                dpkm_err[key].append(float(np.std(arr, ddof=1)))
            else:
                dpkm_metrics[key].append(np.nan)
                dpkm_err[key].append(np.nan)
        dpkm_times_mean.append(float(np.mean(runtimes_list)) if runtimes_list else np.nan)
        dpkm_times_std.append(float(np.std(runtimes_list, ddof=1)) if len(runtimes_list) > 1 else 0.0)

    # Aggregate CSV
    agg_csv = os.path.join(results_dir, "privacy_utility_metrics_aggregated.csv")
    with open(agg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Eps_modes", "Algorithm",
            "ARI", "NMI", "Silhouette", "MSE",
            "Std_ARI", "Std_NMI", "Std_Silhouette", "Std_MSE",
            "Mean_Runtime(s)", "Std_Runtime(s)"
        ])
        for i, eps in enumerate(eps_modes_list):
            writer.writerow([
                eps, "DP-GRAMS-C",
                dpms_metrics["ARI"][i], dpms_metrics["NMI"][i],
                dpms_metrics["Silhouette"][i], dpms_metrics["MSE"][i],
                dpms_err["ARI"][i], dpms_err["NMI"][i],
                dpms_err["Silhouette"][i], dpms_err["MSE"][i],
                dpms_times_mean[i], dpms_times_std[i]
            ])
            writer.writerow([
                eps, "DP-KMeans",
                dpkm_metrics["ARI"][i], dpkm_metrics["NMI"][i],
                dpkm_metrics["Silhouette"][i], dpkm_metrics["MSE"][i],
                dpkm_err["ARI"][i], dpkm_err["NMI"][i],
                dpkm_err["Silhouette"][i], dpkm_err["MSE"][i],
                dpkm_times_mean[i], dpkm_times_std[i]
            ])
    print(f"[{now_str()}] Aggregated privacy–utility metrics saved to: {agg_csv}")

    # 4 separate PU plots
    for metric in ["ARI", "NMI", "Silhouette", "MSE"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(eps_modes_list, dpms_metrics[metric], yerr=dpms_err[metric],
                    marker='o', linestyle='-', linewidth=2, markersize=7, capsize=3,
                    label="DP-GRAMS-C")
        ax.errorbar(eps_modes_list, dpkm_metrics[metric], yerr=dpkm_err[metric],
                    marker='s', linestyle='-', linewidth=2, markersize=7, capsize=3,
                    label="DP-KMeans")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\epsilon_{\mathrm{modes}}$")
        ax.set_ylabel(metric)
        ax.set_title(f"Privacy–Utility: {metric} (MNIST)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        plt.tight_layout()
        out_path = os.path.join(results_dir, f"mnist_privacy_utility_{metric.lower()}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"[{now_str()}] Privacy–utility ({metric}) saved to: {out_path}")

    print(f"[{now_str()}] All MNIST experiments completed. Results in: {results_dir}")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
