# iris_compare_centroids.py
#
# Clustering comparison on the Iris dataset:
# - Non-private Mean Shift (MS) + merging
# - DP-GRAMS-C (dpms_private: DP-GRAMS modes + nearest-center labels)
# - Non-private KMeans
# - DP-KMeans (diffprivlib)
#
# Notes:
# - DP-GRAMS-C uses (epsilon_modes, delta) only for private modes.
# - Cluster labels are assigned deterministically by nearest private centroid

import numpy as np
import os
import sys
import time
import csv
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
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
# Setup
# ---------------------------------------------------------------------
results_dir = "results/iris_centroid_comparison"
os.makedirs(results_dir, exist_ok=True)

sns.set(style="whitegrid", context="talk")

iris = load_iris()
X_raw = iris.data
y_true = iris.target

n, d = X_raw.shape

merge_n_clusters = 3          # Iris has 3 classes
ms_T = int(np.log(n))        # iterations for mean-shift
ms_p = 1                     # use all points as seeds
delta = 1e-5
p_seed = 1                   # DP-GRAMS-C: p0 (seed fraction)
base_rng_seed = 12
n_runs = 20

# We'll cluster in the original feature space,
# but evaluate centroid MSE in standardized space.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
true_means_scaled = np.array([
    X_scaled[y_true == c].mean(axis=0)
    for c in np.unique(y_true)
])

X = X_raw
true_means = np.array([
    X[y_true == c].mean(axis=0)
    for c in np.unique(y_true)
])

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def compute_metrics(y_true, labels, X):
    """Compute ARI, NMI, Silhouette for given clustering."""
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
    return ari, nmi, sil

def timed(func, *args, **kwargs):
    """Return (result, runtime_seconds) for calling func."""
    start = time.time()
    out = func(*args, **kwargs)
    return out, time.time() - start

def relabel_clusters(y_true, y_pred, n_clusters=3):
    """
    Relabel predicted clusters to best match true labels via Hungarian algorithm,
    for comparability across methods.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(n_clusters))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in y_pred])

# ---------------------------------------------------------------------
# 1) Mean Shift baseline (non-private)
# ---------------------------------------------------------------------
rng_ms = np.random.default_rng(base_rng_seed)

(raw_ms_modes, ms_time) = timed(
    mean_shift,
    X,
    None,
    ms_T,
    silverman_bandwidth(X),  # bandwidth for MS
    ms_p,
    rng_ms
)
ms_merged_modes = merge_modes_agglomerative(
    raw_ms_modes,
    n_clusters=merge_n_clusters,
    random_state=base_rng_seed
)

# Assign points to nearest MS centroid
dists_ms = np.linalg.norm(X[:, None, :] - ms_merged_modes[None, :, :], axis=2)
labels_ms = np.argmin(dists_ms, axis=1)
labels_ms = relabel_clusters(y_true, labels_ms, merge_n_clusters)

# ---------------------------------------------------------------------
# 2) DP-GRAMS-C: private modes + deterministic nearest-center labels
# ---------------------------------------------------------------------
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
    clip_multiplier=0.1
)
labels_dpms = relabel_clusters(y_true, labels_dpms, merge_n_clusters)

# ---------------------------------------------------------------------
# 3) Non-private KMeans
# ---------------------------------------------------------------------
kmeans = KMeans(
    n_clusters=merge_n_clusters,
    random_state=base_rng_seed,
    n_init=10
)
(labels_km, km_time) = timed(kmeans.fit_predict, X)
modes_km = kmeans.cluster_centers_
labels_km = relabel_clusters(y_true, labels_km, merge_n_clusters)

# ---------------------------------------------------------------------
# 4) DP-KMeans (diffprivlib)
# ---------------------------------------------------------------------
dp_kmeans = DPKMeans(
    n_clusters=merge_n_clusters,
    epsilon=1.0,
    random_state=base_rng_seed
)
try:
    (_, dpkm_time) = timed(dp_kmeans.fit, X)
except ValueError:
    # If bounds are required, infer from data
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

# ---------------------------------------------------------------------
# Aggregate metrics for all methods
# ---------------------------------------------------------------------
algorithms = ["MS Clustering", "DP-GRAMS-C", "KMeans", "DP-KMeans"]
labels_list = [labels_ms, labels_dpms, labels_km, labels_dpkm]
modes_list = [ms_merged_modes, modes_dpms, modes_km, modes_dpkm]
runtimes = [ms_time, dpms_time, km_time, dpkm_time]

metrics = {}
mse_centroids = {}

for alg, labels, modes in zip(algorithms, labels_list, modes_list):
    metrics[alg] = compute_metrics(y_true, labels, X)
    # Evaluate centroid accuracy in standardized space
    modes_scaled = scaler.transform(modes)
    mse_centroids[alg] = mode_matching_mse(
        true_means_scaled.copy(),
        modes_scaled.copy()
    )

# ---------------------------------------------------------------------
# Save metrics to CSV and TXT
# ---------------------------------------------------------------------
csv_file = os.path.join(results_dir, "clustering_metrics.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Algorithm", "ARI", "NMI", "Silhouette", "MSE", "Runtime(s)"])
    for alg, rt in zip(algorithms, runtimes):
        ari, nmi, sil = metrics[alg]
        writer.writerow([alg, ari, nmi, sil, mse_centroids[alg], rt])

txt_file = os.path.join(results_dir, "clustering_metrics.txt")
with open(txt_file, "w") as f:
    for alg, rt in zip(algorithms, runtimes):
        ari, nmi, sil = metrics[alg]
        f.write(
            f"{alg}: ARI={ari:.4f}, NMI={nmi:.4f}, "
            f"Silhouette={sil:.4f}, MSE={mse_centroids[alg]:.6f}, "
            f"Runtime={rt:.4f}s\n"
        )

print("\nMetrics (raw features for clustering, MSE on scaled features):")
print(f"{'Alg':<12} {'ARI':>6} {'NMI':>6} {'Silhouette':>10} "
      f"{'MSE_centroids':>14} {'Runtime(s)':>12}")
for alg, rt in zip(algorithms, runtimes):
    ari, nmi, sil = metrics[alg]
    print(
        f"{alg:<12} {ari:6.3f} {nmi:6.3f} {sil:10.3f} "
        f"{mse_centroids[alg]:14.6f} {rt:12.4f}"
    )

# ---------------------------------------------------------------------
# PCA visualization of clusterings
# ---------------------------------------------------------------------
pca = PCA(n_components=2, random_state=base_rng_seed)
X_2d = pca.fit_transform(X)
true_means_2d = pca.transform(true_means)
modes_2d_list = [pca.transform(m) for m in modes_list]

fig, axes = plt.subplots(1, 4, figsize=(20, 8))
palette = sns.color_palette("tab10", merge_n_clusters)

global_handles = []
global_labels = []

for ax, alg, labels_pred, modes_2d in zip(axes, algorithms, labels_list, modes_2d_list):
    # Clustered points
    for i, color in enumerate(palette):
        sc = ax.scatter(
            X_2d[labels_pred == i, 0],
            X_2d[labels_pred == i, 1],
            c=[color],
            s=40,
            alpha=0.7
        )
        if len(global_handles) < merge_n_clusters:
            global_handles.append(sc)
            global_labels.append(f"Cluster {i}")

    # True centroids
    true_sc = ax.scatter(
        true_means_2d[:, 0],
        true_means_2d[:, 1],
        marker="X",
        c="magenta",
        s=180,
        linewidths=2
    )
    if "True means" not in global_labels:
        global_handles.append(true_sc)
        global_labels.append("True means")

    # Estimated centroids
    modes_sc = ax.scatter(
        modes_2d[:, 0],
        modes_2d[:, 1],
        marker="X",
        c="blue",
        s=120,
        linewidths=2
    )
    if "Estimated modes" not in global_labels:
        global_handles.append(modes_sc)
        global_labels.append("Estimated modes")

    ax.set_title(alg, fontsize=16)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

legend = fig.legend(
    global_handles,
    global_labels,
    fontsize=10,
    loc="lower center",
    ncol=6,
    bbox_to_anchor=(0.5, 0.01),
    title="Cluster Assignments & Centroids"
)
plt.setp(legend.get_title(), fontsize=12, fontweight="bold")
fig.suptitle("Clustering Comparison on Iris Dataset", fontsize=20, y=0.90)

save_path_clusters = os.path.join(results_dir, "iris_clustering_comparison.png")
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(save_path_clusters, dpi=300, bbox_inches='tight')
plt.show()
print("Clustering comparison plot saved to:", save_path_clusters)

# ---------------------------------------------------------------------
# Privacy–utility curves: DP-GRAMS-C vs DP-KMeans as ε_modes varies
# ---------------------------------------------------------------------
eps_modes_list = [0.1, 0.5, 1, 2, 5, 10]

dpms_metrics = {m: [] for m in ["ARI", "NMI", "Silhouette", "MSE"]}
dpkm_metrics = {m: [] for m in ["ARI", "NMI", "Silhouette", "MSE"]}
dpms_err = {m: [] for m in ["ARI", "NMI", "Silhouette", "MSE"]}
dpkm_err = {m: [] for m in ["ARI", "NMI", "Silhouette", "MSE"]}
dpms_times_mean, dpkm_times_mean = [], []
dpms_times_std, dpkm_times_std = [], []

for eps in eps_modes_list:
    # --- DP-GRAMS-C ---
    ari_list, nmi_list, sil_list, mse_list = [], [], [], []
    dpms_times = []

    for run in range(n_runs):
        rng_run = np.random.default_rng(base_rng_seed + run)
        t0 = time.time()
        modes_dpms_run, labels_dpms_run = dpms_private(
            data=X,
            epsilon_modes=eps,
            delta=delta,
            p_seed=p_seed,
            rng=rng_run,
            k_est=merge_n_clusters,
            bandwidth_multiplier=1.0,
            clip_multiplier=0.01
        )
        t_dpms = time.time() - t0
        dpms_times.append(t_dpms)

        labels_dpms_run = relabel_clusters(y_true, labels_dpms_run, merge_n_clusters)
        run_ari, run_nmi, run_sil = compute_metrics(y_true, labels_dpms_run, X)

        modes_scaled = scaler.transform(modes_dpms_run)
        run_mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

        ari_list.append(run_ari)
        nmi_list.append(run_nmi)
        sil_list.append(run_sil)
        mse_list.append(run_mse)

    dpms_metrics["ARI"].append(np.mean(ari_list))
    dpms_metrics["NMI"].append(np.mean(nmi_list))
    dpms_metrics["Silhouette"].append(np.mean(sil_list))
    dpms_metrics["MSE"].append(np.mean(mse_list))
    dpms_err["ARI"].append(np.std(ari_list, ddof=1))
    dpms_err["NMI"].append(np.std(nmi_list, ddof=1))
    dpms_err["Silhouette"].append(np.std(sil_list, ddof=1))
    dpms_err["MSE"].append(np.std(mse_list, ddof=1))
    dpms_times_mean.append(float(np.mean(dpms_times)))
    dpms_times_std.append(float(np.std(dpms_times, ddof=1)))

    # --- DP-KMeans ---
    ari_list, nmi_list, sil_list, mse_list = [], [], [], []
    dpkm_times = []

    for run in range(n_runs):
        t0 = time.time()
        dp_kmeans_run = DPKMeans(
            n_clusters=merge_n_clusters,
            epsilon=eps,
            random_state=base_rng_seed + run
        )
        try:
            dp_kmeans_run.fit(X)
        except ValueError:
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            dp_kmeans_run.fit(X, bounds=(mins, maxs))
        t_dpkm = time.time() - t0
        dpkm_times.append(t_dpkm)

        labels_dpkm_run = (
            dp_kmeans_run.labels_.astype(int)
            if hasattr(dp_kmeans_run, "labels_")
            else dp_kmeans_run.predict(X)
        )
        labels_dpkm_run = relabel_clusters(y_true, labels_dpkm_run, merge_n_clusters)
        modes_dpkm_run = dp_kmeans_run.cluster_centers_

        run_ari, run_nmi, run_sil = compute_metrics(y_true, labels_dpkm_run, X)
        modes_scaled = scaler.transform(modes_dpkm_run)
        run_mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

        ari_list.append(run_ari)
        nmi_list.append(run_nmi)
        sil_list.append(run_sil)
        mse_list.append(run_mse)

    dpkm_metrics["ARI"].append(np.mean(ari_list))
    dpkm_metrics["NMI"].append(np.mean(nmi_list))
    dpkm_metrics["Silhouette"].append(np.mean(sil_list))
    dpkm_metrics["MSE"].append(np.mean(mse_list))
    dpkm_err["ARI"].append(np.std(ari_list, ddof=1))
    dpkm_err["NMI"].append(np.std(nmi_list, ddof=1))
    dpkm_err["Silhouette"].append(np.std(sil_list, ddof=1))
    dpkm_err["MSE"].append(np.std(mse_list, ddof=1))
    dpkm_times_mean.append(float(np.mean(dpkm_times)))
    dpkm_times_std.append(float(np.std(dpkm_times, ddof=1)))

# ---------------------------------------------------------------------
# Save privacy–utility metrics
# ---------------------------------------------------------------------
csv_file = os.path.join(results_dir, "privacy_utility_metrics.csv")
with open(csv_file, "w", newline="") as f:
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
            dpms_metrics["ARI"][i],
            dpms_metrics["NMI"][i],
            dpms_metrics["Silhouette"][i],
            dpms_metrics["MSE"][i],
            dpms_err["ARI"][i],
            dpms_err["NMI"][i],
            dpms_err["Silhouette"][i],
            dpms_err["MSE"][i],
            dpms_times_mean[i],
            dpms_times_std[i]
        ])
        writer.writerow([
            eps, "DP-KMeans",
            dpkm_metrics["ARI"][i],
            dpkm_metrics["NMI"][i],
            dpkm_metrics["Silhouette"][i],
            dpkm_metrics["MSE"][i],
            dpkm_err["ARI"][i],
            dpkm_err["NMI"][i],
            dpkm_err["Silhouette"][i],
            dpkm_err["MSE"][i],
            dpkm_times_mean[i],
            dpkm_times_std[i]
        ])

print("All Iris experiments completed. Results saved in:", results_dir)

# ---------------------------------------------------------------------
# Privacy–utility plots: 4 separate figures (ARI, NMI, Silhouette, MSE)
# ---------------------------------------------------------------------
metrics_names = ["ARI", "NMI", "Silhouette", "MSE"]

for metric in metrics_names:
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
    ax.set_title(f"Privacy–Utility: {metric}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    plt.tight_layout()
    out_path = os.path.join(
        results_dir,
        f"iris_privacy_utility_{metric.lower()}.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Privacy–utility ({metric}) figure saved to:", out_path)
