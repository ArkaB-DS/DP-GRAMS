# gene50_compare_centroids.py

import numpy as np
import os
import sys
import time
import csv
import zipfile
import tarfile
import urllib.request
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
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
results_dir = "results/gene50_centroid_comparison"
os.makedirs(results_dir, exist_ok=True)

sns.set(style="whitegrid", context="talk")

# Dataset download/extract
DATA_ROOT = Path("data/uci_gene_expression_rnaseq").resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"
UCI_ZIP_PATH = DATA_ROOT / "gene+expression+cancer+rna+seq.zip"
EXTRACT_DIR = DATA_ROOT / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

merge_n_clusters = 5          # BRCA/COAD/KIRC/LUAD/PRAD
delta = 1e-5
p_seed = 0.1
base_rng_seed = 12
n_runs = 20

# Mean Shift params
ms_p = 0.1

# We'll do: Standardize -> PCA50 for clustering
pca_dim = 50

# DP-GRAMS-C params (you can tune clip_multiplier)
dpms_clip_multiplier = 0.1
dpms_bandwidth_multiplier = 1.0

# Privacy-utility sweep
eps_modes_list = [0.25, 0.5, 1.0, 2.5, 5.0]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def compute_metrics(y_true, labels):
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    return ari, nmi

def timed(func, *args, **kwargs):
    start = time.time()
    out = func(*args, **kwargs)
    return out, time.time() - start

def relabel_clusters(y_true, y_pred, n_clusters):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_clusters))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, label) for label in y_pred], dtype=int)

def download_if_needed(url: str, out_path: Path):
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    print(f"[download] Fetching: {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"[download] Saved -> {out_path}")

def unzip_if_needed(zip_path: Path, extract_dir: Path):
    marker = extract_dir / ".unzipped_ok"
    if marker.exists():
        return
    print(f"[extract] Unzipping -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok")
    print("[extract] Unzip done.")

def untar_all_archives(extract_dir: Path):
    marker = extract_dir / ".untar_ok"
    if marker.exists():
        return
    archives = (
        list(extract_dir.rglob("*.tar.gz")) +
        list(extract_dir.rglob("*.tgz")) +
        list(extract_dir.rglob("*.tar"))
    )
    for arch in archives:
        print(f"[extract] Untarring: {arch.name}")
        mode = "r:gz" if arch.suffixes[-2:] == [".tar", ".gz"] or arch.suffix == ".tgz" else "r"
        with tarfile.open(arch, mode) as tf:
            tf.extractall(arch.parent)
    marker.write_text("ok")
    print("[extract] Tar extraction done.")

def find_data_and_labels_csv(extract_dir: Path):
    data_csvs = list(extract_dir.rglob("data.csv"))
    label_csvs = list(extract_dir.rglob("labels.csv"))
    if data_csvs and label_csvs:
        return data_csvs[0], label_csvs[0]
    raise FileNotFoundError("Could not find data.csv and labels.csv after extraction.")

def load_gene_dataset(extract_dir: Path):

    data_csv, labels_csv = find_data_and_labels_csv(extract_dir)
    print(f"[data] Using data file:   {data_csv}")
    print(f"[data] Using labels file: {labels_csv}")

    # --- load data.csv robustly (drop first col: sample id) ---
    raw = np.genfromtxt(
        data_csv,
        delimiter=",",
        dtype=None,
        encoding="utf-8",
        skip_header=1
    )

    if raw.dtype.names is not None:
        fields = list(raw.dtype.names)
        numeric_fields = fields[1:]
        X = np.vstack([raw[name].astype(float) for name in numeric_fields]).T
    else:
        raw = np.asarray(raw)
        X = raw[:, 1:].astype(float)

    # --- labels: use last column if multiple ---
    y_raw = np.genfromtxt(labels_csv, delimiter=",", dtype=str, encoding="utf-8", skip_header=1)
    if y_raw.ndim == 0:
        y_raw = np.array([str(y_raw)])
    if y_raw.ndim > 1:
        y_raw = y_raw[:, -1]
    y_raw = np.array([s.strip() for s in y_raw], dtype=str)

    uniq = sorted(set(y_raw.tolist()))
    name_to_int = {name: i for i, name in enumerate(uniq)}
    y = np.array([name_to_int[name] for name in y_raw], dtype=int)

    return X, y, uniq

def fit_dp_kmeans_version_robust(X, eps, seed, n_clusters, bounds_tuple):
    """
    diffprivlib API differs across versions. Try:
      1) constructor(bounds=...), fit(X)
      2) fit(X, bounds=...)
      3) constructor(data_norm=...), fit(X)
      4) constructor(...), fit(X)
    """
    t0 = time.time()

    # (1) constructor(bounds=...)
    try:
        model = DPKMeans(n_clusters=n_clusters, epsilon=eps, random_state=seed, bounds=bounds_tuple)
        model.fit(X)
        return model, time.time() - t0
    except Exception:
        pass

    # (2) fit(..., bounds=...)
    try:
        model = DPKMeans(n_clusters=n_clusters, epsilon=eps, random_state=seed)
        model.fit(X, bounds=bounds_tuple)
        return model, time.time() - t0
    except Exception:
        pass

    # (3) data_norm fallback
    try:
        data_norm = float(np.max(np.linalg.norm(X, axis=1)))
        model = DPKMeans(n_clusters=n_clusters, epsilon=eps, random_state=seed, data_norm=data_norm)
        model.fit(X)
        return model, time.time() - t0
    except Exception:
        pass

    # (4) plain fit
    model = DPKMeans(n_clusters=n_clusters, epsilon=eps, random_state=seed)
    model.fit(X)
    return model, time.time() - t0

# ---------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------
download_if_needed(UCI_ZIP_URL, UCI_ZIP_PATH)
unzip_if_needed(UCI_ZIP_PATH, EXTRACT_DIR)
untar_all_archives(EXTRACT_DIR)

X_raw, y_true, class_names = load_gene_dataset(EXTRACT_DIR)
n, d = X_raw.shape
print(f"[data] X shape = {X_raw.shape} (n={n}, d={d})")
print(f"[data] classes ({len(class_names)}): {class_names}")

ms_T = int(np.ceil(np.log(max(2, n))))

# ---------------------------------------------------------------------
# Standardize + PCA50 for clustering
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca50 = PCA(n_components=pca_dim, random_state=base_rng_seed)
X = pca50.fit_transform(X_scaled)    # <-- cluster in PCA50 space

# True class means in the clustering space
true_means = np.array([X[y_true == c].mean(axis=0) for c in range(merge_n_clusters)])

# For centroid MSE evaluation in "standardized space" :
# compare back-projected standardized centroids
true_means_scaled = np.array([X_scaled[y_true == c].mean(axis=0) for c in range(merge_n_clusters)])

def pca50_to_scaled(centers_50):
    return pca50.inverse_transform(centers_50)

# ---------------------------------------------------------------------
# Bandwidth (compute once, reuse)
# ---------------------------------------------------------------------
h = silverman_bandwidth(X)

# ---------------------------------------------------------------------
# 1) Mean Shift baseline (non-private)
# ---------------------------------------------------------------------
rng_ms = np.random.default_rng(base_rng_seed)

(raw_ms_modes, ms_time) = timed(
    mean_shift,
    X,
    T=ms_T,
    bandwidth=h,
    p=ms_p,
    seed=rng_ms,
)

ms_merged_modes = merge_modes_agglomerative(
    raw_ms_modes,
    n_clusters=merge_n_clusters,
    random_state=base_rng_seed
)

dists_ms = np.linalg.norm(X[:, None, :] - ms_merged_modes[None, :, :], axis=2)
labels_ms = np.argmin(dists_ms, axis=1).astype(int)
labels_ms = relabel_clusters(y_true, labels_ms, merge_n_clusters)

# ---------------------------------------------------------------------
# 2) DP-GRAMS-C
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
    bandwidth_multiplier=dpms_bandwidth_multiplier,
    clip_multiplier=dpms_clip_multiplier,
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
labels_km = relabel_clusters(y_true, labels_km.astype(int), merge_n_clusters)

# ---------------------------------------------------------------------
# 4) DP-KMeans (diffprivlib) - version robust
# ---------------------------------------------------------------------
mins = X.min(axis=0)
maxs = X.max(axis=0)
bounds_tuple = (mins, maxs)

dp_kmeans, dpkm_time = fit_dp_kmeans_version_robust(
    X=X,
    eps=1.0,
    seed=base_rng_seed,
    n_clusters=merge_n_clusters,
    bounds_tuple=bounds_tuple
)

labels_dpkm = (
    dp_kmeans.labels_.astype(int)
    if hasattr(dp_kmeans, "labels_")
    else dp_kmeans.predict(X).astype(int)
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
    metrics[alg] = compute_metrics(y_true, labels)

    # Evaluate centroid MSE in standardized (gene) space:
    # transform centroids back to standardized gene space, compare to true class means.
    modes_scaled = pca50_to_scaled(modes)
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
    writer.writerow(["Algorithm", "ARI", "NMI", "MSE", "Runtime(s)"])
    for alg, rt in zip(algorithms, runtimes):
        ari, nmi = metrics[alg]
        writer.writerow([alg, ari, nmi, mse_centroids[alg], rt])

txt_file = os.path.join(results_dir, "clustering_metrics.txt")
with open(txt_file, "w") as f:
    f.write(f"Classes: {class_names}\n")
    f.write(f"PCA dims for clustering: {pca_dim}\n")
    f.write(f"Silverman bandwidth h = {float(h):.8f}\n")
    for alg, rt in zip(algorithms, runtimes):
        ari, nmi = metrics[alg]
        f.write(
            f"{alg}: ARI={ari:.4f}, NMI={nmi:.4f}, "
            f"MSE={mse_centroids[alg]:.6f}, Runtime={rt:.4f}s\n"
        )

print("\nMetrics (clustering on PCA50, MSE on standardized gene space):")
print(f"{'Alg':<12} {'ARI':>6} {'NMI':>6} {'MSE_centroids':>14} {'Runtime(s)':>12}")
for alg, rt in zip(algorithms, runtimes):
    ari, nmi = metrics[alg]
    print(
        f"{alg:<12} {ari:6.3f} {nmi:6.3f} "
        f"{mse_centroids[alg]:14.6f} {rt:12.4f}"
    )

# ---------------------------------------------------------------------
# PCA visualization of clusterings (on PCA50 space -> project to 2D)
# ---------------------------------------------------------------------
pca2 = PCA(n_components=2, random_state=base_rng_seed)
X_2d = pca2.fit_transform(X)
true_means_2d = pca2.transform(true_means)
modes_2d_list = [pca2.transform(m) for m in modes_list]

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
palette = sns.color_palette("tab10", merge_n_clusters)

global_handles = []
global_labels = []

for ax, alg, labels_pred, modes_2d in zip(axes, algorithms, labels_list, modes_2d_list):
    for i, color in enumerate(palette[:merge_n_clusters]):
        sc = ax.scatter(
            X_2d[labels_pred == i, 0],
            X_2d[labels_pred == i, 1],
            c=[color],
            s=15,
            alpha=0.7
        )
        if len(global_handles) < merge_n_clusters:
            global_handles.append(sc)
            global_labels.append(f"Cluster {i}")

    true_sc = ax.scatter(
        true_means_2d[:, 0],
        true_means_2d[:, 1],
        marker="X",
        c="magenta",
        s=140,
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
        s=100,
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
    bbox_to_anchor=(0.5, 0.01),
    title="Cluster Assignments & Centroids"
)
plt.setp(legend.get_title(), fontsize=11, fontweight="bold")
fig.suptitle("Clustering Comparison on Cancer RNA-Seq Data", fontsize=18, y=0.92)

save_path_clusters = os.path.join(results_dir, "gene50_clustering_comparison.pdf")
plt.tight_layout(rect=[0, 0.05, 1, 0.93])
plt.savefig(save_path_clusters, dpi=120)
plt.show()
print("Clustering comparison plot saved to:", save_path_clusters)

# ---------------------------------------------------------------------
# Privacy-utility curves: DP-GRAMS-C vs DP-KMeans as epsilon varies
# ---------------------------------------------------------------------
dpms_metrics = {m: [] for m in ["ARI", "NMI", "MSE"]}
dpkm_metrics = {m: [] for m in ["ARI", "NMI", "MSE"]}
dpms_err = {m: [] for m in ["ARI", "NMI", "MSE"]}
dpkm_err = {m: [] for m in ["ARI", "NMI", "MSE"]}
dpms_times_mean, dpkm_times_mean = [], []
dpms_times_std, dpkm_times_std = [], []

for eps in eps_modes_list:
    # --- DP-GRAMS-C ---
    ari_list, nmi_list, mse_list = [], [], []
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
            bandwidth_multiplier=dpms_bandwidth_multiplier,
            clip_multiplier=dpms_clip_multiplier
        )
        t_dpms = time.time() - t0
        dpms_times.append(t_dpms)

        labels_dpms_run = relabel_clusters(y_true, labels_dpms_run, merge_n_clusters)
        run_ari, run_nmi = compute_metrics(y_true, labels_dpms_run)

        modes_scaled = pca50_to_scaled(modes_dpms_run)
        run_mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

        ari_list.append(run_ari)
        nmi_list.append(run_nmi)
        mse_list.append(run_mse)

    dpms_metrics["ARI"].append(float(np.mean(ari_list)))
    dpms_metrics["NMI"].append(float(np.mean(nmi_list)))
    dpms_metrics["MSE"].append(float(np.mean(mse_list)))
    dpms_err["ARI"].append(float(np.std(ari_list, ddof=1)))
    dpms_err["NMI"].append(float(np.std(nmi_list, ddof=1)))
    dpms_err["MSE"].append(float(np.std(mse_list, ddof=1)))
    dpms_times_mean.append(float(np.mean(dpms_times)))
    dpms_times_std.append(float(np.std(dpms_times, ddof=1)))

    # --- DP-KMeans ---
    ari_list, nmi_list, mse_list = [], [], []
    dpkm_times = []

    for run in range(n_runs):
        dp_kmeans_run, t_dpkm = fit_dp_kmeans_version_robust(
            X=X,
            eps=eps,
            seed=base_rng_seed + run,
            n_clusters=merge_n_clusters,
            bounds_tuple=bounds_tuple
        )
        dpkm_times.append(t_dpkm)

        labels_dpkm_run = (
            dp_kmeans_run.labels_.astype(int)
            if hasattr(dp_kmeans_run, "labels_")
            else dp_kmeans_run.predict(X).astype(int)
        )
        labels_dpkm_run = relabel_clusters(y_true, labels_dpkm_run, merge_n_clusters)
        modes_dpkm_run = dp_kmeans_run.cluster_centers_

        run_ari, run_nmi = compute_metrics(y_true, labels_dpkm_run)

        modes_scaled = pca50_to_scaled(modes_dpkm_run)
        run_mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

        ari_list.append(run_ari)
        nmi_list.append(run_nmi)
        mse_list.append(run_mse)

    dpkm_metrics["ARI"].append(float(np.mean(ari_list)))
    dpkm_metrics["NMI"].append(float(np.mean(nmi_list)))
    dpkm_metrics["MSE"].append(float(np.mean(mse_list)))
    dpkm_err["ARI"].append(float(np.std(ari_list, ddof=1)))
    dpkm_err["NMI"].append(float(np.std(nmi_list, ddof=1)))
    dpkm_err["MSE"].append(float(np.std(mse_list, ddof=1)))
    dpkm_times_mean.append(float(np.mean(dpkm_times)))
    dpkm_times_std.append(float(np.std(dpkm_times, ddof=1)))

# ---------------------------------------------------------------------
# Save privacy-utility metrics
# ---------------------------------------------------------------------
csv_file = os.path.join(results_dir, "privacy_utility_metrics.csv")
with open(csv_file, "w", newline="") as f:
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
            dpms_metrics["ARI"][i],
            dpms_metrics["NMI"][i],
            dpms_metrics["MSE"][i],
            dpms_err["ARI"][i],
            dpms_err["NMI"][i],
            dpms_err["MSE"][i],
            dpms_times_mean[i],
            dpms_times_std[i]
        ])
        writer.writerow([
            eps, "DP-KMeans",
            dpkm_metrics["ARI"][i],
            dpkm_metrics["NMI"][i],
            dpkm_metrics["MSE"][i],
            dpkm_err["ARI"][i],
            dpkm_err["NMI"][i],
            dpkm_err["MSE"][i],
            dpkm_times_mean[i],
            dpkm_times_std[i]
        ])

print("All Gene50 experiments completed. Results saved in:", results_dir)

# ---------------------------------------------------------------------
# Privacy-utility plots: ARI, NMI, MSE
# ---------------------------------------------------------------------
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
    ax.set_title(f"Privacy-Utility: {metric} (Cancer RNA-Seq)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    plt.tight_layout()
    out_path = os.path.join(results_dir, f"gene50_privacy_utility_{metric.lower()}.pdf")
    plt.savefig(out_path, dpi=120)
    plt.show()
    print(f"Privacy-utility ({metric}) figure saved to:", out_path)

# ---------------------------------------------------------------------
# Hyperparameter sweeps for DP-GRAMS-C: ARI / NMI / MSE vs C^* and vs m (n fixed)
# Operates in PCA50 clustering space X
# and evaluates MSE in standardized gene space via inverse PCA.
# ---------------------------------------------------------------------

# Hyperparameter sweep settings for DP-GRAMS-C
clip_grid = [0.01, 0.05, 0.1, 0.2, 0.5]
m_frac_grid = [0.05, 0.1, 0.2, 0.5, 1.0]
n_reps_hparam = 20
epsilon_hparam = 1.0

def _run_dpms_single(
    epsilon_modes,
    clip_multiplier,
    m,
    seed,
):
    """
    Single DP-GRAMS-C run for hyperparameter sweeps on Gene (PCA50).
    Returns (MSE_centroids, ARI, NMI, runtime_seconds).
    """
    rng = np.random.default_rng(int(seed))
    t0 = time.time()

    modes_dpms_run, labels_dpms_run = dpms_private(
        data=X,  # PCA50 space
        epsilon_modes=epsilon_modes,
        delta=delta,
        p_seed=p_seed,
        rng=rng,
        k_est=merge_n_clusters,
        bandwidth_multiplier=dpms_bandwidth_multiplier,
        clip_multiplier=clip_multiplier,
        m=m,
    )
    runtime = time.time() - t0

    if modes_dpms_run.size == 0:
        return float("nan"), float("nan"), float("nan"), float(runtime)

    labels_dpms_run = relabel_clusters(y_true, labels_dpms_run, merge_n_clusters)
    ari, nmi = compute_metrics(y_true, labels_dpms_run)

    # MSE in standardized gene space (inverse PCA)
    modes_scaled = pca50_to_scaled(modes_dpms_run)
    mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

    return float(mse), float(ari), float(nmi), float(runtime)


def sweep_clip_multiplier_gene(
    clip_values,
    epsilon,
    n_reps=20,
    base_seed=12345,
):
    """
    Sweep over clip_multiplier (C^*) for DP-GRAMS-C on Gene (PCA50).
    Returns curves for centroid MSE / ARI / NMI vs C^*.
    """
    rng = np.random.default_rng(base_seed)
    results = []

    # Use default minibatch size (m=None) so dp_grams chooses m ~ n/log n
    m_default = None

    for cm in clip_values:
        mses, aris, nmis, times = [], [], [], []
        for _ in range(n_reps):
            seed_run = int(rng.integers(0, 2**31 - 1))
            mse, ari, nmi, rt = _run_dpms_single(
                epsilon_modes=epsilon,
                clip_multiplier=cm,
                m=m_default,
                seed=seed_run,
            )
            mses.append(mse)
            aris.append(ari)
            nmis.append(nmi)
            times.append(rt)

        mses = np.array(mses, dtype=float)
        aris = np.array(aris, dtype=float)
        nmis = np.array(nmis, dtype=float)
        times = np.array(times, dtype=float)

        results.append({
            "clip_multiplier": float(cm),
            "mean_mse": float(np.nanmean(mses)),
            "std_mse": float(np.nanstd(mses)),
            "mean_ari": float(np.nanmean(aris)),
            "std_ari": float(np.nanstd(aris)),
            "mean_nmi": float(np.nanmean(nmis)),
            "std_nmi": float(np.nanstd(nmis)),
            "mean_time": float(np.nanmean(times)),
            "std_time": float(np.nanstd(times)),
        })
    results.sort(key=lambda r: r["clip_multiplier"])
    return results


def sweep_minibatch_gene(
    m_frac_grid,
    epsilon,
    clip_multiplier_fixed=0.01,
    n_reps=20,
    base_seed=54321,
):
    """
    Sweep over minibatch size m (as a fraction of n) for DP-GRAMS-C on Gene (PCA50).
    Returns curves for centroid MSE / ARI / NMI vs m.
    """
    rng = np.random.default_rng(base_seed)
    results = []

    n_samples = X.shape[0]
    m_grid = sorted(set(max(1, int(frac * n_samples)) for frac in m_frac_grid))

    for m_val in m_grid:
        mses, aris, nmis, times = [], [], [], []
        for _ in range(n_reps):
            seed_run = int(rng.integers(0, 2**31 - 1))
            mse, ari, nmi, rt = _run_dpms_single(
                epsilon_modes=epsilon,
                clip_multiplier=clip_multiplier_fixed,
                m=m_val,
                seed=seed_run,
            )
            mses.append(mse)
            aris.append(ari)
            nmis.append(nmi)
            times.append(rt)

        mses = np.array(mses, dtype=float)
        aris = np.array(aris, dtype=float)
        nmis = np.array(nmis, dtype=float)
        times = np.array(times, dtype=float)

        results.append({
            "m": int(m_val),
            "mean_mse": float(np.nanmean(mses)),
            "std_mse": float(np.nanstd(mses)),
            "mean_ari": float(np.nanmean(aris)),
            "std_ari": float(np.nanstd(aris)),
            "mean_nmi": float(np.nanmean(nmis)),
            "std_nmi": float(np.nanstd(nmis)),
            "mean_time": float(np.nanmean(times)),
            "std_time": float(np.nanstd(times)),
        })
    results.sort(key=lambda r: r["m"])
    return results


print("\n[hyperparam] Running DP-GRAMS-C hyperparameter sweeps on Gene (PCA50)...")

clip_results = sweep_clip_multiplier_gene(
    clip_values=clip_grid,
    epsilon=epsilon_hparam,
    n_reps=n_reps_hparam,
    base_seed=12345,
)

m_results = sweep_minibatch_gene(
    m_frac_grid=m_frac_grid,
    epsilon=epsilon_hparam,
    clip_multiplier_fixed=0.01,
    n_reps=n_reps_hparam,
    base_seed=54321,
)

# ---------------------------------------------------------------------
# Save hyperparameter sweep results (text)
# ---------------------------------------------------------------------
clip_txt_path = os.path.join(results_dir, "dpgrams_c_vs_clip_multiplier_gene50.txt")
with open(clip_txt_path, "w") as f:
    header = (
        f"{'clip_mult':>10} | "
        f"{'mean_mse':>10} | {'std_mse':>10} | "
        f"{'mean_ari':>10} | {'std_ari':>10} | "
        f"{'mean_nmi':>10} | {'std_nmi':>10} | "
        f"{'mean_t':>10} | {'std_t':>10}"
    )
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    for r in clip_results:
        line = (
            f"{r['clip_multiplier']:10.3f} | "
            f"{r['mean_mse']:10.4f} | {r['std_mse']:10.4f} | "
            f"{r['mean_ari']:10.4f} | {r['std_ari']:10.4f} | "
            f"{r['mean_nmi']:10.4f} | {r['std_nmi']:10.4f} | "
            f"{r['mean_time']:10.4f} | {r['std_time']:10.4f}"
        )
        f.write(line + "\n")
print(f"[saved] Clip-multiplier sweep results -> {clip_txt_path}")

m_txt_path = os.path.join(results_dir, "dpgrams_c_vs_minibatch_gene50.txt")
with open(m_txt_path, "w") as f:
    header = (
        f"{'m':>10} | "
        f"{'mean_mse':>10} | {'std_mse':>10} | "
        f"{'mean_ari':>10} | {'std_ari':>10} | "
        f"{'mean_nmi':>10} | {'std_nmi':>10} | "
        f"{'mean_t':>10} | {'std_t':>10}"
    )
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    for r in m_results:
        line = (
            f"{r['m']:10d} | "
            f"{r['mean_mse']:10.4f} | {r['std_mse']:10.4f} | "
            f"{r['mean_ari']:10.4f} | {r['std_ari']:10.4f} | "
            f"{r['mean_nmi']:10.4f} | {r['std_nmi']:10.4f} | "
            f"{r['mean_time']:10.4f} | {r['std_time']:10.4f}"
        )
        f.write(line + "\n")
print(f"[saved] Minibatch sweep results -> {m_txt_path}")

# ---------------------------------------------------------------------
# Plots: MSE / ARI / NMI vs C^*
# ---------------------------------------------------------------------
clip_x = [r["clip_multiplier"] for r in clip_results]

# MSE vs C^*
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    clip_x,
    [r["mean_mse"] for r in clip_results],
    yerr=[r["std_mse"] for r in clip_results],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=7,
    capsize=3,
)
ax.set_xlabel(r"Clip Multiplier ($C^*$)")
ax.set_ylabel("Centroid MSE (std gene space)")
ax.set_title("DP-GRAMS-C on Gene50: MSE vs $C^*$")
ax.grid(True, alpha=0.4)
plt.tight_layout()
clip_plot_path_mse = os.path.join(results_dir, "gene50_dpgrams_c_mse_vs_clip_multiplier.pdf")
plt.savefig(clip_plot_path_mse, dpi=120)
plt.show()
print(f"[saved] MSE vs C^* plot -> {clip_plot_path_mse}")

# ARI vs C^*
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    clip_x,
    [r["mean_ari"] for r in clip_results],
    yerr=[r["std_ari"] for r in clip_results],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=7,
    capsize=3,
)
ax.set_xlabel(r"Clip Multiplier ($C^*$)")
ax.set_ylabel("ARI")
ax.set_title("DP-GRAMS-C on Gene50: ARI vs $C^*$")
ax.grid(True, alpha=0.4)
plt.tight_layout()
clip_plot_path_ari = os.path.join(results_dir, "gene50_dpgrams_c_ari_vs_clip_multiplier.pdf")
plt.savefig(clip_plot_path_ari, dpi=120)
plt.show()
print(f"[saved] ARI vs C^* plot -> {clip_plot_path_ari}")

# NMI vs C^*
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    clip_x,
    [r["mean_nmi"] for r in clip_results],
    yerr=[r["std_nmi"] for r in clip_results],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=7,
    capsize=3,
)
ax.set_xlabel(r"Clip Multiplier ($C^*$)")
ax.set_ylabel("NMI")
ax.set_title("DP-GRAMS-C on Gene50: NMI vs $C^*$")
ax.grid(True, alpha=0.4)
plt.tight_layout()
clip_plot_path_nmi = os.path.join(results_dir, "gene50_dpgrams_c_nmi_vs_clip_multiplier.pdf")
plt.savefig(clip_plot_path_nmi, dpi=120)
plt.show()
print(f"[saved] NMI vs C^* plot -> {clip_plot_path_nmi}")

# ---------------------------------------------------------------------
# Plots: MSE / ARI / NMI vs m
# ---------------------------------------------------------------------
m_x = [r["m"] for r in m_results]

# MSE vs m
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    m_x,
    [r["mean_mse"] for r in m_results],
    yerr=[r["std_mse"] for r in m_results],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=7,
    capsize=3,
)
ax.set_xscale("log")
ax.set_xlabel("Minibatch size m")
ax.set_ylabel("Centroid MSE (std gene space)")
ax.set_title("DP-GRAMS-C on Gene50: MSE vs m")
ax.grid(True, alpha=0.4)
plt.tight_layout()
m_plot_path_mse = os.path.join(results_dir, "gene50_dpgrams_c_mse_vs_minibatch.pdf")
plt.savefig(m_plot_path_mse, dpi=120)
plt.show()
print(f"[saved] MSE vs m plot -> {m_plot_path_mse}")

# ARI vs m
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    m_x,
    [r["mean_ari"] for r in m_results],
    yerr=[r["std_ari"] for r in m_results],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=7,
    capsize=3,
)
ax.set_xscale("log")
ax.set_xlabel("Minibatch size m")
ax.set_ylabel("ARI")
ax.set_title("DP-GRAMS-C on Gene50: ARI vs m")
ax.grid(True, alpha=0.4)
plt.tight_layout()
m_plot_path_ari = os.path.join(results_dir, "gene50_dpgrams_c_ari_vs_minibatch.pdf")
plt.savefig(m_plot_path_ari, dpi=120)
plt.show()
print(f"[saved] ARI vs m plot -> {m_plot_path_ari}")

# NMI vs m
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    m_x,
    [r["mean_nmi"] for r in m_results],
    yerr=[r["std_nmi"] for r in m_results],
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=7,
    capsize=3,
)
ax.set_xscale("log")
ax.set_xlabel("Minibatch size m")
ax.set_ylabel("NMI")
ax.set_title("DP-GRAMS-C on Gene50: NMI vs m")
ax.grid(True, alpha=0.4)
plt.tight_layout()
m_plot_path_nmi = os.path.join(results_dir, "gene50_dpgrams_c_nmi_vs_minibatch.pdf")
plt.savefig(m_plot_path_nmi, dpi=120)
plt.show()
print(f"[saved] NMI vs m plot -> {m_plot_path_nmi}")
