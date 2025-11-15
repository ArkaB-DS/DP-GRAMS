#!/usr/bin/env python3
"""
biv_5mix_slides.py

Saves:
 - KDE 3D: results/bivariate_5mix/contour_modes_3d.png
 - Mean Shift frames: results/bivariate_5mix/ms_frames/ms_iter_001.png ... ms_iter_{T+1}.png
 - DP-GRAMS frames:    results/bivariate_5mix/dp_frames/dp_iter_001.png ... dp_iter_{T+1}.png

Follows parameters exactly from your supplied script.
"""
import os, sys, math, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# make sure main_scripts is importable (one directory up)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_scripts.ms import mean_shift
from main_scripts.dp_grams import dp_grams
from main_scripts.merge import merge_modes
from main_scripts.bandwidth import silverman_bandwidth
from main_scripts.mode_matching_mse import mode_matching_mse  # imported to match your environment

sns.set_context("talk")
sns.set_style("white")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11
})

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def generate_5mix(n_samples, seed=None):
    rng = np.random.default_rng(seed)
    means = np.array([[0,0],[6,0],[-6,0],[0,6],[0,-6]])
    dfs = [15,6,10,8,20]
    probs = np.array([0.2]*5)
    scales = np.array([0.1,0.9,1.3,1.0,0.4])
    samples_per_mode = (probs*n_samples).astype(int)
    pts = []
    for m, df, n_pts, scale in zip(means, dfs, samples_per_mode, scales):
        t_samples = rng.standard_t(df, size=(n_pts,2))*scale + m
        pts.append(t_samples)
    return np.vstack(pts), means

# plotting helpers (simple, no annotations other than iteration title)
def contour_backdrop(data, ax, nx=200, cmap="Blues"):
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data.T)
    x_min, x_max = data[:,0].min()-2, data[:,0].max()+2
    y_min, y_max = data[:,1].min()-2, data[:,1].max()+2
    xgrid, ygrid = np.meshgrid(np.linspace(x_min,x_max,nx),
                               np.linspace(y_min,y_max,nx))
    positions = np.vstack([xgrid.ravel(), ygrid.ravel()])
    Z = np.reshape(kde(positions), xgrid.shape)
    ax.contourf(xgrid, ygrid, Z, levels=50, cmap=cmap, alpha=0.25, zorder=0)
    ax.contour(xgrid, ygrid, Z, levels=8, colors='k', linewidths=0.5, alpha=0.5, zorder=1)
    return (x_min,x_max,y_min,y_max)

def save_frame(path, data, true_modes, algo_modes, title, algo_label):
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(6,6))
    bounds = contour_backdrop(data, ax)
    # plot true modes first (green stars, behind)
    ax.scatter(true_modes[:,0], true_modes[:,1], s=220, marker="*", color="#2ca02c",
               edgecolor='k', zorder=3)
    # then algorithm modes in orange on top
    if algo_modes is not None and len(algo_modes)>0:
        ax.scatter(algo_modes[:,0], algo_modes[:,1], s=160, marker="o", color="#ff7f0e",
                   edgecolor='k', zorder=4)
    # legend: only true modes and algo modes
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker="*", color="#2ca02c", label="True modes", markeredgecolor='k', markersize=12, linestyle=""),
        Line2D([0],[0], marker="o", color="#ff7f0e", label=f"{algo_label} modes", markeredgecolor='k', markersize=10, linestyle="")
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # print saved file
    print("Saved:", path)

def kde_3d_and_contour_save(data, true_modes, outpath):
    """Save only the 3D KDE surface plot (no contour subplot)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.stats import gaussian_kde

    ensure_dir(os.path.dirname(outpath))
    kde = gaussian_kde(data.T)
    x_min, x_max = data[:,0].min()-2, data[:,0].max()+2
    y_min, y_max = data[:,1].min()-2, data[:,1].max()+2
    xgrid, ygrid = np.meshgrid(np.linspace(x_min, x_max, 200),
                               np.linspace(y_min, y_max, 200))
    positions = np.vstack([xgrid.ravel(), ygrid.ravel()])
    Z = np.reshape(kde(positions), xgrid.shape)

    fig = plt.figure(figsize=(8,6))
    ax3d = fig.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(xgrid, ygrid, Z, cmap='Blues', alpha=0.9, linewidth=0, antialiased=True)

    ax3d.set_xlabel(r"$x_1$")
    ax3d.set_ylabel(r"$x_2$")
    ax3d.set_zlabel("Density")
    ax3d.set_title("Estimated Density Surface for 5-Modal Bivariate t-Mixture")

    # Optional: mark true modes on surface
    if true_modes is not None and len(true_modes) > 0:
        z_vals = kde(true_modes.T)
        ax3d.scatter(true_modes[:,0], true_modes[:,1], z_vals,
                     s=80, c="#2ca02c", edgecolor='k', marker="*", label="True modes")
        ax3d.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved 3D KDE surface to:", outpath)

def main():
    # parameters exactly as in your script
    epsilon = 1
    delta = 1e-6
    p = 0.1
    base_seed = 42
    n_samples_base = 2000
    results_dir = "results/bivariate_5mix_slides"
    ensure_dir(results_dir)

    data, true_modes = generate_5mix(n_samples_base, seed=base_seed)
    h = silverman_bandwidth(data)
    print(f"[main] Silverman bandwidth h={h:.6f}")

    # Save combined KDE+contour figure (single file)
    kde_out = os.path.join(results_dir, "contour_modes_3d.png")
    kde_3d_and_contour_save(data, true_modes, kde_out)

    # iteration count
    T = int(math.ceil(math.log(max(2, n_samples_base))))
    print("T =", T)

    # initial candidate pool: same logic as your scripts (use p fraction)
    rng = np.random.default_rng(base_seed + 1)
    k_init = max(1, int(n_samples_base * p))
    init_idx = rng.choice(n_samples_base, size=k_init, replace=False)
    init_points = data[init_idx].astype(float)

    ms_dir = os.path.join(results_dir, "ms_frames")
    dp_dir = os.path.join(results_dir, "dp_frames")
    ensure_dir(ms_dir); ensure_dir(dp_dir)

    # start from initial_points, then perform T updates; save frames AFTER each update (iter 1..T)
    ms_points = init_points.copy()
    dp_points = init_points.copy()

    for t in range(1, T+1):
        # update MS by one step (mean_shift with T=1)
        ms_points = mean_shift(data, initial_modes=ms_points, T=1, bandwidth=h, p=p, seed=np.random.default_rng(base_seed + 100 + t))
        ms_path = os.path.join(ms_dir, f"ms_iter_{t:03d}.png")
        save_frame(ms_path, data, true_modes, ms_points, title=f"Iteration {t} / {T}", algo_label="MS")

        # update DP-GRAMS by one step (dp_grams with T=1, initial_modes=dp_points)
        _, dp_new = dp_grams(X=data, epsilon=epsilon, delta=delta,
                             initial_modes=dp_points, T=1, m=None, h=h, p0=p,
                             rng=np.random.default_rng(base_seed + 200 + t),
                            #  clip_multiplier=1e-1,
                             hdp=0.
                             )
        # dp_grams may return empty array if filtered; ensure shape correct
        if dp_new is None:
            dp_new = np.empty((0, data.shape[1]))
        dp_points = dp_new
        dp_path = os.path.join(dp_dir, f"dp_iter_{t:03d}.png")
        save_frame(dp_path, data, true_modes, dp_points, title=f"Iteration {t} / {T}", algo_label="DP-GRAMS")

    # After finishing T iterations save merged result as iteration T+1 (no text about merging)
    merged_ms = merge_modes(ms_points, bandwidth=h, k=1) if len(ms_points)>0 else np.empty((0,2))
    merged_dp = merge_modes(dp_points, bandwidth=h, k=1) if len(dp_points)>0 else np.empty((0,2))

    ms_merge_path = os.path.join(ms_dir, f"ms_iter_{(T+1):03d}.png")
    dp_merge_path = os.path.join(dp_dir, f"dp_iter_{(T+1):03d}.png")

    # save merged frames (title shows iteration T+1)
    # save merged frames (display "Merged", but keep filename as T+1)
    save_frame(ms_merge_path, data, true_modes, merged_ms, title="Merged (Final)", algo_label="MS")
    save_frame(dp_merge_path, data, true_modes, merged_dp, title="Merged (Final)", algo_label="DP-GRAMS")

    print("Frames saved under:", results_dir)
    print(" - MS frames:", ms_dir)
    print(" - DP frames:", dp_dir)
    print("Done.")

if __name__ == "__main__":
    main()
