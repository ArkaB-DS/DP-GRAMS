# dp-grams-c.py
#
# DP-GRAMS-based private clustering:
# ---------------------------------
# - Step 1: Run DP-GRAMS (dp_grams) to obtain differentially private modes.
# - Step 2: Merge the raw private modes:
#       * If k_est is None: spatial merging via merge_modes (distance threshold).
#       * If k_est is given: Agglomerative clustering into exactly k_est clusters.
# - Step 3: Assign each data point to the nearest merged private mode

import numpy as np
from typing import Optional, Tuple
import os
import sys

# Ensure local imports work when run as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_scripts.dp_grams import dp_grams
from main_scripts.merge import merge_modes, merge_modes_agglomerative
from main_scripts.bandwidth import silverman_bandwidth


def _child_rng(master: np.random.Generator) -> np.random.Generator:
    """
    Derive an independent child RNG from a master RNG.
    Keeps stochastic components reproducible but decoupled.
    """
    seed = int(master.integers(0, 2**31 - 1))
    return np.random.default_rng(seed)


def dpms_private(
    data: np.ndarray,
    epsilon_modes: float,
    delta: float = 1e-5,
    p_seed: float = 0.1,
    use_hdp_filter: bool = False,
    hdp: float = 0.1,
    bandwidth_multiplier: float = 1.0,
    clip_multiplier: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    k_est: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Differentially private clustering via DP-GRAMS + deterministic label assignment.

    Parameters
    ----------
    data : np.ndarray, shape (n, d)
        Input dataset.
    epsilon_modes : float
        Privacy budget ε used inside DP-GRAMS to learn private modes.
    delta : float, default=1e-5
        Privacy parameter δ for DP-GRAMS.
    p_seed : float, default=0.1
        Fraction of points used as initial seeds in DP-GRAMS (p0 in the paper).
    use_hdp_filter : bool, default=False
        If True, use high-density point filtering within DP-GRAMS.
    hdp : float, default=0.1
        High-density proportion threshold used when use_hdp_filter=True.
    bandwidth_multiplier : float, default=1.0
        Scale factor applied to Silverman's bandwidth for DP-GRAMS.
    clip_multiplier : float, default=1.0
        Multiplier on the clipping constant C in DP-GRAMS.
    rng : np.random.Generator, optional
        RNG for reproducibility. If None, a new Generator is created.
    k_est : int, optional
        If provided, enforce exactly k_est clusters via agglomerative merging.
        If None, use merge_modes with a distance-based threshold.

    Returns
    -------
    merged_modes : np.ndarray, shape (k, d)
        DP cluster centers (merged private modes).
    labels : np.ndarray, shape (n,)
        Deterministic cluster assignment for each data point:
        labels[i] = argmin_j ||x_i - merged_modes[j]||_2.
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a 2D array of shape (n, d).")

    n, d = data.shape

    # ------------------------------------------------------------
    # Bandwidth for DP-GRAMS
    # ------------------------------------------------------------
    # Use Silverman's rule on the data, scaled if desired.
    base_h = silverman_bandwidth(data)
    h = float(base_h) * float(bandwidth_multiplier)

    # ------------------------------------------------------------
    # Run DP-GRAMS to obtain private modes
    # ------------------------------------------------------------
    child = _child_rng(rng)
    _, dp_modes = dp_grams(
        X=data,
        epsilon=epsilon_modes,
        delta=delta,
        p0=p_seed,
        rng=child,
        h=h,
        hdp=hdp,
        use_hdp_filter=use_hdp_filter,
        clip_multiplier=clip_multiplier,
    )

    if dp_modes.size == 0:
        # Degenerate: no modes found; return empty centers + dummy labels
        return np.empty((0, d)), np.zeros(n, dtype=int)

    # ------------------------------------------------------------
    # Merge DP modes into final cluster centers
    # ------------------------------------------------------------
    if k_est is None:
        # Distance-threshold-based merging (as in DP-GRAMS experiments)
        merged_modes = merge_modes(dp_modes)
    else:
        # Force exactly k_est clusters using agglomerative clustering
        merged_modes = merge_modes_agglomerative(dp_modes, n_clusters=k_est)

    k = merged_modes.shape[0]
    if k == 0:
        # Safety: if merging somehow collapses everything, fall back
        return np.empty((0, d)), np.zeros(n, dtype=int)

    # ------------------------------------------------------------
    # Deterministic cluster assignment by nearest private center
    # ------------------------------------------------------------
    # Compute pairwise distances between points and merged modes
    # shapes: data -> (n, d), merged_modes -> (k, d)
    diffs = data[:, None, :] - merged_modes[None, :, :]   # (n, k, d)
    dists = np.linalg.norm(diffs, axis=2)                 # (n, k)

    labels = np.argmin(dists, axis=1).astype(int)         # (n,)

    return merged_modes, labels
