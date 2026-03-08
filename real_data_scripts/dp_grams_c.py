# dp-grams-c.py

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
    """
    seed = int(master.integers(0, 2**31 - 1))
    return np.random.default_rng(seed)


def dpms_private(
    data: np.ndarray,
    epsilon_modes: float,
    delta: float = 1e-5,
    p_seed: float = 0.1,
    bandwidth_multiplier: float = 1.0,
    clip_multiplier: float = 1.0,
    m: Optional[int] = None,
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
        Privacy budget epsilon used inside DP-GRAMS to learn private modes.
    delta : float, default=1e-5
        Privacy parameter delta for DP-GRAMS.
    p_seed : float, default=0.1
        Fraction of points used as initial seeds in DP-GRAMS (p0 in the paper).
    bandwidth_multiplier : float, default=1.0
        Scale factor applied to Silverman's bandwidth for DP-GRAMS.
    clip_multiplier : float, default=1.0
        Multiplier on the clipping constant C in DP-GRAMS.
    m : int, optional
        Minibatch size for DP-GRAMS. If None, dp_grams defaults to m ~ n / log n.
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
    # Use simple non-private optimal rate (log n / n)^{1/(d+6)}.
    h = (np.log(n) / n) ** (1 / (d + 6))
    h = float(h) * float(bandwidth_multiplier)
    # print("[DEBUG] Bandwidth used:", h)

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
        clip_multiplier=clip_multiplier,
        m=m,
    )

    if dp_modes.size == 0:
        # Degenerate: no modes found; return empty centers + dummy labels
        return np.empty((0, d)), np.zeros(n, dtype=int)

    # ------------------------------------------------------------
    # Merge DP modes into final cluster centers
    # ------------------------------------------------------------
    if k_est is None:
        # Distance-threshold-based merging
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
    diffs = data[:, None, :] - merged_modes[None, :, :]   # (n, k, d)
    dists = np.linalg.norm(diffs, axis=2)                 # (n, k)
    labels = np.argmin(dists, axis=1).astype(int)         # (n,)

    return merged_modes, labels
