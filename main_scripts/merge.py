# merge.py
# ---------------------------------------------------------------------
# Utilities for merging candidate mode estimates.
#
# In the DP-GRAMS framework,
# this corresponds to the *post-processing* stage: multiple noisy
# candidate modes (from random initializations + DP gradient ascent)
# are merged into a stable set of final modes. Importantly, all
# operations here are deterministic functions of already-private
# outputs, so they do NOT consume additional privacy budget.
# ---------------------------------------------------------------------

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from main_scripts.ms import mean_shift
from main_scripts.bandwidth import silverman_bandwidth

def merge_modes(modes, bandwidth=None, k=1):
    """
    Merge nearby mode estimates using a simple radius-based rule.

    This is the "bandwidth-scaled radius grouping" described in the
    paper: candidate modes within 'threshold = k * h' of each other
    are grouped and replaced by their (non-private) average.

    Parameters
    ----------
    modes : np.ndarray
        Array of shape (n_modes, d) containing candidate mode estimates,
        e.g., the outputs x_T for different initializations.
    bandwidth : float, optional
        Kernel bandwidth h. If None, estimated from `modes` via
        Silverman's rule-of-thumb for consistency with KDE / mean-shift.
    k : float, default=1
        Multiplier on h that sets the merge radius. Larger k => more
        aggressive merging (fewer final modes).

    Returns
    -------
    merged : np.ndarray
        Array of merged modes of shape (n_merged, d).
    """
    if bandwidth is None:
        # Use Silverman bandwidth as a heuristic for the spatial scale
        # at which modes should be considered "the same".
        bandwidth = silverman_bandwidth(modes)

    threshold = k * bandwidth
    merged = []
    used = np.zeros(len(modes), dtype=bool)

    # Greedy single-linkage style merging:
    # For each unused mode, collect all modes within `threshold` and
    # replace them with their mean. This is deterministic post-processing.
    for i, mode in enumerate(modes):
        if used[i]:
            continue
        cluster = [mode]
        used[i] = True
        for j in range(i + 1, len(modes)):
            if (not used[j]
                and np.linalg.norm(modes[j] - mode) <= threshold):
                cluster.append(modes[j])
                used[j] = True
        merged.append(np.mean(cluster, axis=0))

    return np.array(merged)


def merge_modes_agglomerative(modes, n_clusters, random_state=None):
    """
    Merge modes using agglomerative clustering when the desired
    number of clusters/modes is known or specified.

    This corresponds to the "when the number of modes is known,
    hierarchical clustering compresses M to that count" option
    mentioned in the paper.

    Parameters
    ----------
    modes : np.ndarray
        Array of shape (n_modes, d) of candidate modes.
    n_clusters : int
        Target number of merged modes.
    random_state : int or None
        Included for API completeness; AgglomerativeClustering with
        Ward linkage itself is deterministic for fixed input.

    Returns
    -------
    merged : np.ndarray
        Array of shape (n_clusters, d) with cluster-mean centroids.
    """
    modes = np.asarray(modes)

    # If no modes are provided, return an empty (0, d) array.
    if modes.size == 0:
        # Assumes 2D shape for consistency; if modes is (0,), this will raise,
        # which is a useful signal to the caller to ensure proper input shape.
        return np.empty((0, modes.shape[1]))

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    labels = clustering.fit_predict(modes)

    # Compute mean of modes in each cluster as final merged mode
    merged = np.array([
        modes[labels == i].mean(axis=0)
        for i in range(n_clusters)
    ])
    return merged


def ms_merge(dp_modes, seed, k=1):
    """
    Refine and merge DP-GRAMS modes using a (non-private) mean-shift pass.

    Pipeline:
      1. Start from differentially private candidate modes `dp_modes`
         produced by DP-GRAMS.
      2. Run a short mean-shift refinement initialized at these modes.
         Since this operates ONLY on already-private outputs, it is
         pure post-processing and does not consume additional (ε, δ).
      3. Merge refined modes with `merge_modes` using bandwidth-based
         radius threshold.

    Parameters
    ----------
    dp_modes : np.ndarray
        Array of shape (n_modes, d), DP-protected mode candidates.
    seed : int
        Seed for the RNG passed to `mean_shift` to ensure reproducibility
        across runs / experiments.
    k : float, default=1
        Multiplier in the merge threshold `k * bandwidth` for `merge_modes`.

    Returns
    -------
    merged_modes : np.ndarray
        Array of merged and refined modes.
    """
    n = len(dp_modes)

    # Estimate bandwidth from the (already-private) mode locations.
    # This aligns with using a consistent kernel scale in both DP-GRAMS
    # and post-processing.
    bw = silverman_bandwidth(dp_modes)

    # One short mean-shift run using dp_modes as both data and initialization.
    # T = log n is consistent with the iteration complexity in the paper.
    ms_dp_modes = mean_shift(
        dp_modes,
        initial_modes=dp_modes,
        T=int(np.log(n)),
        bandwidth=bw,
        seed=np.random.default_rng(seed)
    )

    # Final bandwidth-based merging (no extra privacy cost).
    return merge_modes(ms_dp_modes, bandwidth=bw, k=k)
