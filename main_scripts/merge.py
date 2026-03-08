# merge.py

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
