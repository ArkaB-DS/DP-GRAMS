# ms.py
# ---------------------------------------------------------------------
# Classical (non-private) Mean-Shift algorithm implementation.
#
# This file provides a clean mean-shift routine used in:
#   - Non-private baseline experiments 
#   - Post-processing refinements in DP-GRAMS 
#
# Mean-shift is an iterative, nonparametric mode-seeking algorithm that
# performs gradient ascent on the kernel density estimate (KDE) of the data.
# In DP-GRAMS, this interpretation allows gradient ascent steps to be
# privatized with Gaussian noise.
# ---------------------------------------------------------------------

import numpy as np
from main_scripts.bandwidth import silverman_bandwidth

def mean_shift_step(data, mode, bandwidth):
    """
    Perform one mean-shift update step for a single mode candidate.

    Given current estimate 'mode', compute:
        m(x) = (Σ_i K((X_i - x)/h) X_i) / (Σ_i K((X_i - x)/h))
    where K is the Gaussian kernel. This moves 'mode' toward regions
    of higher estimated density.

    Parameters
    ----------
    data : np.ndarray
        Dataset of shape (n_samples, d).
    mode : np.ndarray
        Current mode estimate of shape (d,).
    bandwidth : float
        Kernel bandwidth (h) controlling the neighborhood size.

    Returns
    -------
    new_mode : np.ndarray
        Updated mode estimate after one mean-shift step.
    """
    # Gaussian kernel weights based on Euclidean distance
    kernel = np.exp(-np.linalg.norm(data - mode, axis=1) ** 2 / (2 * bandwidth ** 2))

    # Weighted average of data points within the kernel window
    return np.sum(kernel[:, np.newaxis] * data, axis=0) / kernel.sum()


def mean_shift(data, initial_modes=None, T=20, bandwidth=None, p=1, seed=None):
    """
    Full mean-shift loop performing T iterations for each mode initialization.

    This classical version corresponds to the non-private algorithm that
    DP-GRAMS reformulates as differentially private gradient ascent.
    It's also reused in post-processing for refining DP modes (ms_merge).

    Parameters
    ----------
    data : np.ndarray
        Input dataset of shape (n_samples, d).
    initial_modes : np.ndarray, optional
        Initial points to start mean-shift from.
        If None, a random subset of data is used.
    T : int, default=20
        Number of mean-shift iterations to perform.
        DP-GRAMS typically uses T = ceil(log n).
    bandwidth : float, optional
        Kernel bandwidth (h). If None, estimated via Silverman's rule.
    p : float, default=1
        Fraction of samples to use for initialization (if initial_modes is None).
        Used in experiments to randomly subsample starting points.
    seed : int or np.random.Generator, optional
        Random seed or RNG for reproducibility.

    Returns
    -------
    modes : np.ndarray
        Array of final mode estimates of shape (n_modes, d).
    """
    rng = np.random.default_rng(seed)

    # Use Silverman's rule-of-thumb if bandwidth not supplied.
    if bandwidth is None:
        bandwidth = silverman_bandwidth(data)

    # Randomly select subset of data points as initializations
    if initial_modes is None:
        n_samples = max(1, int(len(data) * p))
        indices = rng.choice(len(data), size=n_samples, replace=False)
        initial_modes = data[indices]

    modes = initial_modes

    # Iteratively shift each mode toward a local density maximum
    for _ in range(T):
        modes = np.array([
            mean_shift_step(data, mode, bandwidth)
            for mode in modes
        ])

    return modes
