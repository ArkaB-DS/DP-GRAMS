# mode_matching_mse.py
# ---------------------------------------------------------------------
# Utility for evaluating mode estimation accuracy in DP-GRAMS experiments.
#
# This implements the Hungarian (linear sum assignment) matching step
# described in the paper for computing mean-squared error
# (MSE) between true and estimated modes. It provides a fair one-to-one
# correspondence between modes before computing squared error.
# ---------------------------------------------------------------------

import numpy as np
from scipy.optimize import linear_sum_assignment

def mode_matching_mse(true_modes, est_modes, penalty=None):
    """
    Compute mean-squared error (MSE) between true and estimated modes
    with optimal assignment using the Hungarian algorithm.

    Each true mode is matched to the nearest estimated mode to minimize
    total squared distance. This metric is used in all experiments to
    assess mode recovery quality for DP-GRAMS, PMS, and clustering.

    Parameters
    ----------
    true_modes : np.ndarray
        Array of shape (k_true, d), containing ground-truth mode locations.
    est_modes : np.ndarray
        Array of shape (k_est, d), containing estimated (private or non-private) modes.
    penalty : float, optional
        Penalty term applied per unmatched mode when the number of estimated
        and true modes differ. Reflects under-/over-estimation penalties
        (optional; used mainly in ablation tests).

    Returns
    -------
    mse : float
        Mean-squared error between matched true and estimated modes,
        adjusted for unmatched penalties if provided. Returns NaN if
        either array is empty.
    """
    true_modes = np.atleast_2d(true_modes)
    est_modes = np.atleast_2d(est_modes)

    # Handle degenerate cases with no modes detected or no ground truth
    if true_modes.size == 0 or est_modes.size == 0:
        return np.nan

    # Compute squared Euclidean distance matrix between all mode pairs
    dist_matrix = np.sum(
        (true_modes[:, None, :] - est_modes[None, :, :]) ** 2,
        axis=2
    )

    # Replace invalid or infinite values (safety for numerical stability)
    dist_matrix = np.nan_to_num(dist_matrix, nan=1e10, posinf=1e10, neginf=1e10)

    # Solve optimal assignment using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    total_error = dist_matrix[row_ind, col_ind].sum()

    # Optionally penalize mismatched mode counts
    if penalty is not None:
        unmatched = abs(true_modes.shape[0] - est_modes.shape[0])
        total_error += penalty * unmatched

    # Normalize by the maximum number of modes to compute per-mode MSE
    mse = total_error / max(true_modes.shape[0], est_modes.shape[0])
    return mse
