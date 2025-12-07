# pms.py

import numpy as np
from main_scripts.bandwidth import silverman_bandwidth

def partial_mean_shift(X, Y, mesh_points=None, bandwidth=None, T=20):
    """
    Non-private Partial Mean Shift (PMS) for modal regression.

    This is the baseline (non-DP) algorithm corresponding to the
    partial mean-shift / modal regression setup in the paper.

    Setup (simplified 1D X, 1D Y):
      - For each i, we maintain a location y_i (initialized from mesh_points or Y).
      - At each iteration, we:
          * compute a joint kernel in (X, Y)-space centered at (X_i, y_i),
          * update y_i to the kernel-weighted average of the original responses Y.
      - We keep X fixed, only shifting Y-locations, as in standard PMS.

    Parameters
    ----------
    X : array-like, shape (n,)
        Predictor values.
    Y : array-like, shape (n,)
        Response values.
    mesh_points : array-like, optional
        Initial values for the modal locations in Y.
        If None, we start at the observed Y.
        If provided, must have length n.
    bandwidth : float, optional
        Bandwidth h for the Gaussian kernel in (X, Y)-space.
        If None, we estimate it from Y via Silverman's rule.
    T : int
        Number of partial mean-shift iterations.

    Returns
    -------
    Y_new : np.ndarray, shape (n,)
        Updated (non-private) modal regression estimates after T iterations.
    """
    # Ensure 1D numpy arrays
    X = np.asarray(X).reshape(-1)
    Y = np.asarray(Y).reshape(-1)
    n = len(X)

    if len(Y) != n:
        raise ValueError("X and Y must have the same length.")

    # Initialization: one path per sample (classical PMS layout)
    if mesh_points is None:
        mesh_points = Y.copy()
    else:
        mesh_points = np.asarray(mesh_points).reshape(-1)
        if len(mesh_points) != n:
            raise ValueError("mesh_points must have length n (= len(X) = len(Y)).")

    # Bandwidth selection (1D Silverman on Y as a simple, stable choice)
    if bandwidth is None:
        bandwidth = float(silverman_bandwidth(Y.reshape(-1, 1)))

    h = max(1e-6, float(bandwidth))  # small floor to avoid numerical issues
    h2 = h ** 2

    Y_new = mesh_points.copy()

    # Iterative partial mean shift
    for _ in range(T):
        for i in range(n):
            xi = X[i]
            yi = Y_new[i]

            # Joint kernel in (X, Y) centered at (xi, yi):
            #   w_j = exp(-((X_j - xi)^2 + (Y_j - yi)^2) / (2 h^2))
            w = np.exp(-((X - xi) ** 2 + (Y - yi) ** 2) / (2.0 * h2))
            denom = np.sum(w)

            if denom > 0.0:
                # Classical PMS update: use original Y in numerator
                # (y_i moves toward local conditional mode around x_i)
                Y_new[i] = np.sum(w * Y) / denom
            # else: if denom == 0, keep Y_new[i] unchanged

    return Y_new
