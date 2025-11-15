# dp-pms.py
#
# Differentially Private Partial Mean Shift (DP-PMS)
#
# This implements the DP modal regression / partial mean-shift variant
# described in the paper, in a simplified form:
#
# - We treat each (X_i, Y_i) as defining a local conditional mode in Y
#   around X_i, and iteratively update Y_i via a mean-shift-style step.
# - Each update is written as a sum of *normalized* per-sample contributions
#   q_j, which are then clipped to control sensitivity.
# - We add correlated Gaussian noise across mesh points (as in DP-GRAMS),
#   with scale calibrated heuristically as O(C / (n * epsilon)) using
#   the same flavor of analysis as in the main algorithm.
#
# Notes:
# - No mini-batching: each iteration uses all n samples.

import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import cholesky, eigh
from main_scripts.bandwidth import silverman_bandwidth


def gaussian_kernel_1d(u: np.ndarray) -> np.ndarray:
    """
    Standard 1D Gaussian kernel (up to normalization constant):
        K(u) = exp(-u^2 / 2)

    Used implicitly inside the joint (X, Y) kernel below.
    """
    return np.exp(-0.5 * (u ** 2))


def dp_pms(
    X,
    Y,
    mesh_points=None,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    T: int = 20,
    rng=None,
    bandwidth=None,
    verbose: bool = False,
):
    """
    Differentially Private Partial Mean Shift (DP-PMS) for 1D modal regression.

    Parameters
    ----------
    X : array-like, shape (n,)
        Predictor values.
    Y : array-like, shape (n,)
        Response values.
    mesh_points : array-like, optional
        Initial Y-locations to run partial mean shift from.
        For this simple implementation, we require len(mesh_points) == n
        and conceptually pair (X_i, mesh_points[i]) as starting points.
        If None, we initialize mesh_points = Y (identity).
    epsilon : float
        Global privacy parameter ε.
    delta : float
        Global privacy parameter δ.
    T : int
        Number of DP-PMS iterations.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    bandwidth : float, optional
        Bandwidth h for the Gaussian kernel in (X, Y)-space.
        If None, estimated from Y via Silverman's rule.
    verbose : bool
        If True, print basic diagnostic information.

    Returns
    -------
    Y_new : np.ndarray, shape (n,)
        DP-PMS-updated modal regression estimates corresponding to mesh_points.

    Notes
    -----
    - The update at each (X_i, Y_i) is:
          w_j = exp(-((X_j - X_i)^2 + (Y_j - Y_i)^2) / (2 h^2))
          q_j = w_j * (Y_j - Y_i) / sum_l w_l
          q_j is clipped to |q_j| <= C
          ΔY_i = sum_j q_j_clipped
          Y_i <- Y_i + ΔY_i
      i.e., a sum of clipped normalized contributions.

    - We then add correlated Gaussian noise across indices i using a
      kernel matrix in Y (based on mesh_points), following the same
      spirit as the correlated noise construction in DP-GRAMS.

    - The noise scale sigma is chosen proportional to C / (n * epsilon),
      mirroring the heuristic / theoretical scaling used in DP-GRAMS,
      under the convention that the effective per-iteration sensitivity
      behaves like O(C / n).
    """
    # ------------------------------------------------------------
    # Setup & basic validation
    # ------------------------------------------------------------
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X).reshape(-1)
    Y = np.asarray(Y).reshape(-1)
    n = len(X)

    if len(Y) != n:
        raise ValueError("X and Y must have the same length.")

    # Initialize mesh_points: for this simple version we tie one per sample.
    if mesh_points is None:
        mesh_points = Y.copy()
    mesh_points = np.asarray(mesh_points).reshape(-1)
    k = len(mesh_points)

    if k != n:
        raise ValueError("dp_pms expects mesh_points length == len(X) == len(Y).")

    # ------------------------------------------------------------
    # Bandwidth selection
    # ------------------------------------------------------------
    if bandwidth is None:
        # Silverman on Y only (1D) as a simple, stable choice
        bandwidth = float(silverman_bandwidth(Y.reshape(-1, 1)))
    h = max(1e-3, float(bandwidth))  # lower bound to avoid degeneracy
    h2 = h ** 2

    # ------------------------------------------------------------
    # Clipping constant
    # ------------------------------------------------------------
    # This controls |q_j| <= C; chosen to be small relative to the scale
    # induced by bandwidth. You may tune this in experiments.
    C = 1.0 / h / 100.0

    # ------------------------------------------------------------
    # Noise scale (heuristic, DP-GRAMS-style)
    # ------------------------------------------------------------
    # Following the same spirit as DP-GRAMS:
    #   sigma ~ (C / (n * epsilon)) * polylog(n, T, 1/delta)
    eps_safe = max(epsilon, 1e-12)
    sigma = (
        (8.0 * C) / (n * eps_safe)
        * np.sqrt(
            T * np.log(2.0 / delta) * np.log(2.5 * n * max(T, 1) / (n * delta))
        )
    )

    if verbose:
        print(f"[dp_pms] n={n}, k={k}, h={h:.6f}, C={C:.6e}, sigma={sigma:.6e}")

    # ------------------------------------------------------------
    # Correlated noise kernel over mesh points (1D RBF in Y-space)
    # ------------------------------------------------------------
    # We correlate noise across indices i based on proximity of their
    # starting mesh_points in Y, analogous to the correlated noise idea
    # based on kernel similarities.
    D2 = cdist(mesh_points.reshape(-1, 1),
               mesh_points.reshape(-1, 1),
               metric="sqeuclidean")
    K_y = np.exp(-D2 / (2.0 * h2)) + 1e-8 * np.eye(k)

    try:
        # Preferred: Cholesky factor for a valid PSD kernel
        L_base = cholesky(K_y)
    except Exception:
        # Fallback: eigen decomposition with truncation of small negatives
        vals, vecs = eigh(K_y)
        vals[vals < 0.0] = 0.0
        L_base = (vecs * np.sqrt(vals)).astype(float)

    # ------------------------------------------------------------
    # DP-PMS iterations
    # ------------------------------------------------------------
    Y_new = mesh_points.copy()

    for t in range(T):
        # Deterministic partial mean-shift-style updates (no subsampling)
        for i in range(k):
            xi = X[i]
            yi = Y_new[i]

            # Joint kernel in (X, Y) around (xi, yi)
            # Using current Y_new[i] as the center for the response dimension
            w = np.exp(-((X - xi) ** 2 + (Y - yi) ** 2) / (2.0 * h2))
            w_sum = np.sum(w)

            if w_sum > 0.0:
                # Normalized per-sample contributions:
                #   q_j = w_j * (Y_j - yi) / w_sum
                q = (w * (Y - yi)) / (w_sum + 1e-12)

                # Clip each q_j to |q_j| <= C
                abs_q = np.abs(q)
                scales = np.minimum(1.0, C / (abs_q + 1e-12))
                q_clipped = q * scales

                # Sum of clipped contributions -> estimate of shift m(y) - y
                delta_y = np.sum(q_clipped)

                # Update Y_new[i]
                Y_new[i] = yi + delta_y
            # else: if w_sum == 0, we leave Y_new[i] unchanged

        # Add correlated Gaussian noise once per iteration
        if np.isfinite(sigma) and sigma > 0.0:
            if k > 1:
                # Draw base standard normal and correlate via L_base
                G = rng.normal(size=(k, 1))
                noise = (L_base @ G).flatten()
                Y_new += sigma * noise
            else:
                # Single point: scalar Gaussian
                Y_new += rng.normal(loc=0.0, scale=sigma, size=1)

        if verbose:
            max_shift = float(np.max(np.abs(q_clipped))) if w_sum > 0 else 0.0
            print(f"[dp_pms] Iter {t+1}/{T}, max |q_clipped|={max_shift:.3e}")

    return Y_new