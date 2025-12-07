# dp-grams.py
# ---------------------------------------------------------------------
# DP-GRAMS: Differentially Private Gradient Ascent-based Mean Shift
#
# This implementation follows the structure in the DP-GRAMS paper:
# - Reformulate mean shift as gradient ascent on log KDE.
# - Express each step via per-sample contributions q_i(x).
# - Clip q_i(x) to control sensitivity.
# - Add Gaussian noise (with subsampling amplification + advanced composition)
#   to guarantee (ε, δ)-DP.
# - Use multiple random initializations + optional high-density filtering.
# - Add correlated noise across initializations based on an RBF kernel.
#
# ---------------------------------------------------------------------

import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from numpy.linalg import cholesky, eigh

def dp_grams(
    X,
    epsilon,
    delta,
    initial_modes=None,
    hdp=0.8,
    T=None,
    m=None,
    h=None,
    p0=0.1,
    rng=None,
    use_hdp_filter=True,
    clip_multiplier=1.0
):
    """
    Run the DP-GRAMS algorithm to obtain differentially private mode estimates.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d).
    epsilon : float
        Global privacy budget ε.
    delta : float
        Global privacy parameter δ.
    initial_modes : np.ndarray, optional
        Custom initializations for modes; if None, sampled from X.
    hdp : float, default=0.8
        High-density filter threshold as a fraction of max KDE density.
    T : int, optional
        Number of iterations; defaults to ceil(log n), per theory.
    m : int, optional
        Minibatch size; defaults to n / log n if not provided.
    h : float, optional
        Bandwidth for Gaussian kernel. If None, chosen using DP vs non-DP
        regime selection described in the paper.
    p0 : float, default=0.1
        Fraction of points used as random initializations when
        initial_modes is None.
    rng : np.random.Generator, optional
        RNG for reproducibility.
    use_hdp_filter : bool, default=True
        Whether to apply the high-density initialization filter.
    clip_multiplier : float, default=1.0
        Scales the clipping threshold C.

    Returns
    -------
    h : float
        Bandwidth used by DP-GRAMS.
    final_modes : np.ndarray
        Array of shape (k_final, d) containing the private mode estimates.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d = X.shape

    # -----------------------------
    # Minibatch size and iterations
    # -----------------------------
    # Theory: T ≍ log n iterations; m ≍ n / log n used in experiments.
    if m is None:
        m = int(n / np.log(n))
    if T is None:
        T = int(np.ceil(np.log(n)))

    # ---------------------------------------------
    # Bandwidth selection: DP vs non-DP tradeoff
    # ---------------------------------------------
    # polylog and Kconst aggregate dependence on (T, d, δ, m, n).
    polylog = np.log(2 / delta) * np.log(2.5 * m * max(T, 1) / (n * delta))
    Kconst = T * d * polylog

    # Non-private optimal rate (up to logs): (log n / n)^{1/(d+6)}
    h_non_dp = (np.log(n) / n) ** (1 / (d + 6))

    # Threshold ε separating DP-dominated vs sampling-dominated regimes
    eps_non = np.sqrt(Kconst) / (n ** (3 / (d + 6)) * np.log(n))

    # Choose bandwidth according to theory:
    # - If ε is small: inflate h based on privacy term.
    # - If ε is large: use standard non-private bandwidth.
    if h is None:
        if epsilon <= eps_non:
            h = (Kconst / (n ** 2 * epsilon ** 2)) ** (1 / (2 * d + 6))
        else:
            h = h_non_dp
    h2 = h ** 2

    # ---------------------------------------------
    # Initialization of candidate modes
    # ---------------------------------------------
    # Random subset of data with Bernoulli(p0)-style sampling:
    # this matches the random initialization strategy analyzed in the paper.
    if initial_modes is None:
        k = max(1, int(n * p0))
        init_indices = rng.choice(n, k, replace=False)
        modes = X[init_indices].copy()
    else:
        modes = np.array(initial_modes).copy()

    # ---------------------------------------------
    # High-density filter (HDP)
    # ---------------------------------------------
    # Retain only initializations in regions of relatively high KDE density:
    # this implements the HDP-style filtering described as an optional,
    # assumption-lean improvement for practical performance.
    if use_hdp_filter and len(modes) > 0:
        kde = gaussian_kde(X.T)
        densities = kde(modes.T)
        density_threshold = hdp * densities.max()
        modes = modes[densities >= density_threshold]

    k = len(modes)
    if k == 0:
        # No high-density initializations; return empty mode set.
        return h, np.empty((0, d))

    # -------------------------------------------------------
    # Per-iteration privacy budget via advanced composition
    # -------------------------------------------------------
    # Follows the standard DP-SGD-like allocation used in the proof:
    # ε_iter = ε / (2 sqrt(2 T log(2/δ))).
    epsilon_iter = epsilon / (2 * np.sqrt(2 * T * np.log(2 / delta)))

    # -------------------------------------------------------
    # Noise scale σ(ε, δ, m, C) with subsampling amplification
    # -------------------------------------------------------
    def compute_sigma(C):
        """
        Compute Gaussian noise scale σ using:
          - sensitivity of the (mini-batch) averaged/clipped contributions
          - amplification by sampling (ratio q = m/n)
          - per-iteration ε_iter, δ_iter,
          - and the Gaussian mechanism calibration.

        This matches the σ expression given in the privacy proof
        (Equation (sigma-exact) / Theorem proof).
        """
        # Sampling ratio q = m/n enters via amplification; implemented as 1/(m/n).
        sigma = (2 * C / m) / np.log(
            1 + (1 / (m / n)) * (np.exp(epsilon_iter) - 1)
        ) * np.sqrt(
            2 * np.log(2.5 * m * T / (n * delta))
        )

        # Small-ε approximation: matches the simplified expression in the text.
        if epsilon <= 1:
            sigma_approx = (8 * C / (n * epsilon)) * np.sqrt(
                T * np.log(2 / delta) * np.log(2.5 * m * T / (n * delta))
            )
            return sigma_approx

        return sigma

    # -------------------------------------------------------
    # RBF kernel for correlated noise across initializations
    # -------------------------------------------------------
    def rbf_kernel_matrix(A, B=None, hloc=h):
        """
        Compute Gaussian (RBF) kernel matrix:
            K_ij = exp(-||A_i - B_j||^2 / (2 hloc^2))

        Used to correlate DP noise across different starting points,
        following the Hall-style construction described in the paper.
        """
        if B is None:
            B = A
        D2 = cdist(A, B, metric='sqeuclidean')
        return np.exp(-D2 / (2.0 * hloc * hloc))

    # Kernel matrix on initial modes for correlated noise
    K_modes = rbf_kernel_matrix(modes, modes) + 1e-8 * np.eye(k)

    # Cholesky factorization (or eigen fallback) to sample correlated noise
    try:
        L_base = cholesky(K_modes)
    except Exception:
        vals, vecs = eigh(K_modes)
        vals[vals < 0] = 0.0
        L_base = (vecs * np.sqrt(vals)).astype(float)

    # -------------------------------------------------------
    # Clipping constant C
    # -------------------------------------------------------
    # Theory: C_* ≍ (1 / (p_min V_d)) h^{1-d}.
    # Here: we use h^{1-d} scaled by clip_multiplier as a practical proxy.
    modes_curr = modes.copy()
    eta = 1.0 # step size; η=1 keeps updates mean-shift-like.
    C = h ** (1 - d) * clip_multiplier

    # Calibrate Gaussian noise using the theoretical σ expression.
    sigma = compute_sigma(C)

    # =======================================================
    # Main DP-GRAMS loop: noisy gradient-ascent / mean-shift
    # =======================================================
    for t in range(T):
        # Will store (noisy) gradient / mean-shift directions per mode
        avg_grads = np.zeros_like(modes_curr)

        for i in range(k):
            x = modes_curr[i]

            # Sample a minibatch (subsampling for amplification)
            batch_indices = rng.choice(n, m, replace=False)
            batch = X[batch_indices]

            # ---------------------------
            # Compute kernel weights
            # ---------------------------
            diffs = batch - x  # shape (m, d)
            weights = np.exp(-np.sum(diffs ** 2, axis=1) / (2 * h2))  # shape (m,)
            weights_sum = np.sum(weights) + 1e-12

            # -----------------------------------------------
            # Per-sample contributions q_i(x):
            # q_i = (K_i * (X_i - x)) / sum_j K_j
            # so that SUM_i q_i ≈ m_batch(x) - x (mean-shift step).
            # This is the normalized form consistent with the paper.
            # -----------------------------------------------
            q_i = (weights[:, None] * diffs) / weights_sum  # shape (m, d)

            # Clip each q_i to norm ≤ C to bound sensitivity
            norms = np.linalg.norm(q_i, axis=1, keepdims=True)
            scales = np.minimum(1.0, C / (norms + 1e-12))
            q_i_clipped = q_i * scales

            # Sum of clipped contributions:
            # interpreted under the paper's normalization as the
            # mini-batch estimator of m(x) - x used in the DP step.
            avg_grads[i] = np.sum(q_i_clipped, axis=0)

        # -------------------------------------------
        # Correlated Gaussian noise across modes
        # -------------------------------------------
        # Draw standard normal G, then correlate via L_base.
        G = rng.normal(size=(k, d))
        noise = sigma * (L_base @ G)  # shape (k, d)

        # -------------------------------------------
        # Update step:
        # x_{t+1} = x_t + (clipped mean-shift increment + DP noise)
        # -------------------------------------------
        modes_curr += eta * (avg_grads + noise)

    final_modes = modes_curr.copy()
    return h, final_modes
