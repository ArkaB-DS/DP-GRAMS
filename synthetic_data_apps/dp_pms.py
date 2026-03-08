# dp-pms.py

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
    T: int | None = None,
    m: int | None = None,
    clip_multiplier: float = 0.01,
    rng=None,
    bandwidth=None,
    verbose: bool = False,
    init_epsilon_frac: float = 0.1,
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
        For this implementation, we require len(mesh_points) == n
        and conceptually pair (X_i, mesh_points[i]) as starting points.

        If None, we DO DAP initialization on Z=(X,Y)
        and set mesh_points to the jittered Y coordinates.
    epsilon : float
        Global privacy parameter epsilon.
    delta : float
        Global privacy parameter delta.
    T : int or None
        Number of DP-PMS iterations. If None, set to ceil(log n).
    m : int or None
        Minibatch size. If None, set to floor(n / log n).
        If m >= n, we fall back to full-batch updates.
    clip_multiplier : float
        Scales the clipping threshold C. We set
            C = clip_multiplier * (1 / h)
        so the default 0.01 roughly matches the old 1 / (100 h).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    bandwidth : float, optional
        Bandwidth h for the Gaussian kernel in (X, Y)-space.
        If None, estimated from Y via Silverman's rule (Y-only), matching PMS.
    verbose : bool
        If True, print basic diagnostic information.
    init_epsilon_frac : float, default=0.1
        Fraction of epsilon reserved for DP initialization (DP-BALL-EM anchor sampling).
        Remaining epsilon is used for DP-PMS iterations (noise added each iteration).

    Returns
    -------
    Y_new : np.ndarray, shape (n,)
        DP-PMS-updated modal regression estimates corresponding to mesh_points.
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

    # ------------------------------------------------------------
    # Bandwidth selection
    # ------------------------------------------------------------
    if bandwidth is None:
        # Silverman on Y only (1D) as a simple, stable choice
        bandwidth = float(silverman_bandwidth(Y.reshape(-1, 1)))
    h = max(1e-3, float(bandwidth))  # lower bound to avoid degeneracy
    h2 = h ** 2

    # ------------------------------------------------------------
    # Default T and m 
    
    # ------------------------------------------------------------
    if T is None:
        T = int(np.ceil(np.log(max(n, 2))))  # T ~ log n

    if m is None:
        m = int(n / max(np.log(max(n, 2)), 1.0))  # m ~ n / log n
    m = max(1, min(int(m), n))  # clamp to [1, n]

    # ---------------------------------------------------------
    # Split epsilon between initialization and DP-PMS updates
    # ---------------------------------------------------------
    eps_global = float(epsilon)
    delta_global = float(delta)

    epsilon_init = float(init_epsilon_frac) * eps_global
    epsilon_alg = max(1e-12, eps_global - epsilon_init)

    # ------------------------------------------------------------
    # Initialize mesh_points: one per sample
    # ------------------------------------------------------------
    # If mesh_points is None: DAP initialization on Z=(X,Y).
    # We sample k = n anchors (one path per sample) and jitter inside B(h).
    if mesh_points is None:
        # Joint data Z=(X,Y) in R^2
        Z = np.column_stack([X, Y])  # shape (n, 2)
        d_joint = 2

        # k paths = n (one path per sample)
        k = n

        # Density proxy at scale h in joint space: u_i = #{j: ||Z_j - Z_i|| <= h}
        diffs = Z[:, None, :] - Z[None, :, :]
        dist2 = np.sum(diffs ** 2, axis=2)
        counts = np.sum(dist2 <= (h ** 2), axis=1).astype(float)  # includes self
        u = (counts - 1.0) / max(1, (n - 1))

        # Per-draw privacy for anchor sampling )
        eps_draw = epsilon_init / max(1, np.sqrt(k))  

        # Exponential mechanism weights: P(i) ∝ exp((eps_draw/2) * u_i)
        logits = (eps_draw / 2.0) * u
        logits -= np.max(logits)
        w = np.exp(logits)
        w_sum = float(np.sum(w))
        if (not np.isfinite(w_sum)) or (w_sum <= 0.0):
            probs = np.ones(n, dtype=float) / float(n)
        else:
            probs = w / w_sum

        # Sample anchors (WITH replacement, as you wrote)
        anchor_idx = rng.choice(n, size=k, replace=True, p=probs)
        anchors = Z[anchor_idx].copy()

        # Jitter uniformly inside ball of radius h around each anchor
        Z_noise = rng.normal(size=(k, d_joint))
        norms = np.linalg.norm(Z_noise, axis=1, keepdims=True) + 1e-12
        directions = Z_noise / norms
        radii = (rng.random(size=(k, 1)) ** (1.0 / max(1, d_joint))) * h * 0.01
        Z_init = anchors + directions * radii


        ord_X = np.argsort(X)                 # ranks of original X
        ord_Z = np.argsort(Z_init[:, 0])      # ranks of sampled/jittered X

        mesh_points = np.empty(n, dtype=float)
        mesh_points[ord_X] = Z_init[ord_Z, 1]  # assign Y by matching X-quantiles

    mesh_points = np.asarray(mesh_points).reshape(-1)
    k = len(mesh_points)

    if k != n:
        raise ValueError("dp_pms expects mesh_points length == len(X) == len(Y).")

    # ------------------------------------------------------------
    # Clipping constant C 
    # ------------------------------------------------------------
    # Modal regression uses joint dimension (X,Y) = 2, but the update is scalar in Y.
    # You requested C ~ 1/h (and keep it as a sum form).
    C = clip_multiplier * (1.0 / h)

    # ------------------------------------------------------------
    # Noise scale 
    # ------------------------------------------------------------
    eps_safe = max(epsilon_alg, 1e-12)
    delta_safe = max(delta_global, 1e-12)
    q_samp = m / n  # sampling ratio

    # Per-iteration epsilon via advanced composition 
    eps_iter = eps_safe / (2 * np.sqrt(2 * T * np.log(2.0 / delta_safe)))

    def compute_sigma():
        """
        Gaussian noise scale sigma using:
          - sensitivity of the (mini-batch) averaged/clipped contributions
          - amplification by sampling (ratio q_samp = m/n)
          - per-iteration eps_iter, delta_safe
          - Gaussian mechanism calibration 
        """
      
        sigma_exact = (2.0 * C / m) / np.log(
            1.0 + (1.0 / q_samp) * (np.exp(eps_iter) - 1.0)
        ) * np.sqrt(
            2.0 * np.log(2.5 * m * max(T, 1) / (n * delta_safe))
        )

        # Small-epsilon approximation
        if eps_safe <= 1.0:
            sigma_approx = (8.0 * C / (n * eps_safe)) * np.sqrt(
                T * np.log(2.0 / delta_safe)
                * np.log(2.5 * m * max(T, 1) / (n * delta_safe))
            )
            return sigma_approx

        return sigma_exact

    sigma = compute_sigma()

    if verbose:
        print(
            f"[dp_pms] n={n}, k={k}, h={h:.6f}, C={C:.6e}, "
            f"sigma={sigma:.6e}, T={T}, m={m}, "
            f"eps_global={eps_global:.3g}, eps_init={epsilon_init:.3g}, eps_alg={epsilon_alg:.3g}, "
            f"delta={delta_global:.1e}"
        )

    # ------------------------------------------------------------
    # Correlated noise kernel over mesh points (1D RBF in Y-space)
    # ------------------------------------------------------------
    D2 = cdist(
        mesh_points.reshape(-1, 1),
        mesh_points.reshape(-1, 1),
        metric="sqeuclidean"
    )
    K_y = np.exp(-D2 / (2.0 * h2)) + 1e-8 * np.eye(k)

    try:
 
        L_base = cholesky(K_y)
    except Exception:
        vals, vecs = eigh(K_y)
        vals[vals < 0.0] = 0.0
        L_base = (vecs * np.sqrt(vals)).astype(float)

    # ------------------------------------------------------------
    # DP-PMS iterations (mini-batch or full-batch if m == n)
    # ------------------------------------------------------------
    Y_new = mesh_points.copy()

    for t in range(T):
        for i in range(k):
            xi = X[i]
            yi = Y_new[i]

            # Mini-batch indices (or full batch if m == n)
            if m == n:
                batch_idx = np.arange(n)
            else:
                batch_idx = rng.choice(n, size=m, replace=False)

            Xb = X[batch_idx]
            Yb = Y[batch_idx]

            # Joint kernel in (X, Y) around (xi, yi)
            w = np.exp(-((Xb - xi) ** 2 + (Yb - yi) ** 2) / (2.0 * h2))
            w_sum = np.sum(w)

            if w_sum > 0.0:
                # Normalized per-sample contributions:
                #   q_j = w_j * (Y_j - yi) / sum_l w_l
                # Keep it in "sum form" (no extra cancellations).
                q = (w * (Yb - yi)) / (w_sum + 1e-12)

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
                G = rng.normal(size=(k, 1))
                noise = (L_base @ G).flatten()
                Y_new += sigma * noise
            else:
                Y_new += rng.normal(loc=0.0, scale=sigma, size=1)

        if verbose:
            print(f"[dp_pms] Iter {t+1}/{T}")

    return Y_new
