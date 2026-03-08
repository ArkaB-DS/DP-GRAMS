# dp_grams_beta.py

import numpy as np
import math
from scipy.spatial.distance import cdist
from numpy.linalg import cholesky, eigh


# ============================================================
# Orthonormal Legendre basis phi_m on [-1,1]
# phi_m(u) = sqrt((2m+1)/2) * P_m(u)
# ============================================================
def _phi_m(u, m):
    from numpy.polynomial.legendre import Legendre
    Pm = Legendre.basis(m)(u)
    return math.sqrt((2 * m + 1) / 2.0) * Pm


def _dphi_m(u, m):
    from numpy.polynomial.legendre import Legendre
    Pm = Legendre.basis(m)
    Pm_prime = Pm.deriv()(u)
    return math.sqrt((2 * m + 1) / 2.0) * Pm_prime


# ============================================================
# Tsybakov kernel (1D) + derivative
# K(u) = sum_{m=0}^{beta-1} phi_m(0) phi_m(u) I(|u|<=1)
# ============================================================
def _tsybakov_K1(u, beta=4):
    u = np.asarray(u)
    out = np.zeros_like(u, dtype=float)
    mask = (np.abs(u) <= 1.0)
    if not np.any(mask):
        return out
    um = u[mask]
    s = np.zeros_like(um, dtype=float)
    for m in range(beta):
        s += _phi_m(0.0, m) * _phi_m(um, m)
    out[mask] = s
    return out


def _tsybakov_dK1(u, beta=4):
    u = np.asarray(u)
    out = np.zeros_like(u, dtype=float)
    mask = (np.abs(u) <= 1.0)
    if not np.any(mask):
        return out
    um = u[mask]
    s = np.zeros_like(um, dtype=float)
    for m in range(beta):
        s += _phi_m(0.0, m) * _dphi_m(um, m)
    out[mask] = s
    return out


# ============================================================
# Correlation kernel for DP noise across initializations
# ============================================================
def _rbf_kernel_matrix(A, B=None, hloc=1.0):
    if B is None:
        B = A
    D2 = cdist(A, B, metric="sqeuclidean")
    return np.exp(-D2 / (2.0 * hloc * hloc))


# ============================================================
# DP-GRAMS-beta main
# ============================================================
def dp_grams_beta(
    X,
    epsilon,
    delta,
    beta=4,
    initial_modes=None,
    T=None,
    m=None,
    h=None,
    p0=0.1,
    rng=None,
    clip_multiplier=1.0,
    noise_rbf_h=None,
    init_epsilon_frac=0.1,
    return_diagnostics=False,
):

    if rng is None:
        rng = np.random.default_rng()
    X = np.asarray(X)
    n, d = X.shape

    if not (0 < init_epsilon_frac < 1):
        raise ValueError("init_epsilon_frac must be in (0,1).")

    # Split epsilon between initialization and DP updates 
    epsilon_init = float(init_epsilon_frac * epsilon)
    epsilon_alg = float(max(1e-12, epsilon - epsilon_init))


    # -----------------------------
    # Minibatch size and iterations
    # -----------------------------
    if m is None:
        m = max(1, int(n / np.log(max(3, n))))
    if T is None:
        T = int(np.ceil(np.log(max(3, n))))

    # ---------------------------------------------
    # Bandwidth selection 
    # ---------------------------------------------
    # Theory, up to constants:
    #   h_non_dp ~ (log n / n)^{1 / (d + 2 * beta)}
    #   h_dp ~ (K / (n^2 * epsilon^2))^{1 / (2 * d + 2 * beta)} where K is a polylog bundle
    if h is None:
        # Compact polylog bundle used for the bandwidth regime.
        polylog = np.log(2.0 / delta) * np.log(2.5 * m * max(T, 1) / (n * delta))
        Kbundle = max(1.0, float(T * d * polylog))

        h_non_dp = (np.log(max(3, n)) / n) ** (1.0 / (d + 2.0 * beta))
        h_dp = (Kbundle / (n * n * epsilon_alg * epsilon_alg)) ** (1.0 / (2.0 * d + 2.0 * beta))

        # Use the larger regime scale.
        h = h_dp if h_dp > h_non_dp else h_non_dp

    if noise_rbf_h is None:
        noise_rbf_h = h

    # -----------------------------
    # Initialization: DAPI
    # -----------------------------
    if initial_modes is None:
        k = max(1, int(np.floor(n * p0)))

        # Local density proxy at scale h
        h2 = h * h
        diffs = X[:, None, :] - X[None, :, :]
        dist2 = np.sum(diffs**2, axis=2)
        within = (dist2 <= h2)
        u = (np.sum(within, axis=1).astype(float) - 1.0) / max(1, (n - 1))

        # Per-draw epsilon for anchor sampling
        eps_draw = epsilon_init / max(1, np.sqrt(k))

        # Exponential-style weights (sensitivity <= 1):
        # P(i) is proportional to exp((eps_draw / 2) * u_i)
        logits = (eps_draw / 2.0) * u
        logits -= np.max(logits)  # stabilize
        w = np.exp(logits)
        w_sum = float(np.sum(w))
        if (not np.isfinite(w_sum)) or (w_sum <= 0.0):
            probs = np.ones(n, dtype=float) / float(n)
        else:
            probs = w / w_sum

        # Sample anchors with replacement
        anchor_idx = rng.choice(n, size=k, replace=True, p=probs)
        anchors = X[anchor_idx].copy()

        # Jitter inside a ball of radius h
        Z = rng.normal(size=(k, d))
        norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        directions = Z / norms
        radii = (rng.random(size=(k, 1)) ** (1.0 / max(1, d))) * h
        modes = anchors + directions * radii

    else:
        modes = np.asarray(initial_modes).copy()

    k = len(modes)
    if k == 0:
        if return_diagnostics:
            return h, np.empty((0, d)), {"p_hat_nonpos_rate": np.nan, "p_hat_min": np.nan}
        return h, np.empty((0, d))

    # -----------------------------
    # Per-iteration epsilon via composition
    # eps_iter = epsilon / sqrt(8 * T * log(2 / delta))
    # -----------------------------
    eps_iter = float(epsilon_alg) / np.sqrt(8.0 * T * np.log(2.0 / delta))


    # -----------------------------
    # Clipping scale: ~ h^{-(d+1)}
    # -----------------------------
    C = (h ** (-1.0 - d)) * clip_multiplier

    # -----------------------------
    # Noise scale as a function of clipping
    # -----------------------------
    def compute_sigma(C_local: float) -> float:
        log_term = np.log(2.5 * m * T / (n * delta))
        # Small-epsilon approximation
        sigma_approx = (8.0 * C_local / (n * epsilon_alg)) * np.sqrt(
            T * np.log(2.0 / delta) * log_term
        )
        if epsilon_alg <= 1.0:
            return float(sigma_approx)

        # Exact
        denom = np.log(1.0 + (n / m) * (np.exp(eps_iter) - 1.0))
        sigma_exact = ( (C_local / m) / denom ) * np.sqrt(8.0 * log_term)
        return float(sigma_exact)

    sigma = compute_sigma(C)

    # -----------------------------
    # Correlated noise across initializations
    # -----------------------------
    K_modes = _rbf_kernel_matrix(modes, modes, hloc=noise_rbf_h) + 1e-8 * np.eye(k)
    try:
        L_base = cholesky(K_modes)
    except Exception:
        vals, vecs = eigh(K_modes)
        vals[vals < 0] = 0.0
        L_base = (vecs * np.sqrt(vals)).astype(float)

    # -----------------------------
    # Main loop
    # -----------------------------
    modes_curr = modes.copy()
    eta = h * h

    total_phat_checks = 0
    nonpos_phat_count = 0
    min_phat_seen = np.inf  # minimum p_hat for diagnostics

    for _t in range(T):
        avg_steps = np.zeros_like(modes_curr)  # sum of clipped q_i over minibatch

        for i in range(k):
            x = modes_curr[i]
            batch_idx = rng.choice(n, m, replace=False)
            B = X[batch_idx]  # (m,d)

            # u = (x - X_i)/h
            u = (x[None, :] - B) / h  # (m,d)

            # Evaluate K1 and dK1 coordinate-wise
            K1_all = np.zeros_like(u, dtype=float)
            dK1_all = np.zeros_like(u, dtype=float)
            for j in range(d):
                K1_all[:, j] = _tsybakov_K1(u[:, j], beta=beta)
                dK1_all[:, j] = _tsybakov_dK1(u[:, j], beta=beta)

            # Product kernel values K(u_i)
            K_prod = np.prod(K1_all, axis=1)  # (m,)
            S = float(np.sum(K_prod))  # (unscaled) KDE numerator on minibatch

            # Track diagnostics
            p_hat = (S / m) / (h ** d)  # scaled for diagnostics
            total_phat_checks += 1
            min_phat_seen = min(min_phat_seen, p_hat)

            # If S <= 0, take no step
            if S <= 0.0:
                nonpos_phat_count += 1
                avg_steps[i] = 0.0
                continue

            prod_except = np.zeros((m, d), dtype=float)
            for j in range(d):
                if d == 1:
                    prod_except[:, j] = 1.0
                else:
                    prod_except[:, j] = np.prod(K1_all[:, np.arange(d) != j], axis=1)

            grad_u = dK1_all * prod_except  # (m,d)
            q_i = (h * grad_u) / (S + 1e-12)  # (m,d)

            # Clip each q_i to norm <= C
            norms = np.linalg.norm(q_i, axis=1, keepdims=True)
            scales = np.minimum(1.0, C / (norms + 1e-12))
            q_i_clipped = q_i * scales

            avg_steps[i] = np.sum(q_i_clipped, axis=0)

        # Correlated Gaussian noise across modes
        # Columns of G are independent across coordinates.
        # L_base induces correlation across initializations.
        G = rng.normal(size=(k, d))
        noise = sigma * (L_base @ G)

        # Update with eta = h^2
        modes_curr += eta * (avg_steps + noise)

    final_modes = modes_curr.copy()

    if not return_diagnostics:
        return h, final_modes

    diag = {
        "p_hat_nonpos_rate": float(nonpos_phat_count / max(1, total_phat_checks)),
        "p_hat_min": float(min_phat_seen) if np.isfinite(min_phat_seen) else np.nan,
        "sigma": float(sigma),
        "C": float(C),
        "eta": float(eta),
        "T": int(T),
        "m": int(m),
        "k_inits": int(k),
        "beta": int(beta),
    }
    return h, final_modes, diag
