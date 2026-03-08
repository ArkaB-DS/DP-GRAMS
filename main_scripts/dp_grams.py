# dp_grams.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import cholesky, eigh
from sklearn.neighbors import NearestNeighbors

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_scripts.ms import mean_shift
from main_scripts.merge import merge_modes

@dataclass(frozen=True)
class DPGramsInitInfo:
    """DP initialization diagnostics."""
    u: np.ndarray                 # (n,) local neighbor fraction in [0,1]
    probs: np.ndarray             # (n,) exp-mech-like sampling probabilities
    anchor_idx: np.ndarray        # (k,) sampled anchor indices (with replacement)
    anchors: np.ndarray           # (k,d) anchor points X[anchor_idx]
    init_modes: np.ndarray        # (k,d) jittered initializations used by DP-GRAMS
    k: int                        # number of initializations
    epsilon_init: float           # epsilon allocated to init stage
    eps_draw: float               # per-draw epsilon used inside init
    h_used: float                 # bandwidth used (after selection)


def dp_grams(
    X: np.ndarray,
    epsilon: float,
    delta: float,
    *,
    initial_modes: Optional[np.ndarray] = None,
    T: Optional[int] = None,
    m: Optional[int] = None,
    h: Optional[float] = None,
    p0: float = 0.1,
    rng: Optional[np.random.Generator] = None,
    clip_multiplier: float = 1.0,
    init_epsilon_frac: float = 0.1,
    eta: float = 1.0,
    return_init_info: bool = False,
) -> Union[
    Tuple[float, np.ndarray],
    Tuple[float, np.ndarray, DPGramsInitInfo],
]:
    """
    Run DP-GRAMS to obtain differentially private mode estimates.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d).
    epsilon : float
        Global privacy budget epsilon (> 0).
    delta : float
        Global privacy parameter delta in (0,1).
    initial_modes : np.ndarray, optional
        Custom initializations of shape (k,d). If None, uses DP-BALL-EM initialization.
    T : int, optional
        Number of iterations. Defaults to ceil(log n).
    m : int, optional
        Minibatch size. Defaults to floor(n / log n).
    h : float, optional
        Gaussian kernel bandwidth. If None, chosen using the DP vs non-DP regime rule.
    p0 : float
        Fraction of points used for initializations when initial_modes is None.
        Uses k = max(1, floor(p0*n)).
    rng : np.random.Generator, optional
        RNG for reproducibility.
    clip_multiplier : float
        Scales the clipping threshold C = h^(1-d) * clip_multiplier.
    init_epsilon_frac : float
        Fraction of epsilon reserved for DP initialization. Remaining goes to DP-GRAMS updates.
    eta : float
        Step size multiplier applied to (avg_grads + noise).
    return_init_info : bool
        If True, also returns a DPGramsInitInfo object with init diagnostics.

    Returns
    -------
    h_used : float
        Bandwidth used by DP-GRAMS.
    final_modes : np.ndarray
        Array of shape (k, d) containing the private mode estimates (one per init).
    init_info : DPGramsInitInfo (optional)
        Returned only if return_init_info=True.
    """
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n,d). Got shape {X.shape}.")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain at least 2 samples.")
    if not (epsilon > 0):
        raise ValueError("epsilon must be > 0.")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1).")
    if not (0 < p0 <= 1):
        raise ValueError("p0 must be in (0,1].")
    if not (0 < init_epsilon_frac < 1):
        raise ValueError("init_epsilon_frac must be in (0,1).")
    if not np.isfinite(eta):
        raise ValueError("eta must be finite.")

    # -----------------------------
    # Minibatch size and iterations
    # -----------------------------
    # Defaults: T ~ ceil(log n), m ~ n / log n.
    if T is None:
        T = int(np.ceil(np.log(max(2, n))))
    T = max(1, int(T))

    if m is None:
        denom = np.log(max(3, n))
        m = int(n / denom)
    m = int(max(1, min(n, m)))

    # ---------------------------------------------
    # Bandwidth selection
    # ---------------------------------------------
    # polylog and Kconst aggregate dependence on (T, d, delta, m, n).
    polylog = np.log(2.0 / delta) * np.log(2.5 * m * max(T, 1) / (n * delta))
    Kconst = T * d * polylog

    # Non-private optimal rate (up to logs): (log n / n)^{1/(d+6)}
    h_non_dp = (np.log(max(3, n)) / n) ** (1.0 / (d + 6))

    # Threshold epsilon separating DP-dominated vs sampling-dominated regimes
    eps_non = np.sqrt(Kconst) / (n ** (3.0 / (d + 6)) * np.log(max(3, n)))

    # ---------------------------------------------------------
    # Split epsilon between initialization and DP-GRAMS updates
    # ---------------------------------------------------------
    epsilon_init = float(init_epsilon_frac * epsilon)
    epsilon_alg = float(max(1e-12, epsilon - epsilon_init))

    # Choose bandwidth according to theory:
    # - If epsilon is small: inflate h based on privacy term.
    # - If epsilon is large: use standard non-private bandwidth.
    if h is None:
        if epsilon_alg <= eps_non:
            h_used = (Kconst / (n**2 * epsilon_alg**2)) ** (1.0 / (2.0 * d + 6.0))
        else:
            h_used = h_non_dp
    else:
        h_used = float(h)

    if not (h_used > 0) or not np.isfinite(h_used):
        raise ValueError(f"Invalid bandwidth h_used={h_used}.")

    h2 = h_used * h_used

    # ---------------------------------------------
    # Initialization: DAPI
    # ---------------------------------------------
    init_info: Optional[DPGramsInitInfo] = None

    if initial_modes is None:
        k = max(1, int(np.floor(n * p0)))

        # Local density proxy at scale h: leave-one-out neighbor fraction in [0,1]


        X_nn = np.asarray(X, dtype=np.float32, order="C")

        nn = NearestNeighbors(radius=h_used, algorithm="ball_tree", metric="euclidean")
        nn.fit(X_nn)

        # sparse adjacency; sum gives neighbor counts (includes self)
        G = nn.radius_neighbors_graph(X_nn, radius=h_used, mode="connectivity")
        counts = np.asarray(G.sum(axis=1)).ravel()
        # leave-one-out neighbor fraction in [0,1]
        u = (counts - 1.0) / max(1, (n - 1))

        # Per-draw epsilon for anchor sampling (k sequential draws composed)
        eps_draw = epsilon_init / max(1, k)

        # Exponential mechanism weights (sensitivity <= 1):
        # P(i) \propto exp((eps_draw/2) * u_i)
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

        # Jitter uniformly inside ball of radius h around each anchor
        Z = rng.normal(size=(k, d))
        norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        directions = Z / norms
        radii = (rng.random(size=(k, 1)) ** (1.0 / max(1, d))) * h_used 
        modes = anchors + directions * radii

        if return_init_info:
            init_info = DPGramsInitInfo(
                u=u,
                probs=probs,
                anchor_idx=anchor_idx,
                anchors=anchors,
                init_modes=modes.copy(),
                k=k,
                epsilon_init=epsilon_init,
                eps_draw=eps_draw,
                h_used=h_used,
            )
    else:
        modes = np.asarray(initial_modes, dtype=float).copy()
        if modes.ndim != 2 or modes.shape[1] != d:
            raise ValueError(
                f"initial_modes must have shape (k,{d}). Got {modes.shape}."
            )

        if return_init_info:
            # For externally supplied initial modes, we provide a minimal init_info.
            init_info = DPGramsInitInfo(
                u=np.array([], dtype=float),
                probs=np.array([], dtype=float),
                anchor_idx=np.array([], dtype=int),
                anchors=np.empty((0, d), dtype=float),
                init_modes=modes.copy(),
                k=int(modes.shape[0]),
                epsilon_init=epsilon_init,
                eps_draw=0.0,
                h_used=h_used,
            )

    k = int(modes.shape[0])
    if k == 0:
        empty = np.empty((0, d), dtype=float)
        if return_init_info and init_info is not None:
            return h_used, empty, init_info
        return h_used, empty

    # -------------------------------------------------------
    # Per-iteration privacy budget via advanced composition
    # -------------------------------------------------------
    epsilon_iter = epsilon_alg / (2.0 * np.sqrt(2.0 * T * np.log(2.0 / delta)))

    # -------------------------------------------------------
    # Noise scale sigma(epsilon, delta, m, C) with subsampling amplification
    # -------------------------------------------------------
    def compute_sigma(C: float) -> float:
        """
        Compute Gaussian noise scale sigma using:
        - sensitivity of the mini-batch averaged/clipped contributions
        - amplification by sampling (m/n)
        - per-iteration epsilon_iter
        """
        # Numerically safe log term
        q = m / n
        denom = np.log(1.0 + (1.0 / max(1e-12, q)) * (np.exp(epsilon_iter) - 1.0))
        denom = max(1e-12, denom)

        sigma_exact = (2.0 * C / m) / denom * np.sqrt(2.0 * np.log(2.5 * m * T / (n * delta)))

        # Small-epsilon approximation
        if epsilon_alg <= 1.0:
            sigma_approx = (8.0 * C / (n * epsilon_alg)) * np.sqrt(
                T * np.log(2.0 / delta) * np.log(2.5 * m * T / (n * delta))
            )
            return float(sigma_approx)

        return float(sigma_exact)

    # -------------------------------------------------------
    # RBF kernel for correlated noise across initializations
    # -------------------------------------------------------
    def rbf_kernel_matrix(A: np.ndarray, B: Optional[np.ndarray] = None, hloc: float = h_used) -> np.ndarray:
        if B is None:
            B = A
        D2 = cdist(A, B, metric="sqeuclidean")
        return np.exp(-D2 / (2.0 * hloc * hloc))

    K_modes = rbf_kernel_matrix(modes, modes) + 1e-8 * np.eye(k)

    try:
        L_base = cholesky(K_modes)
    except Exception:
        vals, vecs = eigh(K_modes)
        vals = np.maximum(vals, 0.0)
        L_base = (vecs * np.sqrt(vals)).astype(float)

    # -------------------------------------------------------
    # Clipping constant C and noise calibration
    # -------------------------------------------------------
    # Practical proxy: C = h^(1-d) * clip_multiplier
    C = (h_used ** (1.0 - d)) * float(clip_multiplier)
    sigma = compute_sigma(C)

    # =======================================================
    # Main DP-GRAMS loop
    # =======================================================
    modes_curr = modes.copy()

    for _t in range(T):
        avg_grads = np.zeros_like(modes_curr)

        for i in range(k):
            x = modes_curr[i]

            # Subsample minibatch (amplification)
            batch_indices = rng.choice(n, m, replace=False)
            batch = X[batch_indices]

            diffs = batch - x  # (m,d)
            weights = np.exp(-np.sum(diffs**2, axis=1) / (2.0 * h2))  # (m,)
            weights_sum = float(np.sum(weights) + 1e-12)

            # Per-sample contributions
            q_i = (weights[:, None] * diffs) / weights_sum  # (m,d)

            # Clip each q_i to norm <= C
            norms = np.linalg.norm(q_i, axis=1, keepdims=True)
            scales = np.minimum(1.0, C / (norms + 1e-12))
            q_i_clipped = q_i * scales

            avg_grads[i] = np.sum(q_i_clipped, axis=0)

        # Correlated noise across modes (independent across iterations)
        G = rng.normal(size=(k, d))
        noise = sigma * (L_base @ G)  # (k,d)

        # Update
        modes_curr += float(eta) * (avg_grads + noise)

    final_modes = modes_curr.copy()

    if return_init_info and init_info is not None:
        return h_used, final_modes, init_info
    return h_used, final_modes