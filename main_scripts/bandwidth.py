# bandwidth.py
# ---------------------------------------------------------------
# Bandwidth selection utilities for DP-GRAMS.
# These implement Silverman's rule-of-thumb, used in the paper
# for initializing the Gaussian kernel bandwidth h in KDE-based
# mode estimation.
# ---------------------------------------------------------------

import numpy as np
import math

def silverman_bandwidth(data):
    """
    Compute the Silverman bandwidth for 1D or multidimensional data.
    For univariate data, uses the standard scalar formula.
    For multivariate data, computes a matrix-consistent scalar bandwidth.
    
    Parameters
    ----------
    data : np.ndarray
        Input array of shape (n_samples, n_features).
    
    Returns
    -------
    h : float
        Bandwidth value used for Gaussian kernel smoothing
        in mean-shift and DP-GRAMS updates.
    """
    data = np.atleast_2d(data)
    # For single-dimensional inputs, apply the univariate rule
    if data.shape[1] == 1:
        return _silverman_univariate(data[:, 0])
    # Otherwise, use the multivariate extension
    else:
        return _silverman_multivariate(data)

def _silverman_univariate(x):
    """
    Silverman's rule-of-thumb for univariate bandwidth selection:
        h = 0.9 * min(sigma, IQR / 1.34) * n^(-1/5)
    """
    n = len(x)
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.34)
    return 0.9 * sigma * n ** (-1 / 5)

def _silverman_multivariate(X):
    """
    Multivariate Silverman bandwidth approximation:
    Uses the trace of the covariance matrix to compute
    an isotropic scalar bandwidth. This ensures that
    Gaussian kernels remain roughly spherical and scales
    appropriately with data dimensionality d.
    
    Formula derived from Scott's generalization:
        h^2 = (2/d) * tr(Sigma) * [4 / ((2d + 1) * n)]^(2 / (d + 4))
    where Sigma is the covariance matrix.
    
    Returns sqrt(h^2) as scalar bandwidth (for isotropic kernels).
    """
    n, d = X.shape
    cov = np.cov(X.T)
    tr = np.trace(cov)
    # Compute isotropic bandwidth squared
    h2 = (2 / d) * tr * (4 / ((2 * d + 1) * n)) ** (2 / (d + 4))
    return math.sqrt(h2)
