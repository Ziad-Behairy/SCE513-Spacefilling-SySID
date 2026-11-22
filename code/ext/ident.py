"""
ident.py  --  identification of the unknown force law g(u, x1, x2) with a
linear-in-parameters basis-function model and ridge least squares.

Why this model class?
  * linear in theta  -> ordinary (regularised) least squares, Ch. 7 of Ljung;
  * Gaussian RBFs are LOCAL: a basis function only learns where data exist,
    so the model accuracy depends directly on where the data are -- exactly
    the property that the space-filling design is supposed to improve;
  * the linear part [1, z_s] gives a sensible (linear) behaviour where no
    data exist, instead of falling to zero.
"""
import numpy as np
import config as cfg
from ext import ext_config as ec

_ax = np.linspace(-1, 1, ec.RBF_PTS)
CENTRES = np.stack([g.ravel() for g in np.meshgrid(_ax, _ax, _ax, indexing="ij")], axis=1)  # (M,3)


def features(Z):
    """phi(z) for every row of Z (n,3) -> (n, 4+M)."""
    Zs = Z / ec.HALF_RANGE
    D2 = ((Zs[:, None, :] - CENTRES[None, :, :]) ** 2).sum(-1)
    R = np.exp(-0.5 * D2 / ec.RBF_WIDTH ** 2)
    return np.column_stack([np.ones(len(Z)), Zs, R])


def g_true(Z):
    """True force law of Eq. (9): acceleration dx2/dt for z = [u, x1, x2]."""
    u, x1, x2 = Z[:, 0], Z[:, 1], Z[:, 2]
    s, l, a, c, m = cfg.S_SPRING, cfg.L_FREE, cfg.A_ANCHOR, cfg.C_DAMP, cfg.M_MASS
    return (u - s * x1 + s * l * x1 / np.sqrt(x1 ** 2 + a ** 2) - c * x2) / m


def regression_data(u, X):
    """
    One-step regression from one period of measured data (u_k, x_k), k=0..N-1.
    Periodic steady state: x_N = x_0, so the last sample wraps around.
      target y_k = (x2[k+1] - x2[k]) / Ts      regressor z_k = [u_k, x1_k, x2_k]
    """
    x2n = np.roll(X[:, 1], -1)
    y = (x2n - X[:, 1]) / cfg.TS
    Z = np.column_stack([u, X])
    return Z, y


def _ridge(F, y, lam):
    p = F.shape[1]
    reg = lam * np.eye(p)
    reg[0, 0] = 0.0                                   # do not shrink the offset
    return np.linalg.solve(F.T @ F + reg * len(y), F.T @ y)


def fit(Z, y, lambdas=ec.LAMBDAS, folds=ec.CV_FOLDS):
    """Ridge LS; lambda chosen by contiguous-block cross-validation on the
    training data ONLY.  Returns (theta, chosen lambda, cv curve)."""
    F = features(Z)
    n = len(y)
    idx = np.array_split(np.arange(n), folds)
    cv = []
    for lam in lambdas:
        err = 0.0
        for te in idx:
            tr = np.setdiff1d(np.arange(n), te)
            th = _ridge(F[tr], y[tr], lam)
            err += np.sum((F[te] @ th - y[te]) ** 2)
        cv.append(err / n)
    lam = lambdas[int(np.argmin(cv))]
    return _ridge(F, y, lam), float(lam), np.array(cv)


def predict(theta, Z):
    return features(Z) @ theta
