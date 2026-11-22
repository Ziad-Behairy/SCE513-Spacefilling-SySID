"""
surrogate.py  --  simulation of the IDENTIFIED model and its exact adjoint
gradient, so that the phases can be optimised on the identified model
(surrogate-in-the-loop design, extension Q2).  numba is used only for speed;
the equations are the same as system.simulate / simulate_backward with the
true force law replaced by g_hat = theta^T phi(z).
"""
import numpy as np
from numba import njit
import config as cfg
import signals, cost
from ext import ext_config as ec
from ext.ident import CENTRES


@njit(cache=True)
def _g_and_grad(u, x1, x2, theta, C, hr, w2):
    zs0, zs1, zs2 = u / hr[0], x1 / hr[1], x2 / hr[2]
    g = theta[0] + theta[1] * zs0 + theta[2] * zs1 + theta[3] * zs2
    d0, d1, d2 = theta[1], theta[2], theta[3]          # d g / d z_s (linear part)
    for j in range(C.shape[0]):
        e0, e1, e2 = zs0 - C[j, 0], zs1 - C[j, 1], zs2 - C[j, 2]
        r = theta[4 + j] * np.exp(-0.5 * (e0 * e0 + e1 * e1 + e2 * e2) / w2)
        g += r
        d0 -= r * e0 / w2
        d1 -= r * e1 / w2
        d2 -= r * e2 / w2
    return g, d0 / hr[0], d1 / hr[1], d2 / hr[2]


@njit(cache=True)
def simulate_hat(u, x0, theta, C, hr, w2, ts):
    K = u.shape[0]
    X = np.empty((K, 2))
    x1, x2 = x0[0], x0[1]
    for k in range(K):
        X[k, 0] = x1
        X[k, 1] = x2
        g, _, _, _ = _g_and_grad(u[k], x1, x2, theta, C, hr, w2)
        x1, x2 = x1 + ts * x2, x2 + ts * g
        if abs(x1) > 1e6 or abs(x2) > 1e6:     # diverged: freeze (flagged by caller)
            for j in range(k + 1, K):
                X[j, 0] = x1
                X[j, 1] = x2
            return X
    return X


@njit(cache=True)
def backward_hat(u, X, gX, theta, C, hr, w2, ts):
    K = X.shape[0]
    gU = np.empty(K)
    lam1, lam2 = 0.0, 0.0
    for k in range(K - 1, -1, -1):
        _, du, dx1, dx2 = _g_and_grad(u[k], X[k, 0], X[k, 1], theta, C, hr, w2)
        gU[k] = ts * du * lam2
        lam1, lam2 = (gX[k, 0] + lam1 + ts * dx1 * lam2,
                      gX[k, 1] + ts * lam1 + (1.0 + ts * dx2) * lam2)
    return gU


_W2 = ec.RBF_WIDTH ** 2


def sim(theta, u_full, x0=cfg.X0):
    return simulate_hat(u_full, np.asarray(x0, float), theta, CENTRES, ec.HALF_RANGE, _W2, cfg.TS)


def objective_hat(phi, theta, P=cfg.P_PERIODS):
    """Space-filling cost of the samples predicted by the identified model
    and its exact gradient w.r.t. the phases (same chain as cost.objective)."""
    N = cfg.N
    u = signals.multisine(phi)
    u_full = np.tile(u, P)
    X_full = sim(theta, u_full)
    if not np.all(np.isfinite(X_full)) or np.max(np.abs(X_full)) > 1e5:
        return 1e3, np.zeros_like(phi)
    Z = np.column_stack([u, X_full[(P - 1) * N:]])
    C, gZ = cost.cost_and_grad_Z(Z)
    gX = np.zeros_like(X_full)
    gX[(P - 1) * N:] = gZ[:, 1:]
    gU_full = backward_hat(u_full, X_full, gX, theta, CENTRES, ec.HALF_RANGE, _W2, cfg.TS)
    gU = gU_full.reshape(P, N).sum(axis=0) + gZ[:, 0]
    return C, signals.multisine_grad(phi, gU)
