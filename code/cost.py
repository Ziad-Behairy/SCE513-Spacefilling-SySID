"""
cost.py  --  space-filling cost (paper Eqs. 5-8) and its exact gradient,
             plus the full chain phases -> signal -> simulation -> cost.

Samples:  z_k = [u_k, x1_k, x2_k],  k = 1..N  (one steady-state period)
Grid:     z~_i, i = 1..n, 20 points per axis -> n = 8000
Eq. (7):  d_i = sum_k exp( -1/2 (z~_i - z_k)^T Sigma^-1 (z~_i - z_k) )
Eq. (8):  C   = (1/n) sum_i 1 / (eps + d_i)       (lower = better coverage)

Trick: Sigma is diagonal, so dividing every coordinate by its sigma turns
the weighted distance into an ordinary squared distance:
   w = z / sigma,   (z~-z)^T Sigma^-1 (z~-z) = |w~ - w|^2
"""
import numpy as np
import config as cfg
import signals
import system


def make_grid(points=cfg.GRID_PTS):
    """All grid centres z~_i as an (n, 3) array, n = points^3."""
    axes = [np.linspace(lo, hi, points) for lo, hi in zip(cfg.DOMAIN_LO, cfg.DOMAIN_HI)]
    G = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in G], axis=1)


GRID = make_grid()


def densities(Z, grid=GRID):
    """d_i of Eq. (7) for every grid point; also returns the kernel matrix E (n, N)."""
    Wg = grid / cfg.SIGMA
    Wz = Z / cfg.SIGMA
    # |a-b|^2 = |a|^2 + |b|^2 - 2 a.b  (fast matrix form; clip tiny negatives)
    D2 = (Wg ** 2).sum(1)[:, None] + (Wz ** 2).sum(1)[None, :] - 2.0 * Wg @ Wz.T
    E = np.exp(-0.5 * np.maximum(D2, 0.0))
    return E.sum(axis=1), E


def cost(Z, grid=GRID, eps=cfg.EPS):
    """Eq. (8) for a set of samples Z (N, 3)."""
    d, _ = densities(Z, grid)
    return np.mean(1.0 / (eps + d))


def cost_and_grad_Z(Z, grid=GRID, eps=cfg.EPS):
    """
    Cost and dC/dZ (shape (N, 3)).
      dC/dd_i      = -1 / (n (eps + d_i)^2)                    =: a_i
      dd_i/dw_k    = E_ik * (w~_i - w_k)
      dC/dw_k      = sum_i a_i E_ik (w~_i - w_k)
      dC/dz_k      = dC/dw_k / sigma        (because w = z / sigma)
    """
    d, E = densities(Z, grid)
    n = grid.shape[0]
    C = np.mean(1.0 / (eps + d))
    a = -1.0 / (n * (eps + d) ** 2)
    Wg, Wz = grid / cfg.SIGMA, Z / cfg.SIGMA
    # sum_i a_i E_ik w~_i  and  sum_i a_i E_ik  in ONE matrix product:
    # E^T (n->N) times the 4 columns [a*w~_u, a*w~_x1, a*w~_x2, a]
    Mx = E.T @ np.column_stack([a[:, None] * Wg, a])     # (N, 4)
    gW = Mx[:, :3] - Wz * Mx[:, 3:4]
    return C, gW / cfg.SIGMA


def samples_from_phases(phi, P=cfg.P_PERIODS):
    """phases -> one period u -> steady-state states -> samples Z (N, 3)."""
    u = signals.multisine(phi)
    X_last, X_full, u_full = system.steady_state_response(u, P)
    Z = np.column_stack([u, X_last])
    return Z, u, X_full


def objective(phi, P=cfg.P_PERIODS, eps=cfg.EPS):
    """
    The function given to the optimizer: returns (C, dC/dphi).

    Chain rule, step by step:
      1. C depends on Z = [u, x1, x2] of the last period  -> gZ (N, 3)
      2. the state part of gZ is pushed back through all P periods of the
         Euler simulation (system.simulate_backward) -> dC/du at every step
      3. the input itself also appears directly in Z (column 0)
      4. u is periodic: u at step p*N + k equals u_k, so the gradients of the
         P copies are summed into one period
      5. finally dC/dphi via du_k/dphi_l = A cos(...)
    """
    N = cfg.N
    Z, u, X_full = samples_from_phases(phi, P)
    C, gZ = cost_and_grad_Z(Z, eps=eps)
    gX = np.zeros_like(X_full)
    gX[(P - 1) * N:] = gZ[:, 1:]                          # step 1
    gU_full = system.simulate_backward(X_full, gX)        # step 2
    gU = gU_full.reshape(P, N).sum(axis=0) + gZ[:, 0]     # steps 3 + 4
    return C, signals.multisine_grad(phi, gU)             # step 5


def coverage(Z, grid=GRID):
    """Extra metrics (M2): kappa = share of grid points with d_i >= 1,
    and share of samples that lie inside the domain box."""
    d, _ = densities(Z, grid)
    inside = np.all((Z >= cfg.DOMAIN_LO) & (Z <= cfg.DOMAIN_HI), axis=1)
    return np.mean(d >= 1.0), np.mean(inside)
