"""
gimbal.py  --  VALIDATION ON A SECOND, MORE REALISTIC SYSTEM (M5 add-on).

Question: does the surrogate-in-the-loop space-filling design (extension E2)
still help on a system it was never tuned on, and does the better model
matter for a real control task?

Plant: one axis of a robot gimbal (RoboMaster-like pitch axis), simulated.
    J dw/dt = tau - MGL sin(theta) - tau_stop(theta) - B w - TC tanh(w / V0)
      gravity  : centre of mass is not on the axis          (smooth nonlinearity)
      tau_stop : rubber bumpers at the joint limits +-TL   (hard, local nonlinearity)
      friction : viscous + smooth Coulomb                  (nonlinear in w)
States x1 = theta [rad], x2 = w [rad/s], input u = motor torque [N m].
Same discretisation as the course project: forward Euler, Ts = 0.01 s.

Nothing of the mass-spring code is changed: this module has its own domain,
grid, kernel widths, RBF scaling and input band, all chosen by one rule
(domain = about 3 std of the random-phase response; sigma = range / 10).
"""
import numpy as np
from numba import njit
from scipy.optimize import minimize

# ---------------------------------------------------------------- plant
J, MGL, B, TC, V0 = 0.05, 2.0, 0.25, 0.3, 0.5     # kg m^2, N m, N m s, N m, rad/s
KS, TL, WS = 15.0, 1.0, 0.03                      # bumper N m/rad, limit rad, smoothing rad
TS, N, P = 0.01, 2048, 8
LINES = np.arange(4, 101)                          # 0.2 .. 4.9 Hz (gimbal tracking band)
U_STD = 1.2                                        # N m
AMP = U_STD * np.sqrt(2.0 / LINES.size)
_ANG = 2 * np.pi * np.outer(LINES, np.arange(N)) / N

LO = np.array([-4.0, -1.4, -10.0])
HI = -LO
SIGMA = (HI - LO) / 10.0
EPS = 1e-3
_ax = [np.linspace(l, h, 20) for l, h in zip(LO, HI)]
GRID = np.stack([g.ravel() for g in np.meshgrid(*_ax, indexing="ij")], 1)

RBF_PTS = 7
_c = np.linspace(-1, 1, RBF_PTS)
CENTRES = np.stack([g.ravel() for g in np.meshgrid(_c, _c, _c, indexing="ij")], 1)
W2 = (2.0 / (RBF_PTS - 1)) ** 2
LAMBDAS = np.logspace(-8, 1, 19)


@njit(cache=True)
def _sp(x):            # smooth max(0, x) and its derivative
    t = x / WS
    if t > 30:
        return x, 1.0
    e = np.exp(t)
    return WS * np.log1p(e), e / (1.0 + e)


@njit(cache=True)
def g_true(u, x1, x2):
    a, da = _sp(x1 - TL)
    b, db = _sp(-x1 - TL)
    th = np.tanh(x2 / V0)
    g = (u - MGL * np.sin(x1) - KS * (a - b) - B * x2 - TC * th) / J
    dx1 = (-MGL * np.cos(x1) - KS * (da + db)) / J
    dx2 = (-B - TC * (1 - th * th) / V0) / J
    return g, 1.0 / J, dx1, dx2


@njit(cache=True)
def g_hat(u, x1, x2, theta):
    zs0, zs1, zs2 = u / 4.0, x1 / 1.4, x2 / 10.0
    g = theta[0] + theta[1] * zs0 + theta[2] * zs1 + theta[3] * zs2
    d0, d1, d2 = theta[1], theta[2], theta[3]
    for j in range(CENTRES.shape[0]):
        e0, e1, e2 = zs0 - CENTRES[j, 0], zs1 - CENTRES[j, 1], zs2 - CENTRES[j, 2]
        r = theta[4 + j] * np.exp(-0.5 * (e0 * e0 + e1 * e1 + e2 * e2) / W2)
        g += r
        d0 -= r * e0 / W2
        d1 -= r * e1 / W2
        d2 -= r * e2 / W2
    return g, d0 / 4.0, d1 / 1.4, d2 / 10.0


@njit(cache=True)
def _acc(u, x1, x2, theta, use_hat):
    if use_hat:
        return g_hat(u, x1, x2, theta)
    return g_true(u, x1, x2)


@njit(cache=True)
def simulate(u, theta, use_hat):
    K = u.shape[0]
    X = np.empty((K, 2))
    x1, x2 = 0.0, 0.0
    for k in range(K):
        X[k, 0], X[k, 1] = x1, x2
        g, _, _, _ = _acc(u[k], x1, x2, theta, use_hat)
        x1, x2 = x1 + TS * x2, x2 + TS * g
        if abs(x1) > 1e6 or abs(x2) > 1e6:
            for j in range(k + 1, K):
                X[j, 0], X[j, 1] = x1, x2
            return X
    return X


@njit(cache=True)
def backward(u, X, gX, theta, use_hat):
    K = X.shape[0]
    gU = np.empty(K)
    l1, l2 = 0.0, 0.0
    for k in range(K - 1, -1, -1):
        _, du, dx1, dx2 = _acc(u[k], X[k, 0], X[k, 1], theta, use_hat)
        gU[k] = TS * du * l2
        l1, l2 = gX[k, 0] + l1 + TS * dx1 * l2, gX[k, 1] + TS * l1 + (1.0 + TS * dx2) * l2
    return gU


_NO = np.zeros(4 + CENTRES.shape[0])


# ---------------------------------------------------------------- signals
def random_phases(seed):
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, LINES.size)


def schroeder():
    l = np.arange(1, LINES.size + 1)
    return -l * (l - 1) * np.pi / LINES.size


def multisine(phi):
    return AMP * np.sin(_ANG + phi[:, None]).sum(0)


def steady(u, theta=_NO, use_hat=False):
    return simulate(np.tile(u, P), theta, use_hat)[(P - 1) * N:]


# ---------------------------------------------------------------- cost (paper Eqs. 7-8)
def _dens(Z, grid=GRID):
    Wg, Wz = grid / SIGMA, Z / SIGMA
    D2 = (Wg ** 2).sum(1)[:, None] + (Wz ** 2).sum(1)[None, :] - 2 * Wg @ Wz.T
    E = np.exp(-0.5 * np.maximum(D2, 0))
    return E.sum(1), E


def cost(Z, grid=GRID):
    return float(np.mean(1 / (EPS + _dens(Z, grid)[0])))


def kappa(Z):
    return float(np.mean(_dens(Z)[0] >= 1.0))


def objective(phi, theta, use_hat, grid=GRID):
    """space-filling cost and exact phase gradient. grid = the points that must
    be covered: the 20^3 box (paper) or the task points (task-aware design)."""
    u = multisine(phi)
    uf = np.tile(u, P)
    Xf = simulate(uf, theta, use_hat)
    if not np.all(np.isfinite(Xf)) or np.abs(Xf).max() > 1e5:
        return 1e3, np.zeros_like(phi)
    Z = np.column_stack([u, Xf[(P - 1) * N:]])
    d, E = _dens(Z, grid)
    C = np.mean(1 / (EPS + d))
    a = -1 / (len(d) * (EPS + d) ** 2)
    Wg, Wz = grid / SIGMA, Z / SIGMA
    M = E.T @ np.column_stack([a[:, None] * Wg, a])
    gZ = (M[:, :3] - Wz * M[:, 3:4]) / SIGMA
    gX = np.zeros_like(Xf)
    gX[(P - 1) * N:] = gZ[:, 1:]
    gU = backward(uf, Xf, gX, theta, use_hat).reshape(P, N).sum(0) + gZ[:, 0]
    return C, AMP * (np.cos(_ANG + phi[:, None]) @ gU)


def design(phi0, theta, use_hat, iters, grid=GRID):
    r = minimize(lambda p: objective(p, theta, use_hat, grid), phi0, jac=True, method="L-BFGS-B",
                 options={"maxiter": iters, "ftol": 1e-10, "gtol": 1e-6})
    return r.x


# ---------------------------------------------------------------- identification (same as ext/ident.py)
def features(Z):
    Zs = Z / HI
    D2 = ((Zs[:, None, :] - CENTRES[None]) ** 2).sum(-1)
    return np.column_stack([np.ones(len(Z)), Zs, np.exp(-0.5 * D2 / W2)])


def _ridge(F, y, lam):
    R = lam * np.eye(F.shape[1]) * len(y)
    R[0, 0] = 0
    return np.linalg.solve(F.T @ F + R, F.T @ y)


def identify(u, X):
    y = (np.roll(X[:, 1], -1) - X[:, 1]) / TS
    Z = np.column_stack([u, X])
    F = features(Z)
    n = len(y)
    idx = np.array_split(np.arange(n), 5)
    cv = []
    for lam in LAMBDAS:
        e = 0.0
        for te in idx:
            tr = np.setdiff1d(np.arange(n), te)
            e += np.sum((F[te] @ _ridge(F[tr], y[tr], lam) - y[te]) ** 2)
        cv.append(e)
    return _ridge(F, y, LAMBDAS[int(np.argmin(cv))])


# ---------------------------------------------------------------- the control task
@njit(cache=True)
def track(ref, theta, use_hat, kp, kd):
    """Model-based (computed-torque) tracking on the TRUE plant:
       a_des = ref'' + kp e + kd e'  ;  find u with model_acc(u, x) = a_des (grid search + Newton);
       u is saturated at +-5 N m (motor limit). Returns RMS angle error [rad]."""
    K = ref.shape[0]
    x1, x2, u, se = ref[0, 0], ref[0, 1], 0.0, 0.0
    for k in range(K):
        e, ed = ref[k, 0] - x1, ref[k, 1] - x2
        se += e * e
        a_des = ref[k, 2] + kp * e + kd * ed
        best = 1e300                       # coarse search first: a learned model
        for i in range(41):                # need not be monotone in u everywhere
            uc = -5.0 + 0.25 * i
            gc, _, _, _ = _acc(uc, x1, x2, theta, use_hat)
            if abs(gc - a_des) < best:
                best, u = abs(gc - a_des), uc
        for _ in range(4):                 # then Newton refinement
            g, du, _, _ = _acc(u, x1, x2, theta, use_hat)
            if abs(du) < 1e-6:
                break
            u = u - (g - a_des) / du
            u = min(5.0, max(-5.0, u))
        gt, _, _, _ = g_true(u, x1, x2)
        x1, x2 = x1 + TS * x2, x2 + TS * gt
    return np.sqrt(se / K)


def task_points(refs, theta, use_hat, n=3000, seed=0):
    """operating points z = [u, theta, w] the controller will visit: for each
    reference sample, u solves model_acc(u, theta_r, w_r) = alpha_r (same coarse
    search as the controller). With theta = an identified model this needs NO
    knowledge of the true plant."""
    Rr = np.concatenate(refs)
    Rr = Rr[np.random.default_rng(seed).choice(len(Rr), n, replace=False)]
    us = np.linspace(-5, 5, 201)
    Z = np.empty((n, 3))
    for i, (x1, x2, a) in enumerate(Rr):
        g = np.array([_acc(uc, x1, x2, theta, use_hat)[0] for uc in us])
        Z[i] = us[np.argmin(np.abs(g - a))], x1, x2
    return Z


TASKS = {"A": (1.1, 0.2, 1.2),     # normal aiming: up to the joint limit, 0.2-1.2 Hz
         "B": (1.2, 0.5, 1.5)}     # aggressive slews: into the bumpers, 0.5-1.5 Hz


def reference(seed, task="A", T=20.0):
    """smooth random target motion (sum of 4 sines), peak angle amp [rad]."""
    amp, flo, fhi = TASKS[task]
    rng = np.random.default_rng(seed)
    t = np.arange(int(T / TS)) * TS
    f = rng.uniform(flo, fhi, 4)
    a = rng.uniform(0.5, 1.0, 4)
    p = rng.uniform(0, 2 * np.pi, 4)
    s = sum(ai * np.sin(2 * np.pi * fi * t + pi) for ai, fi, pi in zip(a, f, p))
    k = amp / np.abs(s).max()
    w = 2 * np.pi * f
    th = k * s
    om = k * sum(ai * wi * np.cos(wi * t + pi) for ai, wi, pi in zip(a, w, p))
    al = -k * sum(ai * wi ** 2 * np.sin(wi * t + pi) for ai, wi, pi in zip(a, w, p))
    return np.column_stack([th, om, al])
