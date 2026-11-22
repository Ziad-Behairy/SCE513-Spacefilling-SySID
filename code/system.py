"""
system.py  --  the mass-spring-damper system (paper Eqs. 9 and 10)
               plus the hand-written gradient ("adjoint") of the simulation.

State x = [x1, x2] = [position (m), velocity (m/s)];  input u = force F (N).

Continuous time, Eq. (9)  (1/m multiplies ONLY the velocity line, as the
instructor asked -- Eq. (11) in the paper puts 1/m outside the whole vector,
which would give dx1/dt = x2/m and is a typo):
    dx1/dt = x2
    dx2/dt = (1/m) * ( F - s*x1 + s*l*x1/sqrt(x1^2 + a^2) - c*x2 )

Discrete time, forward Euler, Eq. (10):
    x[k+1] = x[k] + Ts * f(x[k], u[k])
"""
import math
import numpy as np
import config as cfg


def f(x1, x2, u):
    """Right-hand side of Eq. (9). Returns (dx1/dt, dx2/dt)."""
    spring = -cfg.S_SPRING * x1 + cfg.S_SPRING * cfg.L_FREE * x1 / math.sqrt(x1 * x1 + cfg.A_ANCHOR ** 2)
    return x2, (u + spring - cfg.C_DAMP * x2) / cfg.M_MASS


def simulate(u, x0=cfg.X0):
    """
    Forward Euler simulation (Eq. 10).

    u  : 1-D array of K input samples (the whole horizon, e.g. 5 periods)
    x0 : initial state [x1, x2]
    returns X : array (K, 2); row k is the state x[k] at time step k
                (X[0] = x0; the state AFTER the last input is not stored,
                 because it has no input partner for z_k).

    We use plain Python floats in the loop because for a 2-state system
    this is faster than NumPy (NumPy has overhead on tiny arrays).
    """
    K = len(u)
    X = np.empty((K, 2))
    x1, x2 = float(x0[0]), float(x0[1])
    uu = u.tolist()
    ts, m, s, l, a2, c = cfg.TS, cfg.M_MASS, cfg.S_SPRING, cfg.L_FREE, cfg.A_ANCHOR ** 2, cfg.C_DAMP
    for k in range(K):
        X[k, 0] = x1
        X[k, 1] = x2
        acc = (uu[k] - s * x1 + s * l * x1 / math.sqrt(x1 * x1 + a2) - c * x2) / m
        x1, x2 = x1 + ts * x2, x2 + ts * acc       # Euler step (both use OLD x)
    return X


def simulate_backward(X, gX):
    """
    Exact gradient of a scalar cost J through the Euler simulation
    (reverse-mode chain rule, also called the "adjoint" method).

    X  : states from simulate(), shape (K, 2)
    gX : dJ/dX[k] coming DIRECTLY from the cost (zero where the cost
         does not look at that sample), shape (K, 2)
    returns gU : dJ/du[k] for every input sample, shape (K,)

    Idea: x[k+1] depends on x[k] and u[k] through
        x1[k+1] = x1[k] + Ts*x2[k]
        x2[k+1] = x2[k] + Ts*f2(x1[k], x2[k], u[k])
    Walking BACKWARDS in time, lam = total dJ/dx[k+1] is pushed to x[k]:
        dJ/dx1[k] = direct + lam1 + Ts*(df2/dx1)*lam2
        dJ/dx2[k] = direct + Ts*lam1 + (1 + Ts*df2/dx2)*lam2
        dJ/du[k]  =          Ts*(df2/du)*lam2 = Ts/m * lam2
    with  df2/dx1 = (-s + s*l*a^2/(x1^2+a^2)^(3/2)) / m,  df2/dx2 = -c/m.
    """
    K = X.shape[0]
    gU = np.empty(K)
    ts, m, s, l, a2, c = cfg.TS, cfg.M_MASS, cfg.S_SPRING, cfg.L_FREE, cfg.A_ANCHOR ** 2, cfg.C_DAMP
    x1s = X[:, 0].tolist()
    g1s, g2s = gX[:, 0].tolist(), gX[:, 1].tolist()
    lam1, lam2 = 0.0, 0.0          # dJ/dx[K] = 0 (state after the horizon is unused)
    d22 = 1.0 - ts * c / m          # constant part of the Jacobian
    for k in range(K - 1, -1, -1):
        x1 = x1s[k]
        df2dx1 = (-s + s * l * a2 / (x1 * x1 + a2) ** 1.5) / m
        gU[k] = ts / m * lam2
        lam1, lam2 = (g1s[k] + lam1 + ts * df2dx1 * lam2,
                      g2s[k] + ts * lam1 + d22 * lam2)
    return gU


def steady_state_response(u_period, P=cfg.P_PERIODS, x0=cfg.X0):
    """
    Apply the periodic input for P periods starting from x0 and return the
    states of the LAST period (shape (N, 2)), plus the full trajectory.
    This is how the "steady-state response" of the paper is approximated.
    """
    u_full = np.tile(u_period, P)
    X_full = simulate(u_full, x0)
    N = len(u_period)
    return X_full[(P - 1) * N:], X_full, u_full


def steady_state_check(u_period, P_max=10, x0=cfg.X0, tol=cfg.SS_TOL):
    """
    M2 steady-state criterion: for p = 2..P_max compare period p with
    period p-1:   rel(p) = max_k |x^(p) - x^(p-1)|_inf / max_k |x^(p)|_inf.
    Steady state is accepted at the first p with rel(p) <= tol.
    Returns (list of rel values for p = 2..P_max, first accepted p or None).
    """
    N = len(u_period)
    X = simulate(np.tile(u_period, P_max), x0)
    rels, p_ok = [], None
    for p in range(2, P_max + 1):
        cur = X[(p - 1) * N:p * N]
        prev = X[(p - 2) * N:(p - 1) * N]
        rel = np.max(np.abs(cur - prev)) / np.max(np.abs(cur))
        rels.append(rel)
        if p_ok is None and rel <= tol:
            p_ok = p
    return rels, p_ok
