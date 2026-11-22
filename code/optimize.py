"""
optimize.py  --  phase optimization.

Main method: SciPy L-BFGS-B (M2 choice; the paper used MATLAB fmincon, which
is not available here).  L-BFGS-B is a quasi-Newton method: it uses the exact
gradient from cost.objective and builds an approximation of the curvature
from the last few gradients.  Phases are periodic (phi and phi + 2*pi give
the same signal), so no bounds are needed.

Cross-check: plain Adam (a first-order method with a fixed learning rate),
to see whether a different optimizer reaches a similar cost.
"""
import time
import numpy as np
from scipy.optimize import minimize
import config as cfg
import cost


def optimize_phases(phi0, maxiter=cfg.OPT_MAXITER, eps=cfg.EPS, verbose=True):
    """
    Minimize the space-filling cost starting from phases phi0 with L-BFGS-B.
    Returns a dict with the final phases and costs, the cost after every
    iteration (history), the cost of every single evaluation (eval_costs;
    each evaluation = one cost + one exact gradient), counts and run time.
    """
    history = []
    eval_costs = []

    def fun(phi):
        C, g = cost.objective(phi, eps=eps)     # (value, gradient), both float64
        eval_costs.append(C)
        return C, g

    def callback(intermediate_result):          # called once per iteration
        history.append(intermediate_result.fun)
        if verbose and len(history) % 50 == 0:
            print(f"  iter {len(history):4d}  C = {history[-1]:.4f}", flush=True)

    t0 = time.time()
    res = minimize(fun, phi0, jac=True, method="L-BFGS-B", callback=callback,
                   options={"maxiter": maxiter, "ftol": cfg.OPT_FTOL,
                            "gtol": cfg.OPT_GTOL, "maxcor": 10})
    history = [eval_costs[0]] + history          # history[k] = cost after k iterations
    return {
        "phi0": phi0, "phi": res.x,
        "C0": eval_costs[0], "C": res.fun,
        "history": np.array(history), "eval_costs": np.array(eval_costs),
        "nit": res.nit, "nfev": len(eval_costs),
        "message": str(res.message), "success": bool(res.success),
        "time_s": time.time() - t0,
    }


def best_within_budget(eval_costs, n_grad=cfg.FMINCON_GRAD_BUDGET):
    """Lowest cost found within the first n_grad gradient evaluations
    (used for the budget-matched comparison with fmincon defaults, H1)."""
    return float(np.min(eval_costs[:n_grad]))


def adam(phi0, lr=cfg.ADAM_LR, steps=cfg.ADAM_STEPS, eps=cfg.EPS):
    """
    Adam: keeps running averages of the gradient (m) and of its square (v)
    and moves each phase by lr * m / sqrt(v).  Standard constants
    beta1 = 0.9, beta2 = 0.999.  Returns final phases and the cost history.
    """
    b1, b2, tiny = 0.9, 0.999, 1e-8
    phi = phi0.copy()
    m = np.zeros_like(phi)
    v = np.zeros_like(phi)
    hist = []
    for t in range(1, steps + 1):
        C, g = cost.objective(phi, eps=eps)
        hist.append(C)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        phi = phi - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + tiny)
    hist.append(cost.objective(phi, eps=eps)[0])
    return phi, np.array(hist)
