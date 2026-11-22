"""
config.py  --  ALL settings of the reproduction in one place.

Why one file?  The M2 protocol says "record every choice".  If every number
lives here, nothing is hidden inside other files, and changing a setting
(for example epsilon) needs a change in exactly one line.

Paper: M. Kiss, R. Toth, M. Schoukens, "Space-filling input design for
nonlinear state-space identification", IFAC-PapersOnLine 58(15), 2024.
"""
import numpy as np

# ---------------------------------------------------------------------------
# 1) Physical system (paper Sec. 4.1, Eq. 9)
# ---------------------------------------------------------------------------
M_MASS = 5.0      # m  [kg]   mass of the cart
S_SPRING = 800.0  # s  [N/m]  spring stiffness
A_ANCHOR = 0.25   # a  [m]    stretched length (height of the spring anchor)
L_FREE = 0.17     # l  [m]    tensionless (natural) length of the spring
C_DAMP = 10.0     # c  [Ns/m] linear damper

FS = 100.0        # sampling frequency [Hz]
TS = 1.0 / FS     # sampling time T_s = 0.01 s (forward Euler step, Eq. 10)

# ---------------------------------------------------------------------------
# 2) Input signal: periodic multisine (paper Sec. 4.2, Eq. 12)
# ---------------------------------------------------------------------------
N = 2048                    # samples per period
L_MIN, L_MAX = 21, 204      # first and last excited frequency line
LINES = np.arange(L_MIN, L_MAX + 1)   # the 184 excited lines l = 21..204
F_LINES = LINES.size                  # F = 184
U_STD = 160.0               # required standard deviation of u [N]
# Equal amplitudes: a sine of amplitude A has power A^2/2, so
# F * A^2 / 2 = U_STD^2  ->  A = U_STD * sqrt(2 / F)  (about 16.68 N).
AMP = U_STD * np.sqrt(2.0 / F_LINES)

# ---------------------------------------------------------------------------
# 3) Steady state (M2 protocol: not specified in the paper)
# ---------------------------------------------------------------------------
# M2 baseline was P = 5.  The first seed-0 optimum violated the criterion at
# p = 5 (met only at p = 6: the optimized data reach |x1| ~ 2, where the
# Euler transient decays slowly).  As the M2 protocol prescribes ("raise P
# if violated"), P was raised to 8 for ALL runs.
P_PERIODS = 8        # simulate 8 periods from rest, keep only the last one
SS_TOL = 1e-6        # accept steady state when period-to-period change
                     # <= SS_TOL * (largest state value)
X0 = np.array([0.0, 0.0])   # start from rest: position 0, velocity 0

# ---------------------------------------------------------------------------
# 4) Domain, grid and cost (paper Sec. 4.3, Eqs. 5-8)
# ---------------------------------------------------------------------------
# order of the three axes everywhere in the code: z = [u, x1, x2]
DOMAIN_LO = np.array([-400.0, -2.0, -20.0])
DOMAIN_HI = np.array([400.0, 2.0, 20.0])
GRID_PTS = 20                         # 20 points per axis -> 20^3 = 8000
SIGMA = np.array([40.0, 0.2, 2.0])    # kernel widths (Sigma = diag(SIGMA^2))
EPS = 1e-3            # epsilon in Eq. 8 (NOT given in the paper; M2 choice)

# ---------------------------------------------------------------------------
# 5) Optimizer (paper used MATLAB fmincon; M2 choice: SciPy L-BFGS-B)
# ---------------------------------------------------------------------------
OPT_MAXITER = 1000
OPT_FTOL = 1e-10
OPT_GTOL = 1e-6
# Budget-matched comparison with the paper's optimizer (hypothesis H1):
# MATLAB fmincon defaults (interior-point): MaxFunctionEvaluations = 3000 and,
# without a user gradient, forward finite differences = 184 + 1 evaluations per
# gradient  ->  3000 / 185 = 16 gradient evaluations.
FMINCON_GRAD_BUDGET = 3000 // (184 + 1)    # = 16

# Adam cross-check on seed 0 (M2 plan)
ADAM_LR = 0.05
ADAM_STEPS = 400

# ---------------------------------------------------------------------------
# 6) Random seeds (M2 protocol)
# ---------------------------------------------------------------------------
SEED_MAIN = 0                      # targets T1-T6
SEEDS_MC = list(range(1, 21))      # target T7 (20 Monte Carlo runs)

# ---------------------------------------------------------------------------
# 7) Values published in the paper (used only for the comparison table)
# ---------------------------------------------------------------------------
PAPER = {
    "C_init": 287.15,
    "C_opt": 3.88,
    "ratio_text": 69.0,             # what the text says
    "ratio_implied": 287.15 / 3.88,  # what the two numbers imply (74.0)
    "CF_schroeder": 1.87,
    "CF_opt": 3.67,
}

RESULTS_DIR = "results"
