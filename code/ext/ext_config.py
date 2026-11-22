"""
ext_config.py  --  every setting of the EXTENSION in one place.

Extension question (bonus):
  Q1  Does a space-filling input give a more accurate identified nonlinear
      state-space model than random-phase or Schroeder inputs with the SAME
      amplitude spectrum (same energy, same frequency lines, same length)?
  Q2  Does the benefit survive when the design model is not known but is
      itself identified from a first random-phase experiment
      ("surrogate-in-the-loop" design, the realistic situation)?
"""
import numpy as np

# measurement noise on the two states, as SNR in dB per channel (inf = noiseless)
SNR_DB = [np.inf, 40.0, 30.0]
NOISE_SEED_OFFSET = 5000          # noise seed = seed + offset (+ 100*level index)

# model class: x1[k+1] = x1[k] + Ts*x2[k]          (known kinematics)
#              x2[k+1] = x2[k] + Ts*g(u, x1, x2)   (unknown force law, black box)
# g = theta^T phi(z),  phi = [1, z_s, RBF_1..RBF_M],  z_s = z / HALF_RANGE
HALF_RANGE = np.array([400.0, 2.0, 20.0])
RBF_PTS = 7                       # 7^3 = 343 Gaussian centres on [-1, 1]^3
RBF_WIDTH = 2.0 / (RBF_PTS - 1)   # width = centre spacing (scaled units)
LAMBDAS = np.logspace(-8, 1, 19)  # ridge values tried by cross-validation
CV_FOLDS = 5                      # contiguous blocks (time series)

# evaluation
N_UNIFORM = 20000                 # uniform test points in D_z (one-step accuracy)
UNIFORM_SEED = 777
TEST_SEEDS = [1001, 1002, 1003, 1004, 1005]   # free-run test multisines
TEST_STD = [160.0, 240.0]          # design level and 1.5x (extrapolation stress)
DIVERGE = 1e3                      # |x| above this = simulation diverged

# surrogate-in-the-loop design (Q2)
SUR_ITERS = 20                     # same budget as the paper-like O20 design
