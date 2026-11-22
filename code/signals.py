"""
signals.py  --  multisine input (paper Eq. 12), Schroeder phases (Eq. 13),
                crest factor, and the gradient of u with respect to phases.
"""
import numpy as np
import config as cfg

# k = 0..N-1 and the angle 2*pi*l*k/N for every excited line l.
# (f0/fs = 1/N because f0 = fs/N.)  Shape: (184, 2048).
_K = np.arange(cfg.N)
_ANGLE = 2.0 * np.pi * np.outer(cfg.LINES, _K) / cfg.N


def random_phases(seed):
    """Initial phases: uniform on [0, 2*pi), one per excited line."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, cfg.F_LINES)


def multisine(phi):
    """Eq. (12): u_k = sum_l A*sin(2*pi*l*k/N + phi_l), one period (N samples)."""
    return cfg.AMP * np.sin(_ANGLE + phi[:, None]).sum(axis=0)


def multisine_grad(phi, g_u):
    """
    Chain rule from dJ/du_k (one period, g_u) to dJ/dphi_l:
        du_k/dphi_l = A*cos(2*pi*l*k/N + phi_l)
        dJ/dphi_l   = A * sum_k g_u[k] * cos(2*pi*l*k/N + phi_l)
    """
    return cfg.AMP * (np.cos(_ANGLE + phi[:, None]) @ g_u)


def schroeder_phases(index="standard"):
    """
    Eq. (13): phi_l = -l(l-1)*pi / N_l, N_l = 184 excited lines.
    The paper does not say whether l counts 1..184 (position in the list of
    excited lines -- the classical Schroeder form) or 21..204 (the actual
    frequency line).  M2 decision: primary = 1..184, alternative also reported.
    """
    l = np.arange(1, cfg.F_LINES + 1) if index == "standard" else cfg.LINES.astype(float)
    return -l * (l - 1) * np.pi / cfg.F_LINES


def crest_factor(u):
    """Crest factor = max|u| / rms(u) over one period (M2 definition)."""
    return np.max(np.abs(u)) / np.sqrt(np.mean(u ** 2))


def amplitude_spectrum(u):
    """Single-sided amplitude of each FFT line: |U_l| * 2/N  (equals A on excited lines)."""
    return np.abs(np.fft.rfft(u)) * 2.0 / len(u)
