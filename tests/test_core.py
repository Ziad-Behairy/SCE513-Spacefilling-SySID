"""Fast unit tests (< 1 min): the M3 diagnostic checks D1, D4, D5, D6, D8
plus the extension building blocks.  Run:  pytest -q"""
import numpy as np
import config as cfg
import cost, signals, system


def test_D1_multisine_amplitudes_and_std():
    u = signals.multisine(signals.random_phases(0))
    assert abs(u.std() - cfg.U_STD) < 1e-9 and abs(u.mean()) < 1e-9
    A = signals.amplitude_spectrum(u)[cfg.LINES]
    assert np.max(np.abs(A - cfg.AMP)) < 1e-9


def test_D4_kernel_is_one_at_grid_point():
    d, _ = cost.densities(cost.GRID[:1].copy())
    assert abs(d[0] - 1.0) < 1e-12


def test_D5_cost_upper_bound():
    far = np.full((5, 3), 1e4)
    assert abs(cost.cost(far) - 1.0 / cfg.EPS) < 1e-9


def test_D6_adjoint_gradient_matches_finite_differences():
    phi = signals.random_phases(3)
    _, g = cost.objective(phi)
    h = 1e-4
    for i in (0, 91, 183):
        p1, p2 = phi.copy(), phi.copy()
        p1[i] += h; p2[i] -= h
        fd = (cost.objective(p1)[0] - cost.objective(p2)[0]) / (2 * h)
        assert abs(fd - g[i]) <= 1e-6 * max(1.0, np.max(np.abs(g)))


def test_D8_grid():
    assert cost.GRID.shape == (8000, 3)


def test_schroeder_crest_factor():
    u = signals.multisine(signals.schroeder_phases())
    assert abs(signals.crest_factor(u) - 1.879) < 1e-3


def test_determinism_seed0_initial_cost():
    Z, _, _ = cost.samples_from_phases(signals.random_phases(0))
    assert abs(cost.cost(Z) - 252.5666) < 1e-3
