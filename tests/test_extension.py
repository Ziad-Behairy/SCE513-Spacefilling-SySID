import numpy as np
import config as cfg
import signals, system
from ext import ident, surrogate


def _fit_on(phi, seed_noise=None):
    u = signals.multisine(phi)
    X = system.steady_state_response(u)[0]
    Z, y = ident.regression_data(u, X)
    return ident.fit(Z, y)[0]


def test_regression_target_is_exact_on_noiseless_data():
    u = signals.multisine(signals.random_phases(1))
    X = system.steady_state_response(u)[0]
    Z, y = ident.regression_data(u, X)
    assert np.max(np.abs(y - ident.g_true(Z))) < 1e-6 * np.max(np.abs(y))


def test_surrogate_with_true_like_model_and_gradient():
    theta = _fit_on(signals.random_phases(2))
    phi = signals.random_phases(4)
    C, g = surrogate.objective_hat(phi, theta)
    h = 1e-4
    for i in (5, 120):
        p1, p2 = phi.copy(), phi.copy()
        p1[i] += h; p2[i] -= h
        fd = (surrogate.objective_hat(p1, theta)[0] - surrogate.objective_hat(p2, theta)[0]) / (2 * h)
        assert abs(fd - g[i]) <= 1e-6 * max(1.0, np.max(np.abs(g)))


def test_identified_model_free_run_is_bounded():
    theta = _fit_on(signals.random_phases(2))
    u = signals.multisine(signals.random_phases(9))
    Xt = system.steady_state_response(u)[0]
    Xh = surrogate.sim(theta, np.tile(u, cfg.P_PERIODS))[-cfg.N:]
    assert np.sqrt(np.mean((Xh[:, 0] - Xt[:, 0]) ** 2)) / Xt[:, 0].std() < 1.0   # sanity: finite, bounded, not diverged


def test_peak_penalty_gradient():
    from ext.run_peak_limited import objective
    phi = signals.random_phases(3)
    umax = 1.9 * cfg.U_STD
    J, g = objective(phi, umax)
    h = 1e-5
    for i in (0, 77, 150):
        p1, p2 = phi.copy(), phi.copy()
        p1[i] += h; p2[i] -= h
        fd = (objective(p1, umax)[0] - objective(p2, umax)[0]) / (2 * h)
        assert abs(fd - g[i]) <= 1e-6 * max(1.0, np.max(np.abs(g)))


def test_iv_estimator_recovers_noise_free_model():
    from ext.run_noise_remedies import fit_iv, fit_ls, e_uni
    u = signals.multisine(signals.random_phases(2))
    X = system.steady_state_response(u)[0]
    assert e_uni(fit_iv(u, X)) < 2 * e_uni(fit_ls(u, X)) + 0.02


def test_gimbal_gradients_match_finite_differences():
    """M5 add-on (code/gimbal): adjoint gradient of the space-filling cost through the
    true gimbal plant and through an identified model, box grid and task grid."""
    from gimbal import gimbal as G
    phi = G.random_phases(3)
    theta = G.identify(G.multisine(phi), G.steady(G.multisine(phi)))
    task = G.task_points([G.reference(3000, "A")], theta, True, n=300)
    d = np.random.default_rng(4).standard_normal(phi.size)
    for th, hat, grid in [(G._NO, False, G.GRID), (theta, True, G.GRID), (theta, True, task)]:
        C, g = G.objective(phi, th, hat, grid)
        h = 1e-6
        fd = (G.objective(phi + h * d, th, hat, grid)[0] - G.objective(phi - h * d, th, hat, grid)[0]) / (2 * h)
        assert abs(fd - g @ d) <= 1e-5 * max(1.0, abs(fd))
