import numpy as np

from biomfg.analysis import doubling_time, estimate_mu_loglinear, fit_logistic
from biomfg.models.growth import (exponential, exponential_analytic, logistic,
                                  logistic_analytic)
from biomfg.parameters import ExponentialParams, LogisticParams
from biomfg.simulation import simulate


def test_exponential_matches_analytic():
    s = simulate(exponential, ExponentialParams(0.4), [0.05], 10.0)
    assert np.allclose(s.y[0], exponential_analytic(s.t, 0.05, 0.4), rtol=1e-6)


def test_logistic_matches_analytic():
    s = simulate(logistic, LogisticParams(0.6, 4.0), [0.05], 16.0)
    assert np.allclose(s.y[0], logistic_analytic(s.t, 0.05, 0.6, 4.0), rtol=1e-6)


def test_logistic_never_exceeds_Xmax():
    s = simulate(logistic, LogisticParams(0.6, 4.0), [0.05], 50.0)
    assert s.y[0].max() <= 4.0 + 1e-6


def test_doubling_time():
    assert np.isclose(doubling_time(np.log(2)), 1.0)


def test_mu_estimate_noiseless_is_exact():
    t = np.linspace(0, 5, 20)
    mu, X0, r2 = estimate_mu_loglinear(t, 0.1 * np.exp(0.5 * t))
    assert np.isclose(mu, 0.5) and np.isclose(X0, 0.1) and r2 > 0.999999


def test_logistic_fit_recovers_parameters_with_noise():
    rng = np.random.default_rng(0)
    t = np.arange(0, 16.1, 1.0)
    X = logistic_analytic(t, 0.05, 0.6, 4.0) * rng.lognormal(0, 0.05, t.size)
    mu, X0, Xmax = fit_logistic(t, X, guess=(0.3, 0.1, 2.0))
    assert abs(mu - 0.6) / 0.6 < 0.1
    assert abs(Xmax - 4.0) / 4.0 < 0.1
