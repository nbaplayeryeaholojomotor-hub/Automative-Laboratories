import numpy as np

from biomfg.analysis import fit_monod
from biomfg.models.monod import monod_batch, monod_mu
from biomfg.models.reactor import (chemostat_steady_state, predict_chemostat,
                                   washout_rate)
from biomfg.parameters import MonodParams, ReactorParams
from biomfg.plant import Plant
from biomfg.simulation import simulate

P = ReactorParams(mu_max=0.5, Ks=0.5, Y=0.5, S_in=10.0)


def test_monod_half_saturation():
    assert np.isclose(monod_mu(0.5, 0.5, 0.5), 0.25)  # mu(Ks) = mu_max/2


def test_batch_mass_balance_conserved():
    s = simulate(monod_batch, MonodParams(0.5, 0.5, 0.5), [0.1, 10.0], 24.0)
    X, S = s.y
    assert np.allclose(X + 0.5 * S, 0.1 + 0.5 * 10.0, rtol=1e-6)
    assert S.min() >= -1e-8


def test_chemostat_reaches_analytic_steady_state():
    D = 0.3
    Xs, Ss = chemostat_steady_state(P, D)
    plant = Plant(P, [1.0, 5.0, 2.0])
    for _ in range(400):
        plant.step_chemostat(D, 0.5)
    assert np.allclose(plant.y[:2], [Xs, Ss], rtol=1e-3)
    assert np.isclose(plant.y[2], 2.0)  # volumen constante


def test_washout_above_critical_dilution():
    plant = Plant(P, [4.0, 1.0, 2.0])
    for _ in range(400):
        plant.step_chemostat(washout_rate(P) * 1.1, 0.5)
    assert plant.y[0] < 1e-3


def test_fed_batch_volume_balance():
    plant = Plant(P, [1.0, 5.0, 2.0])
    plant.step(F_in=0.5, F_out=0.0, dt=4.0)
    assert np.isclose(plant.y[2], 4.0)


def test_fast_predictor_matches_plant():
    D = 0.35
    Xp, Sp = predict_chemostat(3.0, 2.0, np.full((1, 8), D), P, Ts=0.5)
    plant = Plant(P, [3.0, 2.0, 1.0])
    for _ in range(8):
        plant.step_chemostat(D, 0.5)
    assert np.allclose([Xp[0, -1], Sp[0, -1]], plant.y[:2], rtol=1e-3)


def test_fit_monod_recovers_parameters():
    rng = np.random.default_rng(1)
    true = MonodParams(0.5, 0.5, 0.5)
    sim = simulate(monod_batch, true, [0.1, 10.0], 16.0, 800)
    t = np.arange(0, 16.1, 0.5)
    X = np.interp(t, sim.t, sim.y[0]) * (1 + 0.03 * rng.standard_normal(t.size))
    S = np.interp(t, sim.t, sim.y[1]) + 0.1 * rng.standard_normal(t.size)
    r = fit_monod(t, X, np.maximum(S, 0), 10.0, 0.03, 0.1, guess=[0.3, 1.0, 0.4, 0.2])
    mu_max, Ks, Y, X0 = r["est"]
    assert abs(mu_max - 0.5) / 0.5 < 0.1 and abs(Y - 0.5) / 0.5 < 0.1
    assert r["se"][1] > 0
