"""Análisis de curvas de crecimiento: estimar parámetros a partir de datos."""
import numpy as np
from scipy.optimize import least_squares

from .models.growth import logistic
from .parameters import LogisticParams
from .simulation import simulate


def doubling_time(mu):
    """t_d = ln(2) / mu  [h]."""
    return np.log(2.0) / mu


def estimate_mu_loglinear(t, X):
    """Estima mu con una recta sobre ln(X): ln X = ln X0 + mu * t.

    Solo es válido en la fase EXPONENCIAL: hay que pasarle solo esos datos.
    Devuelve (mu, X0, R2).
    """
    t, lnX = np.asarray(t), np.log(np.asarray(X))
    mu, lnX0 = np.polyfit(t, lnX, 1)
    pred = lnX0 + mu * t
    r2 = 1.0 - np.sum((lnX - pred) ** 2) / np.sum((lnX - lnX.mean()) ** 2)
    return mu, np.exp(lnX0), r2


def fit_logistic(t, X, guess):
    """Ajusta (mu, X0, X_max) minimizando el error en ESCALA LOGARÍTMICA.

    Se usa log porque el ruido de una medición de densidad óptica es más bien
    proporcional (±5 %), no constante: así todos los puntos pesan parecido.
    guess = (mu, X0, X_max). Devuelve (mu, X0, X_max).
    """
    t, X = np.asarray(t), np.asarray(X)

    def residuals(theta):
        mu, X0, X_max = theta
        sim = simulate(logistic, LogisticParams(mu, X_max), [X0], t[-1], 400)
        X_model = np.interp(t, sim.t, sim.y[0])
        return np.log(X_model) - np.log(X)

    lower, upper = [1e-6, 1e-9, 1e-6], [10.0, np.inf, np.inf]
    return tuple(least_squares(residuals, guess, bounds=(lower, upper)).x)
