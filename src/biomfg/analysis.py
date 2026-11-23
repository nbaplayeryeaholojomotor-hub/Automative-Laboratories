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


def fit_monod(t, X, S, S0, sigma_X_rel, sigma_S, guess):
    """Ajusta un batch de Monod a datos de biomasa X(t) y sustrato S(t).

    Parámetros estimados: mu_max, Ks, Y, X0 (S0 se supone conocido: es la
    receta del medio). Los residuos se ponderan por el ruido de cada sensor,
    así X y S (unidades distintas) contribuyen de forma comparable.

    Devuelve un dict con estimaciones, errores estándar e IC 95 % aproximados
    (linealización: cov = s^2 (J^T J)^-1). Son una APROXIMACIÓN y pueden ser
    poco fiables si el problema está mal condicionado.
    """
    from .models.monod import monod_batch
    from .parameters import MonodParams

    t, X, S = map(np.asarray, (t, X, S))

    def model(theta):
        mu_max, Ks, Y, X0 = theta
        sim = simulate(monod_batch, MonodParams(mu_max, Ks, Y), [X0, S0], t[-1], 600)
        return np.interp(t, sim.t, sim.y[0]), np.interp(t, sim.t, sim.y[1])

    def residuals(theta):
        Xm, Sm = model(theta)
        rX = (np.log(np.maximum(Xm, 1e-12)) - np.log(X)) / sigma_X_rel
        rS = (Sm - S) / sigma_S
        return np.concatenate([rX, rS])

    names = ["mu_max", "Ks", "Y", "X0"]
    sol = least_squares(residuals, guess, bounds=([1e-3, 1e-4, 0.01, 1e-6], [5, 50, 2, 10]),
                        x_scale=np.abs(guess))
    n, k = sol.fun.size, len(names)
    s2 = np.sum(sol.fun ** 2) / (n - k)
    cov = s2 * np.linalg.pinv(sol.jac.T @ sol.jac)
    se = np.sqrt(np.diag(cov))
    return {"names": names, "est": sol.x, "se": se,
            "ci95": np.c_[sol.x - 1.96 * se, sol.x + 1.96 * se],
            "chi2_red": s2}
