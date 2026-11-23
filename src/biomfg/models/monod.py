"""Cinética de Monod: la tasa de crecimiento depende del sustrato.

    mu(S) = mu_max * S / (Ks + S)

    dX/dt =  mu(S) * X                  (la biomasa crece)
    dS/dt = -mu(S) * X / Y              (el sustrato se consume)

Balance de masa: X + Y*S es constante en un cultivo batch (lo que se pierde
como sustrato aparece como biomasa, con rendimiento Y).
"""
import numpy as np


def monod_mu(S, mu_max, Ks):
    S = np.maximum(S, 0.0)  # el sustrato no puede ser negativo
    return mu_max * S / (Ks + S)


def monod_batch(t, y, p):
    X, S = y
    mu = monod_mu(S, p.mu_max, p.Ks)
    return [mu * X, -mu * X / p.Y]
