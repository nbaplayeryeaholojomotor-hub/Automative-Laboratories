"""Modelos de crecimiento microbiano.

CONVENIO (clave para la arquitectura): un modelo es una función

    f(t, y, params) -> dy/dt

donde `y` es el VECTOR DE ESTADO. Hoy y = [X]. Cuando añadamos sustrato será
y = [X, S], luego [X, S, V, O2, ...]. El simulador no cambia: solo cambian
los modelos.
"""
import numpy as np


def exponential(t, y, p):
    """dX/dt = mu * X   (crecimiento sin límites)."""
    X = y[0]
    return [p.mu * X]


def logistic(t, y, p):
    """dX/dt = mu * X * (1 - X/X_max)   (crecimiento con techo).

    Es fenomenológico: no dice POR QUÉ hay un techo (falta de sustrato,
    toxinas...). Eso lo resolverá Monod en el Milestone 2.
    """
    X = y[0]
    return [p.mu * X * (1.0 - X / p.X_max)]


# Soluciones analíticas: sirven para VERIFICAR el simulador numérico.
def exponential_analytic(t, X0, mu):
    return X0 * np.exp(mu * np.asarray(t))


def logistic_analytic(t, X0, mu, X_max):
    t = np.asarray(t)
    return X_max / (1.0 + ((X_max - X0) / X0) * np.exp(-mu * t))
