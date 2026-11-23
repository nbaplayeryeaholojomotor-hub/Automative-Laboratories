"""Simulador genérico: integra CUALQUIER modelo con la forma f(t, y, params)."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class Result:
    t: np.ndarray  # tiempos [h]
    y: np.ndarray  # estados, forma (n_estados, n_tiempos)


def simulate(model, params, y0, t_end, n_points=200):
    """Resuelve dy/dt = model(t, y, params) desde t=0 hasta t_end [h]."""
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        lambda t, y: model(t, y, params),
        t_span=(0.0, t_end),
        y0=y0,
        t_eval=t_eval,
        rtol=1e-9,   # tolerancias estrictas: el error numérico
        atol=1e-12,  # queda muy por debajo del error biológico
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return Result(t=sol.t, y=sol.y)
