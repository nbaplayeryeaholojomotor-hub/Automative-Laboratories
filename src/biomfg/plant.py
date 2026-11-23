"""La "planta": el biorreactor VIRTUAL que hace de realidad en las simulaciones.

Regla de oro: los controladores y estimadores NUNCA leen el estado interno de
la planta. Solo ven lo que entregan los sensores.
"""
from dataclasses import replace

import numpy as np
from scipy.integrate import solve_ivp

from .models.reactor import reactor_rhs


class Plant:
    def __init__(self, params, y0, mu_scale=lambda t: 1.0, S_in_scale=lambda t: 1.0):
        self.p = params                 # parámetros VERDADEROS (ocultos al controlador)
        self.y = np.array(y0, float)    # [X, S, V]
        self.t = 0.0
        self.mu_scale = mu_scale        # perturbación: factor sobre mu_max(t)
        self.S_in_scale = S_in_scale    # perturbación: factor sobre S_in(t)

    def step(self, F_in, F_out, dt):
        p = replace(self.p,
                    mu_max=self.p.mu_max * self.mu_scale(self.t),
                    S_in=self.p.S_in * self.S_in_scale(self.t))
        sol = solve_ivp(lambda t, y: reactor_rhs(t, y, p, (F_in, F_out)),
                        (0.0, dt), self.y, method="LSODA", rtol=1e-8, atol=1e-10)
        self.y = np.maximum(sol.y[:, -1], 0.0)
        self.t += dt
        return self.y

    def step_chemostat(self, D, dt):
        """Quimiostato: caudales iguales, V constante."""
        F = D * self.y[2]
        return self.step(F, F, dt)
