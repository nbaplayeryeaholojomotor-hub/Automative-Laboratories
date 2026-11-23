"""Controladores: PID discreto y MPC (control predictivo basado en modelo)."""
import numpy as np
from scipy.optimize import minimize

from .models.reactor import predict_chemostat
from dataclasses import replace


class PID:
    """PID discreto, ACCIÓN INVERSA: si X está por encima de la consigna hay que
    SUBIR D (diluir más) -> e = y - sp. Con anti-windup (no integra saturado)."""

    def __init__(self, Kp, Ki, Kd, Ts, u_min, u_max, u0, tau_d=1.0):
        self.Kp, self.Ki, self.Kd, self.Ts = Kp, Ki, Kd, Ts
        self.u_min, self.u_max, self.u0, self.tau_d = u_min, u_max, u0, tau_d
        self.I, self._y_prev, self._d = 0.0, None, 0.0

    def __call__(self, y, sp):
        e = y - sp
        if self._y_prev is not None and self.Kd > 0:  # derivada sobre la medida, filtrada
            a = self.tau_d / (self.tau_d + self.Ts)
            self._d = a * self._d + (1 - a) * (y - self._y_prev) / self.Ts
        self._y_prev = y
        u_unsat = self.u0 + self.Kp * e + self.I + self.Kd * self._d
        u = min(max(u_unsat, self.u_min), self.u_max)
        if u == u_unsat or (u_unsat > self.u_max and e < 0) or (u_unsat < self.u_min and e > 0):
            self.I += self.Ki * e * self.Ts
        return u


class MPC:
    """Horizonte Np pasos; Nc movimientos con bloques de longitud (1, 2, resto).
    Minimiza  sum (X_k - sp)^2 + r * sum (dD)^2  con D en [u_min, u_max].
    Usa el modelo de Monod nominal con mu_max escalado por theta (adaptación)."""

    def __init__(self, p_nominal, Ts, Np=8, r=20.0, u_min=0.0, u_max=0.55):
        self.p, self.Ts, self.Np, self.r = p_nominal, Ts, Np, r
        self.u_min, self.u_max = u_min, u_max
        self.blocks = [1, 2, Np - 3]
        self.U = None

    def _profiles(self, U):  # (n_cand, 3) -> (n_cand, Np)
        return np.repeat(U, self.blocks, axis=1)

    def __call__(self, X, S, theta, sp, D_prev):
        p = replace(self.p, mu_max=self.p.mu_max * theta)
        if self.U is None:
            self.U = np.full(len(self.blocks), D_prev)

        def cost_batch(Ub):
            Dp = self._profiles(Ub)
            Xp, _ = predict_chemostat(X, S, Dp, p, self.Ts)
            dD = np.diff(np.concatenate([np.full((len(Ub), 1), D_prev), Dp], axis=1), axis=1)
            return ((Xp[:, 1:] - sp) ** 2).sum(axis=1) + self.r * (dD ** 2).sum(axis=1)

        def fun(u):
            eps = 1e-4
            batch = np.vstack([u, u + eps * np.eye(len(u))])
            c = cost_batch(batch)
            return c[0], (c[1:] - c[0]) / eps

        res = minimize(fun, self.U, jac=True, method="L-BFGS-B",
                       bounds=[(self.u_min, self.u_max)] * len(self.blocks),
                       options={"maxiter": 25})
        self.U = res.x
        return float(res.x[0])
