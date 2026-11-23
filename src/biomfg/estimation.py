"""Filtro de Kalman extendido (EKF) para un quimiostato.

Estado estimado: z = [X, S, theta]
  X, S  : biomasa y sustrato (lo que NO medimos bien o no medimos)
  theta : factor multiplicativo sobre mu_max (modela "la biología hoy rinde
          theta veces lo esperado"). Evoluciona como paseo aleatorio.
Entradas conocidas: D (dilución) y V (constante). Medidas: X (cada Ts) y
S (cada cierto número de muestras).
"""
import numpy as np


class EKF:
    def __init__(self, p, z0, P0, q_diag, Ts, h_max=0.05):
        self.p, self.Ts = p, Ts
        self.z = np.array(z0, float)
        self.P = np.diag(np.array(P0, float))
        self.Q = np.diag(np.array(q_diag, float)) * Ts  # ruido de proceso por intervalo
        self.n_sub = int(np.ceil(Ts / h_max))

    # --- modelo de proceso discreto (RK4) ---
    def _f(self, z, D):
        p = self.p
        h = self.Ts / self.n_sub
        X, S, th = z

        def rhs(X, S):
            Sp = max(S, 0.0)
            mu = th * p.mu_max * Sp / (p.Ks + Sp)
            return np.array([(mu - D) * X, -mu * X / p.Y + D * (p.S_in - S)])

        x = np.array([X, S])
        for _ in range(self.n_sub):
            k1 = rhs(*x); k2 = rhs(*(x + h / 2 * k1))
            k3 = rhs(*(x + h / 2 * k2)); k4 = rhs(*(x + h * k3))
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return np.array([x[0], x[1], th])

    def predict(self, D):
        z = self.z
        F = np.zeros((3, 3))
        for j in range(3):  # Jacobiano numérico (diferencias finitas)
            eps = 1e-6 * max(abs(z[j]), 1e-3)
            dz = z.copy(); dz[j] += eps
            F[:, j] = (self._f(dz, D) - self._f(z, D)) / eps
        self.z = self._f(z, D)
        self.P = F @ self.P @ F.T + self.Q
        self._clip()

    def _update(self, y, H, R):
        innov = y - H @ self.z
        Sinn = H @ self.P @ H + R
        K = self.P @ H / Sinn
        self.z = self.z + K * innov
        I_KH = np.eye(3) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + R * np.outer(K, K)  # forma de Joseph
        self._clip()
        return innov, innov ** 2 / Sinn  # innovación y NIS (chi² con 1 g.l. si todo cuadra)

    def update_X(self, y, rel_noise):
        R = (rel_noise * max(y, 1e-3)) ** 2
        return self._update(y, np.array([1.0, 0, 0]), R)

    def update_S(self, y, abs_noise):
        return self._update(y, np.array([0, 1.0, 0]), abs_noise ** 2)

    def _clip(self):
        self.z[0] = max(self.z[0], 1e-6)
        self.z[1] = max(self.z[1], 0.0)
        self.z[2] = min(max(self.z[2], 0.05), 3.0)

    @property
    def theta_sd(self):
        return float(np.sqrt(self.P[2, 2]))
