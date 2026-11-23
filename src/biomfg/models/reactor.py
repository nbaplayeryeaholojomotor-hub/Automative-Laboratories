"""Biorreactor con volumen variable: estado y = [X, S, V].

Balances de masa (F_in = caudal de entrada, F_out = caudal de salida, [L/h]):

    dX/dt = mu(S) X - (F_in/V) X
    dS/dt = -mu(S) X / Y + (F_in/V) (S_in - S)
    dV/dt = F_in - F_out

Quimiostato: F_in = F_out = F  ->  V constante y D = F/V (tasa de dilución).
"""
import numpy as np

from .monod import monod_mu


def reactor_rhs(t, y, p, u):
    X, S, V = y
    F_in, F_out = u
    mu = monod_mu(S, p.mu_max, p.Ks)
    return [mu * X - F_in / V * X,
            -mu * X / p.Y + F_in / V * (p.S_in - S),
            F_in - F_out]


def washout_rate(p):
    """D crítica: por encima de ella el cultivo se lava (X -> 0)."""
    return p.mu_max * p.S_in / (p.Ks + p.S_in)


def chemostat_steady_state(p, D):
    """Estado estacionario con X > 0 (solo existe si D < D_crítica)."""
    if D >= washout_rate(p):
        return 0.0, p.S_in  # lavado
    S = p.Ks * D / (p.mu_max - D)
    return p.Y * (p.S_in - S), S


def predict_chemostat(X0, S0, D_profile, p, Ts, h_max=0.05):
    """Predicción RÁPIDA y VECTORIZADA (RK4) para el MPC.

    D_profile: array (n_candidatos, N) con la dilución en cada intervalo Ts.
    Simula todos los candidatos a la vez. Devuelve X, S con forma (n, N+1).
    """
    D_profile = np.atleast_2d(D_profile)
    n, N = D_profile.shape
    n_sub = int(np.ceil(Ts / h_max))
    h = Ts / n_sub
    X = np.full(n, float(X0))
    S = np.full(n, float(S0))
    Xs, Ss = [X.copy()], [S.copy()]

    def f(X, S, D):
        mu = monod_mu(S, p.mu_max, p.Ks)
        return (mu - D) * X, -mu * X / p.Y + D * (p.S_in - S)

    for k in range(N):
        D = D_profile[:, k]
        for _ in range(n_sub):
            k1 = f(X, S, D)
            k2 = f(X + h / 2 * k1[0], S + h / 2 * k1[1], D)
            k3 = f(X + h / 2 * k2[0], S + h / 2 * k2[1], D)
            k4 = f(X + h * k3[0], S + h * k3[1], D)
            X = np.maximum(X + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), 0)
            S = np.maximum(S + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), 0)
        Xs.append(X.copy()); Ss.append(S.copy())
    return np.array(Xs).T, np.array(Ss).T
