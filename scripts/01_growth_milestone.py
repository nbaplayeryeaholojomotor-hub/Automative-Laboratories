"""Milestone 1: simulación, verificación y ajuste de una curva de crecimiento.

Los datos son SINTÉTICOS (generados aquí con parámetros conocidos). Sirven para
comprobar que el método recupera lo que pusimos. NO son evidencia biológica.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from biomfg.analysis import doubling_time, estimate_mu_loglinear, fit_logistic
from biomfg.models.growth import exponential, exponential_analytic, logistic
from biomfg.parameters import ExponentialParams, LogisticParams
from biomfg.simulation import simulate

# --- 1. Verificación: numérico vs analítico -------------------------------
p = ExponentialParams(mu=0.4)
sim = simulate(exponential, p, [0.05], t_end=10.0)
exact = exponential_analytic(sim.t, 0.05, 0.4)
err = np.max(np.abs(sim.y[0] - exact) / exact)
print(f"[1] Error relativo máx. numérico vs analítico: {err:.2e}")
print(f"    t_d = {doubling_time(0.4):.3f} h ; X(10 h) = {sim.y[0, -1]:.3f} g/L")

# --- 2. Datos sintéticos con ruido ----------------------------------------
TRUE = dict(mu=0.6, X0=0.05, X_max=4.0)
rng = np.random.default_rng(42)
t_obs = np.arange(0, 16.1, 1.0)
clean = simulate(logistic, LogisticParams(TRUE["mu"], TRUE["X_max"]),
                 [TRUE["X0"]], 16.0, 400)
X_obs = np.interp(t_obs, clean.t, clean.y[0]) * rng.lognormal(0, 0.05, t_obs.size)

# --- 3. Estimar mu en la fase exponencial (X < 10 % de X_max) -------------
mask = X_obs < 0.1 * TRUE["X_max"]
mu_est, X0_est, r2 = estimate_mu_loglinear(t_obs[mask], X_obs[mask])
print(f"[2] mu (recta en ln X, {mask.sum()} puntos): {mu_est:.3f} "
      f"(real {TRUE['mu']}), R2={r2:.4f}")

# --- 4. Ajuste logístico completo ------------------------------------------
mu_f, X0_f, Xmax_f = fit_logistic(t_obs, X_obs, guess=(0.3, 0.1, 2.0))
print(f"[3] Ajuste logístico: mu={mu_f:.3f}, X0={X0_f:.3f}, X_max={Xmax_f:.2f} "
      f"(real {TRUE['mu']}, {TRUE['X0']}, {TRUE['X_max']})")

# --- 5. Figura --------------------------------------------------------------
fit = simulate(logistic, LogisticParams(mu_f, Xmax_f), [X0_f], 16.0, 400)
free = simulate(exponential, ExponentialParams(mu_est), [X0_est], 16.0, 400)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, log in zip(ax, (False, True)):
    a.plot(t_obs, X_obs, "ko", label="datos sintéticos")
    a.plot(fit.t, fit.y[0], "C0", label="ajuste logístico")
    a.plot(free.t, free.y[0], "C3--", label="exponencial (mu de la recta)")
    a.set_xlabel("t [h]"); a.set_ylabel("X [g/L]")
    if log:
        a.set_yscale("log"); a.set_title("Escala log: exponencial = recta")
    else:
        a.set_ylim(0, 5); a.set_title("Escala lineal"); a.legend()
out = Path(__file__).resolve().parents[1] / "docs/figures/milestone1.png"
fig.tight_layout(); fig.savefig(out, dpi=150)
print(f"[4] Figura guardada en {out}")
