"""Fase 3: biorreactor digital. Estados estacionarios, lavado (washout),
perturbaciones y variabilidad biológica. Parámetros ILUSTRATIVOS."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace

from biomfg.models.reactor import chemostat_steady_state, washout_rate
from biomfg.parameters import ReactorParams
from biomfg.plant import Plant

P = ReactorParams(mu_max=0.5, Ks=0.5, Y=0.5, S_in=10.0)
OUT = Path(__file__).resolve().parents[1] / "docs/figures"
Dc = washout_rate(P)
print(f"D crítica (lavado) = {Dc:.4f} 1/h")

# 1) Barrido de D: simulación vs fórmula analítica
Ds = np.linspace(0.05, 0.6, 12)
sim_X = []
for D in Ds:
    pl = Plant(P, [4.0, 1.0, 1.0])
    for _ in range(600):
        pl.step_chemostat(D, 0.5)
    sim_X.append(pl.y[0])
Dg = np.linspace(0.01, 0.6, 300)
Xa = np.array([chemostat_steady_state(P, d)[0] for d in Dg])
err = max(abs(s - chemostat_steady_state(P, d)[0]) for s, d in zip(sim_X, Ds))
print(f"Error máx. |X_sim - X_analítico| en el barrido: {err:.2e} g/L")
best = Dg[np.argmax(Dg * Xa)]
print(f"Productividad D*X máxima en D = {best:.3f} (D crit = {Dc:.3f}): "
      f"¡el óptimo está pegado al precipicio!")

# 2) Perturbación: mu_max cae un 25 % (t=20 h) con D fija (lazo ABIERTO)
def run(D, scale, T=60):
    pl = Plant(P, [*chemostat_steady_state(P, 0.3)[:1], chemostat_steady_state(P, 0.3)[1], 1.0],
               mu_scale=lambda t: scale if t >= 20 else 1.0)
    ts, Xs = [], []
    for k in range(int(T / 0.25)):
        ts.append(pl.t); Xs.append(pl.y[0]); pl.step_chemostat(D, 0.25)
    return np.array(ts), np.array(Xs)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(Dg, Xa, label="X* analítico"); ax[0].plot(Dg, Dg * Xa, label="productividad D·X*")
ax[0].plot(Ds, sim_X, "ko", ms=4, label="simulación"); ax[0].axvline(Dc, c="r", ls="--")
ax[0].set(xlabel="D [1/h]", ylabel="g/L  |  g/L/h", title="Quimiostato en estado estacionario"); ax[0].legend()
for D, sc, lab in [(0.30, 0.75, "D=0.30, mu_max x0.75 (D crít=0.357)"),
                   (0.35, 0.75, "D=0.35, mu_max x0.75 (casi lavado)"),
                   (0.35, 0.70, "D=0.35, mu_max x0.70 (D crít=0.333: lavado LENTO, -0.017 1/h)")]:
    t, X = run(D, sc); ax[1].plot(t, X, label=lab)
    print(f"{lab}: X(60h) = {X[-1]:.3f} g/L")
ax[1].axvline(20, c="gray", ls=":"); ax[1].set(xlabel="t [h]", ylabel="X [g/L]",
        title="Perturbación a las 20 h, lazo abierto"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fase3_reactor.png", dpi=140)

# 3) Variabilidad biológica: Monte Carlo sobre parámetros (D=0.35 fija)
rng = np.random.default_rng(0)
Xend, washed = [], 0
for _ in range(200):
    q = replace(P, mu_max=P.mu_max * rng.lognormal(0, 0.08), Ks=P.Ks * rng.lognormal(0, 0.15),
                Y=P.Y * rng.lognormal(0, 0.05))
    pl = Plant(q, [4.0, 1.0, 1.0])
    for _ in range(300):
        pl.step_chemostat(0.35, 0.5)
    Xend.append(pl.y[0]); washed += pl.y[0] < 0.1
Xend = np.array(Xend)
print(f"Variabilidad (200 reactores, D=0.35): X* media {Xend.mean():.2f}, sd {Xend.std():.2f}, "
      f"min {Xend.min():.2f}; lavados: {washed}/200")
