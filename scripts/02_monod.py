"""Fase 2: Monod, balance de masa, estimación de parámetros e incertidumbre.

DATOS SINTÉTICOS. Parámetros ILUSTRATIVOS (no de una cepa real).
Pregunta científica: ¿se puede recuperar Ks según cómo diseñemos el experimento?
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from biomfg.analysis import fit_monod
from biomfg.models.monod import monod_batch, monod_mu
from biomfg.parameters import MonodParams
from biomfg.simulation import simulate

TRUE = MonodParams(mu_max=0.5, Ks=0.5, Y=0.5)
SIG_X, SIG_S = 0.03, 0.1
OUT = Path(__file__).resolve().parents[1] / "docs/figures"


def experiment(S0, seed, t_end=16.0):
    rng = np.random.default_rng(seed)
    sim = simulate(monod_batch, TRUE, [0.1, S0], t_end, 800)
    t = np.arange(0, t_end + 0.01, 0.5)
    X = np.interp(t, sim.t, sim.y[0]) * (1 + SIG_X * rng.standard_normal(t.size))
    S = np.maximum(np.interp(t, sim.t, sim.y[1]) + SIG_S * rng.standard_normal(t.size), 0)
    return t, X, S, sim


print("Parámetros reales: mu_max=0.5, Ks=0.5, Y=0.5, X0=0.1")
fits = {}
for S0 in (10.0, 2.0):
    t, X, S, sim = experiment(S0, seed=3)
    r = fit_monod(t, X, S, S0, SIG_X, SIG_S, guess=[0.3, 1.0, 0.4, 0.2])
    fits[S0] = (t, X, S, sim, r)
    print(f"\nExperimento con S0 = {S0} g/L  (chi2 reducido = {r['chi2_red']:.2f})")
    for n, e, se, ci in zip(r["names"], r["est"], r["se"], r["ci95"]):
        print(f"  {n:7s} = {e:7.3f} ± {se:6.3f}   IC95% [{ci[0]:.3f}, {ci[1]:.3f}]")

# Repetición: ¿la incertidumbre calculada es honesta? (cobertura del IC95)
for S0 in (10.0, 2.0):
    cover, rel = 0, []
    for seed in range(40):
        t, X, S, _ = experiment(S0, seed=100 + seed)
        r = fit_monod(t, X, S, S0, SIG_X, SIG_S, guess=[0.3, 1.0, 0.4, 0.2])
        lo, hi = r["ci95"][1]
        cover += lo <= TRUE.Ks <= hi
        rel.append(r["est"][1])
    print(f"\nS0={S0}: Ks estimado en 40 repeticiones: media {np.mean(rel):.2f}, "
          f"sd {np.std(rel):.2f}; el IC95% contiene Ks real en {cover}/40 casos")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
Sg = np.linspace(0, 10, 200)
ax[0].plot(Sg, monod_mu(Sg, 0.5, 0.5)); ax[0].axhline(0.25, ls=":", c="gray")
ax[0].axvline(0.5, ls=":", c="gray"); ax[0].set(xlabel="S [g/L]", ylabel="mu [1/h]",
                                              title="Monod: mu(S); mu(Ks)=mu_max/2")
for a, S0 in zip(ax[1:], (10.0, 2.0)):
    t, X, S, sim, r = fits[S0]
    a.plot(t, X, "ko", ms=3, label="X datos"); a.plot(t, S, "s", c="C1", ms=3, label="S datos")
    a.plot(sim.t, sim.y[0], "k-", lw=.8); a.plot(sim.t, sim.y[1], "C1-", lw=.8)
    a.set(xlabel="t [h]", title=f"Batch, S0={S0} g/L (Ks: {r['est'][1]:.2f}±{r['se'][1]:.2f})")
    a.legend()
fig.tight_layout(); fig.savefig(OUT / "fase2_monod.png", dpi=140)
