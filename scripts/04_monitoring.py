"""Fase 4: sensores simulados, EKF, detección de fallos. Parámetros ILUSTRATIVOS.

Se evalúa con Monte Carlo (muchas semillas): una sola simulación no demuestra
nada sobre falsas alarmas ni retrasos de detección.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from biomfg.loop import Scenario, run_closed_loop
from biomfg.sensors import SensorFault

OUT = Path(__file__).resolve().parents[1] / "docs/figures"
N = 30
T0 = 30.0  # instante del fallo/perturbación

scen = {
    "sin fallo":              Scenario(T=60),
    "proceso: mu_max x0.7":   Scenario(T=60, mu_scale=lambda t: 0.7 if t >= T0 else 1.0),
    "sensor X: sesgo +15 %":  Scenario(T=60, fault=SensorFault("bias", T0, 0.15)),
    "sensor X: deriva 3 %/h": Scenario(T=60, fault=SensorFault("drift", T0, 0.03)),
    "sensor X: atascado":     Scenario(T=60, fault=SensorFault("stuck", T0)),
}
# lazo abierto con D fija: aquí solo monitorizamos, no controlamos
print(f"{'escenario':26s} {'tipo alarma':10s} {'detectado':>10s} {'retraso mediano [h]':>20s} {'antes de T0':>12s}")
store = {}
for name, sc in scen.items():
    logs = [run_closed_loop("open_loop", sc, seed) for seed in range(N)]
    store[name] = logs
    for kind in ("sensor_X", "sensor_S", "frozen_X", "process"):
        times = [l["alarms"].get(kind) for l in logs]
        det = [t for t in times if t is not None and t >= T0]
        pre = sum(1 for t in times if t is not None and t < T0)
        med = f"{np.median(np.array(det) - T0):.2f}" if det else "-"
        print(f"{name:26s} {kind:10s} {len(det):>6d}/{N}  {med:>20s} {pre:>8d}/{N}")

# Detección combinada de fallo de sensor (cualquiera de las 3 alarmas de sensor)
print("\nCualquier alarma de sensor (X, S o congelado):")
for name, logs in store.items():
    anyd = [min([t for k, t in l["alarms"].items() if k != "process" and t >= T0] or [np.inf]) for l in logs]
    ok = [a for a in anyd if np.isfinite(a)]
    print(f"  {name:26s} {len(ok):>3d}/{N}  retraso mediano {np.median(np.array(ok) - T0) if ok else float('nan'):.2f} h")

# consistencia del EKF en el caso sin fallo: NIS medio de X debería ser ~1
nis = np.concatenate([l["nisX"][20:] for l in store["sin fallo"]])
print(f"\nNIS medio de X (sin fallo): {nis.mean():.2f} (ideal ≈ 1); "
      f"fracción > 10.83: {np.mean(nis > 10.83):.4f} (ideal ≈ 0.001)")

fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for a, name in zip(ax.ravel()[:5], scen):
    l = store[name][0]
    a.plot(l["t"], l["yX"], ".", c="lightgray", ms=3, label="sensor OD")
    a.plot(l["t"], l["X"], "k", label="X real"); a.plot(l["t"], l["Xh"], "C0", label="X estimado (EKF)")
    a2 = a.twinx(); a2.plot(l["t"], l["th"], "C3", lw=1); a2.set_ylim(0, 1.5); a2.set_ylabel("theta", color="C3")
    a.axvline(T0, c="gray", ls=":"); a.set_title(name); a.set_xlabel("t [h]")
    for kind, t in l["alarms"].items():
        a.axvline(t, c="C1", ls="--"); a.text(t, a.get_ylim()[0], kind, rotation=90, fontsize=7, va="bottom")
ax[0, 0].legend(fontsize=7)
ax.ravel()[5].axis("off")
fig.tight_layout(); fig.savefig(OUT / "fase4_monitorizacion.png", dpi=130)
