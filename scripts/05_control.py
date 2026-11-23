"""Fase 5: control autónomo. Compara lazo abierto, PID (sensor crudo / EKF),
MPC con modelo nominal y MPC adaptativo (theta del EKF). Parámetros ILUSTRATIVOS.

Escenarios (todos con variabilidad biológica entre semillas):
  A  seguimiento: consigna 4.4 -> 4.0 g/L a las 25 h
  B  A + caída de mu_max x0.75 a las 40 h (perturbación de proceso)
  C  A + sesgo +15 % en el sensor de biomasa a las 40 h (fallo de sensor)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from biomfg.loop import Scenario, metrics, run_closed_loop
from biomfg.sensors import SensorFault

OUT = Path(__file__).resolve().parents[1] / "docs/figures"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
CTRL = ["open_loop", "pid_raw", "pid_ekf", "mpc", "mpc_adaptive"]
SC = {
    "A seguimiento": Scenario(T=60),
    "B mu_max x0.75 @40h": Scenario(T=60, mu_scale=lambda t: 0.75 if t >= 40 else 1.0),
    "C sesgo sensor +15% @40h": Scenario(T=60, fault=SensorFault("bias", 40.0, 0.15)),
}
traces = {}
for sname, sc in SC.items():
    print(f"\n== {sname} ({N} semillas) ==")
    print(f"{'controlador':14s} {'IAE [g·h/L]':>14s} {'error final [g/L]':>19s} {'TV(D)':>12s} {'lavados':>8s}")
    for c in CTRL:
        ms, lg0 = [], None
        for seed in range(N):
            lg = run_closed_loop(c, sc, seed)
            ms.append(metrics(lg)); lg0 = lg0 or lg
        traces[(sname, c)] = lg0
        f = lambda k: (np.mean([m[k] for m in ms]), np.std([m[k] for m in ms]))
        print(f"{c:14s} {f('IAE')[0]:8.1f} ± {f('IAE')[1]:4.1f} {f('X_final_err')[0]:11.3f} ± {f('X_final_err')[1]:5.3f} "
              f"{f('TV_D')[0]:6.2f} ± {f('TV_D')[1]:4.2f} {sum(m['washout'] for m in ms):>5d}/{N}")

fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
for j, sname in enumerate(SC):
    for c in CTRL:
        lg = traces[(sname, c)]
        ax[0, j].plot(lg["t"], lg["X"], label=c, lw=1.2); ax[1, j].plot(lg["t"], lg["D"], lw=1)
    ax[0, j].plot(lg["t"], lg["sp"], "k--", label="consigna"); ax[0, j].set_title(sname)
    ax[0, j].set_ylabel("X real [g/L]"); ax[1, j].set_ylabel("D [1/h]"); ax[1, j].set_xlabel("t [h]")
ax[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fase5_control.png", dpi=130)
