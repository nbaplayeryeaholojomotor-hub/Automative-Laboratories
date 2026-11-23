"""Ajuste del PID por barrido, con semillas de ENTRENAMIENTO (100-103) distintas
de las de evaluación (0-5), para no sobreajustar. Criterio: IAE medio sobre los
escenarios A y B con penalización suave de la vibración de la bomba."""
import itertools
import numpy as np
import biomfg.loop as L

SC = [L.Scenario(T=60), L.Scenario(T=60, mu_scale=lambda t: 0.75 if t >= 40 else 1.0)]
rows = []
for Kp, Ki in itertools.product([0.04, 0.1, 0.2, 0.4], [0.015, 0.05, 0.1, 0.2, 0.4]):
    L.PID_GAINS = dict(Kp=Kp, Ki=Ki, Kd=0.0)
    ms = [L.metrics(L.run_closed_loop("pid_ekf", sc, s)) for sc in SC for s in range(100, 104)]
    iae, tv = np.mean([m["IAE"] for m in ms]), np.mean([m["TV_D"] for m in ms])
    rows.append((iae + 2.0 * tv, Kp, Ki, iae, tv))
    print(f"Kp={Kp:5} Ki={Ki:6}: IAE={iae:6.2f}  TV={tv:5.2f}")
best = min(rows)
print(f"\nMejor (IAE + 2·TV): Kp={best[1]}, Ki={best[2]}  (IAE={best[3]:.2f}, TV={best[4]:.2f})")
