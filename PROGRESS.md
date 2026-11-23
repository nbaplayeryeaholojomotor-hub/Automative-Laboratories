# PROGRESS

## Milestone 1 — Modelo de crecimiento microbiano (batch)
**Hecho (simulación + datos sintéticos):**
- Modelos exponencial y logístico con vector de estado (`models/growth.py`).
- Simulador genérico con `solve_ivp` (`simulation.py`).
- Verificación numérico vs analítico: error relativo ~1e-9.
- Estimación de mu por regresión en ln X y ajuste logístico (`analysis.py`).
- 6 tests pasando. Figura: `docs/figures/milestone1.png`.

**Resultados (datos SINTÉTICOS, ruido lognormal 5 %, mu real 0.6, X_max 4.0):**
mu = 0.598 (recta, 4 puntos) y 0.591 (ajuste); X_max = 4.09.

**Pendiente:**
- Dataset REAL publicado de curva de crecimiento, con procedencia en `data/README.md`.
- Fase de latencia (lag), no modelada.
- Incertidumbre de los parámetros (intervalos de confianza).

**Limitaciones:** el logístico es fenomenológico (no explica el techo);
datos sintéticos no prueban nada sobre biología real.

**Siguiente:** Milestone 2 — Monod: estado y = [X, S], mu = mu(S).

## Fases 2-5 (en curso)
**Hecho y con tests (23 en total):**
- F2 Monod + balance de masa + ajuste con IC95 % (`scripts/02_monod.py`). Hallazgo: mi hipótesis
  "S0 bajo identifica mejor Ks" era FALSA (Ks ±0.10 con S0=10 vs ±0.14 con S0=2).
- F3 biorreactor con volumen variable, quimiostato, lavado (`scripts/03_reactor.py`).
  Corrección: el lavado con D apenas > D_crit es muy lento (-0.017 1/h).
- F4 sensores, EKF [X,S,theta], monitor (`scripts/04_monitoring.py`). Límite: el detector por
  modelo NO ve un sensor atascado (el filtro lo absorbe); se añadió chequeo de señal congelada.
- F5 PID y MPC (`control.py`, `loop.py`). El MPC con r=0.5 hacía vibrar la bomba por ruido;
  se subió a r=20 (compromiso IAE/vibración, no un óptimo).

**Pendiente / sin verificar:** resultados de `scripts/05_control.py` (comparación de controladores,
aún ejecutándose al hacer este commit); parámetros ILUSTRATIVOS, no medidos; dataset real publicado.
