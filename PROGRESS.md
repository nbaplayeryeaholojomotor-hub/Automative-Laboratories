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
