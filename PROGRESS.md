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

**Pendiente / sin verificar:** parámetros ILUSTRATIVOS, no medidos; dataset real publicado.

### Corrección (fase 5)
El PID inicial (Kp=0.04, Ki=0.015) estaba mal ajustado: IAE 13.0 frente a 4.35 con Kp=0.1, Ki=0.1
(barrido en `scripts/05b_tune_pid.py`, semillas de entrenamiento 100-103, evaluación en 0-5).
Con el PID ajustado la ventaja del MPC adaptativo sobre el PID es mucho menor que en la primera
tabla. Observaciones ya verificadas: sesgo +15 % en el sensor engaña a todos los controladores
(error 0.3-0.56 g/L); la bandera "lavado" (X<0.5) no se activa en 20 h porque el lavado es lento.

### Resultados finales fase 5 (6 semillas, media ± sd; datos y parámetros SINTÉTICOS/ILUSTRATIVOS)
IAE [g·h/L] (menor = mejor) | TV(D) = vibración de la bomba

| Controlador | A seguimiento | B mu_max x0.75 | C sesgo sensor +15 % | TV(D) en A |
|---|---|---|---|---|
| lazo abierto | 17.6 ± 10.4 | 43.7 ± 8.2 | 17.6 ± 10.4 | 0.10 |
| PID sobre sensor crudo | 2.6 ± 0.5 | 4.2 ± 0.7 | 12.1 ± 0.4 | 2.12 |
| PID sobre estimación EKF | 3.2 ± 1.0 | 4.8 ± 1.1 | 10.1 ± 0.5 | 0.65 |
| MPC modelo nominal | 6.0 ± 6.8 | 9.6 ± 2.6 | 9.8 ± 2.3 | 0.90 |
| MPC adaptativo (theta) | 3.7 ± 1.7 | 3.9 ± 1.4 | 9.0 ± 0.9 | 1.86 |

**Lo que se puede afirmar:** el lazo cerrado supera claramente al abierto; el MPC adaptativo mejora
claramente al MPC sin adaptar (B: 3.9 vs 9.6); el filtrado por EKF reduce 3x la vibración de la bomba
frente al PID sobre sensor crudo; un sesgo de sensor degrada TODOS los controladores (error 0.3-0.5 g/L).
**Lo que NO se puede afirmar:** que el MPC adaptativo sea mejor que un PID bien ajustado (diferencias
dentro del ruido con 6 semillas). Posible sesgo: el PID se ajustó con un barrido (20 combinaciones) y
el MPC solo en r; horizonte y pesos del MPC no se optimizaron. Sin explicar: los MPC oscilan ±0.1 g/L
(periodo ~3 h); hipótesis NO verificada: interacción con la dinámica de theta del EKF.
La bandera "lavado" (X<0.5) nunca salta: el lavado con D apenas sobre D_crit es lento (>20 h).

**Siguiente:** (1) dataset real publicado con procedencia; (2) cerrar el lazo con el monitor
(reacción automática a alarmas de sensor); (3) explicar/eliminar la oscilación del MPC.
