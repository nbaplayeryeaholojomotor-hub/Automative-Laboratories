# Automative Laboratories — Fabricación biológica autónoma

Proyecto de aprendizaje y desarrollo (bioingeniería): biorreactor digital con modelo, sensores
simulados, estimación de estado, detección de fallos y control. Todo 100 % digital.

```bash
pip install -r requirements.txt
python -m pytest -q                       # 23 tests
PYTHONPATH=src python scripts/01_growth_milestone.py   # F1 crecimiento
PYTHONPATH=src python scripts/02_monod.py              # F2 Monod
PYTHONPATH=src python scripts/03_reactor.py            # F3 biorreactor
PYTHONPATH=src python scripts/04_monitoring.py         # F4 EKF + fallos
PYTHONPATH=src python scripts/05_control.py 6          # F5 control (≈ 15 min)
```
**Aviso:** los parámetros (mu_max, Ks, Y...) son ilustrativos, no medidos; los datos son sintéticos.
Estado, resultados y limitaciones: ver `PROGRESS.md`.
