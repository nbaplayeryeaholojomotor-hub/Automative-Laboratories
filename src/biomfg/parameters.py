"""Parámetros de los modelos, con unidades explícitas.

Cada modelo tiene su propia dataclass. Así un parámetro nunca es "un número
suelto": lleva nombre y unidades, y más adelante podremos añadir la fuente
bibliográfica de cada valor.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExponentialParams:
    mu: float  # tasa específica de crecimiento [1/h]


@dataclass(frozen=True)
class LogisticParams:
    mu: float     # tasa específica máxima de crecimiento [1/h]
    X_max: float  # capacidad de carga: biomasa máxima [g/L]


@dataclass(frozen=True)
class MonodParams:
    """Cultivo discontinuo (batch) con limitación por un sustrato."""
    mu_max: float  # tasa específica máxima [1/h]
    Ks: float      # constante de media saturación [g/L]
    Y: float       # rendimiento: g de biomasa formada por g de sustrato [g/g]


@dataclass(frozen=True)
class ReactorParams:
    """Biorreactor agitado con alimentación (fed-batch / quimiostato).

    ATENCIÓN: los valores usados en los scripts son ILUSTRATIVOS (órdenes de
    magnitud de un heterótrofo genérico). NO son parámetros medidos de ninguna
    cepa concreta.
    """
    mu_max: float  # [1/h]
    Ks: float      # [g/L]
    Y: float       # [g/g]
    S_in: float    # concentración de sustrato en la alimentación [g/L]
