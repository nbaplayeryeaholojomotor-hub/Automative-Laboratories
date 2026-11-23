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
