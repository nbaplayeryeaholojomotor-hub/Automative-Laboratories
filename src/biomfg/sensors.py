"""Sensores simulados: lo único que el monitor y el controlador pueden ver."""
from dataclasses import dataclass

import numpy as np


@dataclass
class SensorFault:
    kind: str         # "bias" | "drift" | "stuck"
    t_start: float    # [h]
    magnitude: float = 0.0  # bias: fracción (0.15 = +15 %); drift: fracción por hora


class BiomassSensor:
    """Tipo densidad óptica (OD): ruido proporcional a la señal. Mide cada Ts."""

    def __init__(self, rel_noise, rng, fault=None):
        self.rel_noise, self.rng, self.fault = rel_noise, rng, fault
        self._last, self._stuck = None, None

    def read(self, t, X_true):
        f, x = self.fault, X_true
        if f is not None and t >= f.t_start:
            if f.kind == "bias":
                x = X_true * (1.0 + f.magnitude)
            elif f.kind == "drift":
                x = X_true * (1.0 + f.magnitude * (t - f.t_start))
            elif f.kind == "stuck":
                if self._stuck is None:
                    self._stuck = self._last if self._last is not None else X_true
                return self._stuck
        reading = max(x * (1.0 + self.rel_noise * self.rng.standard_normal()), 1e-6)
        self._last = reading
        return reading


class SubstrateSensor:
    """Tipo análisis fuera de línea (HPLC/enzimático): preciso pero escaso."""

    def __init__(self, abs_noise, every, rng):
        self.abs_noise, self.every, self.rng = abs_noise, every, rng

    def read(self, k, S_true):
        """k = índice de muestreo. Devuelve None cuando no hay medida."""
        if k % self.every != 0:
            return None
        return max(S_true + self.abs_noise * self.rng.standard_normal(), 0.0)
