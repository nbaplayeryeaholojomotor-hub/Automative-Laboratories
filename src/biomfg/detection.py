"""Detección de anomalías basada en el filtro (innovaciones y theta)."""

NIS_THRESHOLD = 10.83  # chi² con 1 g.l., p = 0.001


class Monitor:
    """Tres alarmas, cada una exige varias muestras seguidas (evita falsas alarmas):

    sensor_X : la innovación normalizada (NIS) de X es anormalmente grande
    sensor_S : ídem para el sustrato
    process  : theta_hat está claramente por debajo de 1 (la biología rinde menos)
    frozen_X : la lectura de X se repite EXACTAMENTE varias veces (sensor congelado).
               Chequeo simple y complementario: el detector por modelo NO ve este
               fallo porque el filtro absorbe una lectura constante moviendo theta.
               Limitación: en un sensor real con cuantización podría repetirse un
               valor sin que sea un fallo.
    """

    def __init__(self, n_consec=3, theta_n=6, theta_z=3.0, theta_margin=0.1):
        self.n_consec, self.theta_n = n_consec, theta_n
        self.theta_z, self.theta_margin = theta_z, theta_margin
        self.count = {"sensor_X": 0, "sensor_S": 0, "process": 0, "frozen_X": 0}
        self._y_prev = None
        self.first_alarm = {}

    def update(self, t, nis_X, nis_S, theta_hat, theta_sd, yX=None):
        if nis_X is not None:
            self.count["sensor_X"] = self.count["sensor_X"] + 1 if nis_X > NIS_THRESHOLD else 0
        if nis_S is not None:
            self.count["sensor_S"] = self.count["sensor_S"] + 1 if nis_S > NIS_THRESHOLD else 0
        if yX is not None:
            same = self._y_prev is not None and abs(yX - self._y_prev) < 1e-12
            self.count["frozen_X"] = self.count["frozen_X"] + 1 if same else 0
            self._y_prev = yX
        low = (1 - theta_hat) > self.theta_margin and (1 - theta_hat) / max(theta_sd, 1e-6) > self.theta_z
        self.count["process"] = self.count["process"] + 1 if low else 0
        need = {"sensor_X": self.n_consec, "sensor_S": self.n_consec, "process": self.theta_n, "frozen_X": self.n_consec}
        for k, c in self.count.items():
            if c >= need[k] and k not in self.first_alarm:
                self.first_alarm[k] = t
