"""Bucle cerrado completo: planta -> sensores -> estimador/monitor -> controlador -> planta."""
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .control import MPC, PID
from .detection import Monitor
from .estimation import EKF
from .models.reactor import chemostat_steady_state
from .parameters import ReactorParams
from .plant import Plant
from .sensors import BiomassSensor, SensorFault, SubstrateSensor

NOMINAL = ReactorParams(mu_max=0.5, Ks=0.5, Y=0.5, S_in=10.0)  # ILUSTRATIVO
TS = 0.25            # [h] periodo de muestreo
REL_X, SD_S, S_EVERY = 0.02, 0.1, 4  # ruido OD 2 %, ruido S 0.1 g/L, S cada 1 h
D_MAX = 0.55         # capacidad máxima de la bomba [1/h]
PID_GAINS = dict(Kp=0.1, Ki=0.1, Kd=0.0)  # ajustados por barrido: scripts/05b_tune_pid.py


@dataclass
class Scenario:
    T: float = 60.0
    sp_fn: Callable = lambda t: 4.4 if t < 25 else 4.0
    mu_scale: Callable = lambda t: 1.0
    fault: SensorFault = None
    variability: bool = True   # parámetros verdaderos distintos del nominal
    D0: float = 0.30


def true_params(rng, variability):
    if not variability:
        return NOMINAL
    return replace(NOMINAL, mu_max=NOMINAL.mu_max * rng.lognormal(0, 0.08),
                   Ks=NOMINAL.Ks * rng.lognormal(0, 0.15), Y=NOMINAL.Y * rng.lognormal(0, 0.05))


def make_ekf(z0_X):
    return EKF(NOMINAL, [z0_X, 1.0, 1.0], [0.5 ** 2, 1.0, 0.2 ** 2],
               q_diag=[1e-3, 1e-3, 0.05 ** 2], Ts=TS)


def run_closed_loop(controller, sc, seed, ekf_factory=make_ekf):
    """controller: 'open_loop' | 'pid_raw' | 'pid_ekf' | 'mpc' | 'mpc_adaptive'."""
    rng = np.random.default_rng(seed)
    p_true = true_params(rng, sc.variability)
    X0, S0 = chemostat_steady_state(p_true, sc.D0)
    plant = Plant(p_true, [X0, S0, 1.0], mu_scale=sc.mu_scale)
    sX = BiomassSensor(REL_X, rng, sc.fault)
    sS = SubstrateSensor(SD_S, S_EVERY, rng)
    ekf = ekf_factory(X0)
    mon = Monitor()
    pid = PID(**PID_GAINS, Ts=TS, u_min=0.0, u_max=D_MAX, u0=sc.D0)
    mpc = MPC(NOMINAL, TS, u_min=0.0, u_max=D_MAX)
    D_prev = sc.D0
    n = int(round(sc.T / TS))
    log = {k: np.zeros(n) for k in ["t", "X", "S", "yX", "Xh", "Sh", "th", "D", "sp", "nisX", "nisS"]}
    for k in range(n):
        t = k * TS
        sp = sc.sp_fn(t)
        yX = sX.read(t, plant.y[0]); yS = sS.read(k, plant.y[1])
        _, nisX = ekf.update_X(yX, REL_X)
        nisS = None
        if yS is not None:
            _, nisS = ekf.update_S(yS, SD_S)
        mon.update(t, nisX, nisS, ekf.z[2], ekf.theta_sd, yX)
        Xh, Sh, th = ekf.z
        if controller == "open_loop":
            D = sc.D0 if sc.sp_fn(t) >= 4.4 else 0.40   # D "de receta" para cada consigna
        elif controller == "pid_raw":
            D = pid(yX, sp)
        elif controller == "pid_ekf":
            D = pid(Xh, sp)
        elif controller == "mpc":
            D = mpc(Xh, Sh, 1.0, sp, D_prev)
        elif controller == "mpc_adaptive":
            D = mpc(Xh, Sh, th, sp, D_prev)
        else:
            raise ValueError(controller)
        for key, v in zip(["t", "X", "S", "yX", "Xh", "Sh", "th", "D", "sp", "nisX", "nisS"],
                          [t, plant.y[0], plant.y[1], yX, Xh, Sh, th, D, sp,
                           nisX, np.nan if nisS is None else nisS]):
            log[key][k] = v
        plant.step_chemostat(D, TS)
        ekf.predict(D)
        D_prev = D
    log["alarms"] = dict(mon.first_alarm)
    return log


def metrics(log):
    dt = log["t"][1] - log["t"][0]
    err = np.abs(log["X"] - log["sp"])
    return {"IAE": float(np.sum(err) * dt), "X_final_err": float(err[-20:].mean()),
            "washout": bool(log["X"].min() < 0.5), "TV_D": float(np.sum(np.abs(np.diff(log["D"]))))}
