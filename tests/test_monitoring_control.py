import numpy as np

from biomfg.control import MPC, PID
from biomfg.detection import Monitor
from biomfg.estimation import EKF
from biomfg.loop import NOMINAL, Scenario, metrics, run_closed_loop, make_ekf
from biomfg.sensors import BiomassSensor, SensorFault, SubstrateSensor


def test_biomass_sensor_noise_level():
    rng = np.random.default_rng(0)
    s = BiomassSensor(0.02, rng)
    r = np.array([s.read(0, 4.0) for _ in range(5000)])
    assert abs(r.mean() - 4.0) < 0.01 and abs(r.std() / 4.0 - 0.02) < 0.003


def test_substrate_sensor_is_sparse():
    s = SubstrateSensor(0.1, 4, np.random.default_rng(0))
    out = [s.read(k, 1.0) for k in range(8)]
    assert [o is not None for o in out] == [True, False, False, False] * 2


def test_ekf_converges_from_wrong_initial_guess():
    # sin fallos y sin variabilidad: tras unas horas X̂ y Ŝ deben coincidir con la planta
    lg = run_closed_loop("open_loop", Scenario(T=30, variability=False), seed=0)
    assert abs(lg["Xh"][-1] - lg["X"][-1]) < 0.1
    assert abs(lg["Sh"][-1] - lg["S"][-1]) < 0.3


def test_ekf_nis_is_consistent_without_faults():
    nis = np.concatenate([run_closed_loop("open_loop", Scenario(T=40), s)["nisX"][20:] for s in range(5)])
    assert 0.6 < nis.mean() < 1.8   # ideal 1; tolerancia por variabilidad de parámetros


def test_process_fault_detected_and_not_mistaken_for_sensor():
    sc = Scenario(T=60, mu_scale=lambda t: 0.7 if t >= 30 else 1.0)
    al = run_closed_loop("open_loop", sc, seed=1)["alarms"]
    assert 30 <= al["process"] <= 36
    assert "sensor_X" not in al and "frozen_X" not in al


def test_frozen_sensor_detected():
    sc = Scenario(T=60, fault=SensorFault("stuck", 30.0))
    al = run_closed_loop("open_loop", sc, seed=1)["alarms"]
    assert 30 <= al["frozen_X"] <= 32


def test_monitor_quiet_without_anomalies():
    m = Monitor()
    for k in range(100):
        m.update(k * 0.25, 0.5, 0.5, 1.0, 0.05, yX=4.0 + 0.01 * k)
    assert m.first_alarm == {}


def test_pid_respects_limits_and_antiwindup():
    pid = PID(0.5, 0.5, 0.0, 0.25, 0.0, 0.55, 0.3)
    for _ in range(200):
        assert 0.0 <= pid(10.0, 4.0) <= 0.55   # error enorme: satura
    assert pid.I < 50                          # el integrador no se dispara
    u = [pid(4.0, 10.0) for _ in range(30)]    # error opuesto: debe recuperarse rápido
    assert u[-1] < 0.55 and min(u) < 0.5


def test_mpc_moves_in_correct_direction_and_within_bounds():
    mpc = MPC(NOMINAL, 0.25, u_min=0.0, u_max=0.55)
    assert mpc(4.8, 0.4, 1.0, 4.0, 0.3) > 0.3   # X demasiado alto -> subir D
    mpc = MPC(NOMINAL, 0.25, u_min=0.0, u_max=0.55)
    assert mpc(3.2, 2.0, 1.0, 4.0, 0.4) < 0.4   # X demasiado bajo -> bajar D


def test_feedback_beats_open_loop_under_disturbance():
    sc = Scenario(T=60, variability=False, mu_scale=lambda t: 0.75 if t >= 40 else 1.0)
    ol = metrics(run_closed_loop("open_loop", sc, 0))
    pid = metrics(run_closed_loop("pid_ekf", sc, 0))
    assert pid["IAE"] < 0.6 * ol["IAE"]
