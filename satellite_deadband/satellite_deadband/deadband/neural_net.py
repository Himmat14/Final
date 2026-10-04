"""
Bounded neural-network test (Workstream D): how badly does a time-indexed network extrapolate?

Data: the report's 400-day natural run (long_run.natural_run: every force, realistic drag), as the
mean SMA (energy including the J2 potential, so J2's 12 km per-orbit swing is removed and only the
slow drag decay is left), one sample per hour.

A small MLP is fitted as SMA = f(t) on the first TRAIN_DAYS (the same training period as steps 6-9)
and then asked for the SMA up to day 400. Inside the training window it fits well; outside it the
tanh units saturate and the prediction flattens, instead of continuing the decay. A straight line
through the same training data (the energy method's model: constant decay rate) is the physics-based
comparison.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .constants import DU, J2, to_days
from .long_run import HORIZONS_DAYS, TRAIN_DAYS, natural_run
from .regression import energy_sma

SAMPLE_EVERY_MIN = 60                # one sample per hour
HIDDEN_LAYERS = (48, 24)


@dataclass
class HorizonTest:
    days: np.ndarray
    sma_true_km: np.ndarray
    sma_nn_km: np.ndarray
    sma_line_km: np.ndarray
    train_days: float
    rows: list                       # per horizon: true, NN and straight-line SMA and their errors


def run_long_horizon_test(seed=0):
    t, states = natural_run()
    keep = slice(None, None, SAMPLE_EVERY_MIN)
    days = to_days(t[keep])
    sma = energy_sma(states[0:3, keep], states[3:6, keep], j2=J2) * DU
    train = days < TRAIN_DAYS

    t_scaler = StandardScaler().fit(days[train, None])
    sma_scaler = StandardScaler().fit(sma[train, None])
    network = MLPRegressor(hidden_layer_sizes=HIDDEN_LAYERS, activation="tanh", solver="adam", max_iter=4000,
                           random_state=seed, alpha=1e-4)
    network.fit(t_scaler.transform(days[train, None]), sma_scaler.transform(sma[train, None]).ravel())
    sma_nn = sma_scaler.inverse_transform(network.predict(t_scaler.transform(days[:, None]))[:, None]).ravel()

    slope, intercept = np.polyfit(days[train], sma[train], 1)
    sma_line = intercept + slope * days

    rows = []
    for horizon in HORIZONS_DAYS:
        i = int(np.argmin(np.abs(days - (TRAIN_DAYS + horizon))))
        rows.append(dict(days_after_training=horizon, true_sma_km=float(sma[i]), nn_sma_km=float(sma_nn[i]),
                         line_sma_km=float(sma_line[i]), nn_err_km=float(abs(sma_nn[i] - sma[i])),
                         line_err_km=float(abs(sma_line[i] - sma[i]))))
    return HorizonTest(days=days, sma_true_km=sma, sma_nn_km=sma_nn, sma_line_km=sma_line, train_days=TRAIN_DAYS,
                       rows=rows)
