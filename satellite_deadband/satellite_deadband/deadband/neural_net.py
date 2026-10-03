"""
Bounded neural-network test (Workstream D): how badly does a time-indexed network
extrapolate?

A small MLP is fit as SMA = f(t) on a short training window, then rolled out far beyond
it. Inside the window it fits well; outside it flattens to a plateau (the tanh
activations saturate) instead of continuing the real secular decay.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .constants import CD_TRUE, DU, from_hours, from_minutes, to_hours
from .physics import osculating_sma_series, propagate

TRAIN_CADENCE_MIN = 5.0          # the project's standard sampling interval
HIDDEN_LAYERS = (48, 24)
N_DENSE_POINTS = 4000


@dataclass
class HorizonTest:
    t_hours: np.ndarray
    sma_true_km: np.ndarray
    sma_pred_km: np.ndarray
    train_end_hours: float
    rows: list                   # one dict per horizon: true / predicted SMA and error


def _sma_km(t_eval):
    return osculating_sma_series(propagate(CD_TRUE, t_eval)) * DU


def run_long_horizon_test(train_hours=2.0, horizons_hours=(1, 2, 4, 6, 8, 10, 12), seed=0):
    train_end = from_hours(train_hours)
    t_dense = np.linspace(0, from_hours(max(horizons_hours)), N_DENSE_POINTS)
    sma_dense = _sma_km(t_dense)

    t_train = np.arange(0, train_end, from_minutes(TRAIN_CADENCE_MIN))
    sma_train = _sma_km(t_train)

    t_scaler = StandardScaler().fit(t_train.reshape(-1, 1))
    sma_scaler = StandardScaler().fit(sma_train.reshape(-1, 1))
    network = MLPRegressor(hidden_layer_sizes=HIDDEN_LAYERS, activation="tanh", solver="adam",
                           max_iter=4000, random_state=seed, alpha=1e-4)
    network.fit(t_scaler.transform(t_train.reshape(-1, 1)),
                sma_scaler.transform(sma_train.reshape(-1, 1)).ravel())

    predicted_scaled = network.predict(t_scaler.transform(t_dense.reshape(-1, 1)))
    sma_pred = sma_scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).ravel()

    rows = []
    for horizon in horizons_hours:
        i = np.argmin(np.abs(t_dense - from_hours(horizon)))
        rows.append(dict(horizon_hours=horizon, true_sma_km=sma_dense[i], pred_sma_km=sma_pred[i],
                         abs_err_km=abs(sma_pred[i] - sma_dense[i]),
                         inside_training_window=bool(from_hours(horizon) <= train_end)))

    return HorizonTest(t_hours=to_hours(t_dense), sma_true_km=sma_dense, sma_pred_km=sma_pred,
                       train_end_hours=train_hours, rows=rows)
