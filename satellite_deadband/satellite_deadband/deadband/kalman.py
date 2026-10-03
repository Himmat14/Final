"""
Step 4 of the report: an Extended Kalman Filter (EKF) that estimates the drag coefficient cd
from noisy POSITION measurements, one measurement at a time.

State:        x = [r (3), v (3), cd]           (cd is treated as an unknown constant)
Dynamics:     r' = v,  v' = gravity + J2 + drag(cd),  cd' = 0
Measurement:  z = r + noise                    (we only see position)

Each measurement step does two things:
    PREDICT: integrate x and its 7x7 state-transition matrix Phi from t_k to t_k+1,
             P <- Phi P Phi^T + Q          (uncertainty grows between measurements)
    UPDATE:  K = P H^T (H P H^T + R)^-1    (Kalman gain: how much to trust the measurement)
             x <- x + K (z - H x),  P <- (I - K H) P   (uncertainty shrinks)

Unlike the least-squares fits, the filter gives a running estimate AND its own uncertainty
(the square root of P[6, 6]), so you can watch cd converge as data arrives.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .constants import DU, J2, MU
from .physics import drag_accel, gravity_accel, j2_accel

N_STATE = 7
H = np.hstack([np.eye(3), np.zeros((3, 4))])   # measurement picks out the position


def acceleration(r, v, cd, j2=J2):
    return gravity_accel(r) + j2_accel(r, j2) + drag_accel(v, cd)


def jacobian(x, j2=J2):
    """A = df/dx for the 7-state model (gravity and drag analytic, J2 by central differences)."""
    r, v, cd = x[0:3], x[3:6], x[6]
    radius, speed = np.linalg.norm(r), np.linalg.norm(v)
    A = np.zeros((N_STATE, N_STATE))
    A[0:3, 3:6] = np.eye(3)                                                     # dr'/dv
    gravity_gradient = -MU * (np.eye(3) / radius**3 - 3 * np.outer(r, r) / radius**5)
    step = 1e-7
    j2_gradient = np.column_stack([(j2_accel(r + step * e, j2) - j2_accel(r - step * e, j2)) / (2 * step)
                                   for e in np.eye(3)])
    A[3:6, 0:3] = gravity_gradient + j2_gradient                                # dv'/dr
    A[3:6, 3:6] = -cd * (speed * np.eye(3) + np.outer(v, v) / speed)            # dv'/dv (drag)
    A[3:6, 6] = -speed * v                                                      # dv'/dcd
    return A


def _rhs_with_stm(t, y):
    """d/dt of [x (7), Phi (49)] : x' = f(x), Phi' = A(x) Phi."""
    x, phi = y[:N_STATE], y[N_STATE:].reshape(N_STATE, N_STATE)
    dx = np.concatenate([x[3:6], acceleration(x[0:3], x[3:6], x[6]), [0.0]])
    return np.concatenate([dx, (jacobian(x) @ phi).ravel()])


def predict(x, P, t0, t1, Q):
    """Propagate the state and covariance from t0 to t1."""
    y0 = np.concatenate([x, np.eye(N_STATE).ravel()])
    y1 = solve_ivp(_rhs_with_stm, [t0, t1], y0, method="DOP853", rtol=1e-10, atol=1e-12).y[:, -1]
    phi = y1[N_STATE:].reshape(N_STATE, N_STATE)
    return y1[:N_STATE], phi @ P @ phi.T + Q


def update(x, P, z, R):
    """Blend the prediction with the measurement z."""
    innovation = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ innovation
    P = (np.eye(N_STATE) - K @ H) @ P
    return x, P


@dataclass
class KalmanRun:
    t: np.ndarray            # measurement times
    estimates: np.ndarray    # (7, N) filtered state after each update
    sigmas: np.ndarray       # (7, N) one-sigma uncertainty of each state
    noise_km: float


def run_ekf(t, measured_positions, true_state0, cd_guess, noise_km, process_noise=1e-12, seed=0):
    """
    Filter a sequence of noisy positions.
        true_state0  : the starting state, perturbed below so the filter does not start at the truth
        cd_guess     : starting guess for cd
        noise_km     : measurement noise standard deviation (also used to build R)
    """
    rng = np.random.default_rng(seed)
    sigma = max(noise_km, 1e-3) / DU                      # never claim a perfect sensor
    x = true_state0.copy()
    x[0:3] += sigma * rng.standard_normal(3)              # start from a noisy first fix
    x = np.concatenate([x, [cd_guess]])
    dt = t[1] - t[0]
    P = np.diag([sigma**2] * 3 + [(sigma / dt) ** 2] * 3 + [(0.5 * cd_guess) ** 2])
    R = np.eye(3) * sigma**2
    Q = np.diag([0, 0, 0, process_noise, process_noise, process_noise, 0]) * dt

    estimates, sigmas = [x.copy()], [np.sqrt(np.diag(P))]
    for k in range(1, len(t)):
        x, P = predict(x, P, t[k - 1], t[k], Q)
        x, P = update(x, P, measured_positions[:, k], R)
        estimates.append(x.copy())
        sigmas.append(np.sqrt(np.abs(np.diag(P))))
    return KalmanRun(t=t, estimates=np.array(estimates).T, sigmas=np.array(sigmas).T, noise_km=noise_km)
