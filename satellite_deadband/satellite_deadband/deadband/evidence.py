"""
Burn "evidence" signals: what a detector sees when it tries to tell burns from coasting.

Two sample-to-sample jumps are used:
    delta SMA   : change in the energy-method osculating SMA   (orbit SIZE changing)
    delta speed : change in raw speed |v|                      (how hard the impulse hit)

A real burn moves both by a lot in one step; drag decay between burns moves them by
very little. The jumps span almost two orders of magnitude and are heavily right-skewed,
so detectors work on log(|jump| + eps) -- a plain Gaussian mixture fit on the raw jumps
collapses both components onto the same point.
"""
from dataclasses import dataclass

import numpy as np

from .controller import DeadbandSimulation
from .physics import osculating_sma_series

NOISE_FLOOR = 1e-12   # added to every noise scale and used as the clip floor


def log_transform(jump):
    """log(jump + eps), with eps a tiny fraction of the smallest typical jump (avoids log 0)."""
    positive = jump[jump > 0]
    eps = np.percentile(positive, 1) * 1e-3 if positive.size else 1e-12
    return np.log(jump + eps)


def add_noise(signal, noise_frac, seed=0):
    """Gaussian noise with std = noise_frac * std(signal). A tiny floor keeps zero-noise runs valid."""
    scale = np.std(signal) * noise_frac + NOISE_FLOOR
    return signal + scale * np.random.default_rng(seed).standard_normal(signal.shape)


def _sample_to_sample_jump(values):
    jump = np.abs(np.diff(values))
    return np.concatenate([[jump[0]], jump])   # repeat the first so the length matches


@dataclass
class BurnEvidence:
    t: np.ndarray            # sample times
    true_label: np.ndarray   # 1 = burn, 0 = coast (ground truth from the simulation)
    jump_sma: np.ndarray     # |delta SMA| per sample (raw)
    jump_speed: np.ndarray   # |delta speed| per sample (raw)

    @property
    def log_sma(self):
        return log_transform(self.jump_sma)

    @property
    def log_speed(self):
        return log_transform(self.jump_speed)

    @property
    def features(self):
        """(N, 2) matrix of [log |delta SMA|, log |delta speed|]."""
        return np.column_stack([self.log_sma, self.log_speed])

    def with_noise(self, noise_frac, seed=0):
        """Copy with measurement noise added to the RAW jumps (speed channel uses seed + 1)."""
        noisy_sma = np.clip(add_noise(self.jump_sma, noise_frac, seed), NOISE_FLOOR, None)
        noisy_speed = np.clip(add_noise(self.jump_speed, noise_frac, seed + 1), NOISE_FLOOR, None)
        return BurnEvidence(self.t, self.true_label, noisy_sma, noisy_speed)


def build_evidence(sim: DeadbandSimulation):
    speed = np.linalg.norm(sim.states[3:6, :], axis=0)
    return BurnEvidence(t=sim.t, true_label=sim.burn_labels,
                        jump_sma=_sample_to_sample_jump(osculating_sma_series(sim.states)),
                        jump_speed=_sample_to_sample_jump(speed))
