"""
Lightweight EigenTrust utilities for PRISM / EENet routing experiments.
"""

from typing import Dict, List, Optional

import numpy as np


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    # normalize rows
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return matrix / row_sums


def eigentrust(
    local_trust: np.ndarray,
    pre_trust: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
    alpha: float = 0.1,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Compute the global EigenTrust vector for a non-negative local trust matrix.
    """
    num_peers = local_trust.shape[0]
    if pre_trust is None:
        pre_trust = np.ones(num_peers, dtype=np.float64) / float(num_peers)
    else:
        pre_trust = np.asarray(pre_trust, dtype=np.float64)
        pre_trust = pre_trust / pre_trust.sum()

    matrix = _normalise_rows(np.asarray(local_trust, dtype=np.float64))
    trust = pre_trust.copy()
    for _ in range(max_iter):
        # iterate trust
        updated = (1.0 - alpha) * matrix.T.dot(trust) + alpha * pre_trust
        if np.abs(updated - trust).sum() < epsilon:
            trust = updated
            break
        trust = updated
    return trust / trust.sum()


def compute_score_calibration(scores: np.ndarray, correct: np.ndarray) -> float:
    """
    Map the Pearson correlation between score and correctness onto [0, 1].
    """
    # not enough data yet
    if len(scores) < 5:
        return 0.5

    score_arr = np.asarray(scores, dtype=np.float64)
    correct_arr = np.asarray(correct, dtype=np.float64)
    if score_arr.std() < 1e-9 or correct_arr.std() < 1e-9:
        return 0.5

    corr = np.corrcoef(score_arr, correct_arr)[0, 1]
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


class EigenTrustTracker:
    """
    Stateful trust tracker for routing experiments.

    Each peer accumulates three evidence signals:
    - `accuracy`: how often its outputs are correct
    - `latency_ok`: how often it respects the target latency budget
    - `calibration`: how honest its confidence scores are
    """

    def __init__(
        self,
        n_peers: int,
        pre_trust: Optional[np.ndarray] = None,
        alpha: float = 0.1,
        epsilon: float = 1e-6,
        trust_scale: float = 0.2,
        decay: float = 0.85,
    ):
        self.n = n_peers
        self.alpha = alpha
        self.epsilon = epsilon
        self.trust_scale = trust_scale
        self.decay = decay
        self.pre_trust = pre_trust

        # start even
        self.C = np.ones((n_peers, n_peers), dtype=np.float64) - np.eye(n_peers)
        self.C /= max(n_peers - 1, 1)

        self.accuracy = np.full(n_peers, 0.5, dtype=np.float64)
        self.latency_ok = np.full(n_peers, 1.0, dtype=np.float64)
        self.calibration = np.full(n_peers, 0.5, dtype=np.float64)

        self.trust = eigentrust(self.C, self.pre_trust, self.epsilon, self.alpha)
        self.history: List[Dict[str, object]] = []

    def update(
        self,
        peer_id: int,
        accuracy: float,
        latency_ok: float,
        score_calibration: float,
        ema_momentum: float = 0.3,
    ) -> np.ndarray:
        momentum = ema_momentum
        self.accuracy[peer_id] = (1.0 - momentum) * self.accuracy[peer_id] + momentum * accuracy
        self.latency_ok[peer_id] = (1.0 - momentum) * self.latency_ok[peer_id] + momentum * latency_ok
        self.calibration[peer_id] = (1.0 - momentum) * self.calibration[peer_id] + momentum * score_calibration

        self._rebuild_local_trust(peer_id)
        self.trust = eigentrust(self.C, self.pre_trust, self.epsilon, self.alpha)
        self.history.append({
            "peer_id": peer_id,
            "accuracy": float(accuracy),
            "latency_ok": float(latency_ok),
            "calibration": float(score_calibration),
            "trust": self.trust.copy(),
        })
        return self.trust

    def trust_scaled_thresholds(self, base_thresholds: np.ndarray) -> np.ndarray:
        """
        Lower thresholds for above-average trust, raise them for below-average trust.
        """
        base_thresholds = np.asarray(base_thresholds, dtype=np.float64)
        trust_mean = self.trust.mean()
        spread = self.trust.max() - self.trust.min()
        deviation = (self.trust - trust_mean) / (spread + 1e-9)
        adjusted = base_thresholds - self.trust_scale * deviation * base_thresholds
        low = min(float(base_thresholds.min()), 0.0)
        high = max(float(base_thresholds.max()), 1.0)
        return np.clip(adjusted, low, high)

    def peer_summary(self) -> Dict[str, Dict[str, float]]:
        return {
            "peer_{0}".format(idx): {
                "trust": round(float(self.trust[idx]), 4),
                "accuracy": round(float(self.accuracy[idx]), 4),
                "latency_ok": round(float(self.latency_ok[idx]), 4),
                "calibration": round(float(self.calibration[idx]), 4),
            }
            for idx in range(self.n)
        }

    def _rebuild_local_trust(self, updated_peer: int) -> None:
        # decay old trust
        self.C *= self.decay

        quality = (
            0.5 * self.accuracy[updated_peer]
            + 0.3 * self.latency_ok[updated_peer]
            + 0.2 * self.calibration[updated_peer]
        )
        for observer in range(self.n):
            if observer == updated_peer:
                continue
            self.C[observer, updated_peer] = max(0.0, quality)

        self.C = _normalise_rows(self.C)
