"""
eigentrust.py — PygenTrust-compatible EigenTrust engine for EENet.

Implements the same public API as PygenTrust (mattyTokenomics/PygenTrust)
with zero external dependencies beyond NumPy, so it drops in whether or
not the pip package is available.

Public API
----------
eigentrust(C, p, epsilon, alpha) -> np.ndarray
    C       : (N, N) raw local-trust matrix  C[i,j] = how much peer i trusts peer j
    p       : (N,)   pre-trust vector (uniform by default)
    epsilon : convergence tolerance          (default 1e-6)
    alpha   : weight on pre-trust vs peers   (default 0.1)
    returns : (N,) global trust vector, sums to 1

EigenTrustTracker
    Stateful wrapper for EENet — accumulates per-peer evidence signals
    across batches, builds C, runs convergence, exposes trust-scaled
    thresholds back to the ExitAssigner.
"""

import numpy as np


# ── Core algorithm ─────────────────────────────────────────────────────────

def _normalise_rows(C: np.ndarray) -> np.ndarray:
    """Row-normalise C so each row sums to 1 (or stays zero for isolates)."""
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)   # avoid /0
    return C / row_sums


def eigentrust(
    C: np.ndarray,
    p: np.ndarray | None = None,
    epsilon: float = 1e-6,
    alpha: float = 0.1,
    max_iter: int = 200,
) -> np.ndarray:
    """
    EigenTrust iteration (Kamvar et al. 2003).

    t_{k+1} = (1 - alpha) * C̃ᵀ t_k  +  alpha * p

    Parameters
    ----------
    C       : (N, N) raw local trust scores (non-negative).
    p       : (N,)  pre-trust vector.  Defaults to uniform 1/N.
    epsilon : convergence threshold on L1 norm of successive iterates.
    alpha   : weight of pre-trust vector (0 = pure peer trust, 1 = pure pre-trust).

    Returns
    -------
    t : (N,) global trust vector, sums to 1.
    """
    N = C.shape[0]
    if p is None:
        p = np.ones(N, dtype=np.float64) / N
    else:
        p = np.asarray(p, dtype=np.float64)
        p = p / p.sum()

    C_norm = _normalise_rows(np.asarray(C, dtype=np.float64))

    t = p.copy()
    for _ in range(max_iter):
        t_new = (1.0 - alpha) * C_norm.T @ t + alpha * p
        if np.abs(t_new - t).sum() < epsilon:
            t = t_new
            break
        t = t_new

    return t / t.sum()


# ── EENet evidence accumulator ─────────────────────────────────────────────

class EigenTrustTracker:
    """
    Accumulates per-peer inference evidence across batches and converts it
    into trust-scaled exit thresholds.

    Peers map 1-to-1 onto EENet exits (Peer 0 → Exit 0, …, Peer K-1 → Exit K-1).
    After every call to `update()` the global trust vector is recomputed and
    `trust_scaled_thresholds()` returns thresholds modulated by each peer's
    reputation.

    Evidence signals collected per peer per batch
    ---------------------------------------------
    accuracy_at_exit : fraction of samples exiting here that were correct
    latency_ok       : fraction of exits that met the latency budget
    score_calibration: Pearson r between raw gk score and accuracy — how honest
                       a peer's exit signal is
    """

    def __init__(
        self,
        n_peers: int,
        pre_trust: np.ndarray | None = None,
        alpha: float = 0.1,
        epsilon: float = 1e-6,
        trust_scale: float = 0.2,
        decay: float = 0.85,
    ):
        """
        Parameters
        ----------
        n_peers     : number of peers / exits (must equal args.num_exits).
        pre_trust   : (n_peers,) seed trust; defaults to uniform.
        alpha       : EigenTrust pre-trust weight.
        epsilon     : EigenTrust convergence tolerance.
        trust_scale : max fractional threshold adjustment from trust  (±trust_scale).
        decay       : exponential decay applied to C each round so old
                      observations matter less over time.
        """
        self.n = n_peers
        self.alpha = alpha
        self.epsilon = epsilon
        self.trust_scale = trust_scale
        self.decay = decay

        self.pre_trust = pre_trust
        # Local trust matrix C[i, j] = how much peer i trusts peer j.
        # Diagonal ignored by EigenTrust; off-diagonals reflect cross-peer trust.
        self.C = np.ones((n_peers, n_peers), dtype=np.float64) - np.eye(n_peers)
        self.C /= max(n_peers - 1, 1)          # uniform initialisation

        # Running evidence per peer (exponential moving average)
        self.accuracy   = np.full(n_peers, 0.5)    # start neutral
        self.latency_ok = np.full(n_peers, 1.0)
        self.calibration = np.full(n_peers, 0.5)

        # Cached trust vector
        self.trust = eigentrust(self.C, self.pre_trust, self.epsilon, self.alpha)

        # History for logging
        self.history: list[dict] = []

    # ── public interface ───────────────────────────────────────────────────

    def update(
        self,
        peer_id: int,
        accuracy: float,
        latency_ok: float,
        score_calibration: float,
        ema_momentum: float = 0.3,
    ) -> np.ndarray:
        """
        Update evidence for `peer_id` and recompute global trust.

        Parameters
        ----------
        peer_id           : which exit / peer is being evaluated (0-indexed).
        accuracy          : fraction correct among samples that exited here.
        latency_ok        : fraction of exits within latency budget (0–1).
        score_calibration : correlation between gk score and accuracy (0–1).
        ema_momentum      : weight on new observation vs old EMA.

        Returns
        -------
        Updated global trust vector (n_peers,).
        """
        m = ema_momentum
        self.accuracy[peer_id]    = (1 - m) * self.accuracy[peer_id]    + m * accuracy
        self.latency_ok[peer_id]  = (1 - m) * self.latency_ok[peer_id]  + m * latency_ok
        self.calibration[peer_id] = (1 - m) * self.calibration[peer_id] + m * score_calibration

        self._rebuild_C(peer_id)
        self.trust = eigentrust(self.C, self.pre_trust, self.epsilon, self.alpha)

        self.history.append({
            'peer': peer_id,
            'accuracy': accuracy,
            'latency_ok': latency_ok,
            'calibration': score_calibration,
            'trust': self.trust.copy(),
        })
        return self.trust

    def trust_scaled_thresholds(self, base_thresholds: np.ndarray) -> np.ndarray:
        """
        Adjust base EENet thresholds by peer trust scores.

        High-trust peers (trust > mean) get lower thresholds → more aggressive
        early exiting.  Low-trust peers get higher thresholds → conservative,
        samples pass through to more trusted downstream peers.

        Parameters
        ----------
        base_thresholds : (n_peers,) thresholds from ExitAssigner.get_threshold().

        Returns
        -------
        (n_peers,) adjusted thresholds, same scale as input.
        """
        t = self.trust
        t_mean = t.mean()
        # deviation from mean, normalised to [-1, 1]
        t_dev = (t - t_mean) / (t.max() - t.min() + 1e-9)
        # lower threshold for trusted peers, higher for untrusted
        adjusted = base_thresholds - self.trust_scale * t_dev * base_thresholds
        return np.clip(adjusted, 0.0, 1.0)

    def peer_summary(self) -> dict:
        """Return a readable dict of current evidence + trust per peer."""
        return {
            f'peer_{k}': {
                'trust':       round(float(self.trust[k]), 4),
                'accuracy':    round(float(self.accuracy[k]), 4),
                'latency_ok':  round(float(self.latency_ok[k]), 4),
                'calibration': round(float(self.calibration[k]), 4),
            }
            for k in range(self.n)
        }

    # ── internal ───────────────────────────────────────────────────────────

    def _rebuild_C(self, updated_peer: int) -> None:
        """
        Rebuild C by having every other peer update its trust score for
        `updated_peer` based on observed evidence.

        Evidence composite score for peer j as seen by peer i:
            quality_j = 0.5 * accuracy_j + 0.3 * latency_ok_j + 0.2 * calibration_j

        Peers that performed above average earn positive trust updates;
        those below average earn negative (but floored at 0).
        """
        # Apply decay to existing C to down-weight stale observations
        self.C *= self.decay

        j = updated_peer
        quality = (
            0.5 * self.accuracy[j]
            + 0.3 * self.latency_ok[j]
            + 0.2 * self.calibration[j]
        )

        for i in range(self.n):
            if i == j:
                continue
            self.C[i, j] = max(0.0, quality)

        # Row-normalise so each row sums to 1 (standard EigenTrust requirement)
        row_sums = self.C.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        self.C = self.C / row_sums


# ── Evidence helpers ────────────────────────────────────────────────────────

def compute_score_calibration(
    scores: np.ndarray,
    correct: np.ndarray,
) -> float:
    """
    Pearson correlation between gk scores and binary correctness.
    Returns 0.5 (neutral) if fewer than 5 samples or zero variance.

    Parameters
    ----------
    scores  : (N,) raw gk / ScoreNormalizer outputs for samples that exited here.
    correct : (N,) binary array — 1 if prediction correct, 0 otherwise.
    """
    if len(scores) < 5:
        return 0.5
    s, c = np.asarray(scores, dtype=float), np.asarray(correct, dtype=float)
    if s.std() < 1e-9 or c.std() < 1e-9:
        return 0.5
    r = np.corrcoef(s, c)[0, 1]
    # Map from [-1, 1] to [0, 1]
    return float(np.clip((r + 1.0) / 2.0, 0.0, 1.0))
