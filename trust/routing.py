from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .eigentrust import EigenTrustTracker, compute_score_calibration


SCENARIO_CONFIGS = {
    # fault settings
    "easy": {
        "faulty_fraction": 0.0,
        "drop_prob": 0.0,
        "corrupt_prob": 0.0,
        "spike_prob": 0.02,
        "spike_scale": 1.15,
    },
    "medium": {
        "faulty_fraction": 0.25,
        "drop_prob": 0.10,
        "corrupt_prob": 0.14,
        "spike_prob": 0.18,
        "spike_scale": 1.75,
    },
    "hard": {
        "faulty_fraction": 0.50,
        "drop_prob": 0.22,
        "corrupt_prob": 0.28,
        "spike_prob": 0.30,
        "spike_scale": 2.40,
    },
}


@dataclass
class PeerProfile:
    peer_id: int
    stage_idx: int
    drop_prob: float
    corrupt_prob: float
    spike_prob: float
    spike_scale: float
    latency_multiplier: float
    faulty: bool


def build_peer_profiles(
    num_stages: int,
    replicas_per_stage: int,
    scenario: str,
    seed: int,
) -> List[PeerProfile]:
    if scenario not in SCENARIO_CONFIGS:
        raise ValueError("Unknown scenario: {0}".format(scenario))

    config = SCENARIO_CONFIGS[scenario]
    rng = np.random.default_rng(seed)
    profiles: List[PeerProfile] = []
    peer_id = 0

    for stage_idx in range(num_stages):
        # choose faulty replicas
        faulty_flags = rng.random(replicas_per_stage) < config["faulty_fraction"]
        if faulty_flags.all():
            # keep one good node
            faulty_flags[rng.integers(0, replicas_per_stage)] = False

        for replica_idx in range(replicas_per_stage):
            faulty = bool(faulty_flags[replica_idx])
            if faulty:
                # choose fault type
                fault_mode = rng.choice(["drop", "corrupt", "slow", "mixed"])
                drop_prob = config["drop_prob"] if fault_mode in ("drop", "mixed") else 0.02
                corrupt_prob = config["corrupt_prob"] if fault_mode in ("corrupt", "mixed") else 0.02
                latency_multiplier = float(rng.uniform(1.15, 1.65 if fault_mode != "slow" else 2.10))
                spike_prob = config["spike_prob"]
            else:
                drop_prob = 0.0
                corrupt_prob = 0.0
                latency_multiplier = float(rng.uniform(0.95, 1.10))
                spike_prob = 0.02

            profiles.append(PeerProfile(
                peer_id=peer_id,
                stage_idx=stage_idx,
                drop_prob=float(drop_prob),
                corrupt_prob=float(corrupt_prob),
                spike_prob=float(spike_prob),
                spike_scale=float(config["spike_scale"]),
                latency_multiplier=latency_multiplier,
                faulty=faulty,
            ))
            peer_id += 1

    return profiles


def _adjust_threshold(
    base_threshold: float,
    peer_trust: float,
    stage_trusts: Iterable[float],
    trust_scale: float,
) -> float:
    stage_trusts = np.asarray(list(stage_trusts), dtype=np.float64)
    if len(stage_trusts) == 0:
        return float(base_threshold)

    mean = stage_trusts.mean()
    spread = stage_trusts.max() - stage_trusts.min()
    deviation = (peer_trust - mean) / (spread + 1e-9)
    # adjust by trust
    adjusted = base_threshold - trust_scale * deviation * base_threshold
    low = min(0.0, base_threshold)
    high = max(1.0, base_threshold)
    return float(np.clip(adjusted, low, high))


def _mutate_prediction(
    probs: np.ndarray,
    rng: np.random.Generator,
    corrupt_prob: float,
) -> np.ndarray:
    updated = probs.copy()
    if rng.random() >= corrupt_prob:
        return updated

    # force wrong class
    sorted_indices = np.argsort(updated)
    wrong_idx = int(sorted_indices[-2] if len(sorted_indices) > 1 else sorted_indices[-1])
    updated = np.full_like(updated, 1e-6)
    updated[wrong_idx] = 1.0 - (len(updated) - 1) * 1e-6
    return updated


def simulate_routing_policy(
    policy: str,
    test_pred: np.ndarray,
    test_scores: np.ndarray,
    test_target: np.ndarray,
    base_thresholds: np.ndarray,
    seconds: List[float],
    scenario: str,
    replicas_per_stage: int = 3,
    trust_update_window: int = 64,
    seed: int = 0,
    trust_scale: float = 0.20,
    trust_alpha: float = 0.1,
    trust_decay: float = 0.85,
    max_attempts_per_stage: int = 1,
) -> Dict[str, object]:
    if policy not in ("random", "trust"):
        raise ValueError("Unknown routing policy: {0}".format(policy))

    rng = np.random.default_rng(seed)
    num_stages = int(test_pred.shape[0])
    num_samples = int(test_pred.shape[1])
    profiles = build_peer_profiles(num_stages, replicas_per_stage, scenario, seed)
    peers_by_stage: Dict[int, List[PeerProfile]] = {stage_idx: [] for stage_idx in range(num_stages)}
    for profile in profiles:
        peers_by_stage[profile.stage_idx].append(profile)

    tracker = EigenTrustTracker(
        n_peers=len(profiles),
        pre_trust=None,
        alpha=trust_alpha,
        epsilon=1e-6,
        trust_scale=trust_scale,
        decay=trust_decay,
    )
    peer_buffers = {
        profile.peer_id: {"scores": [], "correct": [], "latency_ok": []}
        for profile in profiles
    }

    completed = 0
    reliable_completed = 0
    correct_total = 0
    dropped_requests = 0
    total_latency = 0.0
    exit_counts = np.zeros(num_stages, dtype=np.int64)
    trust_trace = []

    for sample_idx in range(num_samples):
        sample_done = False
        sample_reliable = True
        sample_latency = 0.0

        for stage_idx in range(num_stages):
            stage_peers = list(peers_by_stage[stage_idx])
            if policy == "trust":
                # try best node first
                stage_peers.sort(key=lambda item: tracker.trust[item.peer_id], reverse=True)
            else:
                rng.shuffle(stage_peers)

            response_obtained = False
            stage_trusts = [tracker.trust[item.peer_id] for item in stage_peers]
            base_probs = np.asarray(test_pred[stage_idx, sample_idx], dtype=np.float64)
            base_score = float(test_scores[stage_idx, sample_idx])

            # Trust routing uses ranked fallback across all replicas. Random routing
            # gets only blind attempts, so a dropped peer can drop the request.
            attempt_limit = len(stage_peers) if policy == "trust" else max_attempts_per_stage
            for attempt_idx, peer in enumerate(stage_peers):
                if attempt_idx >= attempt_limit:
                    break
                latency = seconds[stage_idx] * peer.latency_multiplier
                if rng.random() < peer.spike_prob:
                    latency *= peer.spike_scale
                sample_latency += latency

                if rng.random() < peer.drop_prob:
                    sample_reliable = False
                    peer_buffers[peer.peer_id]["scores"].append(base_score)
                    peer_buffers[peer.peer_id]["correct"].append(0)
                    peer_buffers[peer.peer_id]["latency_ok"].append(0)
                    continue

                response_obtained = True
                probs = _mutate_prediction(base_probs, rng, peer.corrupt_prob)
                pred_label = int(np.argmax(probs))
                is_correct = int(pred_label == int(test_target[sample_idx]))
                peer_score = base_score
                if peer.corrupt_prob > 0 and not is_correct:
                    peer_score = min(base_score + 0.25, 1.0)

                latency_ok = int(latency <= seconds[stage_idx] * 1.25)
                if not latency_ok:
                    sample_reliable = False

                peer_buffers[peer.peer_id]["scores"].append(peer_score)
                peer_buffers[peer.peer_id]["correct"].append(is_correct)
                peer_buffers[peer.peer_id]["latency_ok"].append(latency_ok)

                threshold = base_thresholds[stage_idx]
                if policy == "trust":
                    threshold = _adjust_threshold(
                        float(base_thresholds[stage_idx]),
                        float(tracker.trust[peer.peer_id]),
                        stage_trusts,
                        tracker.trust_scale,
                    )

                should_exit = stage_idx == num_stages - 1 or peer_score >= threshold
                if should_exit:
                    completed += 1
                    if sample_reliable:
                        reliable_completed += 1
                    correct_total += is_correct
                    total_latency += sample_latency
                    exit_counts[stage_idx] += 1
                    sample_done = True
                    break

                break

            if sample_done:
                break

            if not response_obtained:
                dropped_requests += 1
                total_latency += sample_latency
                sample_done = True
                break

        if not sample_done:
            dropped_requests += 1
            total_latency += sample_latency

        if (sample_idx + 1) % trust_update_window == 0:
            for profile in profiles:
                peer_id = profile.peer_id
                score_buf = peer_buffers[peer_id]["scores"]
                correct_buf = peer_buffers[peer_id]["correct"]
                latency_buf = peer_buffers[peer_id]["latency_ok"]
                if not score_buf:
                    continue
                tracker.update(
                    peer_id=peer_id,
                    accuracy=float(np.mean(correct_buf)),
                    latency_ok=float(np.mean(latency_buf)),
                    score_calibration=compute_score_calibration(
                        np.asarray(score_buf),
                        np.asarray(correct_buf),
                    ),
                    ema_momentum=0.30,
                )
                peer_buffers[peer_id] = {"scores": [], "correct": [], "latency_ok": []}
            trust_trace.append(tracker.trust.copy())

    for profile in profiles:
        peer_id = profile.peer_id
        score_buf = peer_buffers[peer_id]["scores"]
        correct_buf = peer_buffers[peer_id]["correct"]
        latency_buf = peer_buffers[peer_id]["latency_ok"]
        if not score_buf:
            continue
        tracker.update(
            peer_id=peer_id,
            accuracy=float(np.mean(correct_buf)),
            latency_ok=float(np.mean(latency_buf)),
            score_calibration=compute_score_calibration(
                np.asarray(score_buf),
                np.asarray(correct_buf),
            ),
            ema_momentum=0.30,
        )

    total_requests = float(num_samples)
    return {
        "policy": policy,
        "scenario": scenario,
        "seed": seed,
        "accuracy": 100.0 * correct_total / total_requests,
        "accuracy_on_completed": 100.0 * correct_total / max(completed, 1),
        "reliability": 100.0 * reliable_completed / total_requests,
        "dropped_responses": 100.0 * dropped_requests / total_requests,
        "avg_latency_ms": total_latency / total_requests,
        "exit_distribution": (exit_counts / max(num_samples, 1)).tolist(),
        "faulty_peers": int(sum(profile.faulty for profile in profiles)),
        "trust_vector": tracker.trust.tolist(),
        "peer_summary": tracker.peer_summary(),
        "trust_trace": [trace.tolist() for trace in trust_trace],
    }
