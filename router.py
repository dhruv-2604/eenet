#!/usr/bin/env python3
"""
router to elect peers based on trust and route inference requests through them.
then collects results and tracks trust updates over time.

Usage:
    python router.py --policy trust --scenario medium --n-samples 1000 \
        --output-json outputs/p2p_results/medium_trust.json
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import zmq

from p2p_config import peer_id_to_port, peers_by_stage, PEERS_PER_STAGE
from trust.eigentrust import EigenTrustTracker, compute_score_calibration

CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]
NUM_STAGES = 4

_PEERS_BY_STAGE = peers_by_stage()


# ── ZMQ helpers ───────────────────────────────────────────────────────────────

def _make_req(ctx: zmq.Context, peer_id: int, timeout_ms: int) -> zmq.Socket:
    port = peer_id_to_port(peer_id)
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.connect(f"tcp://localhost:{port}")
    return sock


def _rebuild_socket(ctx: zmq.Context, sockets: dict, peer_id: int, timeout_ms: int) -> None:
    sockets[peer_id].close()
    sockets[peer_id] = _make_req(ctx, peer_id, timeout_ms)


def make_all_sockets(ctx: zmq.Context, timeout_ms: int) -> dict:
    return {pid: _make_req(ctx, pid, timeout_ms) for pid in range(16)}


def _send_recv(ctx, sockets, peer_id, msg, timeout_ms):
    try:
        sockets[peer_id].send(pickle.dumps(msg))
        raw = sockets[peer_id].recv()
        return pickle.loads(raw)
    except zmq.error.Again:
        # Timeout
        _rebuild_socket(ctx, sockets, peer_id, timeout_ms)
        return None
    except Exception as exc:
        print(f"[router] peer={peer_id} unexpected error: {exc}")
        _rebuild_socket(ctx, sockets, peer_id, timeout_ms)
        return None

def _elect_peers(stage_idx: int, tracker: EigenTrustTracker, policy: str, rng) -> list:
    """Return peer IDs for this stage ordered by election priority."""
    peer_ids = _PEERS_BY_STAGE[stage_idx]
    if policy == "trust":
        return sorted(peer_ids, key=lambda pid: float(tracker.trust[pid]), reverse=True)
    order = list(peer_ids)
    rng.shuffle(order)
    return order


def flush_trust_buffers(trust_buffers: dict, tracker: EigenTrustTracker) -> None:
    for peer_id, buf in trust_buffers.items():
        if len(buf["scores"]) < 2:
            continue
        scores = np.array(buf["scores"], dtype=np.float64)
        correct = np.array(buf["correct"], dtype=np.float64)
        latency = np.array(buf["latency_ok"], dtype=np.float64)
        tracker.update(
            peer_id=peer_id,
            accuracy=float(np.mean(np.array(correct) * np.array(scores))) if len(scores) > 0 else 0.5,
            latency_ok=float(np.mean(latency)),
            score_calibration=compute_score_calibration(scores, correct),
            ema_momentum=0.30,
        )
        buf["scores"].clear()
        buf["correct"].clear()
        buf["latency_ok"].clear()


def route_inference(
    image_tensor: torch.Tensor,
    target_label: int,
    tracker: EigenTrustTracker,
    policy: str,
    sockets: dict,
    ctx: zmq.Context,
    seconds: list,
    rng,
    request_id: int,
    trust_buffers: dict,
    timeout_ms: int,
    max_fallbacks: int,
):
    """
    Route one image through the 4-stage pipeline.

    Returns (pred, elected_chain) where pred is the int class prediction
    (or None if the inference was fully dropped) and elected_chain is the
    list of peer_ids that actually served each stage.
    """
    feat = image_tensor
    past_scores = None 
    elected_chain = []

    for stage_idx in range(NUM_STAGES):
        base_latency_ms = seconds[stage_idx]
        latency_ok_threshold_ms = base_latency_ms * 1.25 + 5.0
        ordered_peers = _elect_peers(stage_idx, tracker, policy, rng)

        response = None
        elected = None

        for attempt, peer_id in enumerate(ordered_peers):
            if attempt >= max_fallbacks:
                break

            msg = {
                "request_id": request_id,
                "feat": feat,
                "past_scores": past_scores,
                "deadline_ms": float(timeout_ms),
            }

            resp = _send_recv(ctx, sockets, peer_id, msg, timeout_ms)

            if resp is None:
                trust_buffers[peer_id]["scores"].append(0.0)
                trust_buffers[peer_id]["correct"].append(0)
                trust_buffers[peer_id]["latency_ok"].append(0)
                continue

            if resp.get("status") == "dropped":
                trust_buffers[peer_id]["scores"].append(0.0)
                trust_buffers[peer_id]["correct"].append(0)
                trust_buffers[peer_id]["latency_ok"].append(0)
                continue

            elected = peer_id
            response = resp
            elapsed = resp["elapsed_ms"]
            score_val = float(resp["score"].item()) if resp.get("score") is not None else 0.5
            latency_ok = int(elapsed <= latency_ok_threshold_ms)
            is_correct = int(resp["logit"].argmax().item() == target_label)

            trust_buffers[peer_id]["scores"].append(score_val)
            trust_buffers[peer_id]["latency_ok"].append(latency_ok)
            trust_buffers[peer_id]["correct"].append(is_correct)
            break

        if response is None:
            return None, elected_chain

        elected_chain.append(elected)
        if response.get("should_exit") or stage_idx == NUM_STAGES - 1:
            return int(response["logit"].argmax().item()), elected_chain
        feat = response["feat"]
        new_score = response.get("score")      
        if new_score is not None:
            if past_scores is None:
                past_scores = new_score.unsqueeze(1)            
            else:
                past_scores = torch.cat(
                    [past_scores, new_score.unsqueeze(1)], dim=1
                )

    return None, elected_chain

def main() -> dict:
    parser = argparse.ArgumentParser(description="PRISM P2P router")
    parser.add_argument("--policy", default="trust", choices=["trust", "random"])
    parser.add_argument("--scenario", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--timeout-ms", type=int, default=2000,
                        help="ZMQ receive timeout in ms (safety net for truly dead peers)")
    parser.add_argument("--trust-update-window", type=int, default=32,
                        help="Flush trust buffers every N samples")
    parser.add_argument("--max-fallbacks", type=int, default=1,
                        help="Max peers to try per stage before dropping")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--seconds-csv",
        type=str,
        default="outputs/densenet121_4_None_cifar100/seconds.csv",
    )
    args = parser.parse_args()

    seconds: list[float] = []
    with open(args.seconds_csv) as fh:
        for line in fh:
            line = line.strip()
            if line:
                seconds.append(float(line))
    print(f"[router] stage latencies (ms): {seconds}")

    transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    test_dataset = torchvision.datasets.CIFAR100(
        "datasets", train=False, download=False, transform=transform
    )
    n = min(args.n_samples, len(test_dataset))
    subset = torch.utils.data.Subset(test_dataset, list(range(n)))
    test_loader = torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=0
    )
    print(
        f"[router] policy={args.policy}  scenario={args.scenario}  "
        f"n_samples={n}  timeout={args.timeout_ms}ms"
    )

    ctx = zmq.Context()
    sockets = make_all_sockets(ctx, args.timeout_ms)
    print("[router] connected to 16 peer sockets")
    tracker = EigenTrustTracker(n_peers=16, alpha=0.1, trust_scale=0.2, decay=0.85)
    rng = np.random.default_rng(args.seed)

    trust_buffers: dict = defaultdict(lambda: {"scores": [], "correct": [], "latency_ok": []})
    trust_trace: list = []

    correct_total = 0
    dropped_total = 0
    completed = 0
    total_latency_ms = 0.0
    exit_counts = [0] * NUM_STAGES
    request_id = 0
    t_run_start = time.perf_counter()

    for sample_idx, (img, label) in enumerate(test_loader):
        target = int(label.item())
        t_sample = time.perf_counter()

        pred, elected_chain = route_inference(
            image_tensor=img,
            target_label=target,
            tracker=tracker,
            policy=args.policy,
            sockets=sockets,
            ctx=ctx,
            seconds=seconds,
            rng=rng,
            request_id=request_id,
            trust_buffers=trust_buffers,
            timeout_ms=args.timeout_ms,
            max_fallbacks=args.max_fallbacks,
        )
        request_id += 1
        sample_ms = (time.perf_counter() - t_sample) * 1000
        total_latency_ms += sample_ms

        if pred is None:
            dropped_total += 1
        else:
            completed += 1
            if pred == target:
                correct_total += 1
            exit_depth = len(elected_chain) - 1
            if 0 <= exit_depth < NUM_STAGES:
                exit_counts[exit_depth] += 1
        if (sample_idx + 1) % args.trust_update_window == 0:
            flush_trust_buffers(trust_buffers, tracker)
            trust_trace.append({
                "sample": sample_idx + 1,
                "trust": tracker.trust.tolist(),
            })
            acc_pct = 100.0 * correct_total / max(completed, 1)
            drop_pct = 100.0 * dropped_total / (sample_idx + 1)
            print(
                f"[router] [{sample_idx + 1:>5}/{n}]  "
                f"acc={acc_pct:.1f}%  drop={drop_pct:.1f}%  "
                f"last_chain={elected_chain}",
                flush=True,
            )
    flush_trust_buffers(trust_buffers, tracker)
    trust_trace.append({"sample": n, "trust": tracker.trust.tolist()})

    run_secs = time.perf_counter() - t_run_start
    total_f = float(n)
    results = {
        "policy": args.policy,
        "scenario": args.scenario,
        "n_samples": n,
        "accuracy": 100.0 * correct_total / total_f,
        "accuracy_on_completed": 100.0 * correct_total / max(completed, 1),
        "reliability": 100.0 * completed / total_f,
        "dropped_responses": 100.0 * dropped_total / total_f,
        "avg_latency_ms": total_latency_ms / total_f,
        "exit_distribution": [c / max(n, 1) for c in exit_counts],
        "trust_vector": tracker.trust.tolist(),
        "peer_summary": tracker.peer_summary(),
        "trust_trace": trust_trace,
        "run_seconds": run_secs,
    }

    print(f"\n[router] ── DONE ───────────────────────────────────────────")
    print(f"  policy        {args.policy}")
    print(f"  scenario      {args.scenario}")
    print(f"  accuracy      {results['accuracy']:.2f}%")
    print(f"  reliability   {results['reliability']:.2f}%")
    print(f"  dropped       {results['dropped_responses']:.2f}%")
    print(f"  avg_latency   {results['avg_latency_ms']:.2f} ms/sample")
    dist_str = "  ".join(f"e{i+1}={c*100:.1f}%" for i, c in enumerate(results["exit_distribution"]))
    print(f"  exit dist     {dist_str}")
    print(f"  wall time     {run_secs:.1f}s")
    print("\n[router] Trust vector (should diverge in medium/hard scenarios):")
    for stage_idx in range(NUM_STAGES):
        peer_ids = _PEERS_BY_STAGE[stage_idx]
        vals = [f"p{pid}={tracker.trust[pid]:.4f}" for pid in peer_ids]
        print(f"  stage {stage_idx}: {' '.join(vals)}")

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n[router] results → {args.output_json}")

    for sock in sockets.values():
        sock.close()
    ctx.term()

    return results


if __name__ == "__main__":
    main()
