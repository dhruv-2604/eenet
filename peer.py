#!/usr/bin/env python3
"""
runs our distributed peer processes
Loads segment, binds a ZMQ socket, and request servicing from router.
also included real fault injection scenarios for drop, latency spike, and output corruption.
Usage:
    python3 peer.py --peer-id 0 --stage-idx 0 --port 5551 \
        --segment-path outputs/segments/segment1.pt \
        --drop-prob 0.0 --corrupt-prob 0.0 \
        --spike-prob 0.02 --spike-scale 1.15 \
        --latency-multiplier 1.0
"""

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_prism_experiments import Segment1, Segment2, Segment3, Segment4 

import numpy as np
import torch
import zmq


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM peer process")
    parser.add_argument("--peer-id", type=int, required=True)
    parser.add_argument("--stage-idx", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--segment-path", type=str, required=True)
    parser.add_argument("--drop-prob", type=float, default=0.0)
    parser.add_argument("--corrupt-prob", type=float, default=0.0)
    parser.add_argument("--spike-prob", type=float, default=0.02)
    parser.add_argument("--spike-scale", type=float, default=1.15)
    parser.add_argument("--latency-multiplier", type=float, default=1.0)
    parser.add_argument("--faulty", action="store_true")
    parser.add_argument("--device", type=str, default="auto",
                        help="Inference device: auto | cpu | mps | cuda")
    args = parser.parse_args()

    # Resolve inference device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    tag = f"[peer-{args.peer_id}]"
    print(f"{tag} loading {args.segment_path} → device={device}", flush=True)
    segment = torch.load(args.segment_path, map_location="cpu", weights_only=False)
    segment = segment.to(device)
    segment.eval()

    # warm up sim
    _WARMUP_SHAPES = {
        0: (1, 3, 32, 32),
        1: (1, 48, 16, 16),
        2: (1, 96, 8, 8),
        3: (1, 192, 4, 4),
    }
  
    _PAST_SCORE_SIZES = {0: None, 1: 1, 2: 2, 3: None}
    with torch.no_grad():
        dummy = torch.zeros(*_WARMUP_SHAPES[args.stage_idx]).to(device)
        if args.stage_idx == 3:
            segment(dummy)
        else:
            feat_d, logit_d = segment(dummy)
            past_size = _PAST_SCORE_SIZES[args.stage_idx]
            ps_d = (torch.zeros(1, past_size).to(device)
                    if past_size is not None else None)
            segment.compute_score(logit_d, past_scores=ps_d)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://*:{args.port}")
    print(
        f"{tag} ready  port={args.port}  stage={args.stage_idx}  "
        f"faulty={args.faulty}  device={device}  "
        f"drop={args.drop_prob:.2f}  corrupt={args.corrupt_prob:.2f}  "
        f"spike_prob={args.spike_prob:.2f}",
        flush=True,
    )

    rng = np.random.default_rng(args.peer_id * 1000 + 7)

    while True:
        try:
            raw = sock.recv()
        except zmq.ZMQError as exc:
            print(f"{tag} recv error: {exc}", flush=True)
            break

        msg = pickle.loads(raw)
        req_id = msg["request_id"]
        t0 = time.perf_counter()

        # drop scneario
        
        if rng.random() < args.drop_prob:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"{tag} req={req_id} elapsed={elapsed_ms:.1f}ms status=dropped", flush=True)
            try:
                sock.send(pickle.dumps({
                    "request_id": req_id,
                    "peer_id": args.peer_id,
                    "status": "dropped",
                    "elapsed_ms": elapsed_ms,
                }))
            except zmq.ZMQError:
                pass
            continue

        # latency spike scneario
        if rng.random() < args.spike_prob:
            sleep_s = args.latency_multiplier * args.spike_scale * 0.008
            time.sleep(sleep_s)


        feat_in = msg["feat"].to(device)
        raw_past = msg.get("past_scores")
        past_scores = raw_past.to(device) if raw_past is not None else None
        is_corrupted = False

        with torch.no_grad():
            if args.stage_idx == 3:
                logit = segment(feat_in).cpu()
                feat_out = None
                score = None
                should_exit = True
            else:
                feat_out, logit = segment(feat_in)
                score = segment.compute_score(logit, past_scores=past_scores)
                should_exit = bool(segment.should_exit(score))
                feat_out = feat_out.cpu()
                logit = logit.cpu()
                score = score.cpu()

        # corrupt scneario
        if rng.random() < args.corrupt_prob:
            is_corrupted = True
            with torch.no_grad():
                sorted_idx = logit.argsort(dim=1, descending=True)
                wrong_class = sorted_idx[:, 1:2]           # (B, 1)
                logit_bad = torch.full_like(logit, 1e-6)
                logit_bad.scatter_(1, wrong_class, 1.0 - (logit.shape[1] - 1) * 1e-6)
                logit = logit_bad
            should_exit = True   

        elapsed_ms = (time.perf_counter() - t0) * 1000
        status = "corrupted" if is_corrupted else "ok"
        print(f"{tag} req={req_id} elapsed={elapsed_ms:.1f}ms status={status}", flush=True)

        response = {
            "request_id": req_id,
            "peer_id": args.peer_id,
            "feat": feat_out,
            "logit": logit,
            "score": score,
            "should_exit": should_exit,
            "elapsed_ms": elapsed_ms,
            "status": status,
        }
        try:
            sock.send(pickle.dumps(response))
        except zmq.ZMQError as exc:
            print(f"{tag} send error (router may have moved on): {exc}", flush=True)


if __name__ == "__main__":
    main()
