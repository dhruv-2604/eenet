#!/usr/bin/env python3
"""
Spawn 16 PRISM peer processes (4 stages × 4 peers).

We use build_peer_profiles() to build peer profiles.
stop_network used to stop the network by killing all PIDs in the json we write to.
Usage:
    python launch_network.py --scenario medium --seed 0
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p2p_config import STAGE_PORTS, SEGMENT_PATHS, PEERS_PER_STAGE
from trust.routing import build_peer_profiles

PYTHON = sys.executable  


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch PRISM P2P network")
    parser.add_argument("--scenario", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pid-file", default="p2p_pids.json")
    parser.add_argument("--log-dir", default="p2p_logs")
    parser.add_argument("--wait-secs", type=float, default=18.0,
                        help="Seconds to wait for peers to bind before returning")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    root = os.path.dirname(os.path.abspath(__file__))

    profiles = build_peer_profiles(
        num_stages=4,
        replicas_per_stage=PEERS_PER_STAGE,
        scenario=args.scenario,
        seed=args.seed,
    )

    pids: list[int] = []
    for profile in profiles:
        stage_idx = profile.stage_idx
        replica_idx = profile.peer_id % PEERS_PER_STAGE
        port = STAGE_PORTS[stage_idx][replica_idx]
        segment_path = os.path.join(root, SEGMENT_PATHS[stage_idx])
        log_path = os.path.join(args.log_dir, f"peer_{profile.peer_id}.log")

        cmd = [
            PYTHON,
            os.path.join(root, "peer.py"),
            "--peer-id", str(profile.peer_id),
            "--stage-idx", str(stage_idx),
            "--port", str(port),
            "--segment-path", segment_path,
            "--drop-prob", str(profile.drop_prob),
            "--corrupt-prob", str(profile.corrupt_prob),
            "--spike-prob", str(profile.spike_prob),
            "--spike-scale", str(profile.spike_scale),
            "--latency-multiplier", str(profile.latency_multiplier),
        ]
        if profile.faulty:
            cmd.append("--faulty")

        log_fh = open(log_path, "w")
        proc = subprocess.Popen(cmd, cwd=root, stdout=log_fh, stderr=subprocess.STDOUT)
        pids.append(proc.pid)
        print(
            f"  peer {profile.peer_id:>2}  stage={stage_idx}  port={port}  "
            f"faulty={profile.faulty}  pid={proc.pid}"
        )

    with open(args.pid_file, "w") as fh:
        json.dump(
            {"pids": pids, "scenario": args.scenario, "seed": args.seed},
            fh,
            indent=2,
        )

    print(f"\nPIDs → {args.pid_file}")
    print(f"Waiting {args.wait_secs:.0f}s for all peers to load segments and bind…")
    time.sleep(args.wait_secs)
    print("Network ready.\n")


if __name__ == "__main__":
    main()
