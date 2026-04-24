#!/usr/bin/env python3
"""
Kill all PRISM peer processes recorded in a PID file.
Sends SIGTERM first, waits briefly, then SIGKILL any survivors.

Usage:
    python stop_network.py [--pid-file p2p_pids.json]
"""

import argparse
import json
import os
import signal
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop PRISM P2P network")
    parser.add_argument("--pid-file", default="p2p_pids.json")
    parser.add_argument("--wait-secs", type=float, default=2.0)
    args = parser.parse_args()

    if not os.path.exists(args.pid_file):
        print(f"No PID file at {args.pid_file} — nothing to stop.")
        return

    with open(args.pid_file) as fh:
        data = json.load(fh)
    pids: list[int] = data["pids"] if isinstance(data, dict) else data

    killed = 0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            pass   # already gone
        except PermissionError as exc:
            print(f"  permission denied for PID {pid}: {exc}")

    print(f"Sent SIGTERM to {killed} processes. Waiting {args.wait_secs:.0f}s…")
    time.sleep(args.wait_secs)
    survivors = 0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            survivors += 1
        except ProcessLookupError:
            pass

    if survivors:
        print(f"SIGKILL sent to {survivors} surviving processes.")

    try:
        os.remove(args.pid_file)
    except FileNotFoundError:
        pass

    print("Network stopped.")


if __name__ == "__main__":
    main()
