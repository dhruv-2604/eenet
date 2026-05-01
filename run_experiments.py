#!/usr/bin/env python3
"""
launch peers, run experiments, log results.

Usage:
    python3 run_experiments.py --n-samples 5000 --seeds 0,1,2
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PYTHON = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))

SCENARIOS = ["easy", "medium", "hard"]
POLICIES = ["random", "trust"]


def _run(cmd: list, **kwargs) -> int:
    # show command
    label = " ".join(os.path.basename(c) if c.endswith(".py") else c for c in cmd)
    print(f"\n$ {label}")
    result = subprocess.run(cmd, cwd=ROOT, **kwargs)
    return result.returncode


def launch_network(scenario: str, seed: int, pid_file: str, log_dir: str, wait_secs: float) -> int:
    return _run([
        PYTHON, os.path.join(ROOT, "launch_network.py"),
        "--scenario", scenario,
        "--seed", str(seed),
        "--pid-file", pid_file,
        "--log-dir", log_dir,
        "--wait-secs", str(wait_secs),
    ])


def stop_network(pid_file: str) -> None:
    if os.path.exists(pid_file):
        _run([PYTHON, os.path.join(ROOT, "stop_network.py"), "--pid-file", pid_file])


def run_router(policy: str, scenario: str, n_samples: int, output_json: str, seed: int,
               trust_exit_adjustment: float = 0.1) -> int:
    cmd = [
        PYTHON, os.path.join(ROOT, "router.py"),
        "--policy", policy,
        "--scenario", scenario,
        "--n-samples", str(n_samples),
        "--output-json", output_json,
        "--seed", str(seed),
        "--trust-exit-adjustment", str(trust_exit_adjustment),
    ]
    return _run(cmd)


def _mean_std(values: list) -> tuple:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance)


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM distributed experiment suite")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Test images to route per (scenario, policy, seed) run")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated seed list, e.g. '0,1,2'")
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIOS),
                        help="Comma-separated scenarios to run: easy,medium,hard")
    parser.add_argument("--policies", type=str, default=",".join(POLICIES),
                        help="Comma-separated policies to run: random,trust")
    parser.add_argument("--results-dir", default="outputs/results")
    parser.add_argument("--pid-file", default="network_pids.json")
    parser.add_argument("--log-dir", default="network_logs")
    parser.add_argument("--wait-secs", type=float, default=18.0,
                        help="Seconds to wait for peers to start (MPS warmup needs ~15s)")
    parser.add_argument("--trust-exit-adjustment", type=float, default=0.1,
                        help="Trust-adjustment factor for exit thresholds (0.0=off, 0.1=recommended)")
    args = parser.parse_args()

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    scenario_list = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    policy_list = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown_scenarios = sorted(set(scenario_list) - set(SCENARIOS))
    unknown_policies = sorted(set(policy_list) - set(POLICIES))
    if unknown_scenarios:
        parser.error("Unknown scenarios: {0}".format(", ".join(unknown_scenarios)))
    if unknown_policies:
        parser.error("Unknown policies: {0}".format(", ".join(unknown_policies)))
    os.makedirs(args.results_dir, exist_ok=True)
    # print(f"[debug] scenarios={scenario_list} policies={policy_list} seeds={seed_list}")
    # print(f"[debug] results dir={args.results_dir}")
    per_seed_results: dict = {(sc, po): [] for sc in scenario_list for po in policy_list}

    for scenario in scenario_list:
        for seed in seed_list:
            print(f"\n{'='*65}")
            print(f"  SCENARIO={scenario.upper()}  SEED={seed}")
            print(f"{'='*65}")

            for policy in policy_list:
                # restart network
                stop_network(args.pid_file)
                time.sleep(1.0)

                rc = launch_network(
                    scenario=scenario,
                    seed=seed,
                    pid_file=args.pid_file,
                    log_dir=args.log_dir,
                    wait_secs=args.wait_secs,
                )
                if rc != 0:
                    print(
                        f"[orchestrator] launch_network failed for policy={policy} "
                        f"(rc={rc}) — skipping"
                    )
                    continue

                out_json = os.path.join(
                    args.results_dir,
                    f"{scenario}_{policy}_seed{seed}.json",
                )
                print(f"\n── policy={policy} ──────────────────────────────────────────")
                run_router(
                    policy=policy,
                    scenario=scenario,
                    n_samples=args.n_samples,
                    output_json=out_json,
                    seed=seed,
                    trust_exit_adjustment=args.trust_exit_adjustment,
                )
                if os.path.exists(out_json):
                    # collect result
                    with open(out_json) as fh:
                        per_seed_results[(scenario, policy)].append(json.load(fh))

                stop_network(args.pid_file)
                time.sleep(1.0)

            stop_network(args.pid_file)
            time.sleep(2.0)
    metrics = ["accuracy", "reliability", "dropped_responses", "avg_latency_ms"]
    aggregated = []
    for scenario in scenario_list:
        for policy in policy_list:
            runs = per_seed_results[(scenario, policy)]
            if not runs:
                continue
            entry = {"scenario": scenario, "policy": policy, "n_seeds": len(runs)}
            for m in metrics:
                vals = [r[m] for r in runs if m in r]
                mean, std = _mean_std(vals)
                entry[f"{m}_mean"] = mean
                entry[f"{m}_std"] = std
            entry["per_seed"] = runs
            aggregated.append(entry)

    if not aggregated:
        print("\n[orchestrator] No results collected.")
        return

    agg_path = os.path.join(args.results_dir, "aggregated_results.json")
    with open(agg_path, "w") as fh:
        json.dump(aggregated, fh, indent=2)
    print(f"\n[orchestrator] aggregated results → {agg_path}")

    n_seeds = len(seed_list)
    n_samples = args.n_samples
    col_w = 22
    print(f"\n{'='*72}")
    print(f"  DISTRIBUTED EXPERIMENT RESULTS ({n_samples} samples × {n_seeds} seeds)")
    print(f"{'='*72}")
    header = (
        f"{'Scenario':<10} | {'Random (mean±std)':<{col_w}} | "
        f"{'Trust (mean±std)':<{col_w}} | {'Trust gain'}"
    )
    print(header)
    print("-" * len(header))

    for scenario in scenario_list:
        rand_entry = next(
            (e for e in aggregated if e["scenario"] == scenario and e["policy"] == "random"), None
        )
        trust_entry = next(
            (e for e in aggregated if e["scenario"] == scenario and e["policy"] == "trust"), None
        )
        if rand_entry is None or trust_entry is None:
            continue
        rand_str = f"{rand_entry['accuracy_mean']:.1f}% ± {rand_entry['accuracy_std']:.1f}%"
        trust_str = f"{trust_entry['accuracy_mean']:.1f}% ± {trust_entry['accuracy_std']:.1f}%"
        gain = trust_entry["accuracy_mean"] - rand_entry["accuracy_mean"]
        print(
            f"{scenario.capitalize():<10} | {rand_str:<{col_w}} | "
            f"{trust_str:<{col_w}} | {gain:+.1f}%"
        )

    if args.trust_exit_adjustment > 0.0:
        print(f"\n{'='*72}")
        print(f"  Trust exit overrides (adjustment={args.trust_exit_adjustment})")
        print(f"{'='*72}")
        print(f"{'Scenario':<10} | {'Policy':<8} | {'Total overrides':>16} | {'Per-seed':>10}")
        print("-" * 55)
        for entry in aggregated:
            overrides = [r.get("trust_override_count", 0) for r in entry["per_seed"]]
            total = sum(overrides)
            per_seed_str = ", ".join(str(v) for v in overrides)
            print(f"{entry['scenario'].capitalize():<10} | {entry['policy']:<8} | {total:>16} | {per_seed_str}")

    print(f"\n[orchestrator] all results → {args.results_dir}/")


if __name__ == "__main__":
    main()
