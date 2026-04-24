#!/usr/bin/env python3
"""
Generate report-ready comparison tables and trust visualizations.

This script intentionally reads the checked-in experiment outputs instead of
rerunning training or the P2P network. It creates a compact "centralized vs.
distributed" table and trust-over-time plots from the 16-node P2P JSON traces.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PRISM_DIR = Path("outputs/prism_experiments_full")
DEFAULT_P2P_DIR = Path("outputs/16-node-p2p")
DEFAULT_OUT_DIR = Path("outputs/report_assets")
DEFAULT_SCHEDULER_BUDGET = 6.5
NUM_STAGES = 4
PEERS_PER_STAGE = 4


def _read_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


def _nearest_budget_row(budget_df: pd.DataFrame, method: str, budget: float) -> pd.Series:
    frame = budget_df[budget_df["method"] == method].copy()
    if frame.empty:
        raise ValueError(f"No rows for method={method!r} in budget sweep")
    frame["budget_delta"] = (frame["budget_ms"] - budget).abs()
    return frame.sort_values(["budget_delta", "budget_ms"]).iloc[0]


def build_comparison_table(prism_dir: Path, p2p_dir: Path, scheduler_budget: float) -> pd.DataFrame:
    exit_df = pd.read_csv(prism_dir / "exit_metrics.csv")
    budget_df = pd.read_csv(prism_dir / "budget_sweep.csv")
    p2p_summary = _read_json(p2p_dir / "aggregated_results.json")

    final_exit = exit_df.sort_values("exit").iloc[-1]
    eenet_row = _nearest_budget_row(budget_df, "EENet", scheduler_budget)

    rows = [
        {
            "system": "Centralized full model",
            "scenario": "clean",
            "policy": "single process",
            "sample_count": 10000,
            "accuracy_pct": float(final_exit["accuracy_top1"]),
            "reliability_pct": 100.0,
            "dropped_pct": 0.0,
            "avg_latency_ms": float(final_exit["latency_ms"]),
            "latency_scope": "model-only local",
            "source": "exit_metrics.csv final exit",
            "comparison_note": "Full CIFAR-100 test set; model-only timing.",
        },
        {
            "system": "Centralized EENet early exit",
            "scenario": "clean",
            "policy": f"EENet budget {float(eenet_row['budget_ms']):.2f} ms",
            "sample_count": 10000,
            "accuracy_pct": float(eenet_row["accuracy"]),
            "reliability_pct": 100.0,
            "dropped_pct": 0.0,
            "avg_latency_ms": float(eenet_row["avg_latency_ms"]),
            "latency_scope": "model-only local",
            "source": "budget_sweep.csv",
            "comparison_note": "Full CIFAR-100 test set; model-only timing.",
        },
    ]

    for entry in p2p_summary:
        rows.append({
            "system": "Distributed 16-node segments",
            "scenario": str(entry["scenario"]),
            "policy": str(entry["policy"]),
            "sample_count": int(entry["per_seed"][0].get("n_samples", 0)) if entry.get("per_seed") else 0,
            "accuracy_pct": float(entry["accuracy_mean"]),
            "reliability_pct": float(entry["reliability_mean"]),
            "dropped_pct": float(entry["dropped_responses_mean"]),
            "avg_latency_ms": float(entry["avg_latency_ms_mean"]),
            "latency_scope": "wall-clock routed",
            "source": "outputs/16-node-p2p/aggregated_results.json",
            "comparison_note": "P2P routed run; wall-clock includes process/ZMQ overhead.",
        })

    table = pd.DataFrame(rows)
    return table[
        [
            "system",
            "scenario",
            "policy",
            "sample_count",
            "accuracy_pct",
            "reliability_pct",
            "dropped_pct",
            "avg_latency_ms",
            "latency_scope",
            "source",
            "comparison_note",
        ]
    ]


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    display = df.copy()
    for col in ["accuracy_pct", "reliability_pct", "dropped_pct", "avg_latency_ms"]:
        display[col] = display[col].map(lambda value: f"{value:.2f}")
    lines = [
        display.to_markdown(index=False),
        "",
        "Note: centralized rows use full-test model-only metrics; distributed rows use the checked-in routed P2P runs, so latency scopes differ.",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_comparison_table(df: pd.DataFrame, figure_dir: Path) -> None:
    p2p = df[df["system"] == "Distributed 16-node segments"].copy()
    scenarios = ["easy", "medium", "hard"]
    x = np.arange(len(scenarios))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    metric_specs = [
        ("accuracy_pct", "Accuracy (%)", "Accuracy"),
        ("reliability_pct", "Reliability (%)", "Completed Requests"),
        ("dropped_pct", "Dropped (%)", "Dropped Responses"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, metric_specs):
        random_vals = [
            p2p[(p2p["scenario"] == scenario) & (p2p["policy"] == "random")][metric].iloc[0]
            for scenario in scenarios
        ]
        trust_vals = [
            p2p[(p2p["scenario"] == scenario) & (p2p["policy"] == "trust")][metric].iloc[0]
            for scenario in scenarios
        ]
        ax.bar(x - width / 2, random_vals, width=width, color="#9ecae1", label="Random")
        ax.bar(x + width / 2, trust_vals, width=width, color="#3182bd", label="Trust")
        ax.set_xticks(x)
        ax.set_xticklabels([scenario.title() for scenario in scenarios])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(bottom=0)
    axes[0].legend()

    fig.suptitle("Distributed Routing: Random vs Trust Across Fault Scenarios", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "distributed_random_vs_trust.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    central = df[df["system"] != "Distributed 16-node segments"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(central["system"], central["accuracy_pct"], color=["#636363", "#31a354"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Centralized Baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figure_dir / "centralized_baselines.png", dpi=180)
    plt.close(fig)


def _trust_trace_from_result(result: dict) -> tuple[list[int], np.ndarray]:
    traces = result.get("trust_trace", [])
    if not traces:
        return [], np.empty((0, 0))

    if isinstance(traces[0], dict):
        samples = [int(item["sample"]) for item in traces]
        values = np.asarray([item["trust"] for item in traces], dtype=float)
    else:
        samples = list(range(1, len(traces) + 1))
        values = np.asarray(traces, dtype=float)
    return samples, values


def _iter_p2p_result_files(p2p_dir: Path, policy: str = "trust") -> Iterable[Path]:
    return sorted(p2p_dir.glob(f"*_{policy}_seed*.json"))


def plot_trust_traces(p2p_dir: Path, figure_dir: Path) -> None:
    stage_colors = ["#2c7fb8", "#31a354", "#756bb1", "#d95f0e"]

    for path in _iter_p2p_result_files(p2p_dir, policy="trust"):
        result = _read_json(path)
        samples, values = _trust_trace_from_result(result)
        if len(samples) == 0 or values.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        for peer_id in range(values.shape[1]):
            stage = peer_id // PEERS_PER_STAGE
            replica = peer_id % PEERS_PER_STAGE
            ax.plot(
                samples,
                values[:, peer_id],
                color=stage_colors[stage],
                alpha=0.45 + 0.12 * replica,
                linewidth=1.8,
                label=f"stage {stage + 1}" if replica == 0 else None,
            )

        ax.set_xlabel("Samples processed")
        ax.set_ylabel("Trust score")
        ax.set_title(
            "Trust Evolution: {0} Scenario".format(str(result["scenario"]).title())
        )
        ax.grid(alpha=0.25)
        ax.legend(title="Peer stage", ncol=2)
        fig.tight_layout()
        fig.savefig(figure_dir / f"trust_trace_{result['scenario']}_seed{result.get('seed', 0)}.png", dpi=180)
        plt.close(fig)


def plot_final_trust_heatmap(p2p_dir: Path, figure_dir: Path) -> None:
    rows = []
    labels = []
    for path in _iter_p2p_result_files(p2p_dir, policy="trust"):
        result = _read_json(path)
        rows.append(result["trust_vector"])
        labels.append(str(result["scenario"]).title())
    if not rows:
        return

    values = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(NUM_STAGES * PEERS_PER_STAGE))
    ax.set_xticklabels([f"p{i}" for i in range(NUM_STAGES * PEERS_PER_STAGE)], rotation=45)
    for boundary in range(PEERS_PER_STAGE, NUM_STAGES * PEERS_PER_STAGE, PEERS_PER_STAGE):
        ax.axvline(boundary - 0.5, color="white", linewidth=1.5)
    ax.set_title("Final Trust by Peer After Trust Routing")
    ax.set_xlabel("Peer ID, grouped by stage")
    fig.colorbar(im, ax=ax, label="Trust score")
    fig.tight_layout()
    fig.savefig(figure_dir / "final_trust_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PRISM report assets.")
    parser.add_argument("--prism-dir", type=Path, default=DEFAULT_PRISM_DIR)
    parser.add_argument("--p2p-dir", type=Path, default=DEFAULT_P2P_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scheduler-budget", type=float, default=DEFAULT_SCHEDULER_BUDGET)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison_table(args.prism_dir, args.p2p_dir, args.scheduler_budget)
    comparison.to_csv(args.out_dir / "centralized_vs_distributed.csv", index=False)
    write_markdown_table(comparison, args.out_dir / "centralized_vs_distributed.md")
    plot_comparison_table(comparison, figure_dir)
    plot_trust_traces(args.p2p_dir, figure_dir)
    plot_final_trust_heatmap(args.p2p_dir, figure_dir)

    print(f"Wrote comparison table to {args.out_dir / 'centralized_vs_distributed.csv'}")
    print(f"Wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()
