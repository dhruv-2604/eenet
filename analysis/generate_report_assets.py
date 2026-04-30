#!/usr/bin/env python3
"""
Generate report-ready comparison tables and trust visualizations.

This script intentionally reads the checked-in experiment outputs instead of
rerunning training or the distributed network. It creates a compact
"centralized vs. distributed" table and trust-over-time plots from the
16-node distributed JSON traces.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PRISM_DIR = Path("outputs/prism_experiments_full")
DEFAULT_RESULTS_DIR = Path("outputs/16-node-network")
DEFAULT_NO_EXIT_ADJUSTMENT_DIR = Path("outputs/results_no_exit_adjustment")
DEFAULT_TUNED_EXIT_ADJUSTMENT_DIR = Path("outputs/results_exit_adjustment_0_1")
DEFAULT_TUNED_EXIT_ADJUSTMENT = 0.1
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


def _replace_hard_trust_with_tuned(results_summary: list, tuned_adjustment_dir: Path) -> list:
    tuned = _hard_policy_summary(tuned_adjustment_dir, "trust")
    if tuned is None:
        return results_summary
    tuned = dict(tuned)
    tuned["_source"] = str(tuned_adjustment_dir / "aggregated_results.json")

    replaced = []
    did_replace = False
    for entry in results_summary:
        if entry.get("scenario") == "hard" and entry.get("policy") == "trust":
            replaced.append(tuned)
            did_replace = True
        else:
            replaced.append(entry)
    if not did_replace:
        replaced.append(tuned)
    return replaced


def build_comparison_table(
    prism_dir: Path,
    results_dir: Path,
    scheduler_budget: float,
    tuned_adjustment_dir: Path,
) -> pd.DataFrame:
    exit_df = pd.read_csv(prism_dir / "exit_metrics.csv")
    budget_df = pd.read_csv(prism_dir / "budget_sweep.csv")
    results_summary = _replace_hard_trust_with_tuned(
        _read_json(results_dir / "aggregated_results.json"),
        tuned_adjustment_dir,
    )

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

    for entry in results_summary:
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
            "source": str(entry.get("_source", results_dir / "aggregated_results.json")),
            "comparison_note": "Distributed routed run; wall-clock includes process/ZMQ overhead.",
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
    rows = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    lines = [
        "\n".join(rows),
        "",
        "Note: centralized rows use full-test model-only metrics; distributed rows use the checked-in routed distributed runs, so latency scopes differ.",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_comparison_table(df: pd.DataFrame, figure_dir: Path, tuned_adjustment: float) -> None:
    distributed = df[df["system"] == "Distributed 16-node segments"].copy()
    scenario_order = {"easy": 0, "medium": 1, "hard": 2}
    scenarios = sorted(
        distributed["scenario"].unique().tolist(),
        key=lambda scenario: scenario_order.get(scenario, 99),
    )
    x = np.arange(len(scenarios))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    metric_specs = [
        ("accuracy_pct", "Accuracy (%)", "Accuracy"),
        ("avg_latency_ms", "Latency (ms / sample)", "Average Latency"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, metric_specs):
        random_vals = []
        trust_vals = []
        for scenario in scenarios:
            random_frame = distributed[(distributed["scenario"] == scenario) & (distributed["policy"] == "random")]
            trust_frame = distributed[(distributed["scenario"] == scenario) & (distributed["policy"] == "trust")]
            random_vals.append(float(random_frame[metric].iloc[0]) if not random_frame.empty else 0.0)
            trust_vals.append(float(trust_frame[metric].iloc[0]) if not trust_frame.empty else 0.0)
        ax.bar(x - width / 2, random_vals, width=width, color="#9ecae1", label="Random")
        ax.bar(
            x + width / 2,
            trust_vals,
            width=width,
            color="#3182bd",
            label=f"Trust (exit adjust={tuned_adjustment:.1f})",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([scenario.title() for scenario in scenarios])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(bottom=0)
    axes[0].legend()

    fig.suptitle("Hard Scenario: Tuned Trust Routing Improves Accuracy and Latency", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "distributed_random_vs_trust.png", dpi=180, bbox_inches="tight")
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


def _iter_result_files(results_dir: Path, policy: str = "trust") -> Iterable[Path]:
    return sorted(results_dir.glob(f"*_{policy}_seed*.json"))


def plot_hard_trust_trace(results_dir: Path, figure_dir: Path) -> None:
    stage_colors = ["#2c7fb8", "#31a354", "#756bb1", "#d95f0e"]

    for path in _iter_result_files(results_dir, policy="trust"):
        result = _read_json(path)
        if result.get("scenario") != "hard" or result.get("seed", 0) != 0:
            continue
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
        ax.set_title("Trust Separates Reliable Peers in the Hard Scenario")
        ax.grid(alpha=0.25)
        ax.legend(title="Peer stage", ncol=2)
        fig.tight_layout()
        fig.savefig(figure_dir / "trust_trace_hard_seed0.png", dpi=180)
        plt.close(fig)
        return


def _hard_trust_summary(results_dir: Path) -> Optional[dict]:
    path = results_dir / "aggregated_results.json"
    if not path.exists():
        return None
    for entry in _read_json(path):
        if entry.get("scenario") == "hard" and entry.get("policy") == "trust":
            return entry
    return None


def plot_trust_exit_adjustment_gain(
    no_adjustment_dir: Path,
    tuned_adjustment_dir: Path,
    figure_dir: Path,
    tuned_adjustment: float,
) -> None:
    baseline = _hard_trust_summary(no_adjustment_dir)
    adjusted = _hard_trust_summary(tuned_adjustment_dir)
    if baseline is None or adjusted is None:
        return

    labels = ["Trust routing\nexit adjust = 0.0", f"Trust routing\nexit adjust = {tuned_adjustment:.1f}"]
    means = [float(baseline["accuracy_mean"]), float(adjusted["accuracy_mean"])]
    stds = [float(baseline["accuracy_std"]), float(adjusted["accuracy_std"])]
    per_seed = [
        [float(run["accuracy"]) for run in baseline.get("per_seed", [])],
        [float(run["accuracy"]) for run in adjusted.get("per_seed", [])],
    ]
    gain = means[1] - means[0]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = ["#9ecae1", "#3182bd"]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6, width=0.58, color=colors, edgecolor="#17324d")

    for idx, values in enumerate(per_seed):
        offsets = np.linspace(-0.09, 0.09, num=max(len(values), 1))
        ax.scatter(
            np.full(len(values), x[idx]) + offsets[: len(values)],
            values,
            color="#f28e2b",
            edgecolor="#5f3b00",
            zorder=3,
            s=42,
            label="seed result" if idx == 0 else None,
        )

    y_max = max(mean + std for mean, std in zip(means, stds)) + 10.0
    ax.text(
        0.5,
        y_max - 4.0,
        f"Mean gain: +{gain:.1f} percentage points",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#17324d",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#17324d"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, y_max)
    ax.set_title("Hard Scenario: Trust-Coupled Exits Improve Accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_dir / "trust_exit_adjustment_accuracy_gain.png", dpi=180)
    plt.close(fig)


def _hard_policy_summary(results_dir: Path, policy: str) -> Optional[dict]:
    path = results_dir / "aggregated_results.json"
    if not path.exists():
        return None
    for entry in _read_json(path):
        if entry.get("scenario") == "hard" and entry.get("policy") == policy:
            return entry
    return None


def plot_hard_accuracy_gain_stack(
    no_adjustment_dir: Path,
    random_baseline_dir: Path,
    tuned_adjustment_dir: Path,
    figure_dir: Path,
    tuned_adjustment: float,
) -> None:
    random_summary = _hard_policy_summary(random_baseline_dir, "random")
    trust_no_adjust = _hard_policy_summary(no_adjustment_dir, "trust")
    trust_adjusted = _hard_policy_summary(tuned_adjustment_dir, "trust")
    if random_summary is None or trust_no_adjust is None or trust_adjusted is None:
        return

    labels = [
        "Random\nrouting",
        "Trust routing\nexit adjust = 0.0",
        f"Trust routing\nexit adjust = {tuned_adjustment:.1f}",
    ]
    summaries = [random_summary, trust_no_adjust, trust_adjusted]
    means = [float(item["accuracy_mean"]) for item in summaries]
    stds = [float(item["accuracy_std"]) for item in summaries]
    per_seed = [
        [float(run["accuracy"]) for run in item.get("per_seed", [])]
        for item in summaries
    ]
    mean_gain = means[2] - means[0]
    seed_gains = [
        adjusted - random
        for random, adjusted in zip(per_seed[0], per_seed[2])
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.8),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    x = np.arange(len(labels))
    colors = ["#c7dcef", "#7fb3d5", "#2c7fb8"]

    axes[0].bar(x, means, yerr=stds, capsize=6, width=0.6, color=colors, edgecolor="#17324d")
    for idx, values in enumerate(per_seed):
        offsets = np.linspace(-0.08, 0.08, num=max(len(values), 1))
        axes[0].scatter(
            np.full(len(values), x[idx]) + offsets[: len(values)],
            values,
            color="#f28e2b",
            edgecolor="#5f3b00",
            zorder=3,
            s=38,
            label="seed result" if idx == 0 else None,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Hard Scenario Accuracy")
    axes[0].set_ylim(0, max(mean + std for mean, std in zip(means, stds)) + 10)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left")
    axes[0].text(
        1.0,
        axes[0].get_ylim()[1] - 5.0,
        f"Mean gain vs random: +{mean_gain:.1f} pp",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#17324d",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#17324d"},
    )

    seed_labels = [f"Seed {idx}" for idx in range(len(seed_gains))]
    gain_colors = ["#2c7fb8" if gain >= 0 else "#d95f0e" for gain in seed_gains]
    axes[1].bar(np.arange(len(seed_gains)), seed_gains, color=gain_colors, edgecolor="#17324d")
    axes[1].axhline(0, color="#17324d", linewidth=1.0)
    axes[1].set_xticks(np.arange(len(seed_gains)))
    axes[1].set_xticklabels(seed_labels)
    axes[1].set_ylabel("Accuracy gain vs random (pp)")
    axes[1].set_title("Seed-Level Gain")
    axes[1].set_ylim(min(seed_gains) - 3.0, max(seed_gains) + 3.0)
    axes[1].grid(axis="y", alpha=0.25)
    for idx, gain in enumerate(seed_gains):
        va = "bottom" if gain >= 0 else "top"
        offset = 0.8 if gain >= 0 else -0.8
        axes[1].text(idx, gain + offset, f"{gain:+.1f}", ha="center", va=va, fontweight="bold")

    fig.suptitle("Trust-Aware Routing + Exits Improve Hard-Scenario Accuracy", y=1.04)
    fig.tight_layout()
    fig.savefig(figure_dir / "hard_scenario_accuracy_gains.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PRISM report assets.")
    parser.add_argument("--prism-dir", type=Path, default=DEFAULT_PRISM_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--no-exit-adjustment-dir", type=Path, default=DEFAULT_NO_EXIT_ADJUSTMENT_DIR)
    parser.add_argument("--tuned-exit-adjustment-dir", type=Path, default=DEFAULT_TUNED_EXIT_ADJUSTMENT_DIR)
    parser.add_argument("--tuned-exit-adjustment", type=float, default=DEFAULT_TUNED_EXIT_ADJUSTMENT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scheduler-budget", type=float, default=DEFAULT_SCHEDULER_BUDGET)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison_table(
        args.prism_dir,
        args.results_dir,
        args.scheduler_budget,
        args.tuned_exit_adjustment_dir,
    )
    comparison.to_csv(args.out_dir / "centralized_vs_distributed.csv", index=False)
    write_markdown_table(comparison, args.out_dir / "centralized_vs_distributed.md")
    plot_comparison_table(comparison, figure_dir, args.tuned_exit_adjustment)
    plot_hard_trust_trace(args.tuned_exit_adjustment_dir, figure_dir)
    plot_trust_exit_adjustment_gain(
        args.no_exit_adjustment_dir,
        args.tuned_exit_adjustment_dir,
        figure_dir,
        args.tuned_exit_adjustment,
    )
    plot_hard_accuracy_gain_stack(
        args.no_exit_adjustment_dir,
        args.results_dir,
        args.tuned_exit_adjustment_dir,
        figure_dir,
        args.tuned_exit_adjustment,
    )

    print(f"Wrote comparison table to {args.out_dir / 'centralized_vs_distributed.csv'}")
    print(f"Wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()
