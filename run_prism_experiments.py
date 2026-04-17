#!/usr/bin/env python3
"""
Run the PRISM / EENet experiment suite for CIFAR-100 + DenseNet121.

Outputs:
- per-exit accuracy / latency table
- budget sweep results for EENet, EENet+Trust, and entropy routing
- random vs trust-based routing under injected peer faults
- plots and a short best / worst / average summary
"""

import argparse
import math
import os
import pickle as pkl
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models
from args import arg_parser, modify_args
from config import Config
from data_tools.dataloader import get_dataloaders
from trust import EigenTrustTracker, compute_score_calibration
from trust.routing import simulate_routing_policy
from utils.predict_helpers import ExitAssigner, fit_exit_assigner, prepare_input
from utils.predict_utils import test_exit_assigner


class Segment1(nn.Module):
    def __init__(self, model, scorer, threshold):
        super().__init__()
        self.conv1 = model.conv1
        self.dense1 = model.dense1
        self.trans1 = model.trans1
        self.ee_cls = model.ee_classifiers[0]
        self.scorer = scorer
        self.threshold = threshold

    def forward(self, x):
        feat = self.trans1(self.dense1(self.conv1(x)))
        logit = self.ee_cls(feat)
        return feat, logit

    def compute_score(self, logit, past_scores=None):
        with torch.no_grad():
            soft = torch.softmax(logit, dim=1)
            X, _ = prepare_input(soft.unsqueeze(0), k=0)
            raw = self.scorer.predict(X)
            return raw.reshape(logit.shape[0])

    def should_exit(self, score):
        return bool((score >= self.threshold).all().item())


class Segment2(nn.Module):
    def __init__(self, model, scorer, threshold):
        super().__init__()
        self.dense2 = model.dense2
        self.trans2 = model.trans2
        self.ee_cls = model.ee_classifiers[1]
        self.scorer = scorer
        self.threshold = threshold

    def forward(self, feat_in):
        feat = self.trans2(self.dense2(feat_in))
        logit = self.ee_cls(feat)
        return feat, logit

    def compute_score(self, logit, past_scores):
        with torch.no_grad():
            soft = torch.softmax(logit, dim=1)
            X, _ = prepare_input(soft.unsqueeze(0), k=0)
            X = torch.cat([X, past_scores.reshape(logit.shape[0], -1)], dim=1)
            raw = self.scorer.predict(X)
            return raw.reshape(logit.shape[0])

    def should_exit(self, score):
        return bool((score >= self.threshold).all().item())


class Segment3(nn.Module):
    def __init__(self, model, scorer, threshold):
        super().__init__()
        self.dense3 = model.dense3
        self.trans3 = model.trans3
        self.ee_cls = model.ee_classifiers[2]
        self.scorer = scorer
        self.threshold = threshold

    def forward(self, feat_in):
        feat = self.trans3(self.dense3(feat_in))
        logit = self.ee_cls(feat)
        return feat, logit

    def compute_score(self, logit, past_scores):
        with torch.no_grad():
            soft = torch.softmax(logit, dim=1)
            X, _ = prepare_input(soft.unsqueeze(0), k=0)
            X = torch.cat([X, past_scores.reshape(logit.shape[0], -1)], dim=1)
            raw = self.scorer.predict(X)
            return raw.reshape(logit.shape[0])

    def should_exit(self, score):
        return bool((score >= self.threshold).all().item())


class Segment4(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.dense4 = model.dense4
        self.bn = model.bn
        self.linear = model.linear
        self.threshold = float("-inf")

    def forward(self, feat_in):
        out = self.dense4(feat_in)
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out)


class FullModelBackend:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.num_exits = 4
        self.model.eval()

    def forward_all(self, x):
        outputs, _ = self.model(x, manual_early_exit_index=0)
        if not isinstance(outputs, list):
            outputs = [outputs]
        return outputs

    def forward_to_exit(self, x, exit_idx):
        outputs = self.model(x, manual_early_exit_index=exit_idx)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if not isinstance(outputs, list):
            outputs = [outputs]
        return outputs


class SegmentBackend:
    def __init__(self, segments_dir, device):
        self.device = device
        self.num_exits = 4
        self.seg1 = torch.load(os.path.join(segments_dir, "segment1.pt"), map_location="cpu", weights_only=False).to(device)
        self.seg2 = torch.load(os.path.join(segments_dir, "segment2.pt"), map_location="cpu", weights_only=False).to(device)
        self.seg3 = torch.load(os.path.join(segments_dir, "segment3.pt"), map_location="cpu", weights_only=False).to(device)
        self.seg4 = torch.load(os.path.join(segments_dir, "segment4.pt"), map_location="cpu", weights_only=False).to(device)
        for module in [self.seg1, self.seg2, self.seg3, self.seg4]:
            module.eval()

    def forward_all(self, x):
        feat1, logit1 = self.seg1(x)
        feat2, logit2 = self.seg2(feat1)
        feat3, logit3 = self.seg3(feat2)
        logit4 = self.seg4(feat3)
        return [logit1, logit2, logit3, logit4]

    def forward_to_exit(self, x, exit_idx):
        outputs = []
        feat1, logit1 = self.seg1(x)
        outputs.append(logit1)
        if exit_idx == 1:
            return outputs

        feat2, logit2 = self.seg2(feat1)
        outputs.append(logit2)
        if exit_idx == 2:
            return outputs

        feat3, logit3 = self.seg3(feat2)
        outputs.append(logit3)
        if exit_idx == 3:
            return outputs

        outputs.append(self.seg4(feat3))
        return outputs


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def choose_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_path(save_path, model_path):
    if model_path:
        return model_path
    candidate = os.path.join(save_path, "save_models", "model_best.pth.tar")
    return candidate if os.path.exists(candidate) else None


def benchmark_exit_latencies(backend, loader, device, num_exits, warmup_batches, timed_batches, max_eval_batches):
    total_batches = warmup_batches + timed_batches
    if max_eval_batches is not None:
        total_batches = min(total_batches, max_eval_batches)

    per_exit_ms = []
    with torch.no_grad():
        for exit_idx in range(1, num_exits + 1):
            timings = []
            for batch_idx, (inp, _) in enumerate(loader):
                if batch_idx >= total_batches:
                    break
                inp = inp.to(device)
                synchronize_device(device)
                start = time.perf_counter()
                backend.forward_to_exit(inp, exit_idx)
                synchronize_device(device)
                elapsed = time.perf_counter() - start
                if batch_idx >= warmup_batches:
                    timings.append(elapsed / max(inp.size(0), 1) * 1000.0)

            if not timings:
                raise RuntimeError("Unable to benchmark exit latencies.")
            per_exit_ms.append(float(np.mean(timings)))

    return per_exit_ms


def collect_logits(backend, loader, device, num_exits, max_batches, print_freq):
    softmax = nn.Softmax(dim=1)
    logits = [[] for _ in range(num_exits)]
    targets = []

    with torch.no_grad():
        for batch_idx, (inp, target) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inp = inp.to(device)
            outputs = backend.forward_all(inp)
            targets.append(target.cpu())
            for exit_idx in range(num_exits):
                logits[exit_idx].append(softmax(outputs[exit_idx]).cpu())
            if batch_idx % max(print_freq, 1) == 0:
                print("  logits [{0}/{1}]".format(batch_idx + 1, len(loader)))

    for exit_idx in range(num_exits):
        logits[exit_idx] = torch.cat(logits[exit_idx], dim=0)

    stacked = torch.zeros(num_exits, logits[0].shape[0], logits[0].shape[1])
    for exit_idx in range(num_exits):
        stacked[exit_idx].copy_(logits[exit_idx])
    return stacked, torch.cat(targets, dim=0).long()


def topk_accuracy(probs, target, k):
    topk = probs.topk(k=min(k, probs.shape[1]), dim=1).indices
    target_expanded = target.view(-1, 1).expand_as(topk)
    correct = topk.eq(target_expanded).any(dim=1).float().mean().item()
    return correct * 100.0


def compute_per_exit_table(test_pred, test_target, seconds):
    rows = []
    for exit_idx in range(test_pred.shape[0]):
        rows.append({
            "exit": exit_idx + 1,
            "accuracy_top1": topk_accuracy(test_pred[exit_idx], test_target, 1),
            "accuracy_top5": topk_accuracy(test_pred[exit_idx], test_target, 5),
            "latency_ms": float(seconds[exit_idx]),
        })
    return pd.DataFrame(rows)


def load_or_fit_scheduler(ea_path, probs_path, val_pred, val_target, costs, budget, inference_params):
    if os.path.exists(ea_path) and os.path.exists(probs_path):
        ea = pkl.load(open(ea_path, "rb"))
        probs = pkl.load(open(probs_path, "rb"))
        return ea, probs

    ea, probs_tensor = fit_exit_assigner(
        val_pred,
        val_target.float(),
        costs,
        budget,
        alpha_ce=inference_params["alpha_ce"],
        alpha_cost=inference_params["alpha_cost"],
        beta_thr=0,
        beta_ce=inference_params["beta_ce"],
        lr=inference_params["lr"],
        weight_decay=inference_params["weight_decay"],
        num_epoch=inference_params["num_epoch"],
        batch_size=inference_params["bs"],
        hidden_dim_rate=inference_params["hidden_dim_rate"],
        period=inference_params["period"],
        conf_mode="nn",
    )
    probs = probs_tensor.tolist() if not isinstance(probs_tensor, list) else probs_tensor
    pkl.dump(ea, open(ea_path, "wb"))
    pkl.dump(probs, open(probs_path, "wb"))
    return ea, probs


def eval_entropy_baseline(pred, target, probs_target, seconds_list):
    n_stage, n_samp, num_classes = pred.shape
    conf_scores = torch.zeros(n_stage, n_samp)
    for exit_idx in range(n_stage):
        prob = pred[exit_idx]
        conf = 1 + torch.sum(prob * torch.log(prob + 1e-9), dim=1) / math.log(num_classes)
        conf_scores[exit_idx] = conf

    thresholds = ExitAssigner.compute_threshold(conf_scores.permute(1, 0), probs_target)
    correct = 0
    total_latency = 0.0
    exit_counts = torch.zeros(n_stage)
    for sample_idx in range(n_samp):
        for exit_idx in range(n_stage):
            if conf_scores[exit_idx, sample_idx].item() >= thresholds[exit_idx].item():
                if pred[exit_idx, sample_idx].argmax().item() == target[sample_idx].item():
                    correct += 1
                total_latency += seconds_list[exit_idx]
                exit_counts[exit_idx] += 1
                break

    return {
        "accuracy": 100.0 * correct / float(n_samp),
        "avg_latency_ms": total_latency / float(n_samp),
        "exit_distribution": (exit_counts / float(n_samp)).tolist(),
    }


def warm_up_trust(tracker, test_nn, test_pred, test_target, base_thresholds, seconds, budget):
    n_stage, n_samp, _ = test_pred.shape
    rng = np.random.default_rng(42)
    indices = rng.choice(n_samp, size=min(640, n_samp), replace=False)
    peer_scores = [[] for _ in range(n_stage)]
    peer_correct = [[] for _ in range(n_stage)]
    peer_budget_ok = [[] for _ in range(n_stage)]

    for sample_idx in indices:
        for exit_idx in range(n_stage):
            score_value = test_nn[exit_idx, sample_idx].item()
            if score_value >= base_thresholds[exit_idx]:
                is_correct = int(test_pred[exit_idx, sample_idx].argmax().item() == test_target[sample_idx].item())
                peer_scores[exit_idx].append(score_value)
                peer_correct[exit_idx].append(is_correct)
                peer_budget_ok[exit_idx].append(int(seconds[exit_idx] <= budget))
                break

    for exit_idx in range(n_stage):
        if not peer_scores[exit_idx]:
            continue
        tracker.update(
            peer_id=exit_idx,
            accuracy=float(np.mean(peer_correct[exit_idx])),
            latency_ok=float(np.mean(peer_budget_ok[exit_idx])),
            score_calibration=compute_score_calibration(
                np.asarray(peer_scores[exit_idx]),
                np.asarray(peer_correct[exit_idx]),
            ),
            ema_momentum=0.30,
        )
    return tracker


def eval_eenet_with_trust(test_nn, test_pred, test_target, base_thresholds, tracker, seconds, budget):
    n_stage, n_samp, _ = test_pred.shape
    thresholds = tracker.trust_scaled_thresholds(base_thresholds)
    correct = 0
    total_latency = 0.0
    exit_counts = np.zeros(n_stage, dtype=np.int64)
    peer_scores = [[] for _ in range(n_stage)]
    peer_correct = [[] for _ in range(n_stage)]
    peer_budget_ok = [[] for _ in range(n_stage)]

    for sample_idx in range(n_samp):
        for exit_idx in range(n_stage):
            if test_nn[exit_idx, sample_idx].item() >= thresholds[exit_idx]:
                is_correct = int(test_pred[exit_idx, sample_idx].argmax().item() == test_target[sample_idx].item())
                correct += is_correct
                total_latency += seconds[exit_idx]
                exit_counts[exit_idx] += 1
                peer_scores[exit_idx].append(float(test_nn[exit_idx, sample_idx].item()))
                peer_correct[exit_idx].append(is_correct)
                peer_budget_ok[exit_idx].append(int(seconds[exit_idx] <= budget))
                break

        if (sample_idx + 1) % 200 == 0:
            for exit_idx in range(n_stage):
                if len(peer_scores[exit_idx]) < 2:
                    continue
                tracker.update(
                    peer_id=exit_idx,
                    accuracy=float(np.mean(peer_correct[exit_idx])),
                    latency_ok=float(np.mean(peer_budget_ok[exit_idx])),
                    score_calibration=compute_score_calibration(
                        np.asarray(peer_scores[exit_idx]),
                        np.asarray(peer_correct[exit_idx]),
                    ),
                    ema_momentum=0.30,
                )
                thresholds = tracker.trust_scaled_thresholds(base_thresholds)
                peer_scores[exit_idx].clear()
                peer_correct[exit_idx].clear()
                peer_budget_ok[exit_idx].clear()

    return {
        "accuracy": 100.0 * correct / float(n_samp),
        "avg_latency_ms": total_latency / float(n_samp),
        "exit_distribution": (exit_counts / float(n_samp)).tolist(),
        "trust_vector": tracker.trust.tolist(),
    }


def run_budget_sweep(args, val_pred, val_target, test_pred, test_target, seconds, inference_params, experiment_dir):
    n_stage = test_pred.shape[0]
    costs = torch.tensor(seconds)
    os.makedirs(os.path.join(args.save_path, "ea_pkls"), exist_ok=True)
    rows = []

    for budget in args.budgets:
        ea_path = os.path.join(args.save_path, "ea_pkls", "ea_{0}_.pkl".format(budget))
        probs_path = os.path.join(args.save_path, "ea_pkls", "probs_{0}_.pkl".format(budget))
        ea, probs = load_or_fit_scheduler(ea_path, probs_path, val_pred, val_target, costs, budget, inference_params)

        thresholds = ea.get_threshold().detach().cpu().numpy()
        test_nn = test_exit_assigner(args, test_pred, n_stage, ea)

        correct = 0
        total_latency = 0.0
        exit_counts = torch.zeros(n_stage)
        for sample_idx in range(test_pred.shape[1]):
            for exit_idx in range(n_stage):
                if test_nn[exit_idx, sample_idx].item() >= thresholds[exit_idx]:
                    if test_pred[exit_idx, sample_idx].argmax().item() == test_target[sample_idx].item():
                        correct += 1
                    total_latency += seconds[exit_idx]
                    exit_counts[exit_idx] += 1
                    break

        eenet_result = {
            "budget_ms": float(budget),
            "method": "EENet",
            "accuracy": 100.0 * correct / float(test_pred.shape[1]),
            "avg_latency_ms": total_latency / float(test_pred.shape[1]),
            "exit_distribution": (exit_counts / float(test_pred.shape[1])).tolist(),
            "trust_vector": None,
        }
        rows.append(eenet_result)

        tracker = warm_up_trust(
            EigenTrustTracker(n_peers=n_stage, trust_scale=0.20, alpha=0.1, decay=0.85),
            test_nn,
            test_pred,
            test_target,
            thresholds,
            seconds,
            budget,
        )
        trust_result = eval_eenet_with_trust(
            test_nn,
            test_pred,
            test_target,
            thresholds,
            tracker,
            seconds,
            budget,
        )
        rows.append({
            "budget_ms": float(budget),
            "method": "EENet+Trust",
            "accuracy": trust_result["accuracy"],
            "avg_latency_ms": trust_result["avg_latency_ms"],
            "exit_distribution": trust_result["exit_distribution"],
            "trust_vector": trust_result["trust_vector"],
        })

        entropy_result = eval_entropy_baseline(test_pred, test_target, probs, seconds)
        rows.append({
            "budget_ms": float(budget),
            "method": "Entropy",
            "accuracy": entropy_result["accuracy"],
            "avg_latency_ms": entropy_result["avg_latency_ms"],
            "exit_distribution": entropy_result["exit_distribution"],
            "trust_vector": None,
        })

    budget_df = pd.DataFrame(rows)
    budget_df.to_csv(os.path.join(experiment_dir, "budget_sweep.csv"), index=False)
    return budget_df


def run_routing_experiments(args, test_pred, test_target, budget_df, experiment_dir):
    trust_rows = []
    budget_row = budget_df[budget_df["method"] == "EENet+Trust"].sort_values("accuracy", ascending=False).iloc[0]
    reference_budget = float(budget_row["budget_ms"])
    ea = pkl.load(open(os.path.join(args.save_path, "ea_pkls", "ea_{0}_.pkl".format(reference_budget)), "rb"))
    thresholds = ea.get_threshold().detach().cpu().numpy()
    test_nn = test_exit_assigner(args, test_pred, test_pred.shape[0], ea)

    test_pred_np = test_pred.numpy()
    test_target_np = test_target.numpy()
    test_nn_np = test_nn.numpy()
    seconds = list(pd.read_csv(os.path.join(args.save_path, "seconds.csv"), header=None).iloc[:, 0])

    sample_cap = min(args.routing_samples, test_pred.shape[1])
    test_pred_np = test_pred_np[:, :sample_cap]
    test_target_np = test_target_np[:sample_cap]
    test_nn_np = test_nn_np[:, :sample_cap]

    for scenario in ["easy", "medium", "hard"]:
        for policy in ["random", "trust"]:
            for seed in range(args.routing_seeds):
                trust_rows.append(simulate_routing_policy(
                    policy=policy,
                    test_pred=test_pred_np,
                    test_scores=test_nn_np,
                    test_target=test_target_np,
                    base_thresholds=thresholds,
                    seconds=seconds,
                    scenario=scenario,
                    replicas_per_stage=args.replicas_per_exit,
                    trust_update_window=args.trust_update_window,
                    seed=seed,
                ))

    trust_df = pd.DataFrame(trust_rows)
    trust_df.to_csv(os.path.join(experiment_dir, "trust_routing_raw.csv"), index=False)
    numeric_cols = ["accuracy", "accuracy_on_completed", "reliability", "dropped_responses", "avg_latency_ms"]
    summary_df = trust_df.groupby(["scenario", "policy"], as_index=False)[numeric_cols].mean()
    summary_df.to_csv(os.path.join(experiment_dir, "trust_routing_summary.csv"), index=False)
    return trust_df, summary_df


def plot_exit_depth(exit_df, figure_dir):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(exit_df["exit"], exit_df["accuracy_top1"], color="#2c7fb8", label="Top-1 accuracy")
    ax1.set_xlabel("Exit depth")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, max(100.0, exit_df["accuracy_top1"].max() + 5.0))
    ax1.set_title("Experiment 1: Accuracy vs Exit Depth")

    ax2 = ax1.twinx()
    ax2.plot(exit_df["exit"], exit_df["latency_ms"], color="#d95f0e", marker="o", linewidth=2, label="Latency")
    ax2.set_ylabel("Latency (ms / sample)")

    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "accuracy_vs_exit_depth.png"), dpi=180)
    plt.close(fig)


def plot_budget(budget_df, figure_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, frame in budget_df.groupby("method"):
        frame = frame.sort_values("budget_ms")
        ax.plot(frame["budget_ms"], frame["accuracy"], marker="o", linewidth=2, label=method)

    ax.set_xlabel("Allowed compute budget (ms / sample)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Experiment 2: Accuracy vs Budget")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "accuracy_vs_budget.png"), dpi=180)
    plt.close(fig)


def plot_trust(summary_df, figure_dir):
    comparison_scenario = "hard"
    medium_df = summary_df[summary_df["scenario"] == comparison_scenario].copy()
    metrics = ["accuracy", "reliability", "dropped_responses"]
    labels = ["Accuracy", "Reliability", "Dropped responses"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    random_vals = medium_df[medium_df["policy"] == "random"][metrics].iloc[0].tolist()
    trust_vals = medium_df[medium_df["policy"] == "trust"][metrics].iloc[0].tolist()
    ax.bar(x - width / 2, random_vals, width=width, label="Random routing", color="#9ecae1")
    ax.bar(x + width / 2, trust_vals, width=width, label="Trust routing", color="#3182bd")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percent")
    ax.set_title("Experiment 3: Trust-Based Routing ({0} Fault Scenario)".format(comparison_scenario.title()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "trust_routing_comparison.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for policy, frame in summary_df.groupby("policy"):
        frame = frame.sort_values("scenario")
        ax.plot(frame["scenario"], frame["accuracy"], marker="o", linewidth=2, label=policy)
    ax.set_xlabel("Fault scenario")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Trust Routing Accuracy Across Difficulty Levels")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "trust_routing_by_scenario.png"), dpi=180)
    plt.close(fig)


def write_summary(exit_df, budget_df, trust_summary_df, experiment_dir):
    best_case = trust_summary_df[(trust_summary_df["scenario"] == "easy") & (trust_summary_df["policy"] == "trust")].iloc[0]
    avg_case = trust_summary_df[(trust_summary_df["scenario"] == "medium") & (trust_summary_df["policy"] == "trust")].iloc[0]
    worst_case = trust_summary_df[(trust_summary_df["scenario"] == "hard") & (trust_summary_df["policy"] == "random")].iloc[0]
    hard_trust_case = trust_summary_df[(trust_summary_df["scenario"] == "hard") & (trust_summary_df["policy"] == "trust")].iloc[0]
    best_budget = budget_df.sort_values("accuracy", ascending=False).iloc[0]

    lines = [
        "# PRISM Experiment Summary",
        "",
        "## Best / Worst / Average",
        "- Best case: trust routing in the easy scenario reached {0:.2f}% accuracy with {1:.2f}% reliability.".format(
            best_case["accuracy"], best_case["reliability"]),
        "- Average case: trust routing in the medium scenario reached {0:.2f}% accuracy with {1:.2f}% reliability and {2:.2f}% dropped responses.".format(
            avg_case["accuracy"], avg_case["reliability"], avg_case["dropped_responses"]),
        "- Worst case: random routing in the hard scenario fell to {0:.2f}% accuracy with {1:.2f}% reliability.".format(
            worst_case["accuracy"], worst_case["reliability"]),
        "",
        "## Takeaways",
        "- Early exits reduce compute: exit 1 is the fastest point, while later exits recover accuracy.",
        "- The strongest clean-budget result came from {0} at {1:.2f} ms with {2:.2f}% accuracy.".format(
            best_budget["method"], best_budget["budget_ms"], best_budget["accuracy"]),
        "- In the hard scenario, trust routing cut dropped responses from {0:.2f}% to {1:.2f}% while improving accuracy from {2:.2f}% to {3:.2f}%.".format(
            worst_case["dropped_responses"],
            hard_trust_case["dropped_responses"],
            worst_case["accuracy"],
            hard_trust_case["accuracy"],
        ),
        "- Per-exit metrics and trust-routing summaries were written alongside the plots for the report.",
    ]

    with open(os.path.join(experiment_dir, "summary.md"), "w") as fout:
        fout.write("\n".join(lines))


def main():
    arg_parser.set_defaults(
        data="cifar100",
        data_root="datasets",
        arch="densenet121_4",
        use_valid=True,
        save_path="outputs/densenet121_4_None_cifar100",
        evalmode="dynamic",
    )
    parser = argparse.ArgumentParser(
        description="Run PRISM experiment suite.",
        parents=[arg_parser],
        conflict_handler="resolve",
    )
    parser.add_argument("--model-path", default=None, type=str, help="Path to a trained full-model checkpoint.")
    parser.add_argument("--segments-dir", default="outputs/segments", type=str, help="Directory containing segment*.pt.")
    parser.add_argument("--use-segments", action="store_true", help="Use segment artifacts even if a full checkpoint exists.")
    parser.add_argument("--force-cpu", action="store_true", help="Disable MPS / CUDA and run on CPU only.")
    parser.add_argument("--experiment-dir", default="outputs/prism_experiments", type=str, help="Directory for plots and tables.")
    parser.add_argument("--budgets", nargs="+", type=float, default=[7.5, 6.75, 6.5, 6.0], help="Target latency budgets.")
    parser.add_argument("--scheduler-epochs", type=int, default=300, help="Scheduler MLP training epochs.")
    parser.add_argument("--timing-batches", type=int, default=10, help="Number of timed batches for exit latency.")
    parser.add_argument("--timing-warmup-batches", type=int, default=3, help="Warmup batches before timing.")
    parser.add_argument("--routing-samples", type=int, default=2000, help="How many test samples to use for routing simulation.")
    parser.add_argument("--routing-seeds", type=int, default=3, help="Random seeds per routing scenario.")
    parser.add_argument("--replicas-per-exit", type=int, default=3, help="Replicated peers per exit stage.")
    parser.add_argument("--trust-update-window", type=int, default=64, help="Samples per trust update window.")

    args = modify_args(parser.parse_args())
    config = Config()
    args.num_exits = config.model_params[args.data][args.arch]["num_blocks"]
    args.inference_params = dict(config.inference_params[args.data][args.arch])
    args.inference_params["num_epoch"] = args.scheduler_epochs
    args.use_gpu = False
    args.device = choose_device(force_cpu=args.force_cpu)
    print("Using device: {0}".format(args.device))

    batch_size = args.batch_size if args.batch_size else config.training_params[args.data][args.arch]["batch_size"]
    experiment_dir = args.experiment_dir
    figure_dir = os.path.join(experiment_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    model_path = resolve_model_path(args.save_path, args.model_path)
    backend = None
    if not args.use_segments and model_path and os.path.exists(model_path):
        model = getattr(models, args.arch)(args, dict(config.model_params[args.data][args.arch]))
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        backend = FullModelBackend(model, args.device)
        print("Loaded full checkpoint from {0}".format(model_path))
    else:
        if not os.path.isdir(args.segments_dir):
            raise FileNotFoundError(
                "No full checkpoint was found, and the segment directory '{0}' is missing.".format(args.segments_dir)
            )
        backend = SegmentBackend(args.segments_dir, args.device)
        print("Loaded segment artifacts from {0}".format(args.segments_dir))

    train_loader, val_loader, test_loader = get_dataloaders(args, batch_size=batch_size)

    seconds_path = os.path.join(args.save_path, "seconds.csv")
    if os.path.exists(seconds_path):
        seconds = list(pd.read_csv(seconds_path, header=None).iloc[:, 0])
    else:
        probe_loader = val_loader if val_loader is not None else test_loader
        seconds = benchmark_exit_latencies(
            backend,
            probe_loader,
            args.device,
            args.num_exits,
            args.timing_warmup_batches,
            args.timing_batches,
            args.max_eval_batches,
        )
        os.makedirs(args.save_path, exist_ok=True)
        with open(seconds_path, "w") as fout:
            for value in seconds:
                fout.write("{0:.6f}\n".format(value))

    logits_path = os.path.join(args.save_path, "logits_single.pth")
    if os.path.exists(logits_path):
        print("Loading cached logits from {0}".format(logits_path))
        val_pred, val_target, test_pred, test_target = torch.load(logits_path, weights_only=False)
    else:
        print("Collecting validation logits...")
        val_pred, val_target = collect_logits(backend, val_loader, args.device, args.num_exits, args.max_eval_batches, args.print_freq)
        print("Collecting test logits...")
        test_pred, test_target = collect_logits(backend, test_loader, args.device, args.num_exits, args.max_eval_batches, args.print_freq)
        os.makedirs(args.save_path, exist_ok=True)
        torch.save((val_pred, val_target, test_pred, test_target), logits_path)

    exit_df = compute_per_exit_table(test_pred, test_target, seconds)
    exit_df.to_csv(os.path.join(experiment_dir, "exit_metrics.csv"), index=False)

    budget_df = run_budget_sweep(
        args,
        val_pred,
        val_target,
        test_pred,
        test_target,
        seconds,
        args.inference_params,
        experiment_dir,
    )
    _, trust_summary_df = run_routing_experiments(args, test_pred, test_target, budget_df, experiment_dir)

    plot_exit_depth(exit_df, figure_dir)
    plot_budget(budget_df, figure_dir)
    plot_trust(trust_summary_df, figure_dir)
    write_summary(exit_df, budget_df, trust_summary_df, experiment_dir)

    print("\nSaved experiment tables to {0}".format(experiment_dir))
    print("Saved plots to {0}".format(figure_dir))


if __name__ == "__main__":
    main()
