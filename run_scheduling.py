#!/usr/bin/env python3
"""
Phase 2: EENet scheduling optimization for DenseNet121-4exit on CIFAR-100.
Trains exit scoring functions (ScoreNormalizer gk) and exit assignment
functions (ExitAssigner hk) per Algorithm 1 of the EENet paper.
Compares EENet vs entropy-based thresholding (BranchyNet baseline).

Usage:
    cd EENet && python run_scheduling.py
"""
import argparse
import os, sys, math, pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models
from config import Config
from args import arg_parser, modify_args
from data_tools.dataloader import get_dataloaders
from utils.predict_helpers import fit_exit_assigner, prepare_input, ExitAssigner
from utils.predict_utils import test_exit_assigner

schedule_parser = argparse.ArgumentParser(parents=[arg_parser], conflict_handler='resolve')
schedule_parser.set_defaults(
    data='cifar100',
    data_root='datasets',
    arch='densenet121_4',
    use_valid=True,
    evalmode='dynamic',
    save_path='outputs/smoke_cifar100',
)
schedule_parser.add_argument('--budget', dest='budgets', action='append', type=float,
                             help='Target budget in milliseconds. Repeat to evaluate multiple budgets.')
schedule_parser.add_argument('--scheduler-batch-size', default=64, type=int,
                             help='Batch size for logit collection during scheduling.')

args = schedule_parser.parse_args()
args = modify_args(args)
args.print_freq = 100
torch.manual_seed(0)
if args.evaluate_from is None:
    args.evaluate_from = os.path.join(args.save_path, 'save_models', 'model_best.pth.tar')
budgets = args.budgets if args.budgets else [7.5, 6.75, 6.0]

# Prefer MPS for inference; keep scheduler training on CPU (small MLP)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    infer_device = torch.device('mps')
else:
    infer_device = torch.device('cpu')
args.device = infer_device
args.use_gpu = False   # keeps logit tensors on CPU (required by fit_exit_assigner)
print(f"Inference device: {infer_device}")

config = Config()
args.num_exits = config.model_params[args.data][args.arch]['num_blocks']  # 4
args.inference_params = config.inference_params[args.data][args.arch]

# ── Load model ─────────────────────────────────────────────────────────────
model = getattr(models, args.arch)(args, {**config.model_params[args.data][args.arch]})
sd = torch.load(args.evaluate_from, map_location='cpu', weights_only=False)['state_dict']
model.load_state_dict(sd, strict=False)
model = model.to(infer_device)
model.eval()
print("Model loaded from", args.evaluate_from)

# ── Data loaders ───────────────────────────────────────────────────────────
train_loader, val_loader, test_loader = get_dataloaders(args, batch_size=args.scheduler_batch_size)

# ── Collect / cache all-exit logits ───────────────────────────────────────
LOGITS_FILE = os.path.join(args.save_path, 'logits_single.pth')
softmax = nn.Softmax(dim=1)

def collect_logits(loader, tag):
    n_stage = args.num_exits
    logits = [[] for _ in range(n_stage)]
    targets = []
    with torch.no_grad():
        for i, (inp, tgt) in enumerate(loader):
            inp = inp.to(infer_device)
            out, _ = model(inp, manual_early_exit_index=0)
            targets.append(tgt)
            for k in range(n_stage):
                logits[k].append(softmax(out[k]).cpu())
            if i % 50 == 0:
                print(f"  {tag} [{i+1}/{len(loader)}]")
    for k in range(n_stage):
        logits[k] = torch.cat(logits[k], dim=0)
    ts = torch.zeros(n_stage, logits[0].shape[0], logits[0].shape[1])
    for k in range(n_stage):
        ts[k].copy_(logits[k])
    return ts, torch.cat(targets, dim=0).long()

if os.path.exists(LOGITS_FILE):
    print("Loading cached logits from", LOGITS_FILE)
    val_pred, val_target, test_pred, test_target = torch.load(LOGITS_FILE, weights_only=False)
    val_target  = val_target.long()
    test_target = test_target.long()
else:
    print("Collecting validation logits...")
    val_pred,  val_target  = collect_logits(val_loader,  'val')
    print("Collecting test logits...")
    test_pred, test_target = collect_logits(test_loader, 'test')
    torch.save((val_pred, val_target, test_pred, test_target), LOGITS_FILE)
    print("Cached logits →", LOGITS_FILE)

seconds = list(pd.read_csv(os.path.join(args.save_path, 'seconds.csv'), header=None).iloc[:, 0])
costs   = torch.tensor(seconds)
n_stage, n_test, n_cls = test_pred.shape
print(f"\nExit latencies (ms): {seconds}")

# ── Per-exit accuracy on test set (anytime) ───────────────────────────────
print("\n=== Per-exit accuracy on full test set (anytime mode) ===")
for k in range(n_stage):
    acc_k = (test_pred[k].argmax(dim=1) == test_target).float().mean().item() * 100
    print(f"  Exit {k+1}: {acc_k:.2f}% top-1")

# ── Entropy baseline helper ────────────────────────────────────────────────
def eval_entropy_baseline(pred, target, probs_target, seconds_list):
    """
    BranchyNet-style: threshold on normalized entropy confidence.
    Thresholds set so exit distribution ≈ probs_target.
    """
    n_s, n_samp, c = pred.shape
    conf_scores = torch.zeros(n_s, n_samp)
    for k in range(n_s):
        p = pred[k]
        conf = 1 + torch.sum(p * torch.log(p + 1e-9), dim=1) / math.log(c)
        conf_scores[k] = conf

    T = ExitAssigner.compute_threshold(conf_scores.permute(1, 0), probs_target)

    acc, exp_time = 0, 0.0
    exit_counts = torch.zeros(n_s)
    for i in range(n_samp):
        for k in range(n_s):
            if conf_scores[k, i].item() >= T[k].item():
                if pred[k, i].argmax().item() == target[i].item():
                    acc += 1
                exit_counts[k] += 1
                exp_time += seconds_list[k]
                break
    return (
        acc * 100.0 / n_samp,
        exp_time / n_samp,
        (exit_counts / n_samp).tolist(),
    )

# ── Main scheduling loop ───────────────────────────────────────────────────
os.makedirs(os.path.join(args.save_path, 'ea_pkls'), exist_ok=True)
ip = args.inference_params

print(f"\n{'='*80}")
print(f"{'Budget':>8} | {'Method':>10} | {'Accuracy':>9} | {'Avg ms':>8} | Exit distribution [e1,e2,e3,e4]")
print(f"{'='*80}")

for budget in budgets:
    ea_path    = os.path.join(args.save_path, f'ea_pkls/ea_{budget}_.pkl')
    probs_path = os.path.join(args.save_path, f'ea_pkls/probs_{budget}_.pkl')

    # Try loading cached ea; also accept legacy filenames (no trailing _)
    ea, probs = None, None
    for candidate in [ea_path, ea_path.replace('_.pkl', '.pkl')]:
        if os.path.exists(candidate):
            ea    = pkl.load(open(candidate, 'rb'))
            pfn   = candidate.replace('ea_', 'probs_')
            probs = pkl.load(open(pfn, 'rb'))
            print(f"\n[Budget {budget}ms] Loaded cached scheduler from {candidate}")
            break

    if ea is None:
        print(f"\n[Budget {budget}ms] Training EENet scheduler...")
        ea, probs_t = fit_exit_assigner(
            val_pred, val_target.float(), costs, budget,
            alpha_ce=ip['alpha_ce'], alpha_cost=ip['alpha_cost'],
            beta_thr=0, beta_ce=ip['beta_ce'],
            lr=ip['lr'], weight_decay=ip['weight_decay'],
            num_epoch=ip['num_epoch'], batch_size=ip['bs'],
            hidden_dim_rate=ip['hidden_dim_rate'],
            period=ip['period'], conf_mode='nn',
        )
        probs = probs_t.tolist() if not isinstance(probs_t, list) else probs_t
        pkl.dump(ea,    open(ea_path,    'wb'))
        pkl.dump(probs, open(probs_path, 'wb'))
        print(f"  Saved → {ea_path}")

    T = ea.get_threshold().detach().cpu().numpy()
    print(f"  EENet thresholds: {np.round(T, 4)}")
    print(f"  Target exit probs: {[round(p,3) for p in probs]}")

    # ── EENet evaluation on test set ──────────────────────────────────────
    test_nn = test_exit_assigner(args, test_pred, n_stage, ea)  # (n_stage, n_test)

    acc, exp_time, exit_counts = 0, 0.0, torch.zeros(n_stage)
    for i in range(n_test):
        for k in range(n_stage):
            if test_nn[k, i].item() >= T[k]:
                if test_pred[k, i].argmax().item() == test_target[i].item():
                    acc += 1
                exit_counts[k] += 1
                exp_time += seconds[k]
                break
    eenet_acc  = acc * 100.0 / n_test
    eenet_time = exp_time / n_test
    eenet_dist = [round(v.item(), 3) for v in exit_counts / n_test]
    print(f"\n{budget:>8.2f} | {'EENet':>10} | {eenet_acc:>8.2f}% | {eenet_time:>7.2f}ms | {eenet_dist}")

    # ── Entropy baseline (same exit distribution target) ──────────────────
    ent_acc, ent_time, ent_dist = eval_entropy_baseline(test_pred, test_target, probs, seconds)
    ent_dist_r = [round(v, 3) for v in ent_dist]
    print(f"{budget:>8.2f} | {'Entropy':>10} | {ent_acc:>8.2f}% | {ent_time:>7.2f}ms | {ent_dist_r}")
    print(f"  → EENet gain: {eenet_acc - ent_acc:+.2f}%")
    print(f"{'-'*80}")

print("\nScheduling complete.")
