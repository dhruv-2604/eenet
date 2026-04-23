#!/usr/bin/env python3
"""
Phase 2: EENet scheduling optimization for DenseNet121-4exit on CIFAR-100.
Trains exit scoring functions (ScoreNormalizer gk) and exit assignment
functions (ExitAssigner hk) per Algorithm 1 of the EENet paper.
Compares EENet vs entropy-based thresholding (BranchyNet baseline).

Usage:
    python run_scheduling.py
    python run_scheduling.py --save-path outputs/my_run --evaluate-from outputs/my_run/save_models/model_best.pth.tar
"""
import argparse
import math
import os
import pickle as pkl
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils.predict_helpers as _ph
sys.modules['predict_helpers'] = _ph  # fix pickle compat with Colab-saved .pkl files

import models
from args import arg_parser, modify_args
from config import Config
from data_tools.dataloader import get_dataloaders
from utils.predict_helpers import ExitAssigner, fit_exit_assigner
from utils.predict_utils import test_exit_assigner


def synchronize_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def resolve_model_path(args):
    if args.evaluate_from:
        return args.evaluate_from

    model_path = os.path.join(args.save_path, 'save_models', 'model_best.pth.tar')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'No trained checkpoint found at {model_path}. '
            'Run main.py first or pass --evaluate-from.'
        )
    return model_path


def benchmark_exit_latencies(model, loader, device, num_exits, seconds_path, warmup_batches, timed_batches):
    if loader is None:
        raise RuntimeError('A validation or test dataloader is required to benchmark exit latencies.')

    print(f'Benchmarking exit latencies on {device}...')
    per_exit_ms = []
    total_batches = warmup_batches + timed_batches

    model.eval()
    with torch.no_grad():
        for exit_idx in range(1, num_exits + 1):
            timings = []
            for batch_idx, (inp, _) in enumerate(loader):
                if batch_idx >= total_batches:
                    break
                inp = inp.to(device)
                synchronize_device(device)
                start = time.perf_counter()
                model(inp, manual_early_exit_index=exit_idx)
                synchronize_device(device)
                elapsed = time.perf_counter() - start
                if batch_idx >= warmup_batches:
                    timings.append(elapsed / max(inp.size(0), 1) * 1000.0)

            if not timings:
                raise RuntimeError('Unable to benchmark exit latencies because no batches were available.')

            avg_ms = float(np.mean(timings))
            per_exit_ms.append(avg_ms)
            print(f'  Exit {exit_idx}: {avg_ms:.4f} ms/sample')

    with open(seconds_path, 'w') as fout:
        for value in per_exit_ms:
            fout.write(f'{value:.6f}\n')

    print(f'Saved exit latencies to {seconds_path}')
    return per_exit_ms


arg_parser.set_defaults(
    data='cifar100',
    data_root='datasets',
    arch='densenet121_4',
    use_valid=True,
    save_path='outputs/densenet121_4_None_cifar100',
    evalmode='dynamic',
)

scheduler_parser = argparse.ArgumentParser(
    description='Train and evaluate the phase-2 EENet scheduler.',
    parents=[arg_parser],
    conflict_handler='resolve',
)
scheduler_parser.add_argument(
    '--budgets',
    nargs='+',
    type=float,
    default=[7.5, 6.75, 6.0],
    help='Target per-sample latency budgets in milliseconds.',
)
scheduler_parser.add_argument(
    '--scheduler-epochs',
    type=int,
    default=None,
    help='Override the scheduler MLP training epoch count.',
)
scheduler_parser.add_argument(
    '--timing-batches',
    type=int,
    default=10,
    help='Number of timed batches to use when generating seconds.csv.',
)
scheduler_parser.add_argument(
    '--timing-warmup-batches',
    type=int,
    default=3,
    help='Number of warmup batches to skip before timing each exit.',
)

args = scheduler_parser.parse_args()
args = modify_args(args)
torch.manual_seed(args.seed)

model_path = resolve_model_path(args)
save_path = args.save_path
budgets = args.budgets
batch_size = args.batch_size or 64

# Prefer MPS for backbone inference; keep scheduler training on CPU via CPU logits.
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    infer_device = torch.device('mps')
else:
    infer_device = torch.device('cpu')
args.device = infer_device
args.use_gpu = False
print(f'Inference device: {infer_device}')

config = Config()
args.num_exits = config.model_params[args.data][args.arch]['num_blocks']
args.inference_params = dict(config.inference_params[args.data][args.arch])
if args.scheduler_epochs is not None:
    args.inference_params['num_epoch'] = args.scheduler_epochs

model = getattr(models, args.arch)(args, {**config.model_params[args.data][args.arch]})
state_dict = torch.load(model_path, map_location='cpu', weights_only=False)['state_dict']
model.load_state_dict(state_dict, strict=False)
model = model.to(infer_device)
model.eval()
print('Model loaded from', model_path)

train_loader, val_loader, test_loader = get_dataloaders(args, batch_size=batch_size)

os.makedirs(save_path, exist_ok=True)
seconds_path = os.path.join(save_path, 'seconds.csv')
if os.path.exists(seconds_path):
    seconds = list(pd.read_csv(seconds_path, header=None).iloc[:, 0])
    print(f'Loaded cached exit latencies from {seconds_path}')
else:
    probe_loader = val_loader if val_loader is not None else test_loader
    seconds = benchmark_exit_latencies(
        model,
        probe_loader,
        infer_device,
        args.num_exits,
        seconds_path,
        args.timing_warmup_batches,
        args.timing_batches,
    )

logits_file = os.path.join(save_path, 'logits_single.pth')
softmax = nn.Softmax(dim=1)


def collect_logits(loader, tag):
    n_stage = args.num_exits
    logits = [[] for _ in range(n_stage)]
    targets = []
    with torch.no_grad():
        for i, (inp, tgt) in enumerate(loader):
            inp = inp.to(infer_device)
            out, _ = model(inp, manual_early_exit_index=0)
            targets.append(tgt.cpu())
            for k in range(n_stage):
                logits[k].append(softmax(out[k]).cpu())
            if i % max(args.print_freq, 1) == 0:
                print(f'  {tag} [{i + 1}/{len(loader)}]')
    for k in range(n_stage):
        logits[k] = torch.cat(logits[k], dim=0)
    ts = torch.zeros(n_stage, logits[0].shape[0], logits[0].shape[1])
    for k in range(n_stage):
        ts[k].copy_(logits[k])
    return ts, torch.cat(targets, dim=0).long()


if os.path.exists(logits_file):
    print('Loading cached logits from', logits_file)
    val_pred, val_target, test_pred, test_target = torch.load(logits_file, weights_only=False)
    val_target = val_target.long()
    test_target = test_target.long()
else:
    print('Collecting validation logits...')
    val_pred, val_target = collect_logits(val_loader, 'val')
    print('Collecting test logits...')
    test_pred, test_target = collect_logits(test_loader, 'test')
    torch.save((val_pred, val_target, test_pred, test_target), logits_file)
    print('Cached logits ->', logits_file)

costs = torch.tensor(seconds)
n_stage, n_test, n_cls = test_pred.shape
print(f'\nExit latencies (ms): {seconds}')

print('\n=== Per-exit accuracy on full test set (anytime mode) ===')
for k in range(n_stage):
    acc_k = (test_pred[k].argmax(dim=1) == test_target).float().mean().item() * 100
    print(f'  Exit {k + 1}: {acc_k:.2f}% top-1')


def eval_entropy_baseline(pred, target, probs_target, seconds_list):
    n_s, n_samp, c = pred.shape
    conf_scores = torch.zeros(n_s, n_samp)
    for k in range(n_s):
        p = pred[k]
        conf = 1 + torch.sum(p * torch.log(p + 1e-9), dim=1) / math.log(c)
        conf_scores[k] = conf

    thresholds = ExitAssigner.compute_threshold(conf_scores.permute(1, 0), probs_target)

    acc, exp_time = 0, 0.0
    exit_counts = torch.zeros(n_s)
    for i in range(n_samp):
        for k in range(n_s):
            if conf_scores[k, i].item() >= thresholds[k].item():
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


os.makedirs(os.path.join(save_path, 'ea_pkls'), exist_ok=True)
ip = args.inference_params

print(f"\n{'=' * 80}")
print(f"{'Budget':>8} | {'Method':>10} | {'Accuracy':>9} | {'Avg ms':>8} | Exit distribution [e1,e2,e3,e4]")
print(f"{'=' * 80}")

for budget in budgets:
    ea_path = os.path.join(save_path, f'ea_pkls/ea_{budget}_.pkl')
    probs_path = os.path.join(save_path, f'ea_pkls/probs_{budget}_.pkl')

    ea, probs = None, None
    for candidate in [ea_path, ea_path.replace('_.pkl', '.pkl')]:
        if os.path.exists(candidate):
            ea = pkl.load(open(candidate, 'rb'))
            ea_dir, ea_fname = os.path.split(candidate)
            probs_fname = ea_fname.replace('ea_', 'probs_', 1)
            pfn = os.path.join(ea_dir, probs_fname)
            if not os.path.exists(pfn):
                pfn = pfn[:-5] + '.pkl' if pfn.endswith('_.pkl') else pfn[:-4] + '_.pkl'
            probs = pkl.load(open(pfn, 'rb'))
            print(f'\n[Budget {budget}ms] Loaded cached scheduler from {candidate}')
            break

    if ea is None:
        print(f'\n[Budget {budget}ms] Training EENet scheduler...')
        ea, probs_t = fit_exit_assigner(
            val_pred,
            val_target.float(),
            costs,
            budget,
            alpha_ce=ip['alpha_ce'],
            alpha_cost=ip['alpha_cost'],
            beta_thr=0,
            beta_ce=ip['beta_ce'],
            lr=ip['lr'],
            weight_decay=ip['weight_decay'],
            num_epoch=ip['num_epoch'],
            batch_size=ip['bs'],
            hidden_dim_rate=ip['hidden_dim_rate'],
            period=ip['period'],
            conf_mode='nn',
        )
        probs = probs_t.tolist() if not isinstance(probs_t, list) else probs_t
        pkl.dump(ea, open(ea_path, 'wb'))
        pkl.dump(probs, open(probs_path, 'wb'))
        print(f'  Saved -> {ea_path}')

    thresholds = ea.get_threshold().detach().cpu().numpy()
    print(f'  EENet thresholds: {np.round(thresholds, 4)}')
    print(f'  Target exit probs: {[round(p, 3) for p in probs]}')

    test_nn = test_exit_assigner(args, test_pred, n_stage, ea)

    acc, exp_time, exit_counts = 0, 0.0, torch.zeros(n_stage)
    for i in range(n_test):
        for k in range(n_stage):
            if test_nn[k, i].item() >= thresholds[k]:
                if test_pred[k, i].argmax().item() == test_target[i].item():
                    acc += 1
                exit_counts[k] += 1
                exp_time += seconds[k]
                break
    eenet_acc = acc * 100.0 / n_test
    eenet_time = exp_time / n_test
    eenet_dist = [round(v.item(), 3) for v in exit_counts / n_test]
    print(f"\n{budget:>8.2f} | {'EENet':>10} | {eenet_acc:>8.2f}% | {eenet_time:>7.2f}ms | {eenet_dist}")

    ent_acc, ent_time, ent_dist = eval_entropy_baseline(test_pred, test_target, probs, seconds)
    ent_dist_r = [round(v, 3) for v in ent_dist]
    print(f"{budget:>8.2f} | {'Entropy':>10} | {ent_acc:>8.2f}% | {ent_time:>7.2f}ms | {ent_dist_r}")
    print(f'  -> EENet gain: {eenet_acc - ent_acc:+.2f}%')
    print(f"{'-' * 80}")

print('\nScheduling complete.')
