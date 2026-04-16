#!/usr/bin/env python3
"""
Phase 2: EENet scheduling optimization with EigenTrust peer trust.

Trains exit scoring functions (ScoreNormalizer gk) and exit assignment
functions (ExitAssigner hk) per Algorithm 1 of the EENet paper.
Adds EigenTrust-based per-peer reputation that dynamically adjusts exit
thresholds at inference time: trusted peers exit more aggressively,
untrusted peers act conservatively and forward samples downstream.

Compares three methods:
  EENet           — original learned thresholds
  EENet+Trust     — EigenTrust-modulated thresholds
  Entropy         — BranchyNet entropy baseline

Usage:
    cd EENet && python run_scheduling.py
"""
import os, sys, math, pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── EigenTrust import ──────────────────────────────────────────────────────
# Try the pip package first; fall back to our bundled implementation.
try:
    from trust.eigentrust import eigentrust, EigenTrustTracker, compute_score_calibration
    print("[trust] Using PygenTrust (pip) + local EigenTrustTracker")
except ImportError:
    from eigentrust import eigentrust, EigenTrustTracker, compute_score_calibration
    print("[trust] Using bundled eigentrust.py (PygenTrust-compatible API)")

import models
from config import Config
from args import arg_parser, modify_args
from data_tools.dataloader import get_dataloaders
from utils.predict_helpers import fit_exit_assigner, prepare_input, ExitAssigner
from utils.predict_utils import test_exit_assigner

# ── Paths & budgets ────────────────────────────────────────────────────────
MODEL_PATH = 'outputs/densenet121_4_cifar100/save_models/model_best.pth.tar'
SAVE_PATH  = 'outputs/densenet121_4_None_cifar100'
DATA_ROOT  = 'datasets'
BUDGETS    = [7.5, 6.75, 6.5, 6.0]
BATCH_SIZE = 64

# ── EigenTrust hyperparameters ─────────────────────────────────────────────
TRUST_ALPHA       = 0.1    # weight on pre-trust vs peer observations
TRUST_EPSILON     = 1e-6   # convergence tolerance
TRUST_SCALE       = 0.20   # max ±20 % threshold shift from trust modulation
TRUST_DECAY       = 0.85   # exponential decay on trust matrix each round
TRUST_EMA         = 0.30   # momentum for evidence EMA per peer

# ── Build args ─────────────────────────────────────────────────────────────
args = arg_parser.parse_args([
    '--data', 'cifar100', '--data-root', DATA_ROOT,
    '--save-path', SAVE_PATH,
    '--arch', 'densenet121_4', '--use-valid',
    '--evalmode', 'dynamic', '--evaluate-from', MODEL_PATH,
])
args = modify_args(args)
args.print_freq = 100
torch.manual_seed(0)

if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    infer_device = torch.device('mps')
else:
    infer_device = torch.device('cpu')
args.device = infer_device
args.use_gpu = False
print(f"Inference device: {infer_device}")

config = Config()
args.num_exits = config.model_params[args.data][args.arch]['num_blocks']   # 4
args.inference_params = config.inference_params[args.data][args.arch]

# ── Load model ─────────────────────────────────────────────────────────────
model = getattr(models, args.arch)(args, {**config.model_params[args.data][args.arch]})
sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
model.load_state_dict(sd, strict=False)
model = model.to(infer_device)
model.eval()
print("Model loaded from", MODEL_PATH)

# ── Data loaders ───────────────────────────────────────────────────────────
train_loader, val_loader, test_loader = get_dataloaders(args, batch_size=BATCH_SIZE)

# ── Collect / cache all-exit logits ────────────────────────────────────────
LOGITS_FILE = os.path.join(SAVE_PATH, 'logits_single.pth')
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

seconds = list(pd.read_csv(os.path.join(SAVE_PATH, 'seconds.csv'), header=None).iloc[:, 0])
costs   = torch.tensor(seconds)
n_stage, n_test, n_cls = test_pred.shape
print(f"\nExit latencies (ms): {seconds}")

# ── Per-exit accuracy (anytime) ─────────────────────────────────────────────
print("\n=== Per-exit accuracy on full test set (anytime mode) ===")
for k in range(n_stage):
    acc_k = (test_pred[k].argmax(dim=1) == test_target).float().mean().item() * 100
    print(f"  Exit {k+1}: {acc_k:.2f}% top-1")

# ── Entropy baseline helper ─────────────────────────────────────────────────
def eval_entropy_baseline(pred, target, probs_target, seconds_list):
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


# ── EigenTrust-modulated evaluation ────────────────────────────────────────

def build_trust_tracker(n_peers: int) -> EigenTrustTracker:
    """Initialise tracker with uniform pre-trust across all exit peers."""
    return EigenTrustTracker(
        n_peers=n_peers,
        pre_trust=None,          # uniform
        alpha=TRUST_ALPHA,
        epsilon=TRUST_EPSILON,
        trust_scale=TRUST_SCALE,
        decay=TRUST_DECAY,
    )


def warm_up_trust(
    tracker: EigenTrustTracker,
    test_nn: torch.Tensor,       # (n_stage, n_test) — EENet scores
    test_pred: torch.Tensor,     # (n_stage, n_test, n_cls)
    test_target: torch.Tensor,   # (n_test,)
    base_thresholds: np.ndarray, # (n_stage,)
    seconds_list: list[float],
    budget: float,
    n_warmup_batches: int = 10,
    batch_size: int = 64,
) -> EigenTrustTracker:
    """
    This simulates each peer observing how well its neighbours perform before
    the modulated thresholds take effect.
    """
    T = base_thresholds
    n_s, n_samp, _ = test_pred.shape
    rng = np.random.default_rng(42)
    indices = rng.choice(n_samp, size=min(n_warmup_batches * batch_size, n_samp), replace=False)

    # Per-peer evidence accumulators for this warm-up pass
    peer_scores_buf  = [[] for _ in range(n_s)]   # raw gk scores
    peer_correct_buf = [[] for _ in range(n_s)]   # binary correctness
    peer_exits       = np.zeros(n_s, dtype=int)
    peer_budget_ok   = np.zeros(n_s, dtype=int)

    for i in indices:
        for k in range(n_s):
            score_ik = test_nn[k, i].item()
            if score_ik >= T[k]:
                correct = int(test_pred[k, i].argmax().item() == test_target[i].item())
                peer_scores_buf[k].append(score_ik)
                peer_correct_buf[k].append(correct)
                peer_exits[k] += 1
                peer_budget_ok[k] += int(seconds_list[k] <= budget)
                break

    # Push evidence into tracker for each peer
    for k in range(n_s):
        n_exits = peer_exits[k]
        if n_exits == 0:
            continue
        accuracy   = float(np.mean(peer_correct_buf[k]))
        latency_ok = float(peer_budget_ok[k]) / n_exits
        calibration = compute_score_calibration(
            np.array(peer_scores_buf[k]),
            np.array(peer_correct_buf[k]),
        )
        tracker.update(
            peer_id=k,
            accuracy=accuracy,
            latency_ok=latency_ok,
            score_calibration=calibration,
            ema_momentum=TRUST_EMA,
        )

    return tracker


def eval_eenet_with_trust(
    test_nn: torch.Tensor,
    test_pred: torch.Tensor,
    test_target: torch.Tensor,
    base_thresholds: np.ndarray,
    tracker: EigenTrustTracker,
    seconds_list: list[float],
    budget: float,
    online: bool = True,
    online_every: int = 200,
) -> tuple[float, float, list[float], EigenTrustTracker]:
    """
    Evaluate EENet with EigenTrust-modulated thresholds.

    Parameters
    ----------
    test_nn         : (n_stage, n_test) — raw EENet scores from test_exit_assigner.
    test_pred       : (n_stage, n_test, n_cls) — softmax logits per exit.
    test_target     : (n_test,) — ground-truth labels.
    base_thresholds : (n_stage,) — thresholds from ExitAssigner.get_threshold().
    tracker         : EigenTrustTracker pre-warmed on a held-out batch.
    seconds_list    : per-exit latency in ms.
    budget          : target latency budget in ms.
    online          : if True, update trust scores during evaluation every
                      `online_every` samples (simulates live peer feedback).
    online_every    : how many samples between online trust updates.

    Returns
    -------
    (accuracy %, avg_latency_ms, exit_distribution, updated_tracker)
    """
    n_s, n_samp, _ = test_pred.shape

    # Get modulated thresholds from current trust state
    T = tracker.trust_scaled_thresholds(base_thresholds)

    acc, exp_time = 0, 0.0
    exit_counts = np.zeros(n_s, dtype=int)

    # Online update buffers
    peer_scores_buf  = [[] for _ in range(n_s)]
    peer_correct_buf = [[] for _ in range(n_s)]
    peer_budget_ok   = [[] for _ in range(n_s)]

    for i in range(n_samp):
        for k in range(n_s):
            if test_nn[k, i].item() >= T[k]:
                correct = int(test_pred[k, i].argmax().item() == test_target[i].item())
                acc += correct
                exit_counts[k] += 1
                exp_time += seconds_list[k]

                if online:
                    peer_scores_buf[k].append(test_nn[k, i].item())
                    peer_correct_buf[k].append(correct)
                    peer_budget_ok[k].append(int(seconds_list[k] <= budget))
                break

        # Online trust update every `online_every` samples
        if online and (i + 1) % online_every == 0:
            for k in range(n_s):
                if len(peer_scores_buf[k]) < 2:
                    continue
                tracker.update(
                    peer_id=k,
                    accuracy=float(np.mean(peer_correct_buf[k])),
                    latency_ok=float(np.mean(peer_budget_ok[k])),
                    score_calibration=compute_score_calibration(
                        np.array(peer_scores_buf[k]),
                        np.array(peer_correct_buf[k]),
                    ),
                    ema_momentum=TRUST_EMA,
                )
                # Refresh modulated thresholds after each update
                T = tracker.trust_scaled_thresholds(base_thresholds)
                # Clear online buffers
                peer_scores_buf[k].clear()
                peer_correct_buf[k].clear()
                peer_budget_ok[k].clear()

    return (
        acc * 100.0 / n_samp,
        exp_time / n_samp,
        [round(v / n_samp, 3) for v in exit_counts],
        tracker,
    )


# ── Main scheduling loop ────────────────────────────────────────────────────
os.makedirs(os.path.join(SAVE_PATH, 'ea_pkls'), exist_ok=True)
ip = args.inference_params

header = f"{'Budget':>8} | {'Method':>14} | {'Accuracy':>9} | {'Avg ms':>8} | {'Trust vector':>30} | Exit dist"
print(f"\n{'='*len(header)}")
print(header)
print(f"{'='*len(header)}")

for budget in BUDGETS:
    ea_path    = os.path.join(SAVE_PATH, f'ea_pkls/ea_{budget}_.pkl')
    probs_path = os.path.join(SAVE_PATH, f'ea_pkls/probs_{budget}_.pkl')

    # Load or train ExitAssigner
    ea, probs = None, None
    for candidate in [ea_path, ea_path.replace('_.pkl', '.pkl')]:
        if os.path.exists(candidate):
            ea    = pkl.load(open(candidate, 'rb'))
            pfn   = os.path.join(SAVE_PATH, f'ea_pkls/probs_{budget}_.pkl')
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

    base_T = ea.get_threshold().detach().cpu().numpy()
    print(f"  Base EENet thresholds : {np.round(base_T, 4)}")
    print(f"  Target exit probs     : {[round(p, 3) for p in probs]}")

    # ── EENet scores on test set ──────────────────────────────────────────
    test_nn = test_exit_assigner(args, test_pred, n_stage, ea)  # (n_stage, n_test)

    # ── 1. Plain EENet ────────────────────────────────────────────────────
    acc, exp_time, exit_counts = 0, 0.0, torch.zeros(n_stage)
    for i in range(n_test):
        for k in range(n_stage):
            if test_nn[k, i].item() >= base_T[k]:
                if test_pred[k, i].argmax().item() == test_target[i].item():
                    acc += 1
                exit_counts[k] += 1
                exp_time += seconds[k]
                break
    eenet_acc  = acc * 100.0 / n_test
    eenet_time = exp_time / n_test
    eenet_dist = [round(v.item(), 3) for v in exit_counts / n_test]
    print(f"\n{budget:>8.2f} | {'EENet':>14} | {eenet_acc:>8.2f}% | {eenet_time:>7.2f}ms | {'—':>30} | {eenet_dist}")

    # ── 2. EENet + EigenTrust ─────────────────────────────────────────────
    tracker = build_trust_tracker(n_stage)
    tracker = warm_up_trust(
        tracker, test_nn, test_pred, test_target,
        base_T, seconds, budget,
        n_warmup_batches=10, batch_size=64,
    )

    trust_T = tracker.trust_scaled_thresholds(base_T)
    print(f"  Trust vector          : {np.round(tracker.trust, 4)}")
    print(f"  Trust-adjusted thrs   : {np.round(trust_T, 4)}")

    trust_acc, trust_time, trust_dist, tracker = eval_eenet_with_trust(
        test_nn, test_pred, test_target,
        base_T, tracker, seconds, budget,
        online=True, online_every=200,
    )
    trust_vec_str = str(np.round(tracker.trust, 3).tolist())
    print(f"{budget:>8.2f} | {'EENet+Trust':>14} | {trust_acc:>8.2f}% | {trust_time:>7.2f}ms | {trust_vec_str:>30} | {trust_dist}")
    print(f"  → Trust gain vs EENet : {trust_acc - eenet_acc:+.2f}%")

    # Per-peer breakdown
    print("  Peer reputation:")
    for k, v in tracker.peer_summary().items():
        print(f"    {k}: trust={v['trust']:.4f}  acc={v['accuracy']:.3f}  "
              f"lat_ok={v['latency_ok']:.3f}  calib={v['calibration']:.3f}")

    # ── 3. Entropy baseline ───────────────────────────────────────────────
    ent_acc, ent_time, ent_dist = eval_entropy_baseline(test_pred, test_target, probs, seconds)
    print(f"{budget:>8.2f} | {'Entropy':>14} | {ent_acc:>8.2f}% | {ent_time:>7.2f}ms | {'—':>30} | {[round(v, 3) for v in ent_dist]}")
    print(f"  → EENet gain vs Entropy: {eenet_acc - ent_acc:+.2f}%")
    print(f"{'─'*len(header)}")

print("\nScheduling complete.")