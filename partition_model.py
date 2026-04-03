#!/usr/bin/env python3
"""
Partition DenseNet121-4exit into 4 segments for distributed inference.
Each segment bundles its backbone sub-network, exit classifier,
EENet scoring function (gk / ScoreNormalizer), and threshold.

Layer map (CIFAR-100, 32×32 inputs, growth_rate=12):
  Segment 1: conv1 → dense1(6 blocks) → trans1   → ee_cls[0] | scorer[0] | T[0]
             output feat: (B, 48, 16, 16)
  Segment 2: dense2(12 blocks) → trans2           → ee_cls[1] | scorer[1] | T[1]
             output feat: (B, 96, 8, 8)
  Segment 3: dense3(24 blocks) → trans3           → ee_cls[2] | scorer[2] | T[2]
             output feat: (B, 192, 4, 4)
  Segment 4: dense4(16 blocks) → BN → relu → pool → flatten → linear (always exits)
             output: (B, 100)

Inter-segment protocol for distributed inference:
  Each segment sends (feat_activation, score_vector) to the next process.
  score_vector grows by 1 element per exit processed.

Usage:
    cd EENet && python partition_model.py
"""
import os, sys, math, pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models
from config import Config
from args import arg_parser, modify_args
from utils.predict_helpers import prepare_input
from utils.predict_utils import test_exit_assigner

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH   = 'outputs/densenet121_4_cifar100/save_models/model_best.pth.tar'
SAVE_PATH    = 'outputs/densenet121_4_None_cifar100'
SEGMENTS_DIR = 'outputs/segments'
DATA_ROOT    = 'datasets'
BUDGET_FOR_VERIFY = 6.5   # use the 6.5ms scheduler (already trained) for verification
N_VERIFY     = 10
os.makedirs(SEGMENTS_DIR, exist_ok=True)

# ── Build args ─────────────────────────────────────────────────────────────
args = arg_parser.parse_args([
    '--data', 'cifar100', '--data-root', DATA_ROOT,
    '--save-path', SAVE_PATH, '--arch', 'densenet121_4', '--use-valid',
    '--evalmode', 'dynamic', '--evaluate-from', MODEL_PATH,
])
args = modify_args(args)
args.print_freq = 100
args.use_gpu = False
args.device  = torch.device('cpu')
torch.manual_seed(0)

config = Config()
args.num_exits = config.model_params[args.data][args.arch]['num_blocks']  # 4

# ── Load model ─────────────────────────────────────────────────────────────
model = getattr(models, args.arch)(args, {**config.model_params[args.data][args.arch]})
sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)['state_dict']
model.load_state_dict(sd, strict=False)
model.eval()
print("Model loaded.")

# ── Load EENet scheduler ───────────────────────────────────────────────────
# Try both filename variants (with/without trailing _)
for suffix in ['_.pkl', '.pkl']:
    ea_path = os.path.join(SAVE_PATH, f'ea_pkls/ea_{BUDGET_FOR_VERIFY}{suffix}')
    if os.path.exists(ea_path):
        ea = pkl.load(open(ea_path, 'rb'))
        print(f"Loaded scheduler: {ea_path}")
        break
else:
    raise FileNotFoundError(
        f"No scheduler found for budget={BUDGET_FOR_VERIFY}ms. "
        "Run run_scheduling.py first."
    )

T = ea.get_threshold().detach().cpu().numpy()
print(f"EENet thresholds T: {np.round(T, 4)}")

# ──────────────────────────────────────────────────────────────────────────
# Segment module definitions
# Each segment holds: backbone layers + exit classifier + scorer + threshold.
#
# compute_score(logit, past_scores) mirrors test_exit_assigner's per-exit logic:
#   1. Softmax the current exit's logit → (B, C)
#   2. Build feature vector X via prepare_input → (B, C+2)
#   3. Append past accumulated scores → (B, C+2+k) where k = exit index
#   4. Feed to scorer (ScoreNormalizer.predict) → (B, 1, 1)
#   5. Flatten to (B,)
#
# For distributed inference, the caller:
#   - Passes feat_in (activation from previous segment)
#   - Passes past_scores (B, k) tensor of raw scores from exits 0..k-1
#   - Receives feat_out, logit, score_k
#   - Decides to exit if score_k.item() >= self.threshold (for B=1)
# ──────────────────────────────────────────────────────────────────────────

class Segment1(nn.Module):
    """
    Input : raw image (B, 3, 32, 32)
    Output: feat (B, 48, 16, 16), logit (B, 100)
    Owns  : conv1, dense1, trans1, ee_cls[0], scorer[0], T[0]
    """
    def __init__(self, model, scorer, threshold):
        super().__init__()
        self.conv1    = model.conv1
        self.dense1   = model.dense1
        self.trans1   = model.trans1
        self.ee_cls   = model.ee_classifiers[0]
        self.scorer   = scorer
        self.threshold = threshold

    def forward(self, x):
        feat  = self.trans1(self.dense1(self.conv1(x)))
        logit = self.ee_cls(feat)
        return feat, logit

    def compute_score(self, logit, past_scores=None):
        # past_scores: None (first exit, no history)
        with torch.no_grad():
            soft = torch.softmax(logit, dim=1)          # (B, C)
            X, _ = prepare_input(soft.unsqueeze(0), k=0)  # (B, C+2)
            # past_scores is None for exit 0 — no concat needed
            raw = self.scorer.predict(X)                # (B, 1, 1)
            return raw.reshape(logit.shape[0])          # (B,)

    def should_exit(self, score):
        return bool((score >= self.threshold).all().item())


class Segment2(nn.Module):
    """
    Input : feat from Segment1 (B, 48, 16, 16)
    Output: feat (B, 96, 8, 8), logit (B, 100)
    Owns  : dense2, trans2, ee_cls[1], scorer[1], T[1]
    """
    def __init__(self, model, scorer, threshold):
        super().__init__()
        self.dense2    = model.dense2
        self.trans2    = model.trans2
        self.ee_cls    = model.ee_classifiers[1]
        self.scorer    = scorer
        self.threshold = threshold

    def forward(self, feat_in):
        feat  = self.trans2(self.dense2(feat_in))
        logit = self.ee_cls(feat)
        return feat, logit

    def compute_score(self, logit, past_scores):
        # past_scores: (B, 1) — score from exit 0
        with torch.no_grad():
            soft = torch.softmax(logit, dim=1)
            X, _ = prepare_input(soft.unsqueeze(0), k=0)  # (B, C+2)
            X = torch.cat([X, past_scores.reshape(logit.shape[0], -1)], dim=1)  # (B, C+3)
            raw = self.scorer.predict(X)
            return raw.reshape(logit.shape[0])

    def should_exit(self, score):
        return bool((score >= self.threshold).all().item())


class Segment3(nn.Module):
    """
    Input : feat from Segment2 (B, 96, 8, 8)
    Output: feat (B, 192, 4, 4), logit (B, 100)
    Owns  : dense3, trans3, ee_cls[2], scorer[2], T[2]
    """
    def __init__(self, model, scorer, threshold):
        super().__init__()
        self.dense3    = model.dense3
        self.trans3    = model.trans3
        self.ee_cls    = model.ee_classifiers[2]
        self.scorer    = scorer
        self.threshold = threshold

    def forward(self, feat_in):
        feat  = self.trans3(self.dense3(feat_in))
        logit = self.ee_cls(feat)
        return feat, logit

    def compute_score(self, logit, past_scores):
        # past_scores: (B, 2) — scores from exits 0 and 1
        with torch.no_grad():
            soft = torch.softmax(logit, dim=1)
            X, _ = prepare_input(soft.unsqueeze(0), k=0)  # (B, C+2)
            X = torch.cat([X, past_scores.reshape(logit.shape[0], -1)], dim=1)  # (B, C+4)
            raw = self.scorer.predict(X)
            return raw.reshape(logit.shape[0])

    def should_exit(self, score):
        return bool((score >= self.threshold).all().item())


class Segment4(nn.Module):
    """
    Input : feat from Segment3 (B, 192, 4, 4)
    Output: logit (B, 100) — always exits, no scorer needed
    Owns  : dense4, bn, linear
    Note  : threshold = -inf (always exit), no scorer component
    """
    def __init__(self, model):
        super().__init__()
        self.dense4    = model.dense4
        self.bn        = model.bn
        self.linear    = model.linear
        self.threshold = float('-inf')  # always exits

    def forward(self, feat_in):
        out = self.dense4(feat_in)
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out)


# ── Instantiate segments ───────────────────────────────────────────────────
seg1 = Segment1(model, ea.score_normalizers[0], float(T[0]))
seg2 = Segment2(model, ea.score_normalizers[1], float(T[1]))
seg3 = Segment3(model, ea.score_normalizers[2], float(T[2]))
seg4 = Segment4(model)

# ── Layer sequence and activation shapes ──────────────────────────────────
print("\n" + "="*60)
print("LAYER SEQUENCE AND ACTIVATION SHAPES")
print("="*60)

dummy = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    print(f"Input image             : {tuple(dummy.shape)}")
    f0 = model.conv1(dummy)
    print(f"After conv1             : {tuple(f0.shape)}")
    f1_pre = model.dense1(f0)
    print(f"After dense1 (6 blocks) : {tuple(f1_pre.shape)}")
    f1 = model.trans1(f1_pre)
    print(f"After trans1            : {tuple(f1.shape)}  ← EXIT 1 boundary  (~layer 12)")

    f2_pre = model.dense2(f1)
    print(f"After dense2 (12 blocks): {tuple(f2_pre.shape)}")
    f2 = model.trans2(f2_pre)
    print(f"After trans2            : {tuple(f2.shape)}  ← EXIT 2 boundary  (~layer 36)")

    f3_pre = model.dense3(f2)
    print(f"After dense3 (24 blocks): {tuple(f3_pre.shape)}")
    f3 = model.trans3(f3_pre)
    print(f"After trans3            : {tuple(f3.shape)}  ← EXIT 3 boundary  (~layer 84)")

    f4 = model.dense4(f3)
    print(f"After dense4 (16 blocks): {tuple(f4.shape)}")
    f4p = F.adaptive_avg_pool2d(F.relu(model.bn(f4)), (1, 1)).view(1, -1)
    print(f"After bn+pool+flatten   : {tuple(f4p.shape)}  ← EXIT 4 (final, ~layer 121)")

print("\nExit classifier architecture (identical for all exits):")
print("  Conv3×3 → BN → ReLU  (×3 conv layers, with initial channel reduction)")
print("  AdaptiveAvgPool → Flatten → Linear → (B, 100)")

print("\nEENet scorer architecture (ScoreNormalizer) per exit k:")
print("  Input dim: 102 + k  (100 class probs + max_score + entropy + k past scores)")
print("  q_layer_1 : Linear(100, 1)  — weighted class evidence")
print("  q_layer_2 : Linear(2+k, 1)  — weighted confidence features + past")
print("  Output    : raw score estimating P(correct prediction at exit k)")
print("  Threshold : T[k] — if score ≥ T[k], sample exits at k")

print("\nInter-segment communication payload:")
print("  Seg1→Seg2: feat (B,48,16,16)  + score_0 (B,)")
print("  Seg2→Seg3: feat (B,96,8,8)    + scores_01 (B,2)")
print("  Seg3→Seg4: feat (B,192,4,4)   + scores_012 (B,3)  [unused by seg4]")

# ── Save segments as .pt files ─────────────────────────────────────────────
print(f"\nSaving segments to {SEGMENTS_DIR}/")
for idx, seg in enumerate([seg1, seg2, seg3, seg4], start=1):
    torch.save(seg, os.path.join(SEGMENTS_DIR, f'segment{idx}.pt'))
    print(f"  segment{idx}.pt  saved")

# ── Verification ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"VERIFICATION: {N_VERIFY} test images")
print("="*60)

normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                  std=[0.2675, 0.2565, 0.2761])
test_set = datasets.CIFAR100(DATA_ROOT, train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(), normalize
                               ]))
imgs   = torch.stack([test_set[i][0] for i in range(N_VERIFY)])
labels = [test_set[i][1] for i in range(N_VERIFY)]

seg1.eval(); seg2.eval(); seg3.eval(); seg4.eval()
softmax_fn = nn.Softmax(dim=1)

# Step 1: Full model forward pass (single call, all exits)
with torch.no_grad():
    full_logits_batch, _ = model(imgs, manual_early_exit_index=0)
    full_logits = [l.cpu() for l in full_logits_batch]  # list of 4 × (10, 100)

# Step 2: Segmented forward pass (sequential, passing activations between segments)
with torch.no_grad():
    seg_logits = []
    for i in range(N_VERIFY):
        xi = imgs[i:i+1]
        f1i, l1i = seg1(xi)
        f2i, l2i = seg2(f1i)
        f3i, l3i = seg3(f2i)
        l4i       = seg4(f3i)
        seg_logits.append([l.cpu() for l in [l1i, l2i, l3i, l4i]])

# Step 3: Assert logit outputs match (full vs segmented)
print("\n[Test 1] Logit outputs match (full model vs segmented pass):")
tol = 1e-4
all_match = True
for i in range(N_VERIFY):
    for k in range(4):
        diff = (full_logits[k][i] - seg_logits[i][k][0]).abs().max().item()
        if diff > tol:
            print(f"  MISMATCH  img={i} exit={k+1}  max_diff={diff:.6f}")
            all_match = False
if all_match:
    print(f"  ALL {N_VERIFY}×4 outputs match within tol={tol}  ✓")

# Step 4: Compute exit decisions via EENet for both full and segmented
# Build softmax pred tensors of shape (n_stage, N_VERIFY, 100)
full_pred_soft = torch.zeros(4, N_VERIFY, 100)
for k in range(4):
    full_pred_soft[k] = softmax_fn(full_logits[k])

seg_pred_soft = torch.zeros(4, N_VERIFY, 100)
for i in range(N_VERIFY):
    for k in range(4):
        seg_pred_soft[k, i] = softmax_fn(seg_logits[i][k])

# Use test_exit_assigner (the exact same function used in production)
full_nn = test_exit_assigner(args, full_pred_soft, 4, ea)  # (4, N_VERIFY)
seg_nn  = test_exit_assigner(args, seg_pred_soft,  4, ea)  # (4, N_VERIFY)

def decisions_from_nn(nn_arr, thresholds):
    decisions = []
    for i in range(nn_arr.shape[1]):
        for k in range(nn_arr.shape[0]):
            if nn_arr[k, i].item() >= thresholds[k]:
                decisions.append(k + 1)
                break
    return decisions

full_exits = decisions_from_nn(full_nn, T)
seg_exits  = decisions_from_nn(seg_nn,  T)

print(f"\n[Test 2] Exit decisions match (same images exit at same point):")
decision_match = True
print(f"  {'Image':>5} | {'Label':>5} | {'Full exit':>9} | {'Seg exit':>8} | Match")
for i in range(N_VERIFY):
    match_sym = "✓" if full_exits[i] == seg_exits[i] else "✗"
    print(f"  {i:>5} | {labels[i]:>5} | {full_exits[i]:>9} | {seg_exits[i]:>8} | {match_sym}")
    if full_exits[i] != seg_exits[i]:
        decision_match = False

if decision_match:
    print(f"\n  ALL {N_VERIFY} exit decisions match  ✓")
else:
    print(f"\n  WARNING: some decisions differ")

# Step 5: Quick summary of segment files
print(f"\n{'='*60}")
print("SAVED SEGMENT FILES")
print("="*60)
for fname in sorted(os.listdir(SEGMENTS_DIR)):
    fpath = os.path.join(SEGMENTS_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {fname}  ({size_kb:.0f} KB)")

print(f"\nPartitioning complete.")
print(f"Segments are in: {SEGMENTS_DIR}/")
print(f"\nTo load a segment in a peer process:")
print(f"  seg = torch.load('outputs/segments/segment1.pt', weights_only=False)")
print(f"  seg.eval()")
print(f"  feat, logit = seg(img_tensor)")
print(f"  score = seg.compute_score(logit, past_scores=None)")
print(f"  if seg.should_exit(score): return logit.argmax()")
