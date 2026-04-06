# EENet — Decentralized Early Exit CNN Inference

Adaptive inference with early exiting for multi-exit DNNs. Easy samples exit early; harder ones continue deeper. Based on [Ilhan et al., WACV 2024](https://arxiv.org/abs/2301.07099).

## Setup

```
Python 3.8+, PyTorch 1.12+
```

## Train

```bash
python main.py --data-root datasets --data cifar100 --arch densenet121_4 --use-valid
```

## Phase 2: EENet Scheduling

```bash
python run_scheduling.py
```

Trains exit scoring functions (gk) for each budget level and compares against entropy baseline.

## Partition Model

```bash
python partition_model.py
```

Splits the trained model into 4 segments for distributed inference. Each segment includes its exit classifier, scorer, and threshold. Verifies outputs match end-to-end.

## Parameters

All training/inference/model parameters are in `config.py`.
