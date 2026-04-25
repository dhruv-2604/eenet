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

## PRISM Trust Experiments

```bash
python run_prism_experiments.py --use-segments
```

Runs the CIFAR-100 + DenseNet121 experiment suite for:
- accuracy vs. exit depth
- accuracy vs. compute budget
- trust-based vs. random routing under injected faults

The script writes CSV summaries and graphs into `outputs/prism_experiments/`.

## Report Assets

```bash
python analysis/generate_report_assets.py
```

Builds the focused report table and the two highest-signal plots from the
checked-in experiment outputs. The generated files are written to
`outputs/report_assets/`.

Report-ready graph images are also collected in `graphs/` for easier review.

To regenerate the tuned hard-scenario adjusted-exit seed check:

```bash
python run_p2p_experiment.py --scenarios hard --policies trust --n-samples 250 --seeds 0,1,2 --trust-exit-adjustment 0.1 --results-dir outputs/p2p_results_exit_adjustment_0_1
python analysis/generate_report_assets.py --p2p-dir outputs/p2p_results_adjusted --tuned-exit-adjustment-dir outputs/p2p_results_exit_adjustment_0_1 --tuned-exit-adjustment 0.1
```

To compare trust routing with and without trust-coupled exit thresholds:

```bash
python run_p2p_experiment.py --scenarios hard --policies trust --n-samples 250 --seeds 0,1,2 --trust-exit-adjustment 0.0 --results-dir outputs/p2p_results_no_exit_adjustment
python run_p2p_experiment.py --scenarios hard --policies trust --n-samples 250 --seeds 0,1,2 --trust-exit-adjustment 0.1 --results-dir outputs/p2p_results_exit_adjustment_0_1
python analysis/generate_report_assets.py --p2p-dir outputs/p2p_results_adjusted --no-exit-adjustment-dir outputs/p2p_results_no_exit_adjustment --tuned-exit-adjustment-dir outputs/p2p_results_exit_adjustment_0_1 --tuned-exit-adjustment 0.1
```

This also generates `hard_scenario_accuracy_gains.png`, which shows the hard-scenario
gain from random routing to trust-aware routing with tuned trust-coupled exits.

## Partition Model

```bash
python partition_model.py
```

Splits the trained model into 4 segments for distributed inference. Each segment includes its exit classifier, scorer, and threshold. Verifies outputs match end-to-end.

## Parameters

All training/inference/model parameters are in `config.py`.
