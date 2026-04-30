# EENet — Distributed Early Exit CNN Inference

Adaptive inference with early exiting for multi-exit DNNs. Easy samples exit early; harder ones continue deeper. Based on [Ilhan et al., WACV 2024](https://arxiv.org/abs/2301.07099).

The trust scoring implementation is based on EigenTrust, with reference to the
[PygenTrust implementation](https://github.com/mattyTokenomics/PygenTrust).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Requires Python 3.8+.

## Train

```bash
python3 main.py --data-root datasets --data cifar100 --arch densenet121_4 --use-valid
```

## Phase 2: EENet Scheduling

```bash
python3 run_scheduling.py --evaluate-from outputs/report_demo_train/save_models/checkpoint_000.pth.tar
```

Trains exit scoring functions (gk) for each budget level and compares against entropy baseline.

## Distributed Trust Experiments

```bash
python3 run_experiments.py --n-samples 250 --seeds 0,1,2
```

Launches the distributed peer network and routes CIFAR-100 samples through the
segmented DenseNet121 model for:
- easy, medium, and hard fault scenarios
- random vs. trust-based routing
- per-seed JSON logs and aggregated accuracy / reliability / latency summaries

The script writes per-run JSON files and `aggregated_results.json` into
`outputs/results/` by default.

To run only the hard scenario with tuned trust-coupled exits:

```bash
python3 run_experiments.py --scenarios hard --policies random,trust --n-samples 250 --seeds 0,1,2 --trust-exit-adjustment 0.1 --results-dir outputs/results_exit_adjustment_0_1
```

## Report Assets

```bash
python3 analysis/generate_report_assets.py
```

Builds the focused report table and the two highest-signal plots from the
checked-in experiment outputs. The generated files are written to
`outputs/report_assets/`.

Report-ready graph images are also collected in `graphs/` for easier review.

To regenerate the tuned hard-scenario adjusted-exit seed check:

```bash
python3 run_experiments.py --scenarios hard --policies trust --n-samples 250 --seeds 0,1,2 --trust-exit-adjustment 0.1 --results-dir outputs/results_exit_adjustment_0_1
python3 analysis/generate_report_assets.py --results-dir outputs/results --tuned-exit-adjustment-dir outputs/results_exit_adjustment_0_1 --tuned-exit-adjustment 0.1
```

To compare trust routing with and without trust-coupled exit thresholds:

```bash
python3 run_experiments.py --scenarios hard --policies trust --n-samples 250 --seeds 0,1,2 --trust-exit-adjustment 0.0 --results-dir outputs/results_no_exit_adjustment
python3 run_experiments.py --scenarios hard --policies trust --n-samples 250 --seeds 0,1,2 --trust-exit-adjustment 0.1 --results-dir outputs/results_exit_adjustment_0_1
python3 analysis/generate_report_assets.py --results-dir outputs/results --no-exit-adjustment-dir outputs/results_no_exit_adjustment --tuned-exit-adjustment-dir outputs/results_exit_adjustment_0_1 --tuned-exit-adjustment 0.1
```

This also generates `hard_scenario_accuracy_gains_exit_adjust_0_1.png`, which shows
the hard-scenario gain from random routing to trust-aware routing with tuned
trust-coupled exits.

## Partition Model

```bash
python3 partition_model.py
```

Splits the trained model into 4 segments for distributed inference. Each segment includes its exit classifier, scorer, and threshold. Verifies outputs match end-to-end.

## Parameters

All training/inference/model parameters are in `config.py`.
