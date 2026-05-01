# Distributed Trust-Aware Early-Exit CNN Inference

This project extends EENet into a distributed inference system where a multi-exit DenseNet is partitioned across 4 stages and replicated across 16 independent node processes. A central router sends requests through the network, uses EigenTrust scoring to route around unreliable nodes, and adjusts early-exit thresholds based on node trust.

The system builds on two prior ideas: EENet for learned early-exit CNN inference, introduced by [Ilhan et al., WACV 2024](https://arxiv.org/abs/2301.07099), and EigenTrust for reputation-based trust scoring in distributed systems, introduced by [Kamvar et al.](https://nlp.stanford.edu/pubs/eigentrust.pdf). Our implementation adapts EigenTrust to inference by scoring nodes using accuracy, latency compliance, and confidence calibration. The trust implementation also references the [PygenTrust implementation](https://github.com/mattyTokenomics/PygenTrust).


## What We Built

- Segmented a 4 exit DenseNet121 model into 4 sequential inference stages
- Replicated each stage across 4 independent OS processes, for 16 total nodes
- Used ZMQ over TCP for router to node communication
- Added fault injection for dropped, corrupted, and slow node responses
- Adapted EigenTrust to score nodes using accuracy, latency, and confidence calibration
- Added trust aware routing and trust-coupled early exit thresholds


## How This Extends EENet

Original EENet runs early exit inference inside a single reliable model process. Our system keeps the early-exit idea, but moves inference into a distributed environment where different model segments run as separate nodes.

The key difference is that early exit decisions are no longer only about model confidence. They also depend on whether the node producing that confidence has behaved reliably during inference.


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Requires Python 3.8+.

## Training

```bash
python3 main.py --data-root datasets --data cifar100 --arch densenet121_4 --use-valid
```

## EENet Scheduling

```bash
python3 run_scheduling.py --evaluate-from outputs/report_demo_train/save_models/checkpoint_000.pth.tar
```

This trains the exit scoring functions used by EENet and compares them against an entropy baseline.

## Distributed Trust Experiments

```bash
python3 run_experiments.py --n-samples 250 --seeds 0,1,2
```

This starts the 16 node distributed network and routes CIFAR-100 samples through the partitioned model under easy, medium, and hard fault scenarios.
Results are written to `outputs/results/` by default.

To run only the hard scenario:

```bash
python3 run_experiments.py --scenarios hard --policies random,trust --n-samples 250 --seeds 0,1,2
```

The default trust-coupled exit adjustment is `0.1`.

## Report Assets

```bash
python3 analysis/generate_report_assets.py
```

This generates report tables and figures in `outputs/report_assets/`.
Graph images are also collected in `graphs/`.

To compare trust routing with and without trust-coupled exit thresholds:

```bash
python3 run_experiments.py --scenarios hard --policies trust --n-samples 250 --seeds 0,1,2 --trust-exit-adjustment 0.0 --results-dir outputs/results_no_exit_adjustment
python3 run_experiments.py --scenarios hard --policies trust --n-samples 250 --seeds 0,1,2 --results-dir outputs/results_exit_adjustment_0_1
python3 analysis/generate_report_assets.py --results-dir outputs/results --no-exit-adjustment-dir outputs/results_no_exit_adjustment --tuned-exit-adjustment-dir outputs/results_exit_adjustment_0_1
```

This also generates `hard_scenario_accuracy_gains.png`, which shows the hard scenario gain from random routing to trust aware routing with tuned trust coupled exits.

## Model Partitioning

```bash
python3 partition_model.py
```

This splits the trained DenseNet121 model into 4 sequential segments for distributed inference.

All training, inference, and model parameters are defined in `config.py`.
