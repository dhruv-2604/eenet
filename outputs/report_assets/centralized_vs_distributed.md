| system | scenario | policy | sample_count | accuracy_pct | reliability_pct | dropped_pct | avg_latency_ms | latency_scope | source | comparison_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Centralized full model | clean | single process | 10000 | 75.75 | 100.00 | 0.00 | 0.75 | model-only local | exit_metrics.csv final exit | Full CIFAR-100 test set; model-only timing. |
| Centralized EENet early exit | clean | EENet budget 6.50 ms | 10000 | 55.73 | 100.00 | 0.00 | 0.53 | model-only local | budget_sweep.csv | Full CIFAR-100 test set; model-only timing. |
| Distributed 16-node segments | easy | random | 1000 | 59.00 | 100.00 | 0.00 | 16.93 | wall-clock routed | outputs/16-node-network/aggregated_results.json | Distributed routed run; wall-clock includes process/ZMQ overhead. |
| Distributed 16-node segments | easy | trust | 1000 | 59.00 | 100.00 | 0.00 | 16.73 | wall-clock routed | outputs/16-node-network/aggregated_results.json | Distributed routed run; wall-clock includes process/ZMQ overhead. |
| Distributed 16-node segments | medium | random | 1000 | 53.40 | 95.00 | 5.00 | 21.58 | wall-clock routed | outputs/16-node-network/aggregated_results.json | Distributed routed run; wall-clock includes process/ZMQ overhead. |
| Distributed 16-node segments | medium | trust | 1000 | 59.00 | 100.00 | 0.00 | 16.77 | wall-clock routed | outputs/16-node-network/aggregated_results.json | Distributed routed run; wall-clock includes process/ZMQ overhead. |
| Distributed 16-node segments | hard | random | 1000 | 45.80 | 89.00 | 11.00 | 30.95 | wall-clock routed | outputs/16-node-network/aggregated_results.json | Distributed routed run; wall-clock includes process/ZMQ overhead. |
| Distributed 16-node segments | hard | trust | 250 | 57.73 | 88.80 | 11.20 | 29.05 | wall-clock routed | outputs/results_exit_adjustment_0_1/aggregated_results.json | Distributed routed run; wall-clock includes process/ZMQ overhead. |

Note: centralized rows use full-test model-only metrics; distributed rows use the checked-in routed distributed runs, so latency scopes differ.
