# PRISM Experiment Summary

## Best / Worst / Average
- Best case: trust routing in the easy scenario reached 68.75% accuracy with 100.00% reliability.
- Average case: trust routing in the medium scenario reached 66.67% accuracy with 100.00% reliability and 0.00% dropped responses.
- Worst case: random routing in the hard scenario fell to 22.92% accuracy with 100.00% reliability.

## Takeaways
- Early exits reduce compute: exit 1 is the fastest point, while later exits recover accuracy.
- The strongest clean-budget result came from EENet at 6.00 ms with 68.75% accuracy.
- In the hard scenario, trust routing improved accuracy from 22.92% to 39.58%, with reliability remaining high at 96.88%.
- Per-exit metrics and trust-routing summaries were written alongside the plots for the report.
