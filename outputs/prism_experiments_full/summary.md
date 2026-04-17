# PRISM Experiment Summary

## Best / Worst / Average
- Best case: trust routing in the easy scenario reached 66.15% accuracy with 100.00% reliability.
- Average case: trust routing in the medium scenario reached 62.07% accuracy with 99.93% reliability and 0.07% dropped responses.
- Worst case: random routing in the hard scenario fell to 34.97% accuracy with 98.98% reliability.

## Takeaways
- Early exits reduce compute: exit 1 is the fastest point, while later exits recover accuracy.
- The strongest clean-budget result came from EENet at 7.50 ms with 63.79% accuracy.
- In the hard scenario, trust routing cut dropped responses from 1.02% to 0.23% while improving accuracy from 34.97% to 49.15%.
- Per-exit metrics and trust-routing summaries were written alongside the plots for the report.