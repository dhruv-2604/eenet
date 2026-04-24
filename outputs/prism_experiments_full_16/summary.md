# PRISM Experiment Summary

## Best / Worst / Average
- Best case: trust routing in the easy scenario reached 68.75% accuracy with 100.00% reliability.
- Average case: trust routing in the medium scenario reached 66.67% accuracy with 66.67% on-time reliability and 0.00% dropped responses.
- Worst case: random routing in the hard scenario fell to 21.88% accuracy with 12.50% on-time reliability.

## Takeaways
- Early exits reduce compute: exit 1 is the fastest point, while later exits recover accuracy.
- The strongest clean-budget result came from EENet at 6.00 ms with 68.75% accuracy.
- Reliability measures completed responses that stay within the latency budget, not just any returned response.
- In the hard scenario, trust routing eliminated dropped responses from 7.29% to 0.00% while improving accuracy from 21.88% to 41.67%.
- Per-exit metrics and trust-routing summaries were written alongside the plots for the report.
