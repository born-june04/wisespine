# TS mask-based reconstruction: limitation summary
- **N subjects**: 25

## Key rates (what breaks)
| Signal | Meaning | Rate |
|---|---|---:|
| extra_vs_gt | GT coverage mismatch (not necessarily error) | 100.0% |
| transition violations | violates validTransition in predicted order | 16.0% |
| axis outliers | centroid far from estimated spine axis | 56.0% |
| non-bone | HU p90 inside mask below threshold | 88.0% |
| spurious_non_spine_proxy | (axis outlier OR non-bone) | 92.0% |

## Averages (severity)
| Metric | Mean | Median |
|---|---:|---:|
| mean_dice | 0.851 | 0.895 |
| extra_vs_gt_count | 2.28 | 2.00 |
| transition_violations_count | 0.44 | 0.00 |
| spurious_non_spine_proxy_count | 7.32 | 7.00 |

## Top examples (spurious non-spine proxy)
| subject | mean_dice | spurious | non_bone | axis_outliers | extra_vs_gt |
|---|---:|---:|---:|---:|---|
| sub-verse512 | 0.928 | 20 | 20 | 0 | 4 5 6 25 |
| sub-verse560 | 0.756 | 15 | 14 | 1 | 6 7 |
| sub-verse552 | 0.840 | 15 | 14 | 1 | 5 6 7 24 25 |
| sub-verse540 | 0.817 | 14 | 14 | 1 | 6 7 |
| sub-verse555 | 0.943 | 12 | 12 | 2 | 14 15 25 |
