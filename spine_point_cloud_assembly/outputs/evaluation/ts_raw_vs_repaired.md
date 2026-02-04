# TS raw vs TS+repair baseline

- N subjects: **25**

## Aggregate metrics

| Metric | TS raw (mean/median) | TS+repair (mean/median) |
|---|---:|---:|
| NVC(validTransition) | 0.965/1.000 | 0.806/0.889 |
| strict NVC | 0.940/1.000 | 0.714/0.750 |
| SOED (vs GT order) | 0.193/0.182 | 0.468/0.500 |
| Seq length | 16.40/18.00 | 9.08/8.00 |
| Spurious(non-spine) count | 7.32/7.00 | removed 7.32/7.00 |

## Notes
- Repair baseline removes **spurious_non_spine_proxy** (HU non-bone / axis-outlier) and keeps the remaining labels in the original TS order (no GT used).
- The key takeaway is the **trade-off**: removing spurious instances alone can create **gaps** (missing intermediate levels), which hurts continuity (NVC/strict-NVC) and alignment to GT (SOED).
- This is a **heuristic**; Spinal Field aims to solve this as structured inference (and also correct ID confusion).
