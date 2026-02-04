# Phase 0 Baseline Sweep Results

**Subject**: sub-verse563  
**Date**: 2026-01-28  
**Total configs**: 12 (2 ops × 2 radii × 3 p_apply values)

## Clean Baseline
- **Mean Dice**: 0.9057
- **Mean IoU**: 0.8296

## Corruption Impact Summary

| Operation | Radius | p_apply | Corrupt Dice | Corrupt IoU | Δ Dice | Δ IoU |
|-----------|--------|---------|--------------|-------------|--------|-------|
| erode     | 1      | 0.25    | 0.8919       | 0.8074      | -0.014 | -0.022 |
| erode     | 1      | 0.50    | 0.8865       | 0.7985      | -0.019 | -0.031 |
| erode     | 1      | 0.75    | 0.8708       | 0.7743      | -0.035 | -0.055 |
| erode     | 2      | 0.25    | 0.8599       | 0.7623      | -0.046 | -0.067 |
| erode     | 2      | 0.50    | 0.8423       | 0.7355      | -0.063 | -0.094 |
| erode     | 2      | 0.75    | 0.7945       | 0.6672      | -0.111 | -0.162 |
| dilate    | 1      | 0.25    | 0.8938       | 0.8105      | -0.012 | -0.019 |
| dilate    | 1      | 0.50    | 0.8895       | 0.8034      | -0.016 | -0.026 |
| dilate    | 1      | 0.75    | 0.8790       | 0.7864      | -0.027 | -0.043 |
| dilate    | 2      | 0.25    | 0.8719       | 0.7782      | -0.034 | -0.051 |
| dilate    | 2      | 0.50    | 0.8594       | 0.7586      | -0.046 | -0.071 |
| dilate    | 2      | 0.75    | 0.8294       | 0.7128      | -0.076 | -0.117 |

## Key Findings
1. **Erosion is more destructive than dilation** at the same radius/p_apply
2. **Radius=2 causes significantly more degradation** than radius=1
3. **Performance degrades roughly linearly** with p_apply (probability of applying corruption per label)
4. **Worst case** (erode r=2 p=0.75): 11% Dice drop, 16% IoU drop

## Next Steps (Phase 1)
- Implement proxy abnormal transformations (vertebra-level rigid transforms)
- Run TS on proxy abnormal CT to generate "teacher failure" distribution
- Design RL adversary to mimic teacher failures while maximizing assembly challenge
