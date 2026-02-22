# Bias Audit Report

_Generated: 2026-02-22 00:29:33 UTC_

## Run summary

- Accuracy: **0.7240**
- Train size: **750**
- Test size: **250**
- Fairness threshold: **0.05**
- Minimum group size: **30**

## Fairness by sex

### Disparity summary

| Metric | Difference (max-min) | Ratio (min/max) | Flag |
|---|---:|---:|---|
| selection_rate | 0.1108 | 0.8591 | YES |
| true_positive_rate | 0.1007 | 0.8843 | YES |
| false_positive_rate | 0.1000 | 0.8276 | YES |

### By-group metrics

| Group | false_positive_rate | selection_rate | true_positive_rate |
|---|---:|---:|---:|
| female | 0.4800 | 0.6753 | 0.7692 |
| male | 0.5800 | 0.7861 | 0.8699 |

## Fairness by age group

### Disparity summary

| Metric | Difference (max-min) | Ratio (min/max) | Flag |
|---|---:|---:|---|
| selection_rate | 0.3902 | 0.6098 | YES |
| true_positive_rate | 0.3636 | 0.6364 | YES |
| false_positive_rate | 0.6190 | 0.3810 | YES |

### By-group metrics

| Group | false_positive_rate | selection_rate | true_positive_rate |
|---|---:|---:|---:|
| adult | 0.6061 | 0.8000 | 0.8627 |
| middle | 0.3810 | 0.7031 | 0.8605 |
| senior | 1.0000 | 1.0000 | 1.0000 |
| young | 0.5789 | 0.6098 | 0.6364 |

## Findings

The following disparities exceeded the configured threshold:

- **sex**: **selection_rate** difference = **0.1108**
- **sex**: **true_positive_rate** difference = **0.1007**
- **sex**: **false_positive_rate** difference = **0.1000**
- **age_group**: **selection_rate** difference = **0.3902**
- **age_group**: **true_positive_rate** difference = **0.3636**
- **age_group**: **false_positive_rate** difference = **0.6190**
